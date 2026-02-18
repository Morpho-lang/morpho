/** @file help.c
 *  @author T J Atherton
 *
 *  @brief Morpho help
*/

#include <ctype.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <varray.h>

#include "classes.h"
#include "help.h"
#include "resources.h"
#include "common.h"
#include "dictionary.h"
#include "list.h"

#include "error.h"
#include "lex.h"
#include "parse.h"
#include "file.h"
#include "memory.h"

#ifdef MORPHO_INCLUDE_HELP

/* **********************************************************************
 * Data structures and static data
 * ********************************************************************** */

DEFINE_VARRAY(md_block, md_block);
DEFINE_VARRAY(md_file, md_file);
DEFINE_VARRAY(md_topic, md_topic);

static varray_md_file s_files;
static varray_md_topic s_topics;
static dictionary s_names;  /* name (interned on build) -> topic index or list of indices */

/** The interactive help system uses a collection of Markdown files, located in
 *  MORPHO_HELPFOLDER, that define available topics. Help files are all
 *  valid Markdown, although only a subset is used, and the help system interprets
 *  Markdown syntax in special ways:
 *
 *  Headers defined with #, ##, etc are used to identify discrete topics.
 *  Successive levels of header are used to create subtopics.
 *
 *  Link definitions are used to include metadata:
 *
 *  [tag]: # (<TAG>)      is used to define additional synonyms for the topic.
 *
 *  The help system also recognizes code blocks etc. */

/* **********************************************************************
 * Markdown lexer
 * **********************************************************************
 * Token table is sorted for longest-match (e.g. ### before ## before #).
 * The preprocessor captures any run of characters that does not match a
 * defined token as a single MD_TEXT token. Whitespace is not skipped,
 * so newlines and indentation are visible to the parser. */

enum {
    MD_TEXT,
    MD_HASH,
    MD_HASH2,
    MD_HASH3,
    MD_COLON,
    MD_LEFTPAREN,
    MD_RIGHTPAREN,
    MD_LEFTSQUAREBRACE,
    MD_RIGHTSQUAREBRACE,
    MD_BACKTICK,
    MD_ASTERISK,
    MD_PLUS,
    MD_DASH,
    MD_UNDERSCORE,
    MD_ASTERISK2,
    MD_UNDERSCORE2,
    MD_ASTERISK3,
    MD_UNDERSCORE3,
    MD_THEMATIC_BREAK,
    MD_FOURSPACES,
    MD_TAB,
    MD_NEWLINE,
    MD_EOF
};

bool md_lexnewline(lexer *l, token *tok, error *err) {
    lex_newline(l);
    return true;
}


tokendefn mdtokens[] = {
    { "#",          MD_HASH                     , NULL },
    { "##",         MD_HASH2                    , NULL },
    { "###",        MD_HASH3                    , NULL },
    { ":",          MD_COLON                    , NULL },
    { "(",          MD_LEFTPAREN                , NULL },
    { ")",          MD_RIGHTPAREN               , NULL },
    { "[",          MD_LEFTSQUAREBRACE          , NULL },
    { "]",          MD_RIGHTSQUAREBRACE         , NULL },
    { "`",          MD_BACKTICK                 , NULL },
    { "*",          MD_ASTERISK                 , NULL },
    { "+",          MD_PLUS                     , NULL },
    { "-",          MD_DASH                     , NULL },
    { "_",          MD_UNDERSCORE               , NULL },
    { "**",         MD_ASTERISK2                , NULL },
    { "__",         MD_UNDERSCORE2              , NULL },
    { "***",        MD_ASTERISK3                , NULL },
    { "___",        MD_UNDERSCORE3              , NULL },
    { "    ",       MD_FOURSPACES               , NULL },
    { "\t",         MD_TAB                      , NULL },
    { "\r\n",       MD_NEWLINE                  , md_lexnewline },
    { "\n",         MD_NEWLINE                  , md_lexnewline },
    { "\r",         MD_NEWLINE                  , md_lexnewline },
    { "",           TOKEN_NONE                  , NULL }
};

/** Check if a character is ASCII punctuation (CommonMark spec 2.4: escapable with backslash). */
static bool md_isasciipunct(char c) {
    return (unsigned char)c <= 0x7F && ispunct((unsigned char)c);
}

/** Check if current position is a thematic break and record it as MD_THEMATIC_BREAK token.
 * Returns true if a thematic break token was recorded, false otherwise. */
static bool md_lexthematicbreak(lexer *l, token *tok) {
    char c = lex_peek(l);
    if (c != '-' && c != '*' && c != '_') return false;
    
    int count = 0; // Count consecutive matching characters using peekahead (must be 3+)
    while (lex_peekahead(l, count) == c && count < 100) count++; // reasonable limit
    if (count < 3) return false;
    
    int spaces = 0; // Check for optional spaces after break characters
    while (lex_peekahead(l, count + spaces) == ' ') spaces++;
    
    char next = lex_peekahead(l, count + spaces); // Must be followed by newline or EOF
    if (next != '\n' && next != '\r' && next != '\0') return false;
    
    size_t total_len = (size_t)count + (size_t)spaces; // Calculate total length to advance
    if (next == '\r' && lex_peekahead(l, count + spaces + 1) == '\n') {
        total_len += 2; // \r\n
    } else if (next == '\n' || next == '\r') {
        total_len += 1; // \n or \r
    }
    
    lex_advanceby(l, total_len); // Advance by total length and record token
    lex_recordtoken(l, MD_THEMATIC_BREAK, tok);
    return true;
}

/** Lexer token preprocessor function */
bool md_lexpreprocess(lexer *l, token *tok, error *err) {
    if (md_lexthematicbreak(l, tok)) return true; // Check for thematic break before processing other tokens
    
    while (!lex_identifytoken(l, false, NULL) && !lex_isatend(l)) {
        char c = lex_peek(l);
        if (c == '\\') {
            char next = lex_peekahead(l, 1);
            if (md_isasciipunct(next)) {
                lex_advance(l); // Skip \ and include punctuation as literal
                lex_advance(l);
                continue;
            } else if (next == '\n' || next == '\r' || next == '\0') {
                lex_advance(l); // Skip \; newline will be tokenized as MD_NEWLINE in next iteration
                continue;
            }
            // Backslash before non-punctuation: include \ as literal
        }
        lex_advance(l);
    }
    if (l->current > l->start) {
        lex_recordtoken(l, MD_TEXT, tok);
        return true;
    }
    return false;
}

/* -------------------------------------------------------
 * Lexer initialization
 * ------------------------------------------------------- */

void help_initializemdlexer(lexer *l, const char *src) {
    lex_init(l, src, 0);
    lex_settokendefns(l, mdtokens);
    lex_setprefn(l, md_lexpreprocess);
    lex_setwhitespacefn(l, NULL);
    lex_seteof(l, MD_EOF);
}

/* -------------------------------------------------------
 * Markdown parse table (for inline syntax)
 * ------------------------------------------------------- */

/** Parses inline code span. Content is ignored until closing backtick. */
bool md_parseinlinecode(parser *p, void *out) {
    while (!parse_checktoken(p, MD_BACKTICK) && !parse_checktoken(p, MD_EOF))
        parse_advance(p);
    if (!parse_checktokenadvance(p, MD_BACKTICK)) {
        parse_error(p, false, MD_UNCLOSEDINLINECODE);
        return false;
    }
    return true;
}

bool md_parsebold(parser *p, void *out) {
    while (parse_checktokenadvance(p, MD_TEXT));
    if (parse_checktoken(p, MD_ASTERISK2) ||
        parse_checktoken(p, MD_UNDERSCORE2)) parse_advance(p);
    return true;
}

/** Parses italic: called after consuming opening * or _; consumes TEXT then closing delimiter. */
bool md_parseitalic(parser *p, void *out) {
    tokentype delim = p->previous.type; // MD_ASTERISK or MD_UNDERSCORE
    while (parse_checktokenadvance(p, MD_TEXT));
    if (!parse_checktokenadvance(p, delim)) {
        parse_error(p, false, MD_UNCLOSEDITALIC);
        return false;
    }
    return true;
}

parserule md_rules[] = {
    PARSERULE_PREFIX(MD_ASTERISK,        md_parseitalic       ),
    PARSERULE_PREFIX(MD_UNDERSCORE,      md_parseitalic       ),
    PARSERULE_PREFIX(MD_ASTERISK2,       md_parsebold         ),
    PARSERULE_PREFIX(MD_UNDERSCORE2,     md_parsebold         ),
    PARSERULE_PREFIX(MD_BACKTICK,        md_parseinlinecode   ),
    PARSERULE_UNUSED(TOKEN_NONE)
};

/* -------------------------------------------------------
 * Parse output: build AST (blocks + topic list)
 * ------------------------------------------------------- */

/** Output context for the parser: current file, topic list, and base for span offsets. */
typedef struct {
    md_file *file;
    varray_md_topic *topics;
    int file_index;
    int current_header_level; // 1–3, set before md_parseheader
    int current_topic_index;  // Index of most recently created topic, or -1
} md_parseout;

/* Forward declarations */
static void md_push_simpleblock(parser *p, md_parseout *out, size_t block_start, md_blocktype type);
static void md_push_blank(parser *p, md_parseout *out);

static void md_push_header(parser *p, md_parseout *out, size_t block_start, size_t title_start, size_t title_len);
static void md_push_paragraph(parser *p, md_parseout *out, size_t block_start);
static void md_push_code(parser *p, md_parseout *out, size_t block_start);
static void md_push_list(parser *p, md_parseout *out, size_t block_start);
static void md_push_link(parser *p, md_parseout *out, size_t block_start, size_t label_start, size_t label_len, size_t target_start, size_t target_len);

/** Span from the token we just consumed (p->previous). */
static md_span md_span_previous(parser *p, const char *base) {
    md_span s;
    s.start = (size_t)(p->previous.start - base);
    s.length = p->previous.length;
    return s;
}

/** End position of the token we just consumed. */
static size_t md_end_previous(parser *p, const char *base) {
    return (size_t)(p->previous.start - base) + p->previous.length;
}

/** Clamp span length to fit within source length. Returns clamped length. */
static size_t md_clamp_span(size_t start, size_t len, size_t src_len) {
    if (start >= src_len) return 0;
    if (start + len > src_len) return src_len - start;
    return len;
}

/* -------------------------------------------------------
 * Markdown parse rules
 * ------------------------------------------------------- */

/** Consume newline or require EOF; return true if valid line end. */
static bool md_parselineend(parser *p) {
    if (parse_checktoken(p, MD_NEWLINE)) return parse_advance(p);
    return parse_checktoken(p, MD_EOF);
}

/** Check if a token type is a 'textual' token (incl. * _ for italic, ** __ for bold, ` for code) */
tokentype _inlinetokens[] = {
    MD_TEXT, MD_COLON, MD_BACKTICK, MD_LEFTPAREN, MD_RIGHTPAREN,
    MD_ASTERISK, MD_UNDERSCORE, MD_ASTERISK2, MD_UNDERSCORE2
};
int _ninlinetokens = sizeof(_inlinetokens)/sizeof(tokentype);

/** Token types that can appear literally in paragraph/list text (consumed without inline rules). */
tokentype _literalintext[] = {
    MD_HASH, MD_HASH2, MD_HASH3, MD_LEFTSQUAREBRACE, MD_RIGHTSQUAREBRACE,
    MD_FOURSPACES, MD_TAB, MD_PLUS, MD_DASH
};
int _nliteralintext = sizeof(_literalintext)/sizeof(tokentype);

/** True if current token can be consumed as text (MD_TEXT or inline formatting tokens). */
bool md_checktexttoken(parser *p) {
    return parse_checktokenmulti(p, _ninlinetokens, _inlinetokens);
}

/** True if current token can be consumed as literal in paragraph/list (no prefix rule). */
static bool md_checkliteralintext(parser *p) {
    return parse_checktokenmulti(p, _nliteralintext, _literalintext);
}

/** Consume text content tokens (MD_TEXT and literal tokens like - : etc.). 
 * If apply_inline_rules is true, applies inline formatting rules (bold/italic/code) to MD_TEXT tokens.
 * Returns true if we should continue consuming (more text/literal available), false if we hit non-text/non-literal. */
static bool md_consumetextcontent(parser *p, bool apply_inline_rules, void *out) {
    if (md_checktexttoken(p)) {
        parse_advance(p);
        if (apply_inline_rules) {
            parserule *rule = parse_getrule(p, p->previous.type);
            if (rule && rule->prefix && !rule->prefix(p, out)) return false;
        }
        return true;
    } else if (md_checkliteralintext(p)) {
        parse_advance(p);
        return true;
    }
    return false;
}

/** Parses text written in markdown; stops at a non-textual token. Line ends with NEWLINE or EOF. */
bool md_parsetext(parser *p, void *out) {
    md_parseout *ctx = (md_parseout *) out;
    const char *base = ctx->file->source;
    size_t block_start = (size_t)(p->current.start - base);

    for (;;) {
        while (md_consumetextcontent(p, true, out));
        if (md_parselineend(p)) break;
        parse_error(p, true, MD_EXPECTLINEEND);
        return false;
    }
    md_push_paragraph(p, ctx, block_start);
    return true;
}

/** Parses a markdown header (title then newline or EOF). Accepts MD_TEXT and literal tokens (e.g. - for hyphen) for the title. */
bool md_parseheader(parser *p, void *out) {
    md_parseout *ctx = (md_parseout *) out;
    const char *base = ctx->file->source;
    size_t block_start = (size_t)(p->previous.start - base); // # token already consumed

    if (!parse_checktokenadvance(p, MD_TEXT)) {
        parse_error(p, false, MD_EXPECTHEADERTEXT);
        return false;
    }
    size_t title_start = (size_t)(p->previous.start - base);
    size_t title_len = p->previous.length;
    while (md_consumetextcontent(p, false, out)) {
        title_len = (size_t)((p->previous.start - base) + p->previous.length - title_start);
    }

    if (!md_parselineend(p)) {
        parse_error(p, true, MD_EXPECTLINEEND);
        return false;
    }
    md_push_header(p, ctx, block_start, title_start, title_len);
    return true;
}

/** Parses markdown code block (indented lines until newline or EOF). */
bool md_parsecode(parser *p, void *out) {
    md_parseout *ctx = (md_parseout *) out;
    size_t block_start = (size_t)(p->previous.start - ctx->file->source); // 4 spaces / tab already consumed

    while (!parse_checktoken(p, MD_NEWLINE) && !parse_checktoken(p, MD_EOF)) parse_advance(p);
    if (!md_parselineend(p)) {
        parse_error(p, true, MD_EXPECTLINEEND);
        return false;
    }
    md_push_code(p, ctx, block_start);
    return true;
}

/** Parses a thematic break (---, ***, or ___). Token already consumed by lexer (includes newline). */
static bool md_parsethematicbreak(parser *p, void *out) {
    md_parseout *ctx = (md_parseout *) out;
    const char *base = ctx->file->source;
    size_t block_start = (size_t)(p->previous.start - base);
    md_push_simpleblock(p, ctx, block_start, MD_BLOCK_THEMATIC_BREAK);
    return true;
}

/** Parses a markdown list (list item line; records as list block). */
bool md_parselist(parser *p, void *out) {
    md_parseout *ctx = (md_parseout *) out;
    const char *base = ctx->file->source;
    size_t block_start = (size_t)(p->previous.start - base); // *, +, or - already consumed

    for (;;) {
        while (md_consumetextcontent(p, true, out));
        if (md_parselineend(p)) break;
        parse_error(p, true, MD_EXPECTLINEEND);
        return false;
    }
    md_push_list(p, ctx, block_start);
    return true;
}

/** Parses the rest of a link definition after ]:  (e.g. " # (target)" or " # (subtopics)"). Consumes until newline or EOF. */
bool md_parseurl(parser *p, void *out) {
    while (!parse_checktoken(p, MD_NEWLINE) && !parse_checktoken(p, MD_EOF)) {
        parse_advance(p);
    }
    if (parse_checktoken(p, MD_NEWLINE)) return parse_advance(p);
    return true; // EOF is valid end of link line
}

/** Parses inline link [text](url)... after [ and label and ] are consumed. Consumes ( url ) and rest of line, pushes paragraph. */
static bool md_parseinlinelink(parser *p, void *out, size_t block_start) {
    md_parseout *ctx = (md_parseout *) out;
    parse_advance(p); // consume (
    while (!parse_checktoken(p, MD_RIGHTPAREN) && !parse_checktoken(p, MD_NEWLINE) && !parse_checktoken(p, MD_EOF))
        parse_advance(p);
    if (!parse_checktokenadvance(p, MD_RIGHTPAREN)) {
        parse_error(p, false, MD_LINKEXPECTBRACKET);
        return false;
    }
    while (!parse_checktoken(p, MD_NEWLINE) && !parse_checktoken(p, MD_EOF))
        parse_advance(p); // consume rest of line (e.g. trailing ".")
    if (!md_parselineend(p)) {
        parse_error(p, true, MD_EXPECTLINEEND);
        return false;
    }
    md_push_paragraph(p, ctx, block_start);
    return true;
}

static bool help_createalias(const char *base, size_t label_start, size_t label_len, size_t target_start, size_t target_len, int topic_index);

/** Parses [label]: target (link def) or [text](url) an inline link. Caller has already consumed [. */
bool md_parselink(parser *p, void *out) {
    md_parseout *ctx = (md_parseout *) out;
    const char *base = ctx->file->source;
    size_t block_start = (size_t)(p->previous.start - base);

    if (!parse_checktokenadvance(p, MD_TEXT)) { // Parse label text
        parse_error(p, false, MD_LINKEXPECTTEXT);
        return false;
    }
    size_t label_start = (size_t)(p->previous.start - base);
    size_t label_len = p->previous.length;

    if (!parse_checktokenadvance(p, MD_RIGHTSQUAREBRACE)) { // Parse closing ]
        parse_error(p, false, MD_LINKEXPECTBRACKET);
        return false;
    }
    if (parse_checktoken(p, MD_LEFTPAREN)) return md_parseinlinelink(p, out, block_start); // Inline link [text](url)

    if (!parse_checktokenadvance(p, MD_COLON)) { // Parse : for link definition
        parse_error(p, false, MD_LINKEXPECTCOLON);
        return false;
    }
    size_t target_start = (size_t)(p->current.start - base);
    if (!md_parseurl(p, out)) { // Parse target URL until newline/EOF
        parse_error(p, true, MD_EXPECTLINEEND);
        return false;
    }
    size_t target_end = md_end_previous(p, base);
    size_t target_len = target_end > target_start ? (size_t)(target_end - target_start) : 0;
    
    // Check if this is a tag link definition and create alias if so
    help_createalias(base, label_start, label_len, target_start, target_len, ctx->current_topic_index);
    
    md_push_link(p, ctx, block_start, label_start, label_len, target_start, target_len);
    return true;
}

/** Parse a markdown 'block' */
bool md_parseblock(parser *p, void *out) {
    md_parseout *ctx = (md_parseout *) out;
    // Dispatch by first token. Check block-start alternatives first.
    if (parse_checktokenadvance(p, MD_HASH)) {
        ctx->current_header_level = 1;
        return md_parseheader(p, out);
    } else if (parse_checktokenadvance(p, MD_HASH2)) {
        ctx->current_header_level = 2;
        return md_parseheader(p, out);
    } else if (parse_checktokenadvance(p, MD_HASH3)) {
        ctx->current_header_level = 3;
        return md_parseheader(p, out);
    } else if (parse_checktokenadvance(p, MD_FOURSPACES) ||
               parse_checktokenadvance(p, MD_TAB)) {
        return md_parsecode(p, out);
    } else if (parse_checktokenadvance(p, MD_THEMATIC_BREAK)) {
        return md_parsethematicbreak(p, out);
    } else if (parse_checktoken(p, MD_ASTERISK) ||
               parse_checktoken(p, MD_PLUS) ||
               parse_checktoken(p, MD_DASH)) {
        // CommonMark: list marker must be followed by space or tab; else treat as paragraph.
        char c = lex_peek(p->lex);
        if (c == ' ' || c == '\t') {
            parse_advance(p);
            return md_parselist(p, out);
        }
        return md_parsetext(p, out);
    } else if (parse_checktokenadvance(p, MD_LEFTSQUAREBRACE)) {
        return md_parselink(p, out);
    } else if (parse_checktokenadvance(p, MD_NEWLINE)) { // blank line
        md_push_blank(p, ctx);
        return true;
    } else if (parse_checktoken(p, MD_EOF)) {
        return true; // let outer loop exit
    } else if (md_checktexttoken(p)) {
        return md_parsetext(p, out);
    } else {
        parse_error(p, false, MD_UNEXPECTEDTOKEN);
        return false;
    }
}

/** Base markdown parse type */
bool md_parse(parser *p, void *out) {
    while (!parse_checktoken(p, MD_EOF)) {
        PARSE_CHECK(md_parseblock(p, out));
    }
    
    return true;
}

/* -------------------------------------------------------
 * Initialize a Markdown parser
 * ------------------------------------------------------- */

/** Initializes a parser to parse Markdown (help format). */
void help_initializemdparser(parser *p, lexer *l, error *err, void *out) {
    parse_init(p, l, err, out);
    parse_setbaseparsefn(p, md_parse);
    parse_setparsetable(p, md_rules);
    parse_setskipnewline(p, false, TOKEN_NONE);
}

/* **********************************************************************
 * Markdown AST: block and file life cycle
 * ********************************************************************** */

void md_block_clear(md_block *b) {
    varray_intclear(&b->children);
}

void md_file_init(md_file *f) {
    f->source = NULL;
    f->sourcelen = 0;
    f->filename = NULL;
    varray_md_blockinit(&f->blocks);
}

void md_file_clear(md_file *f) {
    if (f->source) MORPHO_FREE(f->source);
    f->source = NULL;
    f->sourcelen = 0;
    if (f->filename) MORPHO_FREE(f->filename);
    f->filename = NULL;
    for (unsigned int i = 0; i < f->blocks.count; i++) md_block_clear(&f->blocks.data[i]);
    varray_md_blockclear(&f->blocks);
}

/* -------------------------------------------------------
 * Name index (single dict: name/alias -> index or List of indices)
 * ------------------------------------------------------- */

static bool help_name_referenced_by(value name, unsigned int keep_count) {
    for (unsigned int j = 0; j < keep_count; j++)
        if (MORPHO_ISSAME(s_topics.data[j].name, name)) return true;
    return false;
}

/** Remove topic_index from s_names (rollback). Frees the name key only if the entry is removed and no topic in [0, keep_count) has that name. */
static void help_nameindex_remove(int topic_index, unsigned int keep_count) {
    if (topic_index < 0 || (unsigned int) topic_index >= s_topics.count) return;
    value name = s_topics.data[topic_index].name;
    if (!MORPHO_ISOBJECT(name)) return;
    value v;
    if (!dictionary_get(&s_names, name, &v)) return;
    bool remove_entry = false;
    if (MORPHO_ISINTEGER(v)) {
        if (MORPHO_GETINTEGERVALUE(v) != topic_index) return;
        remove_entry = true;
    } else if (MORPHO_ISLIST(v)) {
        list_remove(MORPHO_GETLIST(v), MORPHO_INTEGER(topic_index));
        remove_entry = (list_length(MORPHO_GETLIST(v)) == 0);
    }
    if (remove_entry) {
        if (MORPHO_ISLIST(v)) morpho_freeobject(v);
        dictionary_remove(&s_names, name);
        if (!help_name_referenced_by(name, keep_count)) morpho_freeobject(name);
    }
}

/** Add topic_index under name (buf, len). Look up with static key; if present use existing key and update value, else allocate and insert. Returns canonical name value. */
static value help_nameindex_add(const char *buf, size_t len, int topic_index) {
    objectstring key_obj = MORPHO_STATICSTRINGWITHLENGTH(buf, len);
    value name = MORPHO_OBJECT(&key_obj), v;
    value key = dictionary_getkey(&s_names, name, &v);
    if (MORPHO_ISNIL(key)) {
        key = object_stringfromcstring(buf, len);
        if (!MORPHO_ISSTRING(key)) return MORPHO_NIL;
        if (!dictionary_insert(&s_names, key, MORPHO_INTEGER(topic_index))) {
            morpho_freeobject(key);
            return MORPHO_NIL;
        }
    } else {
        if (MORPHO_ISNIL(v)) {
            dictionary_insert(&s_names, key, MORPHO_INTEGER(topic_index));
        } else if (MORPHO_ISINTEGER(v)) {
            if (MORPHO_GETINTEGERVALUE(v) == topic_index) return key;
            value new[2] = { v, MORPHO_INTEGER(topic_index) };
            objectlist *list = object_newlist(2, new);
            if (list) {
                if (!dictionary_insert(&s_names, key, MORPHO_OBJECT(list)))
                    morpho_freeobject(MORPHO_OBJECT(list));
            }
        } else if (MORPHO_ISLIST(v)) list_append(MORPHO_GETLIST(v), MORPHO_INTEGER(topic_index));
    } 
    return key;
}

/** Get first topic index from dict value (integer or first element of list). Returns -1 if not found or invalid. */
static int help_indexvalue_first(value v) {
    if (MORPHO_ISINTEGER(v)) return MORPHO_GETINTEGERVALUE(v);
    if (MORPHO_ISLIST(v)) {
        objectlist *list = MORPHO_GETLIST(v);
        if (list_length(list) == 0) return -1;
        value first;
        if (list_getelement(list, 0, &first) && MORPHO_ISINTEGER(first))
            return MORPHO_GETINTEGERVALUE(first);
    }
    return -1;
}

/** Fill indices[] from dict value (integer or list). Returns number of indices written (at most max). */
static int help_indexvalue_collect(value v, int indices[], int max) {
    if (MORPHO_ISINTEGER(v)) {
        if (max > 0) indices[0] = MORPHO_GETINTEGERVALUE(v);
        return 1;
    } else if (MORPHO_ISLIST(v)) {
        objectlist *list = MORPHO_GETLIST(v);
        unsigned int list_len = list_length(list);
        int written = 0;
        for (unsigned int i = 0; i < list_len && written < max; i++) {
            value el;
            if (list_getelement(list, (int) i, &el) && MORPHO_ISINTEGER(el)) {
                indices[written++] = MORPHO_GETINTEGERVALUE(el);
            }
        }
        return written;
    }
    return 0;
}

int help_findtopic(const char *name) {
    objectstring key_str = MORPHO_STATICSTRING(name);
    value key = MORPHO_OBJECT(&key_str);
    value v;
    if (!dictionary_get(&s_names, key, &v)) return -1;
    return help_indexvalue_first(v);
}

int help_findallbyname(const char *name, int indices[], int max) {
    objectstring key_str = MORPHO_STATICSTRING(name);
    value key = MORPHO_OBJECT(&key_str);
    value v;
    if (!dictionary_get(&s_names, key, &v)) return 0;
    return help_indexvalue_collect(v, indices, max);
}

/** Find child of parent topic with given name. Returns topic index or -1. */
static int help_findchild(int parent_idx, const char *name) {
    objectstring key_str = MORPHO_STATICSTRING(name);
    value key = MORPHO_OBJECT(&key_str);
    for (unsigned int j = 0; j < s_topics.count; j++) {
        if (s_topics.data[j].parent_topic != parent_idx) continue;
        if (MORPHO_ISEQUAL(s_topics.data[j].name, key)) return (int) j;
    }
    return -1;
}

/* -------------------------------------------------------
 * Block construction (spans, init, push)
 * ------------------------------------------------------- */

/** End offset of the block from the last consumed token. */
static size_t md_block_end(parser *p, const char *base) {
    return md_end_previous(p, base);
}

/** Set a span to (start, length). */
static void md_span_set(md_span *s, size_t start, size_t length) {
    s->start = start;
    s->length = length;
}

/** Trim leading and trailing whitespace from (start, len) in place. */
static void md_trimspan(const char *base, size_t *start, size_t *len) {
    while (*len && (unsigned char) base[*start] <= ' ') { (*start)++; (*len)--; }
    while (*len && (unsigned char) base[*start + *len - 1] <= ' ') (*len)--;
}

/** Lowercase len bytes from src into buf and null-terminate. Caller must provide buf of size at least len + 1. */
static void help_lowercase_into(const char *src, size_t len, char *buf) {
    for (size_t i = 0; i < len; i++) buf[i] = (char) tolower((unsigned char) src[i]);
    buf[len] = '\0';
}

/** Initialize common block fields (type, span, parent, children). */
static void md_block_init(md_block *b, md_blocktype type, size_t block_start, size_t block_end) {
    b->type = type;
    md_span_set(&b->span, block_start, block_end - block_start);
    b->parent = -1;
    varray_intinit(&b->children);
}

/** Set header block fields (level and title span). */
static void md_block_set_header(md_block *b, int level, size_t title_start, size_t title_len) {
    b->as.header.level = level;
    md_span_set(&b->as.header.title, title_start, title_len);
}

/** Set link-def block fields (label and target spans). */
static void md_block_set_link_def(md_block *b, size_t label_start, size_t label_len, size_t target_start, size_t target_len) {
    md_span_set(&b->as.link_def.label, label_start, label_len);
    md_span_set(&b->as.link_def.target, target_start, target_len);
}

/** Push a simple block (paragraph, code, list); type and span only. */
static void md_push_simpleblock(parser *p, md_parseout *out, size_t block_start, md_blocktype type) {
    const char *base = out->file->source;
    md_block b;
    md_block_init(&b, type, block_start, md_block_end(p, base));
    varray_md_blockwrite(&out->file->blocks, b);
}

static void md_push_header(parser *p, md_parseout *out, size_t block_start, size_t title_start, size_t title_len) {
    const char *base = out->file->source;
    size_t block_end = md_block_end(p, base);

    md_block b;
    md_block_init(&b, MD_BLOCK_HEADER, block_start, block_end);
    md_block_set_header(&b, out->current_header_level, title_start, title_len);
    varray_md_blockwrite(&out->file->blocks, b);

    // Topic: trimmed title span, lowercased Morpho string (stack buffer)
    size_t ns = title_start, nl = title_len;
    md_trimspan(base, &ns, &nl);
    if (nl > MORPHO_MAX_HELPQUERY_LENGTH) {
        out->current_topic_index = -1;
        return;
    }
    char buf[nl + 1];
    help_lowercase_into(base + ns, nl, buf);
    int topic_index = (int) out->topics->count;
    value name = help_nameindex_add(buf, nl, topic_index);
    if (MORPHO_ISNIL(name)) {
        out->current_topic_index = -1;
        return;
    }
    md_topic topic = {
        .name = name,
        .file_index = out->file_index,
        .block_index = out->file->blocks.count - 1,
        .level = out->current_header_level,
        .parent_topic = -1
    };
    varray_md_topicwrite(out->topics, topic);
    out->current_topic_index = topic_index;
}

static void md_push_paragraph(parser *p, md_parseout *out, size_t block_start) {
    md_push_simpleblock(p, out, block_start, MD_BLOCK_PARAGRAPH);
}
static void md_push_code(parser *p, md_parseout *out, size_t block_start) {
    md_push_simpleblock(p, out, block_start, MD_BLOCK_CODE);
}
static void md_push_list(parser *p, md_parseout *out, size_t block_start) {
    md_push_simpleblock(p, out, block_start, MD_BLOCK_LIST);
}

/** Push a blank line as MD_BLOCK_BLANK. Call after consuming the newline token. */
static void md_push_blank(parser *p, md_parseout *out) {
    const char *base = out->file->source;
    size_t block_start = (size_t)(p->previous.start - base);
    md_push_simpleblock(p, out, block_start, MD_BLOCK_BLANK);
}

static void md_push_link(parser *p, md_parseout *out, size_t block_start, size_t label_start, size_t label_len, size_t target_start, size_t target_len) {
    const char *base = out->file->source;
    size_t block_end = md_block_end(p, base);
    md_block b;
    md_block_init(&b, MD_BLOCK_LINK_DEF, block_start, block_end);
    md_block_set_link_def(&b, label_start, label_len, target_start, target_len);
    varray_md_blockwrite(&out->file->blocks, b);
}

/* **********************************************************************
 * Help topic aliases
 * ********************************************************************** */

/** Create an alias from a tag link definition. Inserts alias -> topic index into s_names. */
static bool help_createalias(const char *base, size_t label_start, size_t label_len, size_t target_start, size_t target_len, int topic_index) {
    if (topic_index < 0 || (unsigned int) topic_index >= s_topics.count) return false;
    if (label_len < 3) return false;
    const char *label = base + label_start;
    if (tolower((unsigned char) label[0]) != 't' || tolower((unsigned char) label[1]) != 'a' || tolower((unsigned char) label[2]) != 'g') return false;
    const char *target = base + target_start, *open = NULL, *close = NULL;
    for (const char *p = target; p < target + target_len && !close; p++) {
        if (!open && *p == '(') open = p + 1;
        else if (open && *p == ')') { close = p; break; }
    }
    if (!open || !close || close == open) return false;
    size_t len = (size_t)(close - open);
    if (len > MORPHO_MAX_HELPQUERY_LENGTH) return false;
    char alias_buf[len + 1];
    help_lowercase_into(open, len, alias_buf);
    return !MORPHO_ISNIL(help_nameindex_add(alias_buf, len, topic_index));
}

/* **********************************************************************
 * Help topic lookup and content range
 * ********************************************************************** */

/** Maximum number of query segments (e.g. "System respondsto" -> 2). */
#define MORPHO_HELP_QUERY_MAXSEGMENTS 8
#define MORPHO_HELP_MAX_MULTIMATCH 8
/** Max edit distance to suggest a topic (only suggest if close enough). */
#define MORPHO_HELP_SUGGEST_MAXDIST 3
/** Edit distance return when string too long (no match). */
#define MORPHO_HELP_EDIT_NOMATCH 255

/* -------------------------------------------------------
 * Query parsing
 * ------------------------------------------------------- */

/** Parse query into lowercased segments in qbuf; segs[] points into qbuf. Returns nsegs (0 if none). */
static int help_parsequery(const char *query, char *qbuf, const char *segs[]) {
    // Copy query, lowercase, replace '.' and ' ' with nul
    size_t qlen = 0;
    while (query[qlen] && qlen < MORPHO_MAX_HELPQUERY_LENGTH - 1) {
        char c = query[qlen];
        qbuf[qlen] = (c == '.' || c == ' ') ? '\0' : (char) tolower((unsigned char) c);
        qlen++;
    }
    qbuf[qlen] = '\0';
    // Collect segment start pointers (skip nuls, take runs)
    int nsegs = 0;
    const char *p = qbuf;
    while (nsegs < MORPHO_HELP_QUERY_MAXSEGMENTS && p < qbuf + qlen) {
        while (p < qbuf + qlen && *p == '\0') p++;
        if (p >= qbuf + qlen) break;
        segs[nsegs++] = p;
        while (p < qbuf + qlen && *p != '\0') p++;
    }
    return nsegs;
}

/* -------------------------------------------------------
 * Topic hierarchy (parent, display path, subtopic)
 * ------------------------------------------------------- */

/** Find parent topic index (same file, level < current, block_index < current, largest block_index). */
static int help_findparent(int idx) {
    if (idx < 0 || (unsigned int) idx >= s_topics.count) return -1;
    const md_topic *t = &s_topics.data[idx];
    int level = t->level;
    if (level <= 1) return -1;
    // Same file, strictly earlier block, lower level; keep one with largest block_index
    int parent = -1;
    unsigned int best_block = 0;
    for (unsigned int i = 0; i < s_topics.count; i++) {
        if (s_topics.data[i].file_index != t->file_index) continue;
        if (s_topics.data[i].level >= level) continue;
        if (s_topics.data[i].block_index >= t->block_index) continue;
        if (parent < 0 || s_topics.data[i].block_index > best_block) {
            parent = (int) i;
            best_block = s_topics.data[i].block_index;
        }
    }
    return parent;
}

/** Write full display path for topic (e.g. "System.clock") into buf. */
static void help_topic_displaypath(int idx, char *buf, size_t bufsize) {
    if (!bufsize) return;
    if (idx < 0 || (unsigned int) idx >= s_topics.count) {
        buf[0] = '\0';
        return;
    }
    const md_topic *t = &s_topics.data[idx];
    const char *name = MORPHO_ISSTRING(t->name) ? MORPHO_GETCSTRING(t->name) : NULL;
    if (!name) {
        buf[0] = '\0';
        return;
    }
    int p = (t->level > 1) ? help_findparent(idx) : -1;
    if (p >= 0) {
        help_topic_displaypath(p, buf, bufsize);
        size_t len = strlen(buf);
        if (len + 1 < bufsize) snprintf(buf + len, bufsize - len, ".%s", name);
    } else {
        snprintf(buf, bufsize, "%s", name);
    }
}

/* -------------------------------------------------------
 * Edit distance and suggestions
 * ------------------------------------------------------- */

/** Levenshtein distance between two strings (for "did you mean").
 * If max_dist is provided (> 0), returns early if distance exceeds it (returns max_dist + 1).
 * This allows early exit when we only care about distances within a threshold. */
static unsigned int help_editdistance(const char *a, const char *b, unsigned int max_dist) {
    size_t na = strlen(a), nb = strlen(b);
    if (na == 0) return (unsigned int) nb;
    if (nb == 0) return (unsigned int) na;
    if (na > MORPHO_HELP_EDITMAXLEN || nb > MORPHO_HELP_EDITMAXLEN) return MORPHO_HELP_EDIT_NOMATCH;
    
    // Early exit: if length difference exceeds max_dist, distance must be at least that
    size_t len_diff = (na > nb) ? na - nb : nb - na;
    if (max_dist > 0 && len_diff > max_dist) return max_dist + 1;
    
    // Two-row DP: prev row and current row, swap each iteration
    unsigned int row0[MORPHO_HELP_EDITMAXLEN + 1], row1[MORPHO_HELP_EDITMAXLEN + 1];
    unsigned int *prev = row0, *curr = row1;
    for (size_t j = 0; j <= nb; j++) prev[j] = (unsigned int) j;
    for (size_t i = 1; i <= na; i++) {
        curr[0] = (unsigned int) i;
        for (size_t j = 1; j <= nb; j++) {
            unsigned int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            unsigned int del = prev[j] + 1, ins = curr[j - 1] + 1, sub = prev[j - 1] + cost;
            unsigned int min = (del < ins) ? del : ins;
            curr[j] = (min < sub) ? min : sub;
        }
        // Early exit: if the final column (curr[nb]) exceeds max_dist, we can stop
        if (max_dist > 0 && curr[nb] > max_dist) return max_dist + 1;
        { unsigned int *t = prev; prev = curr; curr = t; }
    }
    return prev[nb];
}

/** Find top-level topic name closest to name; NULL if none within distance. */
static const char *help_findclosesttopic(const char *name) {
    const char *best = NULL;
    unsigned int best_d = MORPHO_HELP_SUGGEST_MAXDIST + 1;
    for (unsigned int i = 0; i < s_topics.count; i++) {
        if (s_topics.data[i].level != 1 || !MORPHO_ISSTRING(s_topics.data[i].name)) continue;
        const char *tn = MORPHO_GETCSTRING(s_topics.data[i].name);
        unsigned int d = help_editdistance(name, tn, best_d - 1);
        if (d < best_d) { best_d = d; best = tn; }
    }
    return best;
}

/** Find subtopic name under parent closest to name; NULL if none within distance. */
static const char *help_findclosestsubtopic(int parent_idx, const char *name) {
    if (parent_idx < 0 || (unsigned int) parent_idx >= s_topics.count) return NULL;
    const char *best = NULL;
    unsigned int best_d = MORPHO_HELP_SUGGEST_MAXDIST + 1;
    for (unsigned int j = 0; j < s_topics.count; j++) {
        if (s_topics.data[j].parent_topic != parent_idx) continue;
        if (!MORPHO_ISSTRING(s_topics.data[j].name)) continue;
        const char *tn = MORPHO_GETCSTRING(s_topics.data[j].name);
        unsigned int d = help_editdistance(name, tn, best_d - 1);
        if (d < best_d) { best_d = d; best = tn; }
    }
    return best;
}

/* -------------------------------------------------------
 * Topic content and rendering
 * ------------------------------------------------------- */

/** Number of blocks that form this topic's content (header + following until next header of any level). */
static unsigned int help_topiccontentnblocks(const md_topic *topic, const md_file *file) {
    unsigned int i = topic->block_index;
    unsigned int n = file->blocks.count;
    // Advance until we hit any next header (so a # topic doesn't swallow all ## sections)
    while (i < n) {
        const md_block *b = &file->blocks.data[i];
        if (b->type == MD_BLOCK_HEADER && i > topic->block_index)
            break;
        i++;
    }
    return i - topic->block_index;
}

/** Fill a help_topic view from resolved topic and file. */
static void help_topicfill(help_topic *out, const md_topic *topic, const md_file *file) {
    out->topic = topic;
    out->file = file;
    out->content_blocks = &file->blocks.data[topic->block_index];
    out->nblocks = help_topiccontentnblocks(topic, file);
}

/** Validate topic and get source pointer/length; return false if invalid. */
static bool help_topicsrc(const help_topic *t, const char **src, size_t *src_len) {
    if (!t || !t->file || !t->file->source) return false;
    *src = t->file->source;
    *src_len = t->file->sourcelen;
    return true;
}

bool help_topictotext(const help_topic *t, varray_char *result) {
    if (!result) return false;
    const char *src;
    size_t src_len;
    if (!help_topicsrc(t, &src, &src_len)) return false;
    for (unsigned int i = 0; i < t->nblocks; i++) {
        const md_block *b = &t->content_blocks[i];
        size_t len = md_clamp_span(b->span.start, b->span.length, src_len);
        if (len == 0) continue;

        // Per-block: header (title only, no #), paragraph/list/code (span), link/blank skip
        switch (b->type) {
            case MD_BLOCK_HEADER: {
                size_t tlen = md_clamp_span(b->as.header.title.start, b->as.header.title.length, src_len);
                const char *p = src + b->as.header.title.start;
                while (tlen && (*p == '#' || (unsigned char)*p <= ' ')) { p++; tlen--; } // strip leading # and space
                if (tlen > 0) {
                    varray_charadd(result, (char *) p, (int) tlen);
                    varray_charwrite(result, '\n');
                }
                break;
            }
            case MD_BLOCK_PARAGRAPH:
            case MD_BLOCK_LIST:
            case MD_BLOCK_CODE:
                varray_charadd(result, (char *) src + b->span.start, (int) len);
                if (len > 0 && src[b->span.start + len - 1] != '\n') varray_charadd(result, "\n", 1);
                break;
            case MD_BLOCK_LINK_DEF:
            case MD_BLOCK_THEMATIC_BREAK:
                break; // omit from plain text (thematic breaks render as blank lines)
            case MD_BLOCK_BLANK:
                varray_charwrite(result, '\n');
                break;
        }
    }
    varray_charwrite(result, '\0');
    return true;
}

bool help_topicrawmd(const help_topic *t, varray_char *result) {
    if (!result) return false;
    const char *src;
    size_t src_len;
    if (!help_topicsrc(t, &src, &src_len)) return false;
    for (unsigned int i = 0; i < t->nblocks; i++) {
        const md_block *b = &t->content_blocks[i];
        size_t len = md_clamp_span(b->span.start, b->span.length, src_len);
        if (len == 0) continue;
        varray_charadd(result, (char *) src + b->span.start, (int) len);
        if (src[b->span.start + len - 1] != '\n') varray_charadd(result, "\n", 1);
    }
    return true;
}

/* **********************************************************************
 * Morpho help files (load and parse)
 * ********************************************************************** */

/** Parse a markdown file and append blocks/topics to s_topics. */
bool help_parse(md_file *file, int file_index) {
    error err;
    error_init(&err);
    md_parseout parseout = {
        .file = file,
        .topics = &s_topics,
        .file_index = file_index,
        .current_header_level = 1,
        .current_topic_index = -1
    };
    lexer l;
    help_initializemdlexer(&l, file->source);
    parser p;
    help_initializemdparser(&p, &l, &err, &parseout);
    bool ok = parse(&p);
    parse_clear(&p);
    lex_clear(&l);
    if (!ok && morpho_checkerror(&err)) {
        fprintf(stderr, "Help parse error [%s]: %s", err.id ? err.id : "?", err.msg);
        bool has_line = (err.line != ERROR_POSNUNIDENTIFIABLE);
        bool has_posn = (err.posn != ERROR_POSNUNIDENTIFIABLE);
        if (has_line || has_posn) {
            fprintf(stderr, " (");
            if (has_line) fprintf(stderr, "line %d", err.line + 1);
            if (has_posn) fprintf(stderr, "%sposition %d", has_line ? ", " : "", err.posn + 1);
            fprintf(stderr, ")");
        }
        if (file->filename) fprintf(stderr, " in %s", file->filename);
        fprintf(stderr, "\n");
    }
    return ok;
}

/** Loads a help file into the AST (appends to s_files and s_topics). */
bool help_load(char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) return false;
    varray_char contents;
    varray_charinit(&contents);
    if (!file_readintovarray(f, &contents)) {
        fclose(f);
        varray_charclear(&contents);
        return false;
    }
    fclose(f);
    size_t len = (contents.count > 0) ? contents.count - 1 : 0; // Keep contents.data; mdfile will own it (freed in md_file_clear). Do not clear contents.
    md_file mdfile;
    md_file_init(&mdfile);
    mdfile.source = contents.data;
    mdfile.sourcelen = len;
    mdfile.filename = morpho_strdup(filename);
    unsigned int n_topics_before = s_topics.count;
    bool ok = help_parse(&mdfile, (int) s_files.count);
    if (ok) {
        varray_md_filewrite(&s_files, mdfile);
    } else {
        md_file_clear(&mdfile);
        while (s_topics.count > n_topics_before) {
            help_nameindex_remove((int) s_topics.count - 1, n_topics_before);
            s_topics.count--;
        }
    }
    return ok;
}

/** Finds and loads all help files; sets parent_topic for each topic. Name index is built incrementally during parse. */
void help_findfiles(void) {
    varray_value files;
    varray_valueinit(&files);
    if (morpho_listresources(MORPHO_RESOURCE_HELP, &files)) {
        for (unsigned int i = 0; i < files.count; i++) {
            if (MORPHO_ISSTRING(files.data[i])) help_load(MORPHO_GETCSTRING(files.data[i]));
        }
        for (unsigned int i = 0; i < files.count; i++) morpho_freeobject(files.data[i]);
    }
    varray_valueclear(&files);
    for (unsigned int i = 0; i < s_topics.count; i++)
        s_topics.data[i].parent_topic = help_findparent(i);
}

/* **********************************************************************
 * Hints for failed queries
 * ********************************************************************** */

/** Append "Topic 'query' not found [. Did you mean 'suggest'?]." to result. suggest is full path or NULL. */
static void help_hintappend(varray_char *result, const char *query, const char *suggest) {
    char buf[MORPHO_HELP_HINTBUFSIZE];
    int n = suggest
        ? snprintf(buf, sizeof(buf), "Topic '%s' not found. Did you mean '%s'?", query, suggest)
        : snprintf(buf, sizeof(buf), "Topic '%s' not found.", query);
    if (n > 0 && (size_t) n < sizeof(buf)) varray_charadd(result, buf, n);
}

/** Append "Topic 'query' had multiple matches: did you mean 'A' or 'B'?" (or "..., 'A', 'B', or 'C'?"). */
static void help_hintappend_multi(varray_char *result, const char *query, int indices[], int count) {
    char buf[MORPHO_HELP_HINTBUFSIZE];
    char path[MORPHO_MAX_HELPQUERY_LENGTH];
    int n = snprintf(buf, sizeof(buf), "Topic '%s' had multiple matches: did you mean ", query);
    // First: "'A'"; middle: ", 'B'"; last: " or 'C'?"
    for (int i = 0; i < count && n < (int) sizeof(buf) - 4; i++) {
        help_topic_displaypath(indices[i], path, sizeof(path));
        if (i == 0) n += snprintf(buf + n, sizeof(buf) - (size_t) n, "'%s'", path);
        else if (i == count - 1) n += snprintf(buf + n, sizeof(buf) - (size_t) n, " or '%s'?", path);
        else n += snprintf(buf + n, sizeof(buf) - (size_t) n, ", '%s'", path);
    }
    if (n > 0 && (size_t) n < sizeof(buf)) varray_charadd(result, buf, n);
}

/** Build a hint for a failed query and append to result (caller may clear result first). */
void help_queryhint(const char *query, varray_char *result) {
    if (!result) return;
    char qbuf[MORPHO_MAX_HELPQUERY_LENGTH];
    const char *segs[MORPHO_HELP_QUERY_MAXSEGMENTS];
    int nsegs = help_parsequery(query, qbuf, segs);
    if (nsegs == 0) {
        varray_charadd(result, MORPHO_HELP_NOTFOUND, strlen(MORPHO_HELP_NOTFOUND));
        varray_charwrite(result, '\0');
        return;
    }
    // Single segment: multi-match hint, or "not found" + closest, or NOTFOUND
    if (nsegs == 1) {
        int multi[MORPHO_HELP_MAX_MULTIMATCH];
        int n = help_findallbyname(segs[0], multi, MORPHO_HELP_MAX_MULTIMATCH);
        if (n >= 2) {
            help_hintappend_multi(result, query, multi, n);
            varray_charwrite(result, '\0');
            return;
        }
        if (n == 0) {
            help_hintappend(result, query, help_findclosesttopic(segs[0]));
            varray_charwrite(result, '\0');
            return;
        }
        varray_charadd(result, MORPHO_HELP_NOTFOUND, strlen(MORPHO_HELP_NOTFOUND));
        varray_charwrite(result, '\0');
        return;
    }
    // Multi-segment: resolve first, then walk subtopics; on first failure suggest path.closest
    int idx = help_findtopic(segs[0]);
    if (idx < 0) {
        help_hintappend(result, query, help_findclosesttopic(segs[0]));
        varray_charwrite(result, '\0');
        return;
    }
    char path[MORPHO_MAX_HELPQUERY_LENGTH];
    int path_len = snprintf(path, sizeof(path), "%s", segs[0]);
    if (path_len < 0 || path_len >= (int) sizeof(path)) path_len = (int) sizeof(path) - 1;

    for (int i = 1; i < nsegs; i++) {
        int next = help_findchild(idx, segs[i]);
        if (next < 0) {
            const char *closest = help_findclosestsubtopic(idx, segs[i]);
            char suggest[MORPHO_MAX_HELPQUERY_LENGTH] = {0};
            if (closest) {
                int needed = path_len + 1 + (int) strlen(closest);
                if (needed < (int) sizeof(suggest))
                    snprintf(suggest, sizeof(suggest), "%s.%s", path, closest);
            }
            help_hintappend(result, query, suggest[0] ? suggest : NULL);
            varray_charwrite(result, '\0');
            return;
        }
        idx = next;
        // Append "." + segs[i] to path for next level
        if (path_len >= (int) sizeof(path) - 1) continue; // No room
        int n = snprintf(path + path_len, (size_t)(sizeof(path) - path_len), ".%s", segs[i]);
        if (n > 0) path_len += (n < (int) sizeof(path) - path_len) ? n : (int) sizeof(path) - 1 - path_len;
    }
    varray_charadd(result, MORPHO_HELP_NOTFOUND, strlen(MORPHO_HELP_NOTFOUND));
    varray_charwrite(result, '\0');
}

/* **********************************************************************
 * Help API
 * ********************************************************************** */

static bool s_help_files_loaded = false;

static void help_ensureloaded(void) {
    if (!s_help_files_loaded) {
        help_findfiles();
        s_help_files_loaded = true;
    }
}

/** Interface to the morpho help system. Query may be "Topic" or "Topic subtopic" / "Topic.subtopic". */
bool morpho_helpastopic(const char *query, help_topic *out) {
    help_ensureloaded();
    char qbuf[MORPHO_MAX_HELPQUERY_LENGTH];
    const char *segs[MORPHO_HELP_QUERY_MAXSEGMENTS];
    int nsegs = help_parsequery(query, qbuf, segs);
    if (nsegs <= 0) return false;

    // Single segment: dict gives 0, 1, or many; if many caller shows "did you mean?"
    int idx;
    if (nsegs == 1) {
        int multi[MORPHO_HELP_MAX_MULTIMATCH];
        int n = help_findallbyname(segs[0], multi, MORPHO_HELP_MAX_MULTIMATCH);
        if (n == 1) {
            idx = multi[0];
        } else if (n == 0) {
            return false;
        } else {
            return false; // multiple matches: caller shows hint
        }
    } else {
        idx = help_findtopic(segs[0]); // first match
        if (idx < 0) return false;
    }

    // Resolve remaining segments by walking children
    for (int i = 1; i < nsegs; i++) {
        idx = help_findchild(idx, segs[i]);
        if (idx < 0) return false;
    }

    const md_topic *topic = &s_topics.data[idx];
    if (topic->file_index < 0 || (unsigned int) topic->file_index >= s_files.count) return false;
    const md_file *file = &s_files.data[topic->file_index];
    if (topic->block_index >= file->blocks.count) return false;
    help_topicfill(out, topic, file);
    return true;
}

bool morpho_helpastext(const char *query, varray_char *result) {
    help_topic t;
    if (morpho_helpastopic(query, &t)) return help_topictotext(&t, result);
    help_queryhint(query, result);
    return false;
}

bool morpho_helpasmd(const char *query, varray_char *result) {
    help_topic t;
    if (morpho_helpastopic(query, &t)) return help_topicrawmd(&t, result);
    help_queryhint(query, result);
    return false;
}

/** Fill out with top-level topic names (level == 1); each name added once (by identity). */
void morpho_helptopics(varray_value *out) {
    if (!out) return;
    help_ensureloaded();
    for (unsigned int i = 0; i < s_topics.count; i++) {
        if (s_topics.data[i].level != 1 || !MORPHO_ISSTRING(s_topics.data[i].name)) continue;
        value name = s_topics.data[i].name;
        unsigned int idx;
        if (varray_valuefindsame(out, name, &idx)) continue;
        varray_valuewrite(out, name);
    }
}

/* **********************************************************************
 * Initialization/finalization
 * ********************************************************************** */

/** @brief Initialize help system */
void help_initialize(void) {
    // Markdown parser errors
    morpho_defineerror(MD_UNCLOSEDITALIC, ERROR_PARSE, MD_UNCLOSEDITALIC_MSG);
    morpho_defineerror(MD_UNCLOSEDINLINECODE, ERROR_PARSE, MD_UNCLOSEDINLINECODE_MSG);
    morpho_defineerror(MD_EXPECTLINEEND, ERROR_PARSE, MD_EXPECTLINEEND_MSG);
    morpho_defineerror(MD_EXPECTHEADERTEXT, ERROR_PARSE, MD_EXPECTHEADERTEXT_MSG);
    morpho_defineerror(MD_LINKEXPECTTEXT, ERROR_PARSE, MD_LINKEXPECTTEXT_MSG);
    morpho_defineerror(MD_LINKEXPECTBRACKET, ERROR_PARSE, MD_LINKEXPECTBRACKET_MSG);
    morpho_defineerror(MD_LINKEXPECTCOLON, ERROR_PARSE, MD_LINKEXPECTCOLON_MSG);
    morpho_defineerror(MD_UNEXPECTEDTOKEN, ERROR_PARSE, MD_UNEXPECTEDTOKEN_MSG);

    varray_md_fileinit(&s_files);
    varray_md_topicinit(&s_topics);
    dictionary_init(&s_names);
    morpho_addfinalizefn(help_finalize);
}

/** @brief Finalization: free all files, topics, and name index. */
void help_finalize(void) {
    for (unsigned int i = 0; i < s_files.count; i++)
        md_file_clear(&s_files.data[i]);
    varray_md_fileclear(&s_files);
    dictionary_freecontents(&s_names, true, true);
    dictionary_clear(&s_names);
    varray_md_topicclear(&s_topics);
}

#endif
