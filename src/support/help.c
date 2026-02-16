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

#include "error.h"
#include "lex.h"
#include "parse.h"
#include "file.h"
#include "memory.h"

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
    MD_FOURSPACES,
    MD_TAB,
    MD_NEWLINE,
    MD_EOF
};

/* Markdown parser error codes (registered in help_initialize) */
#define MD_UNCLOSEDITALIC       "MDUnclItal"
#define MD_UNCLOSEDITALIC_MSG   "Unclosed italic (missing closing * or _)."
#define MD_UNCLOSEDINLINECODE   "MDUnclCode"
#define MD_UNCLOSEDINLINECODE_MSG "Unclosed inline code (missing closing `)."
#define MD_EXPECTLINEEND        "MDExpLnEnd"
#define MD_EXPECTLINEEND_MSG    "Expected end of line."
#define MD_EXPECTHEADERTEXT     "MDExpHdrTxt"
#define MD_EXPECTHEADERTEXT_MSG "Expected header text after #."
#define MD_LINKEXPECTTEXT       "MDLnkExpTxt"
#define MD_LINKEXPECTTEXT_MSG   "Link definition expects label text after [."
#define MD_LINKEXPECTBRACKET    "MDLnkExpBr"
#define MD_LINKEXPECTBRACKET_MSG "Link definition expects ] after label."
#define MD_LINKEXPECTCOLON      "MDLnkExpCol"
#define MD_LINKEXPECTCOLON_MSG  "Link definition expects : after ]."
#define MD_UNEXPECTEDTOKEN      "MDUnexpTok"
#define MD_UNEXPECTEDTOKEN_MSG  "Unexpected markdown token."

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

/** Lexer token preprocessor function */
bool md_lexpreprocess(lexer *l, token *tok, error *err) {
    while (!lex_identifytoken(l, false, NULL) && !lex_isatend(l)) {
        lex_advance(l);
    }
    if (l->current > l->start) {
        lex_recordtoken(l, MD_TEXT, tok);
        return true;
    }
    return false;
}

/* -------------------------------------------------------
 * Initialize a Markdown lexer
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
    tokentype delim = p->previous.type; /* MD_ASTERISK or MD_UNDERSCORE */
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
    int current_header_level;  /* 1–3, set before md_parseheader */
} md_parseout;

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

bool md_checktexttoken(parser *p) {
    return parse_checktokenmulti(p, _ninlinetokens, _inlinetokens);
}

/** True if current token can be consumed as literal in paragraph/list (no prefix rule). */
static bool md_checkliteralintext(parser *p) {
    return parse_checktokenmulti(p, _nliteralintext, _literalintext);
}

/** Parses text written in markdown; stops at a non-textual token. Line ends with NEWLINE or EOF. */
bool md_parsetext(parser *p, void *out) {
    md_parseout *ctx = (md_parseout *) out;
    const char *base = ctx->file->source;
    size_t block_start = (size_t)(p->current.start - base);

    for (;;) {
        /* Consume text tokens, applying inline rules (bold, italic, code) via prefix handlers. */
        while (md_checktexttoken(p)) {
            parse_advance(p);
            parserule *rule = parse_getrule(p, p->previous.type);
            if (rule && rule->prefix && !rule->prefix(p, out)) return false;
        }
        if (md_parselineend(p)) break;
        // Allow block-style tokens to appear literally in prose (e.g. "+" in "minimization + retriangulation").
        if (md_checkliteralintext(p)) {
            parse_advance(p);
            continue;
        }
        parse_error(p, true, MD_EXPECTLINEEND);
        return false;
    }
    md_push_paragraph(p, ctx, block_start);
    return true;
}

/** Parses a markdown header (title then newline or EOF). Accepts one or more MD_TEXT tokens for the title. */
bool md_parseheader(parser *p, void *out) {
    md_parseout *ctx = (md_parseout *) out;
    const char *base = ctx->file->source;
    size_t block_start = (size_t)(p->previous.start - base);  /* # token already consumed */

    if (!parse_checktokenadvance(p, MD_TEXT)) {
        parse_error(p, false, MD_EXPECTHEADERTEXT);
        return false;
    }
    size_t title_start = (size_t)(p->previous.start - base);
    size_t title_len = p->previous.length;
    while (parse_checktokenadvance(p, MD_TEXT)) {
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
    size_t block_start = (size_t)(p->previous.start - ctx->file->source);  /* 4 spaces / tab already consumed */

    while (!parse_checktoken(p, MD_NEWLINE) && !parse_checktoken(p, MD_EOF)) parse_advance(p);
    if (!md_parselineend(p)) {
        parse_error(p, true, MD_EXPECTLINEEND);
        return false;
    }
    md_push_code(p, ctx, block_start);
    return true;
}

/** Parses a markdown list (list item line; records as list block). */
bool md_parselist(parser *p, void *out) {
    md_parseout *ctx = (md_parseout *) out;
    const char *base = ctx->file->source;
    size_t block_start = (size_t)(p->previous.start - base);  /* *, +, or - already consumed */

    for (;;) {
        // Reuse text parsing (inline rules) but record as list block
        while (md_checktexttoken(p)) {
            parse_advance(p);
            parserule *rule = parse_getrule(p, p->previous.type);
            if (rule && rule->prefix) {
                if (!rule->prefix(p, out)) return false;
            }
        }
        if (md_parselineend(p)) break;
        // Allow block-style tokens literally in list item (e.g. "+" in "minimization + retriangulation").
        if (md_checkliteralintext(p)) {
            parse_advance(p);
            continue;
        }
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
    return true; /* EOF is valid end of link line */
}

/** Parses inline link [text](url)... after [ and label and ] are consumed. Consumes ( url ) and rest of line, pushes paragraph. */
static bool md_parseinlinelink(parser *p, void *out, size_t block_start) {
    md_parseout *ctx = (md_parseout *) out;
    parse_advance(p);  /* consume ( */
    while (!parse_checktoken(p, MD_RIGHTPAREN) && !parse_checktoken(p, MD_NEWLINE) && !parse_checktoken(p, MD_EOF))
        parse_advance(p);
    if (!parse_checktokenadvance(p, MD_RIGHTPAREN)) {
        parse_error(p, false, MD_LINKEXPECTBRACKET);
        return false;
    }
    while (!parse_checktoken(p, MD_NEWLINE) && !parse_checktoken(p, MD_EOF))
        parse_advance(p);
    if (!md_parselineend(p)) {
        parse_error(p, true, MD_EXPECTLINEEND);
        return false;
    }
    md_push_paragraph(p, ctx, block_start);
    return true;
}

/** Parses [label]: target (link def) or [text](url) an inline link. Caller has already consumed [. */
bool md_parselink(parser *p, void *out) {
    md_parseout *ctx = (md_parseout *) out;
    const char *base = ctx->file->source;
    size_t block_start = (size_t)(p->previous.start - base);

    if (!parse_checktokenadvance(p, MD_TEXT)) {
        parse_error(p, false, MD_LINKEXPECTTEXT);
        return false;
    }
    size_t label_start = (size_t)(p->previous.start - base);
    size_t label_len = p->previous.length;

    if (!parse_checktokenadvance(p, MD_RIGHTSQUAREBRACE)) {
        parse_error(p, false, MD_LINKEXPECTBRACKET);
        return false;
    }
    if (parse_checktoken(p, MD_LEFTPAREN)) return md_parseinlinelink(p, out, block_start);

    if (!parse_checktokenadvance(p, MD_COLON)) {
        parse_error(p, false, MD_LINKEXPECTCOLON);
        return false;
    }
    size_t target_start = (size_t)(p->current.start - base);
    if (!md_parseurl(p, out)) {
        parse_error(p, true, MD_EXPECTLINEEND);
        return false;
    }
    size_t target_end = md_end_previous(p, base);
    size_t target_len = target_end > target_start ? (size_t)(target_end - target_start) : 0;
    md_push_link(p, ctx, block_start, label_start, label_len, target_start, target_len);
    return true;
}

/** Parse a markdown 'block' */
bool md_parseblock(parser *p, void *out) {
    md_parseout *ctx = (md_parseout *) out;
    /* Dispatch by first token. Check block-start alternatives first. */
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
    } else if (parse_checktoken(p, MD_ASTERISK) ||
               parse_checktoken(p, MD_PLUS) ||
               parse_checktoken(p, MD_DASH)) {
        /* CommonMark: list marker must be followed by space or tab; else treat as paragraph. */
        char c = lex_peek(p->lex);
        if (c == ' ' || c == '\t') {
            parse_advance(p);
            return md_parselist(p, out);
        }
        return md_parsetext(p, out);
    } else if (parse_checktokenadvance(p, MD_LEFTSQUAREBRACE)) {
        return md_parselink(p, out);
    } else if (parse_checktokenadvance(p, MD_NEWLINE)) { /* blank line */
        return true;
    } else if (parse_checktoken(p, MD_EOF)) {
        return true; /* let outer loop exit */
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
 * Markdown AST: definitions and topic list helpers
 * ********************************************************************** */

DEFINE_VARRAY(md_block, md_block);
DEFINE_VARRAY(md_file, md_file);
DEFINE_VARRAY(md_topic, md_topic);

void md_block_clear(md_block *b) {
    varray_intclear(&b->children);
}

void md_file_init(md_file *f) {
    f->source = NULL;
    f->sourcelen = 0;
    varray_md_blockinit(&f->blocks);
}

void md_file_clear(md_file *f) {
    if (f->source) MORPHO_FREE(f->source);
    f->source = NULL;
    f->sourcelen = 0;
    for (unsigned int i = 0; i < f->blocks.count; i++) md_block_clear(&f->blocks.data[i]);
    varray_md_blockclear(&f->blocks);
}

int md_topic_compare(const void *a, const void *b) {
    const md_topic *ta = (const md_topic *) a;
    const md_topic *tb = (const md_topic *) b;
    return strcmp(ta->name, tb->name);
}

void help_sorttopics(varray_md_topic *topics) {
    if (topics->data && topics->count > 0)
        qsort(topics->data, topics->count, sizeof(md_topic), md_topic_compare);
}

int help_findtopic(varray_md_topic *topics, const char *name) {
    if (!topics->data || topics->count == 0) return -1;
    md_topic key = { .name = (char *) name, .file_index = 0, .block_index = 0, .level = 0, .parent_topic = -1 };
    md_topic *found = (md_topic *) bsearch(&key, topics->data, topics->count, sizeof(md_topic), md_topic_compare);
    return found ? (int) (found - topics->data) : -1;
}

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

    // Topic: trimmed title span, lowercased, for index lookup
    size_t ns = title_start, nl = title_len;
    md_trimspan(base, &ns, &nl);
    char *name = (char *) MORPHO_MALLOC(nl + 1);
    if (name) {
        for (size_t i = 0; i < nl; i++)
            name[i] = (char) tolower((unsigned char) base[ns + i]);
        name[nl] = '\0';
        md_topic topic = {
            .name = name,
            .file_index = out->file_index,
            .block_index = out->file->blocks.count - 1,
            .level = out->current_header_level,
            .parent_topic = -1
        };
        varray_md_topicwrite(out->topics, topic);
    }
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

static void md_push_link(parser *p, md_parseout *out, size_t block_start, size_t label_start, size_t label_len, size_t target_start, size_t target_len) {
    const char *base = out->file->source;
    size_t block_end = md_block_end(p, base);
    md_block b;
    md_block_init(&b, MD_BLOCK_LINK_DEF, block_start, block_end);
    md_block_set_link_def(&b, label_start, label_len, target_start, target_len);
    varray_md_blockwrite(&out->file->blocks, b);
}

/* **********************************************************************
 * Help topic lookup and content range
 * ********************************************************************** */

static varray_md_file s_files;
static varray_md_topic s_topics;

/** Maximum number of query segments (e.g. "System respondsto" -> 2). */
#define MORPHO_HELP_QUERY_MAXSEGMENTS 8
/** Max edit distance to suggest a topic (only suggest if close enough). */
#define MORPHO_HELP_SUGGEST_MAXDIST 3
/** Edit distance return when string too long (no match). */
#define MORPHO_HELP_EDIT_NOMATCH 255

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
    if (!bufsize || idx < 0 || (unsigned int) idx >= s_topics.count) { if (bufsize) buf[0] = '\0'; return; }
    const md_topic *t = &s_topics.data[idx];
    if (!t->name) { buf[0] = '\0'; return; }
    int p = (t->level > 1) ? help_findparent(idx) : -1;
    if (p >= 0) {
        // Recurse for parent path, then append ".name"
        help_topic_displaypath(p, buf, bufsize);
        size_t len = strlen(buf);
        if (len + 1 < bufsize) snprintf(buf + len, bufsize - len, ".%s", t->name);
    } else {
        // Top-level or no parent: just the topic name
        size_t n = strlen(t->name);
        if (n >= bufsize) n = bufsize - 1;
        memcpy(buf, t->name, n);
        buf[n] = '\0';
    }
}

/** Find all topic indices with given name. Fills indices[] up to max, returns count. */
#define MORPHO_HELP_MAX_MULTIMATCH 8
static int help_findallbyname(const char *name, int indices[], int max) {
    int n = 0;
    for (unsigned int i = 0; i < s_topics.count && n < max; i++) {
        if (s_topics.data[i].name && strcmp(s_topics.data[i].name, name) == 0)
            indices[n++] = (int) i;
    }
    return n;
}

/** Find topic index by file and block index. Returns -1 if not found. */
static int help_findtopicbyblock(int file_index, unsigned int block_index) {
    for (unsigned int i = 0; i < s_topics.count; i++) {
        if (s_topics.data[i].file_index == file_index && s_topics.data[i].block_index == block_index)
            return (int) i;
    }
    return -1;
}

/** True if header title span (trimmed, lowercased) equals name. */
static bool help_headertitleeq(const char *base, size_t start, size_t len, const char *name) {
    md_trimspan(base, &start, &len);
    while (len && *name && (char) tolower((unsigned char) base[start]) == *name) {
        start++;
        len--;
        name++;
    }
    return len == 0 && *name == '\0';
}

/** Find first subtopic of parent with given name (same file, block after parent, level > parent). Returns topic index or -1. */
static int help_findsubtopic(int parent_idx, const char *name) {
    if (parent_idx < 0 || (unsigned int) parent_idx >= s_topics.count) return -1;
    const md_topic *parent = &s_topics.data[parent_idx];
    if (parent->file_index < 0 || (unsigned int) parent->file_index >= s_files.count) return -1;
    const md_file *file = &s_files.data[parent->file_index];
    const char *base = file->source;
    size_t base_len = file->sourcelen;

    // Scan blocks after parent; stop when we hit same-or-higher level (left section)
    for (unsigned int i = parent->block_index + 1; i < file->blocks.count; i++) {
        const md_block *b = &file->blocks.data[i];
        if (b->type != MD_BLOCK_HEADER) continue;
        if (b->as.header.level <= parent->level) return -1;  // left section
        size_t start = b->as.header.title.start;
        size_t len = b->as.header.title.length;
        if (start + len > base_len) len = base_len - start;
        if (help_headertitleeq(base, start, len, name))
            return help_findtopicbyblock(parent->file_index, i);
    }
    return -1;
}

/** Levenshtein distance between two strings (for "did you mean"). */
static unsigned int help_editdistance(const char *a, const char *b) {
    size_t na = strlen(a), nb = strlen(b);
    if (na == 0) return (unsigned int) nb;
    if (nb == 0) return (unsigned int) na;
    if (na > MORPHO_HELP_EDITMAXLEN || nb > MORPHO_HELP_EDITMAXLEN) return MORPHO_HELP_EDIT_NOMATCH;
    // Two-row DP: prev row and current row, swap each iteration
    unsigned int row0[MORPHO_HELP_EDITMAXLEN + 1], row1[MORPHO_HELP_EDITMAXLEN + 1];
    unsigned int *prev = row0, *curr = row1;
    for (size_t j = 0; j <= nb; j++) prev[j] = (unsigned int) j;
    for (size_t i = 1; i <= na; i++) {
        curr[0] = (unsigned int) i;
        for (size_t j = 1; j <= nb; j++) {
            unsigned int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            unsigned int del = prev[j] + 1, ins = curr[j - 1] + 1, sub = prev[j - 1] + cost;
            curr[j] = (del < ins) ? (del < sub ? del : sub) : (ins < sub ? ins : sub);
        }
        { unsigned int *t = prev; prev = curr; curr = t; }
    }
    return prev[nb];
}

static const char MORPHO_HELP_NOTFOUND[] = "Topic not found.";

/** Find top-level topic name closest to name; NULL if none within distance. */
static const char *help_findclosesttopic(const char *name) {
    const char *best = NULL;
    unsigned int best_d = MORPHO_HELP_SUGGEST_MAXDIST + 1;
    for (unsigned int i = 0; i < s_topics.count; i++) {
        if (s_topics.data[i].level != 1 || !s_topics.data[i].name) continue;
        unsigned int d = help_editdistance(name, s_topics.data[i].name);
        if (d < best_d) { best_d = d; best = s_topics.data[i].name; }
    }
    return best;
}

/** Find subtopic name under parent closest to name; NULL if none within distance. */
static const char *help_findclosestsubtopic(int parent_idx, const char *name) {
    if (parent_idx < 0 || (unsigned int) parent_idx >= s_topics.count) return NULL;
    const md_topic *parent = &s_topics.data[parent_idx];
    if (parent->file_index < 0 || (unsigned int) parent->file_index >= s_files.count) return NULL;
    const md_file *file = &s_files.data[parent->file_index];
    const char *best = NULL;
    unsigned int best_d = MORPHO_HELP_SUGGEST_MAXDIST + 1;
    for (unsigned int i = parent->block_index + 1; i < file->blocks.count; i++) {
        const md_block *b = &file->blocks.data[i];
        if (b->type != MD_BLOCK_HEADER) continue;
        if (b->as.header.level <= parent->level) break;
        int ti = help_findtopicbyblock(parent->file_index, i);
        if (ti < 0 || !s_topics.data[ti].name) continue;
        unsigned int d = help_editdistance(name, s_topics.data[ti].name);
        if (d < best_d) { best_d = d; best = s_topics.data[ti].name; }
    }
    return best;
}

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
static bool help_topicsrc(const help_topic *t, varray_char *result, const char **src, size_t *src_len) {
    if (!t || !t->file || !t->file->source || !result) return false;
    *src = t->file->source;
    *src_len = t->file->sourcelen;
    return true;
}

bool help_topictotext(const help_topic *t, varray_char *result) {
    const char *src;
    size_t src_len;
    if (!help_topicsrc(t, result, &src, &src_len)) return false;
    for (unsigned int i = 0; i < t->nblocks; i++) {
        const md_block *b = &t->content_blocks[i];
        if (b->span.start >= src_len) continue;
        size_t len = b->span.length;
        if (b->span.start + len > src_len) len = src_len - b->span.start;

        // Per-block: header (title only, no #), paragraph/list/code (span), link/blank skip
        switch (b->type) {
            case MD_BLOCK_HEADER: {
                const char *p = src + b->as.header.title.start;
                size_t tlen = b->as.header.title.length;
                if (b->as.header.title.start + tlen > src_len) tlen = src_len - b->as.header.title.start;
                while (tlen && (*p == '#' || (unsigned char)*p <= ' ')) { p++; tlen--; }  // strip leading # and space
                if (tlen > 0) varray_charadd(result, p, (int) tlen);
                varray_charadd(result, "\n\n", 2);
                break;
            }
            case MD_BLOCK_PARAGRAPH:
            case MD_BLOCK_LIST:
            case MD_BLOCK_CODE:
                varray_charadd(result, src + b->span.start, (int) len);
                if (len > 0 && src[b->span.start + len - 1] != '\n') varray_charadd(result, "\n", 1);
                break;
            case MD_BLOCK_LINK_DEF:
            case MD_BLOCK_BLANK:
                break;  // omit from plain text
        }
    }
    return true;
}

bool help_topicrawmd(const help_topic *t, varray_char *result) {
    const char *src;
    size_t src_len;
    if (!help_topicsrc(t, result, &src, &src_len)) return false;
    for (unsigned int i = 0; i < t->nblocks; i++) {
        const md_block *b = &t->content_blocks[i];
        if (b->span.start >= src_len) continue;
        size_t len = b->span.length;
        if (b->span.start + len > src_len) len = src_len - b->span.start;
        if (len == 0) continue;
        varray_charadd(result, src + b->span.start, (int) len);
        if (src[b->span.start + len - 1] != '\n') varray_charadd(result, "\n", 1);
    }
    return true;
}

bool help_parse(md_file *file, varray_md_topic *topics, int file_index) {
    error err;
    error_init(&err);
    md_parseout parseout = {
        .file = file,
        .topics = topics,
        .file_index = file_index,
        .current_header_level = 1
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
        if (err.line != ERROR_POSNUNIDENTIFIABLE || err.posn != ERROR_POSNUNIDENTIFIABLE) {
            fprintf(stderr, " (");
            if (err.line != ERROR_POSNUNIDENTIFIABLE) fprintf(stderr, "line %d", err.line + 1);
            if (err.posn != ERROR_POSNUNIDENTIFIABLE)
                fprintf(stderr, "%sposition %d", err.line != ERROR_POSNUNIDENTIFIABLE ? ", " : "", err.posn + 1);
            fprintf(stderr, ")");
        }
        fprintf(stderr, "\n");
    }
    return ok;
}

/* **********************************************************************
 * Morpho help files
 * ********************************************************************** */

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
    // Own a copy of the source so each file in s_files has independent storage.
    size_t len = (contents.count > 0) ? contents.count - 1 : 0;
    char *copy = (char *) MORPHO_MALLOC(len + 1);
    if (!copy) {
        varray_charclear(&contents);
        return false;
    }
    memcpy(copy, contents.data, len + 1);
    varray_charclear(&contents);
    // Parse appends topics to s_topics with file_index = s_files.count. If parse fails we must
    // remove those topics so they don't point at the next file we append.
    md_file mdfile;
    md_file_init(&mdfile);
    mdfile.source = copy;
    mdfile.sourcelen = len;
    unsigned int n_topics_before = s_topics.count;
    bool ok = help_parse(&mdfile, &s_topics, (int) s_files.count);
    if (ok) {
        varray_md_filewrite(&s_files, mdfile);
    } else {
        fprintf(stderr, "Help file failed to parse: %s\n", filename);
        md_file_clear(&mdfile);
        /* Remove topics added during the failed parse; they have file_index = s_files.count */
        while (s_topics.count > n_topics_before) {
            md_topic *t = &s_topics.data[s_topics.count - 1];
            if (t->name) MORPHO_FREE(t->name);
            t->name = NULL;
            s_topics.count--;
        }
    }
    return ok;
}

/** Finds and loads all help files; populates s_files and s_topics, then sorts topics. */
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
    help_sorttopics(&s_topics);
}

/* **********************************************************************
 * Help API
 * ********************************************************************** */

/** Interface to the morpho help system. Query may be "Topic" or "Topic subtopic" / "Topic.subtopic". */
bool morpho_helpastopic(const char *query, help_topic *out) {
    char qbuf[MORPHO_MAX_HELPQUERY_LENGTH];
    const char *segs[MORPHO_HELP_QUERY_MAXSEGMENTS];
    int nsegs = help_parsequery(query, qbuf, segs);
    if (nsegs <= 0) return false;

    // Single segment: resolve by name; fail if 0 or multiple matches (caller shows hint)
    int idx;
    if (nsegs == 1) {
        int multi[MORPHO_HELP_MAX_MULTIMATCH];
        int n = help_findallbyname(segs[0], multi, MORPHO_HELP_MAX_MULTIMATCH);
        if (n == 1) {
            idx = multi[0];
        } else if (n == 0) {
            idx = help_findtopic(&s_topics, segs[0]);  /* fallback: bsearch by name */
            if (idx < 0) return false;
        } else {
            return false;  /* multiple matches: caller will show hint */
        }
    } else {
        idx = help_findtopic(&s_topics, segs[0]);
        if (idx < 0) return false;
    }

    // Resolve remaining segments as subtopics
    for (int i = 1; i < nsegs; i++) {
        idx = help_findsubtopic(idx, segs[i]);
        if (idx < 0) return false;
    }

    const md_topic *topic = &s_topics.data[idx];
    if (topic->file_index < 0 || (unsigned int) topic->file_index >= s_files.count) return false;
    const md_file *file = &s_files.data[topic->file_index];
    if (topic->block_index >= file->blocks.count) return false;
    help_topicfill(out, topic, file);
    return true;
}

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
        if (i == 0)
            n += snprintf(buf + n, sizeof(buf) - (size_t) n, "'%s'", path);
        else if (i == count - 1)
            n += snprintf(buf + n, sizeof(buf) - (size_t) n, " or '%s'?", path);
        else
            n += snprintf(buf + n, sizeof(buf) - (size_t) n, ", '%s'", path);
    }
    if (n > 0 && (size_t) n < sizeof(buf)) varray_charadd(result, buf, n);
}

/** Build a hint for a failed query and append to result (caller may clear result first). */
static void help_queryhint(const char *query, varray_char *result) {
    if (!result) return;
    char qbuf[MORPHO_MAX_HELPQUERY_LENGTH];
    const char *segs[MORPHO_HELP_QUERY_MAXSEGMENTS];
    int nsegs = help_parsequery(query, qbuf, segs);
    if (nsegs == 0) {
        varray_charadd(result, MORPHO_HELP_NOTFOUND, (int) (sizeof(MORPHO_HELP_NOTFOUND) - 1));
        return;
    }
    // Single segment: multi-match hint, or "not found" + closest, or NOTFOUND
    if (nsegs == 1) {
        int multi[MORPHO_HELP_MAX_MULTIMATCH];
        int n = help_findallbyname(segs[0], multi, MORPHO_HELP_MAX_MULTIMATCH);
        if (n >= 2) {
            help_hintappend_multi(result, query, multi, n);
            return;
        }
        if (n == 0) {
            help_hintappend(result, query, help_findclosesttopic(segs[0]));
            return;
        }
        varray_charadd(result, MORPHO_HELP_NOTFOUND, (int) (sizeof(MORPHO_HELP_NOTFOUND) - 1));
        return;
    }
    // Multi-segment: resolve first, then walk subtopics; on first failure suggest path.closest
    int idx = help_findtopic(&s_topics, segs[0]);
    if (idx < 0) {
        help_hintappend(result, query, help_findclosesttopic(segs[0]));
        return;
    }
    char path[MORPHO_MAX_HELPQUERY_LENGTH];
    int path_len = (int) strlen(segs[0]);
    if (path_len >= (int) sizeof(path)) path_len = (int) sizeof(path) - 1;
    memcpy(path, segs[0], (size_t) path_len);
    path[path_len] = '\0';

    for (int i = 1; i < nsegs; i++) {
        int next = help_findsubtopic(idx, segs[i]);
        if (next < 0) {
            const char *closest = help_findclosestsubtopic(idx, segs[i]);
            char suggest[MORPHO_MAX_HELPQUERY_LENGTH];
            if (closest && path_len + 1 + (int) strlen(closest) < (int) sizeof(suggest))
                snprintf(suggest, sizeof(suggest), "%s.%s", path, closest);
            else
                suggest[0] = '\0';
            help_hintappend(result, query, suggest[0] ? suggest : NULL);
            return;
        }
        idx = next;
        // Append "." + segs[i] to path for next level
        if (path_len > 0 && path_len < (int) sizeof(path) - 1) {
            path[path_len++] = '.';
            int seglen = (int) strlen(segs[i]);
            if (path_len + seglen >= (int) sizeof(path)) seglen = (int) sizeof(path) - 1 - path_len;
            if (seglen > 0) { memcpy(path + path_len, segs[i], (size_t) seglen); path_len += seglen; }
            path[path_len] = '\0';
        }
    }
    varray_charadd(result, MORPHO_HELP_NOTFOUND, (int) (sizeof(MORPHO_HELP_NOTFOUND) - 1));
}

bool morpho_helpastext(char *query, varray_char *result) {
    help_topic t;
    if (morpho_helpastopic(query, &t)) return help_topictotext(&t, result);
    help_queryhint(query, result);
    return false;
}

bool morpho_helpasmd(char *query, varray_char *result) {
    help_topic t;
    if (morpho_helpastopic(query, &t)) return help_topicrawmd(&t, result);
    help_queryhint(query, result);
    return false;
}

/* **********************************************************************
 * Initialization/finalization
 * ********************************************************************** */

/** @brief Initialize help system */
void help_initialize(void) {
    /* Markdown parser errors */
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
    help_findfiles();
    morpho_addfinalizefn(help_finalize);
}

/** @brief Finalization: free all files and topics. */
void help_finalize(void) {
    for (unsigned int i = 0; i < s_files.count; i++)
        md_file_clear(&s_files.data[i]);
    varray_md_fileclear(&s_files);
    for (unsigned int i = 0; i < s_topics.count; i++) {
        if (s_topics.data[i].name) MORPHO_FREE(s_topics.data[i].name);
    }
    varray_md_topicclear(&s_topics);
}
