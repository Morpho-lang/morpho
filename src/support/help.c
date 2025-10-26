/** @file help.c
 *  @author T J Atherton
 *
 *  @brief Morpho help
*/

#include <ctype.h>
#include <stdio.h>
#include <stdbool.h>
#include <varray.h>

#include "classes.h"
#include "help.h"
#include "resources.h"

#include "lex.h"
#include "parse.h"
#include "file.h"

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
 * ********************************************************************** */

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
    MD_BOLD,
    MD_ITALIC,
    MD_BOLDITALIC,
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
    { "*",          MD_ITALIC                   , NULL },
    { "_",          MD_ITALIC                   , NULL },
    { "**",         MD_BOLD                     , NULL },
    { "__",         MD_BOLD                     , NULL },
    { "***",        MD_BOLDITALIC               , NULL },
    { "___",        MD_BOLDITALIC               , NULL },
    { "\n",         MD_NEWLINE                  , md_lexnewline },
    { "",           TOKEN_NONE                  , NULL }
};

/** Lexer token preprocessor function */
bool md_lexpreprocess(lexer *l, token *tok, error *err) {
    // Keep going until we match a token or reach the end
    while (!lex_identifytoken(l, false, NULL) &&
           !lex_isatend(l)) {
        lex_advance(l);
    }
    
    // If we captured anything, record it as text
    if (l->current>l->start) {
        lex_recordtoken(l, MD_TEXT, tok);
        return true;
    }
    
    return false;
}

/* -------------------------------------------------------
 * Initialize a Markdown lexer
 * ------------------------------------------------------- */

void help_initializemdlexer(lexer *l, char *src) {
    lex_init(l, src, 0);
    lex_settokendefns(l, mdtokens);
    lex_setprefn(l, md_lexpreprocess);
    lex_setwhitespacefn(l, NULL);
    lex_seteof(l, MD_EOF);
}

/* -------------------------------------------------------
 * Markdown parse rules
 * ------------------------------------------------------- */

/** Parses a markdown header  */
bool md_parseheader(parser *p, void *out) {
    PARSE_CHECK(parse_checktokenadvance(p, MD_TEXT));
    PARSE_CHECK(parse_checktokenadvance(p, MD_NEWLINE));
    return true;
}

bool md_parseurl(parser *p, void *out) { // TODO: This is a placeholder
    PARSE_CHECK(parse_checktokenadvance(p, MD_TEXT));
    PARSE_CHECK(parse_checktokenadvance(p, MD_HASH));
    PARSE_CHECK(parse_checktokenadvance(p, MD_TEXT));
    return true;
}

/** Parses a markdown link  */
bool md_parselink(parser *p, void *out) {
    PARSE_CHECK(parse_checktokenadvance(p, MD_TEXT));
    PARSE_CHECK(parse_checktokenadvance(p, MD_RIGHTSQUAREBRACE));
    PARSE_CHECK(parse_checktokenadvance(p, MD_COLON));
    PARSE_CHECK(md_parseurl(p, out));
    if (parse_checktokenadvance(p, MD_LEFTPAREN)) {
        PARSE_CHECK(parse_checktokenadvance(p, MD_TEXT));
        PARSE_CHECK(parse_checktokenadvance(p, MD_RIGHTPAREN));
    }
    PARSE_CHECK(parse_checktokenadvance(p, MD_NEWLINE));
    
    return true;
}

/** Parses a markdown paragraph  */
bool md_parseparagraph(parser *p, void *out) {
    return true;
}

/** Parse a markdown 'block'  */
bool md_parseblock(parser *p, void *out) {
    if (parse_checktokenadvance(p, MD_TEXT)) {
        return md_parseparagraph(p, out);
    } else if (parse_checktokenadvance(p, MD_LEFTSQUAREBRACE)) {
        return md_parselink(p, out);
    } else if (parse_checktokenadvance(p, MD_HASH)) {
        return md_parseheader(p, out);
    } else if (parse_checktokenadvance(p, MD_NEWLINE)) { // A blank line
        return true;
    } else {
        return md_parseparagraph(p, out);
    }
    
    return false;
}

/** Base markdown parse type */
bool md_parse(parser *p, void *out) {
    while(md_parseblock(p, out)) {
    }
    
    return true;
}

/* -------------------------------------------------------
 * Markdown parse table
 * ------------------------------------------------------- */

parserule md_rules[] = {
    PARSERULE_PREFIX(MD_HASH,            NULL           ),
    PARSERULE_PREFIX(MD_LEFTSQUAREBRACE, NULL           ),
    PARSERULE_PREFIX(MD_BOLD,            NULL           ),
    PARSERULE_UNUSED(TOKEN_NONE)
};

/* -------------------------------------------------------
 * Initialize a Markdown parser
 * ------------------------------------------------------- */

/** Initializes a parser to parse JSON */
void help_initializemdparser(parser *p, lexer *l, error *err, void *out) {
    parse_init(p, l, err, out);
    parse_setbaseparsefn(p, md_parse);
    parse_setparsetable(p, md_rules);
    parse_setskipnewline(p, false, TOKEN_NONE);
}

/* **********************************************************************
 * Parse help files
 * ********************************************************************** */

bool help_parse(char *src) {
    error err;
    error_init(&err);
    
    lexer l;
    help_initializemdlexer(&l, src);
    
    /*token tok;
    while (lex(&l, &tok, &err)) {
        if (tok.type==MD_EOF) break;
    }*/
    
    parser p;
    help_initializemdparser(&p, &l, &err, NULL);
    
    parse(&p);
    
    return true;
}

/* **********************************************************************
 * Morpho help files
 * ********************************************************************** */

/** Loads a help file
 *  @param file     file to load
 *  @returns true if any help entries were successfully loaded */
bool help_load(char *filename) {
    bool success=false;
    
    FILE *f = fopen(filename, "r");
    if (f) {
        varray_char contents;
        varray_charinit(&contents);
        
        if (file_readintovarray(f, &contents)) {
            success=help_parse(contents.data);
        }
        
        varray_charclear(&contents);
        
        fclose(f);
    }
    
    return success;
}

/** Searches for help files
 *  @returns true if any help files were successfully processed. */
void help_findfiles(void) {
    varray_value files;
    varray_valueinit(&files);
    
    if (morpho_listresources(MORPHO_RESOURCE_HELP, &files)) {
        for (int i=0; i<files.count; i++) {
            if (MORPHO_ISSTRING(files.data[i])) help_load(MORPHO_GETCSTRING(files.data[i]));
        }
    }
    
    varray_valueclear(&files);
}

/* **********************************************************************
 * Morpho help files
 * ********************************************************************** */

/** Interface to the morpho help system */
bool morpho_help(char *query, varray_char *result) {
    return false;
}

/* **********************************************************************
 * Initialization/finalization
 * ********************************************************************** */

/** @brief Initialization/finalization */
void help_initialize(void) {
    help_findfiles();
    
    morpho_addfinalizefn(help_finalize);
}

/** @brief Initialization/finalization */
void help_finalize(void) {
}
