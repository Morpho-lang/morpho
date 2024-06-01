/** @file lex.c
 *  @author T J Atherton 
 *
 *  @brief Lexer
*/

#include <string.h>
#include <ctype.h>

#include "lex.h"

extern tokendefn standardtokens[];
extern int nstandardtokens;

/* **********************************************************************
 * Lexer library functions
 * ********************************************************************** */

/* -------------------------------------------------------
 * Comparison functions for token definitions
 * ------------------------------------------------------- */

/** Compare two token definitions */
int _lex_tokndefncmp(const void *ldefn, const void *rdefn) {
    tokendefn *a = (tokendefn *) ldefn;
    tokendefn *b = (tokendefn *) rdefn;
    
    return strcmp(a->string, b->string);
}

/** Compare a string  with the contents of a token definition */
int _lex_tokendefnwithstringcmp(const void *lstr, const void *rdefn) {
    char *a = (char *) lstr;
    tokendefn *b = (tokendefn *) rdefn;
    
    return strcmp(a, b->string);
}

/** Compare the contents of a token with the contents of a token definition */
int _lex_tokendefnwithtokencmp(const void *ltok, const void *rdefn) {
    token *tok = (token *) ltok;
    tokendefn *b = (tokendefn *) rdefn;
    
    // Compare token with token definition
    int cmp = strncmp(tok->start, b->string, tok->length);
    
    // If we see a match, ensure that we're not simply matching the initial part of the definition.
    if (cmp==0 && b->string[tok->length]!='\0') cmp = -b->string[tok->length]; // Mimic behavior of strcmp
    
    return cmp;
}

/** Compare a character with the first character of a token definition */
int _lex_tokendefnwithtokenfirstcharcmp(const void *l, const void *rdefn) {
    char *c = (char *) l;
    tokendefn *b = (tokendefn *) rdefn;
    
    return *c -  b->string[0];
}


DEFINE_VARRAY(tokendefn, tokendefn);

/* -------------------------------------------------------
 * Library functions to support writing custom lexers
 * ------------------------------------------------------- */

/** @brief Records a token
 *  @param[in]  l     The lexer in use
 *  @param[in]  type  Type of token to record
 *  @param[out] tok   Token structure to fill out */
void lex_recordtoken(lexer *l, tokentype type, token *tok) {
    tok->type=type;
    tok->start=l->start;
    tok->length=(int) (l->current - l->start);
    tok->line=l->line;
    tok->posn=l->posn - tok->length;
}

/** @brief Advances the lexer by one character, returning the character */
char lex_advance(lexer *l) {
    char c = *(l->current);
    l->current++;
    l->posn++;
    return c;
}

/** @brief Advances the lexer by n characters, returning the last character */
char lex_advanceby(lexer *l, size_t n) {
    l->current+=n;
    l->posn+=n;
    return *(l->current-1);
}

/** @brief Reverses the current character in the lexer by one. */
bool lex_back(lexer *l) {
    if (l->current==l->start) return false;
    l->current--;
    l->posn--;
    return true;
}

/** @brief Checks if we're at the end of the string. Doesn't advance. */
bool lex_isatend(lexer *l) {
    return (*(l->current) == '\0');
}

/** @brief Checks if a character is alphanumeric or underscore. */
bool lex_isalpha(char c) {
    return (isalpha(c) || (c=='_'));
}

/** @brief Checks if a character is a digit.  */
bool lex_isdigit(char c) {
    return isdigit(c);
}

/** @brief Checks if a character is whitespace.
    @warning: The morpho lexer does not consider newlines to be whitespace. */
bool lex_isspace(char c) {
    return (c==' ') || (c=='\t') || (c=='\n') || (c=='\r');
}

/** @brief Returns the next character */
char lex_peek(lexer *l) {
    return *(l->current);
}

/** @brief Returns n characters ahead. Caller should check that this is meaningfull. */
char lex_peekahead(lexer *l, int n) {
    return *(l->current + n);
}

/** @brief Returns the previous character */
char lex_peekprevious(lexer *l) {
    if (l->current==l->start) return '\0';
    return *(l->current - 1);
}

/** @brief Advance line counter */
void lex_newline(lexer *l) {
    l->line++; l->posn=0;
}


/** @brief Attempts to find a matching token for the current token.
 *  @param[in] l The lexer in use
 *  @param[out] defn Type of token, if found
 *  @returns true if the token matched, false if not */
bool lex_matchtoken(lexer *l, tokendefn **defn) {
    token tok = { .start = l->start, .length = (int) (l->current - l->start) };

    tokendefn *def = bsearch(&tok, l->defns, l->ndefns, sizeof(tokendefn),
                              _lex_tokendefnwithtokencmp);
    
    if (def && defn) *defn = def;
    
    return def;
}

/** @brief Attempts to identify a token from the current point, advances if it finds one.
 *  @param[in] l The lexer in use
 *  @param[out] defn Type of token, if found
 *  @returns true if the token matched, false if not */
bool lex_identifytoken(lexer *l, tokendefn **defn) {
    char c = lex_peek(l);
    
    // Match first character
    tokendefn *def = bsearch(&c, l->defns, l->ndefns, sizeof(tokendefn),
                              _lex_tokendefnwithtokenfirstcharcmp);
    if (!def) return false;
    
    tokendefn *last = l->defns+l->ndefns-1;
    // Now find the last definition that matches this character
    while (def<last && (def+1)->string[0]==c) def++;
    
    // Test each in turn, working backwards to match the longest token we can
    for (; def->string[0]==c && def>=l->defns; def--) {
        size_t len=strlen(def->string);
        if (strncmp(def->string, l->current, len)==0) {
            lex_advanceby(l, len);
            if (defn) *defn = def;
            return true;
        }
    }
    
    return false;
}

/* **********************************************************************
 * Morpho lexer
 * ********************************************************************** */

// Process functions we'll be using
bool lex_string(lexer *l, token *tok, error *err);
bool lex_processnewline(lexer *l, token *tok, error *err);
bool lex_processinterpolation(lexer *l, token *tok, error *err);

/* -------------------------------------------------------
 * Morpho token definitions
 * ------------------------------------------------------- */

tokendefn standardtokens[] = {
    { "(",          TOKEN_LEFTPAREN         , NULL },
    { ")",          TOKEN_RIGHTPAREN        , NULL },
    { "[",          TOKEN_LEFTSQBRACKET     , NULL },
    { "]",          TOKEN_RIGHTSQBRACKET    , NULL },
    { "{",          TOKEN_LEFTCURLYBRACKET  , NULL },
    { "}",          TOKEN_RIGHTCURLYBRACKET , lex_processinterpolation },
    { ";",          TOKEN_SEMICOLON         , NULL },
    { ":",          TOKEN_COLON             , NULL },
    { ",",          TOKEN_COMMA             , NULL },
    { "^",          TOKEN_CIRCUMFLEX        , NULL },
    { "?",          TOKEN_QUESTION          , NULL },
    { "@",          TOKEN_AT                , NULL },
    { "#",          TOKEN_HASH              , NULL },
    { ".",          TOKEN_DOT               , NULL },
    { "..",         TOKEN_DOTDOT            , NULL },
    { "...",        TOKEN_DOTDOTDOT         , NULL },
    { "+",          TOKEN_PLUS              , NULL },
    { "+=",         TOKEN_PLUSEQ            , NULL },
    { "-",          TOKEN_MINUS             , NULL },
    { "-=",         TOKEN_MINUSEQ           , NULL },
    { "*",          TOKEN_STAR              , NULL },
    { "*=",         TOKEN_STAREQ            , NULL },
    { "/",          TOKEN_SLASH             , NULL },
    { "/=",         TOKEN_SLASHEQ           , NULL },
    { "==",         TOKEN_EQ                , NULL },
    { "=",          TOKEN_EQUAL             , NULL },
    { "!",          TOKEN_EXCLAMATION       , NULL },
    { "!=",         TOKEN_NEQ               , NULL },
    { "<",          TOKEN_LT                , NULL },
    { "<=",         TOKEN_LTEQ              , NULL },
    { ">",          TOKEN_GT                , NULL },
    { ">=",         TOKEN_GTEQ              , NULL },
    { "&",          TOKEN_AMP               , NULL },
    { "&&",         TOKEN_DBLAMP            , NULL },
    { "|",          TOKEN_VBAR              , NULL },
    { "||",         TOKEN_DBLVBAR           , NULL },
    { "\"",         TOKEN_QUOTE             , lex_string },
    { "\n",         TOKEN_NEWLINE           , lex_processnewline },
    { "and",        TOKEN_DBLAMP            , NULL },
    { "as",         TOKEN_AS                , NULL },
    { "break",      TOKEN_BREAK             , NULL },
    { "class",      TOKEN_CLASS             , NULL },
    { "continue",   TOKEN_CONTINUE          , NULL },
    { "catch",      TOKEN_CATCH             , NULL },
    { "do",         TOKEN_DO                , NULL },
    { "else",       TOKEN_ELSE              , NULL },
    { "false",      TOKEN_FALSE             , NULL },
    { "for",        TOKEN_FOR               , NULL },
    { "fn",         TOKEN_FUNCTION          , NULL },
    { "help",       TOKEN_QUESTION          , NULL },
    { "if",         TOKEN_IF                , NULL },
    { "in",         TOKEN_IN                , NULL },
    { "is",         TOKEN_IS                , NULL },
    { "import",     TOKEN_IMPORT            , NULL },
    { "im",         TOKEN_IMAG              , NULL },
    { "nil",        TOKEN_NIL               , NULL },
    { "or",         TOKEN_DBLVBAR           , NULL },
    { "print",      TOKEN_PRINT             , NULL },
    { "return",     TOKEN_RETURN            , NULL },
    { "self",       TOKEN_SELF              , NULL },
    { "super",      TOKEN_SUPER             , NULL },
    { "true",       TOKEN_TRUE              , NULL },
    { "try",        TOKEN_TRY               , NULL },
    { "var",        TOKEN_VAR               , NULL },
    { "while",      TOKEN_WHILE             , NULL },
    { "with",       TOKEN_WITH              , NULL },
    { "",           TOKEN_NONE              , NULL }  // Token list should be terminated by an empty token
};

int nstandardtokens;

/* -------------------------------------------------------
 * Morpho lexing functions
 * ------------------------------------------------------- */

/** @brief Skips multiline comments
 * @param[in]  l    the lexer
 * @param[out] tok  token record to fill out (if necessary)
 * @param[out] err  error struct to fill out on errors
 * @returns true on success, false if an error occurs */
bool lex_skipmultilinecomment(lexer *l, token *tok, error *err) {
    unsigned int level=0;
    unsigned int startline = l->line, startpsn = l->posn;
    
    do {
        char c = lex_peek(l);
        switch (c) {
            case '\0':
                /* If we come to the end of the file, the token is marked as incomplete. */
                morpho_writeerrorwithid(err, LEXER_UNTERMINATEDCOMMENT, NULL, startline, startpsn);
                lex_recordtoken(l, TOKEN_INCOMPLETE, tok);
                return false;
            case '\n':
                /* Advance the line counter. */
                lex_newline(l);
                break;
            case '/':
                if (lex_peekahead(l, 1)=='*') {
                    level++; lex_advance(l);
                }
                break;
            case '*':
                if (lex_peekahead(l, 1)=='/') {
                    level--; lex_advance(l);
                }
                break;
            default:
                break;
        }
        /* Now advance the counter */
        lex_advance(l);
    } while (level>0);
    
    return true;
}

/** @brief Skips comments
 *  @param[in]  l    the lexer
 *  @param[out] tok  token record to fill out (if necessary)
 *  @param[out] err  error struct to fill out on errors
 *  @returns true on success, false if an error occurs */
bool lex_skipcomment(lexer *l, token *tok, error *err) {
    char c = lex_peekahead(l, 1);
    if (c == '/') {
        while (lex_peek(l) != '\n' && !lex_isatend(l)) lex_advance(l);
        return true;
    } else if (c == '*') {
        return lex_skipmultilinecomment(l, tok, err);
    }
    return false;
}

/** @brief Detect and skip a shebang line
 *  @param[in]  l    the lexer
 *  @param[out] tok  token record to fill out (if necessary)
 *  @param[out] err  error struct to fill out on errors
 *  @returns true on success, false if an error occurs */
bool lex_skipshebang(lexer *l) {
    if (lex_peek(l)=='#' && lex_peekahead(l, 1)=='!') {
        while (lex_peek(l) != '\n' && !lex_isatend(l)) lex_advance(l);
    }
    return true;
}

/** @brief Skips whitespace
 *  @param[in]  l    the lexer
 *  @param[out] tok  token record to fill out (if necessary)
 *  @param[out] err  error struct to fill out on errors
 *  @returns true on success, false if an error occurs */
bool lex_skipwhitespace(lexer *l, token *tok, error *err) {
    do {
        switch (lex_peek(l)) {
            case ' ':
            case '\t':
            case '\r':
                lex_advance(l);
                break;
            case '/':
                if (!lex_skipcomment(l, tok, err)) return true;
                break;
            default:
                return true;
        }
    } while (true);
    return true;
}

/** @brief Lex strings
 *  @param[in]  l    the lexer
 *  @param[out] tok  token record to fill out
 *  @param[out] err  error struct to fill out on errors
 *  @returns true on success, false if an error occurs */
bool lex_string(lexer *l, token *tok, error *err) {
    unsigned int startline = l->line, startpsn = l->posn;
    
    char first = lex_peekprevious(l);
    
    while (lex_peek(l) != '"' && !lex_isatend(l)) {
        if (lex_peek(l) == '\n') lex_newline(l);
        
        /* Detect string interpolation */
        if (l->stringinterpolation && lex_peek(l) == '$' && lex_peekahead(l, 1) == '{') {
            lex_advance(l); lex_advance(l);
            lex_recordtoken(l, TOKEN_INTERPOLATION, tok);
            if (first=='"') l->interpolationlevel++;
            return true;
        }

        /* Detect an escaped character */
        if (lex_peek(l)=='\\') {
            lex_advance(l);
        }
        
        lex_advance(l);
    }
    
    if (lex_isatend(l)) {
        /* Unterminated string */
        morpho_writeerrorwithid(err, LEXER_UNTERMINATEDSTRING, NULL, startline, startpsn);
        lex_recordtoken(l, TOKEN_INCOMPLETE, tok);
        return false;
    }
    
    lex_advance(l); /* Closing quote */
    
    if (l->stringinterpolation && l->interpolationlevel>0 && first=='}') l->interpolationlevel--;
    
    lex_recordtoken(l, TOKEN_STRING, tok);
    return true;
}

/** @brief Lex numbers
 *  @param[in]  l    the lexer
 *  @param[out] tok  token record to fill out
 *  @param[out] err  error struct to fill out on errors
 *  @returns true on success, false if an error occurs */
bool lex_number(lexer *l, token *tok, error *err) {
    tokentype type=l->inttype;
    while (lex_isdigit(lex_peek(l))) lex_advance(l);
    
    /* Fractional part */
    char next = '\0';
    if (lex_peek(l)!='\0') next=lex_peekahead(l, 1); // Prevent looking beyond buffer
    if (lex_peek(l) == '.' && (lex_isdigit(next) || lex_isspace(next) || next=='\0') ) {
        type=TOKEN_NUMBER;
        lex_advance(l); /* Consume the '.' */
        while (lex_isdigit(lex_peek(l))) lex_advance(l);
    }
    
    /* Exponent */
    if (lex_peek(l) == 'e' || lex_peek(l) == 'E') {
        type=l->flttype;
        lex_advance(l); /* Consume the 'e' */
        
        /* Optional sign */
        if (lex_peek(l) == '+' || lex_peek(l) == '-') lex_advance(l);
        
        /* Exponent digits */
        while (lex_isdigit(lex_peek(l))) lex_advance(l);
    }
    
    /* Imaginary Numbers */
    if (lex_peek(l) =='i' && lex_peekahead(l, 1) == 'm'){
        /* mark this as an imaginary number*/
        type = l->imagtype;
        lex_advance(l); /* Consume the 'i' */
        lex_advance(l); /* Consume the 'm' */
    }
    
    lex_recordtoken(l, type, tok);
    
    return true;
}

/** @brief Checks whether a symbol matches a given string, and if so records a token
 *  @param[in]  l      the lexer
 *  @param[in]  start  offset to start comparing from
 *  @param[in]  length length to compare
 *  @param[in]  match  string to match with
 *  @param[in]  type   token type to use if the match is successful
 *  @returns type or symboltype if the match was not successful */
tokentype lex_checksymbol(lexer *l, int start, int length, char *match, tokentype type) {
    int toklength = (int) (l->current - l->start);
    int expectedlength = start + length;
    
    /* Compare, but don't bother calling memcmp if the lengths are different */
    if ((toklength == expectedlength) && (memcmp(l->start+start, match, length) == 0))
            return type;
    
    return l->symboltype;
}

tokentype lex_typeforsymboltoken(lexer *l) {
    tokentype t = l->symboltype;
    tokendefn *def;
    
    if (lex_matchtoken(l, &def)) t = def->type;
    
    return t;
}

/** @brief Lex symbols
 *  @param[in]  l    the lexer
 *  @param[out] tok  token record to fill out
 *  @param[out] err  error struct to fill out on errors
 *  @returns true on success, false if an error occurs */
bool lex_symbol(lexer *l, token *tok, error *err) {
    while (lex_isalpha(lex_peek(l)) || lex_isdigit(lex_peek(l))) lex_advance(l);
    
    tokentype typ = l->symboltype;
    if (l->matchkeywords) typ = lex_typeforsymboltoken(l);
    
    lex_recordtoken(l, typ, tok);
    
    return true;
}

/* -------------------------------------------------------
 * Morpho preprocessing functions
 * ------------------------------------------------------- */

/** @brief Process function for newline tokens */
bool lex_preprocess(lexer *l, token *tok, error *err) {
    char c = lex_peek(l);
    if (lex_isalpha(c)) return lex_symbol(l, tok, err);
    if (lex_isdigit(c)) return lex_number(l, tok, err);
    return false;
}

/** @brief Process function for newline tokens */
bool lex_processnewline(lexer *l, token *tok, error *err) {
    lex_newline(l);
    return true;
}

/** @brief Process function for interpolation tokens */
bool lex_processinterpolation(lexer *l, token *tok, error *err) {
    if (l->stringinterpolation && l->interpolationlevel>0) {
        return lex_string(l, tok, err);
    }
    return true;
}

/* **********************************************************************
 * Initialize/clear a lexer
 * ********************************************************************** */

/** @brief Initializes a lexer with a given starting point
 *  @param l     The lexer to initialize
 *  @param start Starting point to lex from
 *  @param line  The current line number */
void lex_init(lexer *l, const char *start, int line) {
    l->current=start;
    l->start=start;
    l->line=line;
    l->posn=0;
    l->matchkeywords=true;
    l->stringinterpolation=true;
    l->interpolationlevel=0;
    l->prefn=lex_preprocess;
    l->whitespacefn=lex_skipwhitespace;
    l->eoftype=TOKEN_EOF;
    l->inttype=TOKEN_INTEGER;
    l->flttype=TOKEN_NUMBER;
    l->imagtype=TOKEN_IMAG;
    l->symboltype=TOKEN_SYMBOL;
    l->defns=standardtokens;   // Use the standard morpho tokens by default
    l->ndefns=nstandardtokens;
    varray_tokendefninit(&l->defnstore); // Alternative definitions will be held here
}

/** @brief Clears a lexer */
void lex_clear(lexer *l) {
    l->current=NULL;
    l->start=NULL;
    l->posn=0;
    l->interpolationlevel=0;
    varray_tokendefnclear(&l->defnstore);
}

/* **********************************************************************
 * Configure the lexer
 * ********************************************************************** */

/** Sets the lexer to use a specific set of token definitions
 * @param[in]  l    the lexer
 * @param[out] defns  List of token definitons, terminated by a null or null length string
 * @warning: The lexer does not duplicate the token definition strings, so these should be preserved. */
void lex_settokendefns(lexer *l, tokendefn *defns) {
    int n;
    for (n=0; ; n++) if (defns[n].string == NULL || strlen(defns[n].string)==0) break;
    
    l->defnstore.count=0;
    varray_tokendefnadd(&l->defnstore, defns, n);
    
    l->defns=l->defnstore.data;
    l->ndefns=n;
    
    qsort(l->defns, l->ndefns, sizeof(tokendefn), _lex_tokndefncmp);
}

/** @brief Sets the token type representing End Of File */
void lex_seteof(lexer *l, tokentype eoftype) {
    l->eoftype = eoftype;
}

/** @brief Gets the token type representing End Of File */
tokentype lex_eof(lexer *l) {
    return l->eoftype;
}

/** @brief Sets the token type representing integers, floats and complex */
void lex_setnumbertype(lexer *l, tokentype inttype, tokentype flttype, tokentype imagtype) {
    l->inttype=inttype;
    l->flttype=flttype;
    l->imagtype=imagtype;
}

/** @brief Sets the token type representing symbols */
void lex_setsymboltype(lexer *l, tokentype symboltype) {
    l->symboltype=symboltype;
}

/** @brief Gets the token type representing symbols */
tokentype lex_symboltype(lexer *l) {
    return l->symboltype;
}

/** @brief Choose whether the lexer should perform string interpolation. */
void lex_setstringinterpolation(lexer *l, bool interpolation) {
    l->stringinterpolation=interpolation;
}

/** @brief Choose whether the lexer should attempt to match keywords or simply return them as symbols. */
void lex_setmatchkeywords(lexer *l, bool match) {
    l->matchkeywords=match;
}

/** @brief Choose whether the lexer should attempt to match keywords or simply return them as symbols. */
bool lex_matchkeywords(lexer *l) {
    return l->matchkeywords;
};

/** @brief Provide a processing function to skip whitespace and comments. */
void lex_setwhitespacefn(lexer *l, processtokenfn whitespacefn) {
    l->whitespacefn = whitespacefn;
}

/** @brief Provide a processing function to identify tokens prior to matching. */
void lex_setprefn(lexer *l, processtokenfn prefn) {
    l->prefn = prefn;
}

/* **********************************************************************
 * Lexer public interface
 * ********************************************************************** */

/** @brief Identifies the next token
 *  @param[in]  l     The lexer in use
 *  @param[out] err   An error block to fill out on an error
 *  @param[out] tok   Token structure to fill out
 *  @returns true on success or false on failure  */
bool lex(lexer *l, token *tok, error *err) {
    bool success=false;
    
    // Handle leading whitespace
    if (l->whitespacefn) {
        if (!((l->whitespacefn) (l, tok, err))) return false;
    }
    
    // Set beginning of the token
    l->start=l->current;
    
    // Check whether we're at the end of the source string
    if (lex_isatend(l)) {
        lex_recordtoken(l, l->eoftype, tok);
        return true;
    }
    
    // If the lexer has a prefn, call that and check whether it handled the token.
    if (l->prefn) {
        success=(l->prefn) (l, tok, err);
        if (err->cat!=ERROR_NONE) return false; // It raised an error, so should return
        if (success) return true;
    }
    
    tokendefn *defn=NULL;
    if (lex_identifytoken(l, &defn)) {
        lex_recordtoken(l, defn->type, tok);
    } else {
        morpho_writeerrorwithid(err, LEXER_UNRECOGNIZEDTOKEN, NULL, l->line, l->posn);
        return false;
    }
    
    // If the token type provides a process function, call it
    if (defn->processfn) return (defn->processfn) (l, tok, err);

    return true;
}

/* **********************************************************************
 * Initialization/finalization
 * ********************************************************************** */

/** @brief Initialization/finalization */
void lex_initialize(void) {
    // Ensure standardtokens is sorted; this is then used by default to reduce cost of initializing a lexer.
    int n;
    for (n=0; ; n++) if (standardtokens[n].string == NULL || strlen(standardtokens[n].string)==0) break;
    qsort(standardtokens, n, sizeof(tokendefn), _lex_tokndefncmp);
    
    // Retain the number of standardtokens
    nstandardtokens = n;
    
    /* Lexer errors */
    morpho_defineerror(LEXER_UNRECOGNIZEDTOKEN, ERROR_LEX, LEXER_UNRECOGNIZEDTOKEN_MSG);
    morpho_defineerror(LEXER_UNTERMINATEDCOMMENT, ERROR_LEX, LEXER_UNTERMINATEDCOMMENT_MSG);
    morpho_defineerror(LEXER_UNTERMINATEDSTRING, ERROR_LEX, LEXER_UNTERMINATEDSTRING_MSG);
}
