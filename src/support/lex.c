/** @file lex.c
 *  @author T J Atherton 
 *
 *  @brief Lexer
*/

#include <string.h>
#include <ctype.h>

#include "lex.h"

/* **********************************************************************
 * Standard token definitions
 * ********************************************************************** */

tokendefn standardtokens[] = {
    { "and",        TOKEN_DBLAMP   },
    { "as",         TOKEN_AS       },
    { "break",      TOKEN_BREAK    },
    { "class",      TOKEN_CLASS    },
    { "continue",   TOKEN_CONTINUE },
    { "catch",      TOKEN_CATCH    },
    { "do",         TOKEN_DO       },
    { "else",       TOKEN_ELSE     },
    { "false",      TOKEN_FALSE    },
    { "for",        TOKEN_FOR      },
    { "fn",         TOKEN_FUNCTION },
    { "help",       TOKEN_QUESTION },
    { "if",         TOKEN_IF       },
    { "in",         TOKEN_IN       },
    { "is",         TOKEN_IS       },
    { "import",     TOKEN_IMPORT   },
    { "im",         TOKEN_IMAG     },
    { "nil",        TOKEN_NIL      },
    { "or",         TOKEN_DBLVBAR  },
    { "print",      TOKEN_PRINT    },
    { "return",     TOKEN_RETURN   },
    { "self",       TOKEN_SELF     },
    { "super",      TOKEN_SUPER    },
    { "true",       TOKEN_TRUE     },
    { "try",        TOKEN_TRY      },
    { "var",        TOKEN_VAR      },
    { "while",      TOKEN_WHILE    },
    { "with",       TOKEN_WITH     },
#ifdef MORPHO_LOXCOMPATIBILITY
    { "fun",        TOKEN_FUNCTION },
    { "this",       TOKEN_SELF     },
#endif
    { "",           TOKEN_NONE     }  // Token list should be terminated by an empty token
};

int nstandardtokens;

/* **********************************************************************
 * Work with token definitions
 * ********************************************************************** */

/** Compare two token definitions */
int _lex_matchtokndefn(const void *ldefn, const void *rdefn) {
    tokendefn *a = (tokendefn *) ldefn;
    tokendefn *b = (tokendefn *) rdefn;
    
    return strcmp(a->string, b->string);
}

/** Compare a string  with the contents of a token definition */
int _lex_matchtokendefnwithstring(const void *lstr, const void *rdefn) {
    char *a = (char *) lstr;
    tokendefn *b = (tokendefn *) rdefn;
    
    return strcmp(a, b->string);
}

/** Compare the contents of a token with the contents of a token definition */
int _lex_matchtokendefnwithtoken(const void *ltok, const void *rdefn) {
    token *tok = (token *) ltok;
    tokendefn *b = (tokendefn *) rdefn;
    
    // Compare token with token definition
    int cmp = strncmp(tok->start, b->string, tok->length);
    
    // If we see a match, ensure that we're not simply matching the initial part of the definition.
    if (cmp==0 && b->string[tok->length]!='\0') cmp = -b->string[tok->length]; // Mimic behavior of strcmp
    
    return cmp;
}

DEFINE_VARRAY(tokendefn, tokendefn);

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
#ifdef MORPHO_STRINGINTERPOLATION
    l->interpolationlevel=0;
#endif
    l->defns=standardtokens;   // Use the standard morpho tokens by default
    l->ndefns=nstandardtokens;
    varray_tokendefninit(&l->defnstore); // Alternative definitions will be held here
}

/** @brief Clears a lexer */
void lex_clear(lexer *l) {
    l->current=NULL;
    l->start=NULL;
    l->posn=0;
    l->matchkeywords=true;
#ifdef MORPHO_STRINGINTERPOLATION
    l->interpolationlevel=0;
#endif
    varray_tokendefnclear(&l->defnstore);
}

/* **********************************************************************
 * Internal functions
 * ********************************************************************** */

/** @brief Tests if the current prototoken matches a known token.
 *  @param[in]  l       The lexer in use
 *  @param[out] type Type of token, if found
 *  @returns true if the token matched, false if not */
bool lex_matchtoken(lexer *l, tokentype *type) {
    token tok = { .start = l->start, .length = (int) (l->current - l->start) };

    tokendefn *def = bsearch(&tok, l->defns, l->ndefns, sizeof(tokendefn), _lex_matchtokendefnwithtoken);
    
    if (def && type) *type = def->type;
    
    return def;
}

/** @brief Records a token
 *  @param[in]  l     The lexer in use
 *  @param[in]  type  Type of token to record
 *  @param[out] tok   Token structure to fill out */
void lex_recordtoken(lexer *l, tokentype type, token *tok) {
    tok->type=type;
    tok->start=l->start;
    tok->length=(int) (l->current - l->start);
    tok->line=l->line;
    tok->posn=l->posn;
}

/** @brief Checks if we're at the end of the string. Doesn't advance. */
bool lex_isatend(lexer *l) {
    return (*(l->current) == '\0');
}

/** @brief Checks if a character is a digit. Doesn't advance. */
bool lex_isdigit(char c) {
    return (c>='0' && c<= '9');
}

/** @brief Checks if a character is alphanumeric or underscore.  Doesn't advance. */
bool lex_isalpha(char c) {
    return (c>='a' && c<= 'z') || (c>='A' && c<= 'Z') || (c=='_');
}

/** @brief Checks if a character is whitespace.  Doesn't advance. */
bool lex_isnumber(char c) {
    return isdigit(c);
}

/** @brief Checks if a character is whitespace.  Doesn't advance. */
bool lex_isspace(char c) {
    return (c==' ') || (c=='\t') || (c=='\n') || (c=='\r');
}

/** @brief Advances the lexer by one character, returning the character */
char lex_advance(lexer *l) {
    char c = *(l->current);
    l->current++;
    l->posn++;
    return c;
}

/** @brief Returns the previous character */
char lex_previous(lexer *l) {
    if (l->current==l->start) return '\0';
    return *(l->current - 1);
}

/** @brief Returns the next character */
char lex_peek(lexer *l) {
    return *(l->current);
}

/** @brief Returns n characters ahead. Caller should check that this is meaningfull. */
char lex_peekahead(lexer *l, int n) {
    return *(l->current + n);
}

/** @brief Handle line counting */
void lex_newline(lexer *l) {
    l->line++; l->posn=0;
}

/** @brief Advances the lexer by one character if it is equal to c
 * @returns true if the character matched, false otherwise */
bool lex_match(lexer *l, char c) {
    if (lex_isatend(l)) return false;
    if (*(l->current) == c) {
        l->current++;
        return true;
    }
    return false;
}

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
                morpho_writeerrorwithid(err, COMPILE_UNTERMINATEDCOMMENT, startline, startpsn);
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
/*
            case '\n':
                lex_newline(l);
                lex_advance(l);
                break;
*/
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
    
#ifdef MORPHO_STRINGINTERPOLATION
    char first = lex_previous(l);
#endif
    
    while (lex_peek(l) != '"' && !lex_isatend(l)) {
        if (lex_peek(l) == '\n') lex_newline(l);
        
#ifdef MORPHO_STRINGINTERPOLATION
        /* Detect string interpolation */
        if (lex_peek(l) == '$' && lex_peekahead(l, 1) == '{') {
            lex_advance(l); lex_advance(l);
            lex_recordtoken(l, TOKEN_INTERPOLATION, tok);
            if (first=='"') l->interpolationlevel++;
            return true;
        }
#endif
        /* Detect an escaped character */
        if (lex_peek(l)=='\\') {
            lex_advance(l);
        }
        
        lex_advance(l);
    }
    
    if (lex_isatend(l)) {
        /* Unterminated string */
        morpho_writeerrorwithid(err, COMPILE_UNTERMINATEDSTRING, startline, startpsn);
        lex_recordtoken(l, TOKEN_INCOMPLETE, tok);
        return false;
    }
    
    lex_advance(l); /* Closing quote */
    
#ifdef MORPHO_STRINGINTERPOLATION
    if (l->interpolationlevel>0 && first=='}') l->interpolationlevel--;
#endif
    
    lex_recordtoken(l, TOKEN_STRING, tok);
    return true;
}

/** @brief Lex numbers
 *  @param[in]  l    the lexer
 *  @param[out] tok  token record to fill out
 *  @param[out] err  error struct to fill out on errors
 *  @returns true on success, false if an error occurs */
bool lex_number(lexer *l, token *tok, error *err) {
    tokentype type=TOKEN_INTEGER;
    while (lex_isdigit(lex_peek(l))) lex_advance(l);
    
    /* Fractional part */
    char next = '\0';
    if (lex_peek(l)!='\0') next=lex_peekahead(l, 1); // Prevent looking beyond buffer
    if (lex_peek(l) == '.' && (lex_isnumber(next)
#ifndef MORPHO_LOXCOMPATIBILITY
                               || lex_isspace(next) || next=='\0'
#endif
                               ) ) {
        type=TOKEN_NUMBER;
        lex_advance(l); /* Consume the '.' */
        while (lex_isdigit(lex_peek(l))) lex_advance(l);
    }
    
    /* Exponent */
    if (lex_peek(l) == 'e' || lex_peek(l) == 'E') {
        type=TOKEN_NUMBER;
        lex_advance(l); /* Consume the 'e' */
        
        /* Optional sign */
        if (lex_peek(l) == '+' || lex_peek(l) == '-') lex_advance(l);
        
        /* Exponent digits */
        while (lex_isdigit(lex_peek(l))) lex_advance(l);
    }
    
    /* Imaginary Numbers */
    if (lex_peek(l) =='i' && lex_peekahead(l, 1) == 'm'){
        /* mark this as an imaginary number*/
        type = TOKEN_IMAG;
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
 *  @returns type or TOKEN_SYMBOL if the match was not successful */
tokentype lex_checksymbol(lexer *l, int start, int length, char *match, tokentype type) {
    int toklength = (int) (l->current - l->start);
    int expectedlength = start + length;
    
    /* Compare, but don't bother calling memcmp if the lengths are different */
    if ((toklength == expectedlength) && (memcmp(l->start+start, match, length) == 0))
            return type;
    
    return TOKEN_SYMBOL;
}

tokentype lex_symboltype(lexer *l) {
    tokentype t = TOKEN_SYMBOL;
    
    lex_matchtoken(l, &t);
    
    return t;
}

/** @brief Lex symbols
 *  @param[in]  l    the lexer
 *  @param[out] tok  token record to fill out
 *  @param[out] err  error struct to fill out on errors
 *  @returns true on success, false if an error occurs */
bool lex_symbol(lexer *l, token *tok, error *err) {
    while (lex_isalpha(lex_peek(l)) || lex_isdigit(lex_peek(l))) lex_advance(l);
    
    tokentype typ = TOKEN_SYMBOL;
    if (l->matchkeywords) typ = lex_symboltype(l);
    
    lex_recordtoken(l, typ, tok);
    
    return true;
}

/* **********************************************************************
 * Customize the lexer
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
    
    qsort(l->defns, l->ndefns, sizeof(tokendefn), _lex_matchtokndefn);
}

/** @brief Choose whether the lexer should attempt to match keywords or simply return them as symbols. */
void lex_setmatchkeywords(lexer *l, bool match) {
    l->matchkeywords=match;
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
    /* Handle leading whitespace */
    if (! lex_skipwhitespace(l, tok, err)) return false; /* Check for failure */
    
    l->start=l->current;
    
    if (lex_isatend(l)) {
        lex_recordtoken(l, TOKEN_EOF, tok);
        return true;
    }
    
    char c = lex_advance(l);
    if (lex_isalpha(c)) return lex_symbol(l, tok, err);
    if (lex_isdigit(c)) return lex_number(l, tok, err);
    
    switch(c) {
        /* Single character tokens */
        case '(': lex_recordtoken(l, TOKEN_LEFTPAREN, tok); return true;
        case ')': lex_recordtoken(l, TOKEN_RIGHTPAREN, tok); return true;
        case '[': lex_recordtoken(l, TOKEN_LEFTSQBRACKET, tok); return true;
        case ']': lex_recordtoken(l, TOKEN_RIGHTSQBRACKET, tok); return true;
        case '{': lex_recordtoken(l, TOKEN_LEFTCURLYBRACKET, tok); return true;
        case '}':
#ifdef MORPHO_STRINGINTERPOLATION
            if (l->interpolationlevel>0) {
                return lex_string(l, tok, err);
            }
#endif
            lex_recordtoken(l, TOKEN_RIGHTCURLYBRACKET, tok);
            return true;
        case ';': lex_recordtoken(l, TOKEN_SEMICOLON, tok); return true;
        case ':': lex_recordtoken(l, TOKEN_COLON, tok); return true;
        case ',': lex_recordtoken(l, TOKEN_COMMA, tok); return true;
        case '^': lex_recordtoken(l, TOKEN_CIRCUMFLEX, tok); return true;
        case '?': lex_recordtoken(l, TOKEN_QUESTION, tok); return true;
        case '@': lex_recordtoken(l, TOKEN_AT, tok); return true;
        case '#': lex_recordtoken(l, TOKEN_HASH, tok); return true;
            
        /* Possible double character tokens */
        case '.':
            if (lex_match(l, '.')) {
                if (lex_match(l, '.')) {
                    lex_recordtoken(l, TOKEN_DOTDOTDOT , tok);
                } else lex_recordtoken(l, TOKEN_DOTDOT , tok);
            } else lex_recordtoken(l, TOKEN_DOT , tok);
            return true;
        case '+':
            lex_recordtoken(l, ( lex_match(l, '=') ? TOKEN_PLUSEQ : TOKEN_PLUS ), tok);
            return true;
        case '-':
            lex_recordtoken(l, ( lex_match(l, '=') ? TOKEN_MINUSEQ : TOKEN_MINUS ), tok);
            return true;
        case '*':
            lex_recordtoken(l, (lex_match(l, '=') ? TOKEN_STAREQ : TOKEN_STAR), tok);
            return true;
        case '/':
            lex_recordtoken(l, (lex_match(l, '=') ? TOKEN_SLASHEQ : TOKEN_SLASH), tok);
            return true;
        case '=':
            lex_recordtoken(l, (lex_match(l, '=') ? TOKEN_EQ : TOKEN_EQUAL), tok);
            return true;
        case '!':
            lex_recordtoken(l, (lex_match(l, '=') ? TOKEN_NEQ : TOKEN_EXCLAMATION), tok);
            return true;
        case '<':
            lex_recordtoken(l, (lex_match(l, '=') ? TOKEN_LTEQ : TOKEN_LT), tok);
            return true;
        case '>':
            lex_recordtoken(l, (lex_match(l, '=') ? TOKEN_GTEQ : TOKEN_GT), tok);
            return true;
        case '&':
            lex_recordtoken(l, (lex_match(l, '&') ? TOKEN_DBLAMP : TOKEN_AMP), tok);
            return true;
        case '|':
            lex_recordtoken(l, (lex_match(l, '|') ? TOKEN_DBLVBAR : TOKEN_VBAR), tok);
            return true;
        
        /* Strings */
        case '"':
            return lex_string(l, tok, err);
            
        /* Newlines */
        case '\n':
            lex_recordtoken(l, TOKEN_NEWLINE, tok);
            lex_newline(l);
            return true;
    }
    
    return false;
}

/** @brief Initialization/finalization */
void lexer_initialize(void) {
    // Ensure standardtokens is sorted; this is then used by default to reduce cost of initializing a lexer.
    int n;
    for (n=0; ; n++) if (standardtokens[n].string == NULL || strlen(standardtokens[n].string)==0) break;
    qsort(standardtokens, n, sizeof(tokendefn), _lex_matchtokndefn);
    
    // Retain the number of standardtokens
    nstandardtokens = n;
}

void lexer_finalize(void) {
    
}
