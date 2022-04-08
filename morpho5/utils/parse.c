/** @file parse.c
 *  @author T J Atherton 
 *
 *  @brief Lexer and parser
*/

#include <string.h>
#include <ctype.h>
#include "parse.h"
#include "object.h"
#include "common.h"
#include "cmplx.h"

/* **********************************************************************
 * Lexer
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
#ifdef MORPHO_STRINGINTERPOLATION
    l->interpolationlevel=0;
#endif
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
static bool lex_isatend(lexer *l) {
    return (*(l->current) == '\0');
}

/** @brief Checks if a character is a digit. Doesn't advance. */
static bool lex_isdigit(char c) {
    return (c>='0' && c<= '9');
}

/** @brief Checks if a character is alphanumeric or underscore.  Doesn't advance. */
static bool lex_isalpha(char c) {
    return (c>='a' && c<= 'z') || (c>='A' && c<= 'Z') || (c=='_');
}

/** @brief Checks if a character is whitespace.  Doesn't advance. */
static bool lex_isnumber(char c) {
    return isdigit(c);
}

/** @brief Checks if a character is whitespace.  Doesn't advance. */
static bool lex_isspace(char c) {
    return (c==' ') || (c=='\t') || (c=='\n') || (c=='\r');
}

/** @brief Advances the lexer by one character, returning the character */
static char lex_advance(lexer *l) {
    char c = *(l->current);
    l->current++;
    l->posn++;
    return c;
}

/** @brief Returns the previous character */
static char lex_previous(lexer *l) {
    if (l->current==l->start) return '\0';
    return *(l->current - 1);
}

/** @brief Returns the next character */
static char lex_peek(lexer *l) {
    return *(l->current);
}

/** @brief Returns n characters ahead. Caller should check that this is meaningfull. */
static char lex_peekahead(lexer *l, int n) {
    return *(l->current + n);
}

/** @brief Handle line counting */
static void lex_newline(lexer *l) {
    l->line++; l->posn=0;
}

/** @brief Advances the lexer by one character if it is equal to c
 * @returns true if the character matched, false otherwise */
static bool lex_match(lexer *l, char c) {
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
static bool lex_skipmultilinecomment(lexer *l, token *tok, error *err) {
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
static bool lex_skipcomment(lexer *l, token *tok, error *err) {
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
static bool lex_skipwhitespace(lexer *l, token *tok, error *err) {
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
static bool lex_string(lexer *l, token *tok, error *err) {
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
static bool lex_number(lexer *l, token *tok, error *err) {
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
static tokentype lex_checksymbol(lexer *l, int start, int length, char *match, tokentype type) {
    int toklength = (int) (l->current - l->start);
    int expectedlength = start + length;
    
    /* Compare, but don't bother calling memcmp if the lengths are different */
    if ((toklength == expectedlength) && (memcmp(l->start+start, match, length) == 0))
            return type;
    
    return TOKEN_SYMBOL;
}

/** @brief Determines if a token matches any of the reserved words */
tokentype lex_symboltype(lexer *l) {
    switch (l->start[0]) {
        case 'a': {
            tokentype type = lex_checksymbol(l, 1, 2, "nd", TOKEN_DBLAMP);
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 1, "s", TOKEN_AS);
            return type;
        }
        case 'b': return lex_checksymbol(l, 1, 4, "reak", TOKEN_BREAK);
        case 'c': {
            tokentype type = lex_checksymbol(l, 1, 4, "lass", TOKEN_CLASS);
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 7, "ontinue", TOKEN_CONTINUE);
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 4, "atch", TOKEN_CATCH);
            return type;
        }
        case 'd': return lex_checksymbol(l, 1, 1, "o", TOKEN_DO);
        case 'e': return lex_checksymbol(l, 1, 3, "lse", TOKEN_ELSE);
        case 'f': {
            tokentype type = lex_checksymbol(l, 1, 2, "or", TOKEN_FOR);
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 4, "alse", TOKEN_FALSE);
#ifdef MORPHO_LOXCOMPATIBILITY
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 2, "un", TOKEN_FUNCTION);
#else
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 1, "n", TOKEN_FUNCTION);
#endif
            return type;
        }
        case 'h': return lex_checksymbol(l, 1, 3, "elp", TOKEN_QUESTION);
        case 'i': {
            tokentype type = lex_checksymbol(l, 1, 1, "f", TOKEN_IF);
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 1, "n", TOKEN_IN);
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 1, "s", TOKEN_IS);
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 5, "mport", TOKEN_IMPORT);
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 1, "m", TOKEN_IMAG);
            return type;
        }
        case 'n': return lex_checksymbol(l, 1, 2, "il", TOKEN_NIL);
        case 'o': return lex_checksymbol(l, 1, 1, "r", TOKEN_DBLVBAR);
        case 'p': return lex_checksymbol(l, 1, 4, "rint", TOKEN_PRINT);
        case 'r': return lex_checksymbol(l, 1, 5, "eturn", TOKEN_RETURN);
        case 's': {
            tokentype type = lex_checksymbol(l, 1, 3, "elf", TOKEN_SELF);
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 4, "uper", TOKEN_SUPER);
            return type;
        }
        case 't': {
            tokentype type = lex_checksymbol(l, 1, 3, "rue", TOKEN_TRUE);
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 2, "ry", TOKEN_TRY);
#ifdef MORPHO_LOXCOMPATIBILITY
            if (type==TOKEN_SYMBOL) type = lex_checksymbol(l, 1, 3, "his", TOKEN_SELF);
#endif
            return type;
        }
        case 'v': return lex_checksymbol(l, 1, 2, "ar", TOKEN_VAR);
        case 'w': return lex_checksymbol(l, 1, 4, "hile", TOKEN_WHILE);
    }
    
    return TOKEN_SYMBOL;
}

/** @brief Lex symbols
 *  @param[in]  l    the lexer
 *  @param[out] tok  token record to fill out
 *  @param[out] err  error struct to fill out on errors
 *  @returns true on success, false if an error occurs */
static bool lex_symbol(lexer *l, token *tok, error *err) {
    while (lex_isalpha(lex_peek(l)) || lex_isdigit(lex_peek(l))) lex_advance(l);
    
    /* It's a symbol for now... */
    lex_recordtoken(l, lex_symboltype(l), tok);
    
    return true;
}

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

/* **********************************************************************
 * Parser
 * ********************************************************************** */

/* -------------------------------
 * Initialize a parser
 * ------------------------------- */

/** @brief Initialize a parser
 *  @param p       the parser to initialize
 *  @param lex   lexer to use
 *  @param err   an error structure to fill out if necessary
 *  @param tree syntaxtree to fill out */
void parse_init(parser *p, lexer *lex, error *err, syntaxtree *tree) {
    p->current = TOKEN_BLANK;
    p->previous = TOKEN_BLANK;
    p->left = SYNTAXTREE_UNCONNECTED;
    p->lex=lex;
    p->err=err;
    p->tree=tree;
    syntaxtree_clear(tree);
    p->nl=false;
}

/* ------------------------------------------
 * Parser implementation functions (parselets)
 * ------------------------------------------- */

static void parse_error(parser *p, bool use_prev, errorid id, ... );
static bool parse_advance(parser *p);
static bool parse_consume(parser *p, tokentype type, errorid id);
void parse_synchronize(parser *p);

/* --- Prototypes --- */

static syntaxtreeindx parse_precedence(parser *p, precedence precendence);
static syntaxtreeindx parse_expression(parser *p);
static syntaxtreeindx parse_statement(parser *p);
static syntaxtreeindx parse_declaration(parser *p);
static syntaxtreeindx parse_declarationmulti(parser *p, int n, tokentype *end);

static syntaxtreeindx parse_integer(parser *p);
static syntaxtreeindx parse_number(parser *p);
static syntaxtreeindx parse_complex(parser *p);
static syntaxtreeindx parse_bool(parser *p);
static syntaxtreeindx parse_string(parser *p);
static syntaxtreeindx parse_dictionary(parser *p);
static syntaxtreeindx parse_interpolation(parser *p);
static syntaxtreeindx parse_nil(parser *p);
static syntaxtreeindx parse_symbol(parser *p);
static value parse_symbolasvalue(parser *p);
static syntaxtreeindx parse_self(parser *p);
static syntaxtreeindx parse_super(parser *p);
static syntaxtreeindx parse_variable(parser *p, errorid id);
static syntaxtreeindx parse_grouping(parser *p);
static syntaxtreeindx parse_unary(parser *p);
static syntaxtreeindx parse_binary(parser *p);
static syntaxtreeindx parse_assignby(parser *p);
static syntaxtreeindx parse_call(parser *p);
static syntaxtreeindx parse_index(parser *p);
static syntaxtreeindx parse_list(parser *p);
static syntaxtreeindx parse_anonymousfunction(parser *p);
static syntaxtreeindx parse_switch(parser *p);

static syntaxtreeindx parse_vardeclaration(parser *p);
static syntaxtreeindx parse_functiondeclaration(parser *p);
static syntaxtreeindx parse_classdeclaration(parser *p);
static syntaxtreeindx parse_importdeclaration(parser *p);

static syntaxtreeindx parse_printstatement(parser *p);
static syntaxtreeindx parse_expressionstatement(parser *p);
static syntaxtreeindx parse_blockstatement(parser *p);
static syntaxtreeindx parse_ifstatement(parser *p);
static syntaxtreeindx parse_whilestatement(parser *p);
static syntaxtreeindx parse_forstatement(parser *p);
static syntaxtreeindx parse_dostatement(parser *p);
static syntaxtreeindx parse_breakstatement(parser *p);
static syntaxtreeindx parse_returnstatement(parser *p);
static syntaxtreeindx parse_trystatement(parser *p);
static syntaxtreeindx parse_breakpointstatement(parser *p);

static syntaxtreeindx parse_statementterminator(parser *p);

static parserule *parse_getrule(parser *p, tokentype type);

syntaxtreeindx syntaxtree_addnode(syntaxtree *tree, syntaxtreenodetype type, value content, int line, int posn, syntaxtreeindx left, syntaxtreeindx right);

/* --- Utility functions --- */

/** Adds a node to the syntax tree. */
static inline syntaxtreeindx parse_addnode(parser *p, syntaxtreenodetype type, value content, token *tok, syntaxtreeindx left, syntaxtreeindx right) {
    syntaxtreeindx new = syntaxtree_addnode(p->tree, type, content, tok->line, tok->posn, left, right);
    p->left=new; /* Record this for a future infix operator to catch */
    return new;
}

/** Checks whether the current token matches a specified tokentype */
static bool parse_checktoken(parser *p, tokentype type) {
    return p->current.type==type;
}

/** Checks whether the current token matches any of the specified tokentypes */
static bool parse_checktokenmulti(parser *p, int n, tokentype *type) {
    for (int i=0; i<n; i++) {
        if (p->current.type==type[i]) return true;
    }
    
    return false;
}

/** Checks whether the current token matches a given type and advances if so. */
static bool parse_matchtoken(parser *p, tokentype type) {
    if (!parse_checktoken(p, type)) return false;
    parse_advance(p);
    return true;
}

/** Checks whether a possible statement terminator is next */
static bool parse_checkstatementterminator(parser *p) {
    return (parse_checktoken(p, TOKEN_SEMICOLON)
#ifdef MORPHO_NEWLINETERMINATORS
            || (p->nl)
            || parse_checktoken(p, TOKEN_EOF)
            || parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET)
#endif
            || parse_checktoken(p, TOKEN_IN)
            ) ;
}

/** Parse a statement terminator  */
static syntaxtreeindx parse_statementterminator(parser *p) {
    if (parse_checktoken(p, TOKEN_SEMICOLON)) {
        parse_advance(p);
#ifdef MORPHO_NEWLINETERMINATORS
    } else if (p->nl || parse_checktoken(p, TOKEN_EOF) || parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET)) {
#endif
    } else if (parse_checktoken(p, TOKEN_IN) || parse_checktoken(p, TOKEN_ELSE)) {
    } else {
        parse_error(p, true, COMPILE_MISSINGSEMICOLONEXP);
    }
    return SYNTAXTREE_UNCONNECTED;
}

/* --- Implementations --- */

/** Parses an expression */
syntaxtreeindx parse_expression(parser *p) {
    return parse_precedence(p, PREC_ASSIGN);
}

/** Parses an expression that may include an anonymous function */
syntaxtreeindx parse_pseudoexpression(parser *p) {
    if (parse_matchtoken(p, TOKEN_FUNCTION)) {
        return parse_anonymousfunction(p);
    } else {
        return parse_expression(p);
    }
}

/** @brief Parse statements
 *  @details Statements are things that are allowed inside control flow statements */
syntaxtreeindx parse_statement(parser *p) {
    if (parse_matchtoken(p, TOKEN_PRINT)) {
        return parse_printstatement(p);
    } else if (parse_matchtoken(p, TOKEN_IF)) {
        return parse_ifstatement(p);
    } else if (parse_matchtoken(p, TOKEN_WHILE)) {
        return parse_whilestatement(p);
    } else if (parse_matchtoken(p, TOKEN_FOR)) {
        return parse_forstatement(p);
    } else if (parse_matchtoken(p, TOKEN_DO)) {
        return parse_dostatement(p);
    } else if (parse_matchtoken(p, TOKEN_BREAK)) {
        return parse_breakstatement(p);
    } else if (parse_matchtoken(p, TOKEN_CONTINUE)) {
        return parse_breakstatement(p);
    } else if (parse_matchtoken(p, TOKEN_RETURN)) {
        return parse_returnstatement(p);
    } else if (parse_matchtoken(p, TOKEN_TRY)) {
        return parse_trystatement(p);
    } else if (parse_matchtoken(p, TOKEN_LEFTCURLYBRACKET)) {
        return parse_blockstatement(p);
    } else if (parse_matchtoken(p, TOKEN_AT)) {
        return parse_breakpointstatement(p);
    } else {
        return parse_expressionstatement(p);
    }
    return SYNTAXTREE_UNCONNECTED;
}

/** @brief Parse declarations
 *  @details Declarations define something (e.g. a variable or a function) OR
 *           a regular statement. They are *not* allowed in control flow statements. */
syntaxtreeindx parse_declaration(parser *p) {
    syntaxtreeindx ret=SYNTAXTREE_UNCONNECTED;
    
    if (parse_matchtoken(p, TOKEN_FUNCTION)) {
        ret=parse_functiondeclaration(p);
    } else if (parse_matchtoken(p, TOKEN_VAR)) {
        ret=parse_vardeclaration(p);
    } else if (parse_matchtoken(p, TOKEN_CLASS)) {
        ret=parse_classdeclaration(p);
    } else if (parse_matchtoken(p, TOKEN_IMPORT)) {
        ret=parse_importdeclaration(p);
    } else {
        ret=parse_statement(p);
    }
    
    if (!ERROR_SUCCEEDED(*(p->err))) {
        parse_synchronize(p);
    }
    return ret;
}

/** Parses multiple declarations, separated by ; separators
 *  @param p    the parser
 *  @param end  token type to terminate on [use TOKEN_EOF if no special terminator]
 *  @returns    the syntaxtreeindx of the parsed expression */
static syntaxtreeindx parse_declarationmulti(parser *p, int n, tokentype *end) {
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED, current=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    
    while (!parse_checktokenmulti(p, n, end) && !parse_checktoken(p, TOKEN_EOF)) {
        current=parse_declaration(p);
        
        /* If we now have more than one node, insert a sequence node */
        if (last!=SYNTAXTREE_UNCONNECTED) {
            current = parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, last, current);
        }
        
        last = current;
    }
    
    return current;
}

/** Parses an integer */
syntaxtreeindx parse_integer(parser *p) {
    long f = strtol(p->previous.start, NULL, 10);
    return parse_addnode(p, NODE_INTEGER, MORPHO_INTEGER(f), &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses a number */
syntaxtreeindx parse_number(parser *p) {
    double f = strtod(p->previous.start, NULL);
    return parse_addnode(p, NODE_FLOAT, MORPHO_FLOAT(f), &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}
syntaxtreeindx parse_complex(parser *p) {

    double f;
    if (p->previous.length==2) { // just a bare im symbol
        f = 1;
    } else {
        f = strtod(p->previous.start,NULL);
    }
    value c = MORPHO_OBJECT(object_newcomplex(0,f));
    return parse_addnode(p, NODE_IMAG, c, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses a bool */
static syntaxtreeindx parse_bool(parser *p) {
    return parse_addnode(p, NODE_BOOL,
        MORPHO_BOOL((p->previous.type==TOKEN_TRUE ? true : false)), &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Turn a token into a string */
static value parse_stringfromtoken(parser *p, unsigned int start, unsigned int length) {
    char str[p->previous.length];
    const char *in=p->previous.start;
    unsigned int k=0;
    
    for (unsigned int i=start; i<length; i++) {
        if (in[i]!='\\') { // Escape characters
            str[k]=in[i]; k++;
        } else {
            if (i<length-1) switch (in[i+1]) {
                case 'n':
                    str[k]='\n'; break;
                case 't':
                    str[k]='\t'; break;
                case 'r':
                    str[k]='\r'; break;
                default:
                    str[k]=in[i+1]; break;
            }
            i++; k++;
        }
    }
    
    return object_stringfromcstring(str, k);
}


/** Parses a string */
static syntaxtreeindx parse_string(parser *p) {
    value s = parse_stringfromtoken(p, 1, p->previous.length-1);
    if (MORPHO_ISNIL(s)) parse_error(p, true, ERROR_ALLOCATIONFAILED, OBJECT_STRINGLABEL);
    return parse_addnode(p, NODE_STRING, s, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** @brief: Parses a dictionary.
 * @details Dictionaries are a list of key/value pairs,  { key : value, key: value } */
static syntaxtreeindx parse_dictionary(parser *p) {
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED;
    last=parse_addnode(p, NODE_DICTIONARY, MORPHO_NIL, &p->current, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
    
    do {
        syntaxtreeindx key, val, pair;
        token tok=p->current; // Keep track of the token that corresponds to each key/value pair
        
        /* Parse the key/value pair */
        key=parse_expression(p);
        if (!parse_consume(p, TOKEN_COLON, PARSE_DCTSPRTR)) break;
        val=parse_expression(p);
        
        /* Create an entry node */
        pair=parse_addnode(p, NODE_DICTENTRY, MORPHO_NIL, &tok, key, val);
        
        /* These are linked into a chain of dictionary nodes */
        last=parse_addnode(p, NODE_DICTIONARY, MORPHO_NIL, &tok, last, pair);
        
        if (!parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET)) {
            if (!parse_consume(p, TOKEN_COMMA, PARSE_DCTENTRYSPRTR)) break;
        }
    } while(!parse_matchtoken(p, TOKEN_RIGHTCURLYBRACKET) && !parse_checktoken(p, TOKEN_EOF));
    
    return last;
}

/** Parses a string interpolation. */
static syntaxtreeindx parse_interpolation(parser *p) {
    token tok = p->previous;
    
    /* First copy the string */
    //value s = object_stringfromcstring(tok.start+1, tok.length-3);
    value s = parse_stringfromtoken(p, 1, tok.length-2);
    if (MORPHO_ISNIL(s)) parse_error(p, true, ERROR_ALLOCATIONFAILED, OBJECT_STRINGLABEL);
    
    syntaxtreeindx left=SYNTAXTREE_UNCONNECTED, right=SYNTAXTREE_UNCONNECTED;
    
    left = parse_expression(p);
    if (parse_matchtoken(p, TOKEN_STRING)) {
        right = parse_string(p);
    } else if (parse_matchtoken(p, TOKEN_INTERPOLATION)) {
        right = parse_interpolation(p);
    } else {
        parse_error(p, false, COMPILE_INCOMPLETESTRINGINT);
    }
    
    return parse_addnode(p, NODE_INTERPOLATION, s, &tok, left, right);
}

/** Parses nil */
static syntaxtreeindx parse_nil(parser *p) {
    return parse_addnode(p, NODE_NIL,
        MORPHO_NIL, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses a symbol */
static syntaxtreeindx parse_symbol(parser *p) {
    value s = object_stringfromcstring(p->previous.start, p->previous.length);
    if (MORPHO_ISNIL(s)) parse_error(p, true, ERROR_ALLOCATIONFAILED, OBJECT_SYMBOLLABEL);

    return parse_addnode(p, NODE_SYMBOL, s, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses a symbol into a value */
static value parse_symbolasvalue(parser *p) {
    value s = object_stringfromcstring(p->previous.start, p->previous.length);
    if (MORPHO_ISNIL(s)) parse_error(p, true, ERROR_ALLOCATIONFAILED, OBJECT_SYMBOLLABEL);

    return s;
}

/** Parses a variable name, or raises and error if a symbol isn't found */
static syntaxtreeindx parse_variable(parser *p, errorid id) {
    parse_consume(p, TOKEN_SYMBOL, id);
    return parse_symbol(p);
}

/** Parses a self token */
static syntaxtreeindx parse_self(parser *p) {
    return parse_addnode(p, NODE_SELF, MORPHO_NIL, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses a super token */
static syntaxtreeindx parse_super(parser *p) {
    if (!parse_checktoken(p, TOKEN_DOT)) {
        parse_error(p, false, COMPILE_EXPECTDOTAFTERSUPER);
    }

    return parse_addnode(p, NODE_SUPER, MORPHO_NIL, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses an expression in parentheses */
static syntaxtreeindx parse_grouping(parser *p) {
    syntaxtreeindx indx;
    indx = parse_addnode(p, NODE_GROUPING, MORPHO_NIL, &p->previous, parse_expression(p), SYNTAXTREE_UNCONNECTED);
    parse_consume(p, TOKEN_RIGHTPAREN, COMPILE_MISSINGPARENTHESIS);
    return indx;
}

/** Parse a unary operator */
static syntaxtreeindx parse_unary(parser *p) {
    token start = p->previous;
    syntaxtreenodetype nodetype=NODE_LEAF;
    
    /* Determine which operator */
    switch (start.type) {
        case TOKEN_MINUS: nodetype = NODE_NEGATE; break;
        case TOKEN_EXCLAMATION: nodetype = NODE_NOT; break;
        case TOKEN_AT: nodetype = NODE_BREAKPOINT; break;
        default:
            UNREACHABLE("unhandled unary operator [Check parser definition table]");
    }
    
    /* Now add this node */
    return parse_addnode(p, nodetype, MORPHO_NIL, &start, parse_precedence(p, PREC_UNARY), SYNTAXTREE_UNCONNECTED);
}

/** Parse a binary operator */
static syntaxtreeindx parse_binary(parser *p) {
    token start = p->previous;
    syntaxtreenodetype nodetype=NODE_LEAF;
    enum {LEFT, RIGHT} assoc = LEFT; /* for left associative operators */
    
    /* Determine which operator */
    switch (start.type) {
        case TOKEN_EQUAL:
            nodetype = NODE_ASSIGN;
            assoc = RIGHT;
            break;
            
        case TOKEN_PLUS:        nodetype = NODE_ADD; break;
        case TOKEN_MINUS:       nodetype = NODE_SUBTRACT; break;
        case TOKEN_STAR:        nodetype = NODE_MULTIPLY; break;
        case TOKEN_SLASH:       nodetype = NODE_DIVIDE; break;
        case TOKEN_CIRCUMFLEX:
            nodetype = NODE_POW;
            assoc = RIGHT; 
            break;
        
        case TOKEN_EQ:          nodetype = NODE_EQ; break;
        case TOKEN_NEQ:         nodetype = NODE_NEQ; break;
        case TOKEN_LT:          nodetype = NODE_LT; break;
        case TOKEN_GT:          nodetype = NODE_GT; break;
        case TOKEN_LTEQ:        nodetype = NODE_LTEQ; break;
        case TOKEN_GTEQ:        nodetype = NODE_GTEQ; break;
            
        case TOKEN_DOT:         nodetype = NODE_DOT; break;
            
        case TOKEN_DBLAMP:      nodetype = NODE_AND; break;
        case TOKEN_DBLVBAR:     nodetype = NODE_OR; break;
        default:
            UNREACHABLE("unhandled binary operator [Check parser definition table]");
    }
    
    parserule *rule=parse_getrule(p, start.type);
    syntaxtreeindx left=p->left;
    syntaxtreeindx right=SYNTAXTREE_UNCONNECTED;
    
    /* Check if we have a right hand side. */
    if (parse_checktoken(p, TOKEN_EOF)) {
        parse_error(p, true, COMPILE_INCOMPLETEEXPRESSION);
    } else {
        if (nodetype==NODE_ASSIGN &&
            parse_matchtoken(p, TOKEN_FUNCTION)) {
            right=parse_anonymousfunction(p);
        }
        right = parse_precedence(p, rule->precedence + (assoc == LEFT ? 1 : 0));
    }
    
    /* Now add this node */
    return parse_addnode(p, nodetype, MORPHO_NIL, &start, left, right);
}

/** Parse operators like +=, -=, *= etc. */
static syntaxtreeindx parse_assignby(parser *p) {
    token start = p->previous;
    syntaxtreenodetype nodetype=NODE_LEAF;
    
    /* Determine which operator */
    switch (start.type) {
        case TOKEN_PLUSEQ:      nodetype = NODE_ADD; break;
        case TOKEN_MINUSEQ:     nodetype = NODE_SUBTRACT; break;
        case TOKEN_STAREQ:      nodetype = NODE_MULTIPLY; break;
        case TOKEN_SLASHEQ:     nodetype = NODE_DIVIDE; break;
        default:
            UNREACHABLE("unhandled assignment operator [Check parser definition table]");
    }
    
    parserule *rule=parse_getrule(p, start.type);
    syntaxtreeindx left=p->left;
    syntaxtreeindx right=SYNTAXTREE_UNCONNECTED;
    
    /* Check if we have a right hand side. */
    if (parse_checktoken(p, TOKEN_EOF)) {
        parse_error(p, true, COMPILE_INCOMPLETEEXPRESSION);
    } else {
        right = parse_precedence(p, rule->precedence);
    }
    
    right=parse_addnode(p, nodetype, MORPHO_NIL, &start, left, right);
    
    /* Now add this node */
    return parse_addnode(p, NODE_ASSIGN, MORPHO_NIL, &start, left, right);
}

/** Parses a range */
static syntaxtreeindx parse_range(parser *p) {
    token start = p->previous;
    bool inclusive = (start.type==TOKEN_DOTDOT);
    
    syntaxtreeindx left=p->left;
    syntaxtreeindx right=parse_expression(p);
    syntaxtreeindx one=SYNTAXTREE_UNCONNECTED;
    if (!inclusive) {
        one=parse_addnode(p, NODE_INTEGER, MORPHO_INTEGER(1), &start, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
        right=parse_addnode(p, NODE_SUBTRACT, MORPHO_NIL, &start, right, one);
    }
    syntaxtreeindx out=parse_addnode(p, NODE_RANGE, MORPHO_NIL, &start, left, right);
    
    if (parse_matchtoken(p, TOKEN_COLON)) {
        syntaxtreeindx step=parse_expression(p);
        if (!inclusive) p->tree->tree.data[right].right=step;
        out=parse_addnode(p, NODE_RANGE, MORPHO_NIL, &start, out, step);
    }
    
    return out;
}

/** @brief Parses an argument list
 * @param[in]  p     the parser
 * @param[in]  rightdelimiter  token type that denotes the end of the arguments list
 * @param[out] nargs the number of arguments
 * @returns indx of the arguments list
 * @details Note that the arguments are output in reverse order, i.e. the
 *          first argument is deepest in the tree. */
static syntaxtreeindx parse_arglist(parser *p, tokentype rightdelimiter, unsigned int *nargs) {
    syntaxtreeindx prev=SYNTAXTREE_UNCONNECTED, current=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    unsigned int n=0;
    bool varg=false;
    
    if (!parse_checktoken(p, rightdelimiter)) {
        do {
            if (parse_matchtoken(p, TOKEN_DOTDOTDOT)) {
				// If we are trying to index something 
				// then ... represents an open range
				if (rightdelimiter == TOKEN_RIGHTSQBRACKET){
					
				}
                else if (varg) parse_error(p, true, PARSE_ONEVARPR);
                varg = true;
            } else if (varg) parse_error(p, true, PARSE_VARPRLST);
            
            current=parse_pseudoexpression(p);

            if (varg) current=parse_addnode(p, NODE_RANGE, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, current);
            
            n++;
            
            current=parse_addnode(p, NODE_ARGLIST, MORPHO_NIL, &start, prev, current);
            
            prev = current;
        } while (parse_matchtoken(p, TOKEN_COMMA));
    }
    
    /* Output the number of args */
    if (nargs) *nargs=n;
    
    return current;
}

/** Parse a function call */
static syntaxtreeindx parse_call(parser *p) {
    token start = p->previous;
    syntaxtreeindx left=p->left;
    syntaxtreeindx right;
    unsigned int nargs;
    
    /* Abandon parsing the function call if a newline is detected between the symbol and opening parenthesis */
//    if (parse_checkprevnl(p)) {
//        return p->left;
//    }
    
    right=parse_arglist(p, TOKEN_RIGHTPAREN, &nargs);
    
    parse_consume(p, TOKEN_RIGHTPAREN, COMPILE_CALLRGHTPARENMISSING);
    
    return parse_addnode(p, NODE_CALL, MORPHO_NIL, &start, left, right);
}

/** Parse index
        index
       /          \
  symbol         indices
 */
static syntaxtreeindx parse_index(parser *p) {
    token start = p->previous;
    syntaxtreeindx left=p->left;
    syntaxtreeindx right;
    unsigned int nindx;
    
    right=parse_arglist(p, TOKEN_RIGHTSQBRACKET, &nindx);
    
    parse_consume(p, TOKEN_RIGHTSQBRACKET, COMPILE_CALLRGHTPARENMISSING);
    
    return parse_addnode(p, NODE_INDEX, MORPHO_NIL, &start, left, right);
}

/** Parse list  */
static syntaxtreeindx parse_list(parser *p) {
    unsigned int nindx;
    token start = p->previous;
    
    syntaxtreeindx right=parse_arglist(p, TOKEN_RIGHTSQBRACKET, &nindx);
    parse_consume(p, TOKEN_RIGHTSQBRACKET, COMPILE_CALLRGHTPARENMISSING);

    return parse_addnode(p, NODE_LIST, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, right);
}

/** Parses an anonymous function */
static syntaxtreeindx parse_anonymousfunction(parser *p) {
    token start = p->previous;
    syntaxtreeindx args=SYNTAXTREE_UNCONNECTED,
                   body=SYNTAXTREE_UNCONNECTED;
    
    /* Parameter list */
    parse_consume(p, TOKEN_LEFTPAREN, COMPILE_FNLEFTPARENMISSING);
    args=parse_arglist(p, TOKEN_RIGHTPAREN, NULL);
    parse_consume(p, TOKEN_RIGHTPAREN, COMPILE_FNRGHTPARENMISSING);
    
    /* Function body */
    body=parse_expression(p);
    
    return parse_addnode(p, NODE_FUNCTION, MORPHO_NIL, &start, args, body);
}

/** @brief: Parses a switch block
 * @details Switch blocks are key/statement pairs. Each pair is stored in a NODE_DICTIONARY list */
static syntaxtreeindx parse_switch(parser *p) {
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED;
    
    while(!parse_matchtoken(p, TOKEN_RIGHTCURLYBRACKET) && !parse_checktoken(p, TOKEN_EOF)) {
        syntaxtreeindx key, statements, pair;
        token tok=p->current; // Keep track of the token that corresponds to each key/value pair
        
        /* Parse the key/value pair */
        key=parse_expression(p);
        if (!parse_consume(p, TOKEN_COLON, PARSE_SWTCHSPRTR)) break;
        tokentype terminators[] = { TOKEN_STRING, TOKEN_INTEGER, TOKEN_NUMBER, TOKEN_TRUE, TOKEN_FALSE, TOKEN_NIL, TOKEN_RIGHTCURLYBRACKET };
        statements=parse_declarationmulti(p, 7, terminators);
        
        /* Create an entry node */
        pair=parse_addnode(p, NODE_DICTENTRY, MORPHO_NIL, &tok, key, statements);
        
        /* These are linked into a chain of sequence nodes */
        last=parse_addnode(p, NODE_DICTIONARY, MORPHO_NIL, &tok, last, pair);
        
    };
    
    return last;
}

/* -------------------------------
 * Declarations
 * ------------------------------- */

/** Parses a variable declaration */
static syntaxtreeindx parse_vardeclaration(parser *p) {
    syntaxtreeindx symbol, initializer, out=SYNTAXTREE_UNCONNECTED, last=SYNTAXTREE_UNCONNECTED;
    
    do {
        token start = p->previous;
        
        symbol=parse_variable(p, COMPILE_VAREXPECTED);
        
        if (parse_matchtoken(p, TOKEN_LEFTSQBRACKET)) {
            symbol=parse_index(p);
        }
        
        if (parse_matchtoken(p, TOKEN_EQUAL)) {
            initializer=parse_pseudoexpression(p);
        } else initializer=SYNTAXTREE_UNCONNECTED;
        
        out=parse_addnode(p, NODE_DECLARATION, MORPHO_NIL, &start, symbol, initializer);
        
        if (last!=SYNTAXTREE_UNCONNECTED) {
            out=parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, last, out);
        }
        
        last=out;
    } while (parse_matchtoken(p, TOKEN_COMMA));
    
    parse_statementterminator(p);
    
    return out;
}

/** Parses a function declaration */
static syntaxtreeindx parse_functiondeclaration(parser *p) {
    value name=MORPHO_NIL;
    token start = p->previous;
    syntaxtreeindx args=SYNTAXTREE_UNCONNECTED,
                   body=SYNTAXTREE_UNCONNECTED;
    
    /* Function name */
    if (parse_matchtoken(p, TOKEN_SYMBOL)) {
        name=parse_symbolasvalue(p);
    } else parse_error(p, false, COMPILE_FNNAMEMISSING);
    
    /* Parameter list */
    parse_consume(p, TOKEN_LEFTPAREN, COMPILE_FNLEFTPARENMISSING);
    args=parse_arglist(p, TOKEN_RIGHTPAREN, NULL);
    parse_consume(p, TOKEN_RIGHTPAREN, COMPILE_FNRGHTPARENMISSING);
    
    /* Function body */
    parse_consume(p, TOKEN_LEFTCURLYBRACKET, COMPILE_FNLEFTCURLYMISSING);
    body=parse_blockstatement(p);
    
    return parse_addnode(p, NODE_FUNCTION, name, &start, args, body);
}

/* Parses a class declaration */
static syntaxtreeindx parse_classdeclaration(parser *p) {
    value name=MORPHO_NIL;
    value sname=MORPHO_NIL;
    syntaxtreeindx sclass=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    /* Class name */
    if (parse_matchtoken(p, TOKEN_SYMBOL)) {
        name=parse_symbolasvalue(p);
    } else parse_error(p, false, COMPILE_EXPECTCLASSNAME);
    
    /* Extract a superclass name */
    if (parse_matchtoken(p, TOKEN_LT) || parse_matchtoken(p, TOKEN_IS)) {
        parse_consume(p, TOKEN_SYMBOL, COMPILE_EXPECTSUPER);
        sname=parse_symbolasvalue(p);
        sclass=parse_addnode(p, NODE_SYMBOL, sname, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
    }
    
    parse_consume(p, TOKEN_LEFTCURLYBRACKET, COMPILE_CLASSLEFTCURLYMISSING);
    /* Method declarations */
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED, current=SYNTAXTREE_UNCONNECTED;
    
    while (!parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET) && !parse_checktoken(p, TOKEN_EOF)) {
        current=parse_functiondeclaration(p);
        
        /* If we now have more than one node, insert a sequence node */
        if (last!=SYNTAXTREE_UNCONNECTED) {
            current = parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, last, current);
        }
        
        last = current;
    }
    
    parse_consume(p, TOKEN_RIGHTCURLYBRACKET, COMPILE_CLASSRGHTCURLYMISSING);
    
    return parse_addnode(p, NODE_CLASS, name, &start, sclass, current);
}

/** Parse an import declaration.
 *          IMPORT
 *         /              \
 *     module           FOR   or as
 *                    \
 *                   ( items )
 */
static syntaxtreeindx parse_importdeclaration(parser *p) {
    syntaxtreeindx modulename=SYNTAXTREE_UNCONNECTED, right=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    if (parse_matchtoken(p, TOKEN_STRING)) {
        modulename=parse_string(p);
    } else if (parse_matchtoken(p, TOKEN_SYMBOL)){
        modulename=parse_symbol(p);
    } else {
        parse_error(p, true, COMPILE_IMPORTMISSINGNAME);
        return SYNTAXTREE_UNCONNECTED;
    }
    
    if (!parse_checkstatementterminator(p)) {
        if (parse_matchtoken(p, TOKEN_AS)) {
            if (parse_matchtoken(p, TOKEN_SYMBOL)) {
                right=parse_symbol(p);
            } else parse_error(p, true, COMPILE_IMPORTASSYMBL);
        } else if (parse_matchtoken(p, TOKEN_FOR)) {
            do {
                if (parse_matchtoken(p, TOKEN_SYMBOL)) {
                    syntaxtreeindx symbl=parse_symbol(p);
                    right=parse_addnode(p, NODE_FOR, MORPHO_NIL, &p->previous, right, symbl);
                } else parse_error(p, true, COMPILE_IMPORTFORSYMBL);
            } while (parse_matchtoken(p, TOKEN_COMMA));
        } else {
            parse_error(p, true, COMPILE_IMPORTUNEXPCTDTOK);
        }
    }
    
    parse_statementterminator(p);
    
    return parse_addnode(p, NODE_IMPORT, MORPHO_NIL, &start, modulename, right);
}

/* -------------------------------
 * Statements
 * ------------------------------- */

/** Parse a print statement */
static syntaxtreeindx parse_printstatement(parser *p) {
    token start = p->previous;
    syntaxtreeindx left = parse_pseudoexpression(p);
    parse_statementterminator(p);
    return parse_addnode(p, NODE_PRINT, MORPHO_NIL, &start, left, SYNTAXTREE_UNCONNECTED);
}

/** Parse an expression statement */
static syntaxtreeindx parse_expressionstatement(parser *p) {
    syntaxtreeindx out = parse_expression(p);
    parse_statementterminator(p);
    return out;
}

/** @brief Parse a block statement.
 *  @details This wraps up a sequence of statements in a SCOPE node:
 *                     SCOPE
 *                    /     \
 *                   -       body
 **/
static syntaxtreeindx parse_blockstatement(parser *p) {
    syntaxtreeindx body = SYNTAXTREE_UNCONNECTED,
                   scope = SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    tokentype terminator[] = { TOKEN_RIGHTCURLYBRACKET };
    
    body = parse_declarationmulti(p, 1, terminator);
    if (parse_checktoken(p, TOKEN_EOF)) {
        parse_error(p, false, COMPILE_INCOMPLETEEXPRESSION);
    } else {
        parse_consume(p, TOKEN_RIGHTCURLYBRACKET, COMPILE_MISSINGSEMICOLONEXP);
    }
    
    scope=parse_addnode(p, NODE_SCOPE, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, body);
    
    return scope;
}

/** Parse an if statement */
static syntaxtreeindx parse_ifstatement(parser *p) {
    syntaxtreeindx  cond=SYNTAXTREE_UNCONNECTED,
                    then=SYNTAXTREE_UNCONNECTED,
                    els=SYNTAXTREE_UNCONNECTED,
                    out=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    parse_consume(p, TOKEN_LEFTPAREN, COMPILE_IFLFTPARENMISSING);
    cond=parse_expression(p);
    parse_consume(p, TOKEN_RIGHTPAREN, COMPILE_IFRGHTPARENMISSING);
    
    token thentok = p->current;
    then=parse_statement(p);
    
    if (parse_checktoken(p, TOKEN_ELSE)) {
        parse_advance(p);
        els=parse_statement(p);
        
        /* Create an additional node that contains both statements */
        then = parse_addnode(p, NODE_THEN, MORPHO_NIL, &thentok, then, els);
    }
    
    out=parse_addnode(p, NODE_IF, MORPHO_NIL, &start, cond, then);
    
    return out;
}

/** Parse a while statement */
static syntaxtreeindx parse_whilestatement(parser *p) {
    syntaxtreeindx  cond=SYNTAXTREE_UNCONNECTED,
                    body=SYNTAXTREE_UNCONNECTED,
                    out=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    parse_consume(p, TOKEN_LEFTPAREN, COMPILE_WHILELFTPARENMISSING);
    cond=parse_expression(p);
    parse_consume(p, TOKEN_RIGHTPAREN, COMPILE_IFRGHTPARENMISSING);
    body=parse_statement(p);
    
    out=parse_addnode(p, NODE_WHILE, MORPHO_NIL, &start, cond, body);
    
    return out;
}

/** Parse a for statement. */
static syntaxtreeindx parse_forstatement(parser *p) {
    syntaxtreeindx init=SYNTAXTREE_UNCONNECTED, // Initializer
                   cond=SYNTAXTREE_UNCONNECTED, // Condition
                   body=SYNTAXTREE_UNCONNECTED, // Loop body
                   final=SYNTAXTREE_UNCONNECTED; // Final statement
    syntaxtreeindx out=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    bool forin=false;
 
    parse_consume(p, TOKEN_LEFTPAREN, COMPILE_FORLFTPARENMISSING);
    if (parse_matchtoken(p, TOKEN_SEMICOLON)) {
        
    } else if (parse_matchtoken(p, TOKEN_VAR)) {
        init=parse_vardeclaration(p);
    } else {
        init=parse_expression(p);
        while (parse_matchtoken(p, TOKEN_COMMA)) {
            syntaxtreeindx new=parse_expressionstatement(p);
            init=parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &p->current, init, new);
        }
        parse_matchtoken(p, TOKEN_SEMICOLON);
    }
    
    if (parse_matchtoken(p, TOKEN_IN)) {
        /* If its an for..in loop, parse the collection */
        cond=parse_expression(p);
        forin=true;
    } else {
        /* Otherwise, parse the condition and final clause in a traditional for loop. */
        if (!parse_matchtoken(p, TOKEN_SEMICOLON)) {
            cond=parse_expressionstatement(p);
        }
        
        if (!parse_checktoken(p, TOKEN_RIGHTPAREN)) {
            final=parse_expression(p);
        }
    }
    parse_consume(p, TOKEN_RIGHTPAREN, COMPILE_FORRGHTPARENMISSING);
    
    if (!parse_checkstatementterminator(p)) {
        body=parse_statement(p);
    }
    
    if (forin) {
        /* A for..in loop is parsed as follows:
         *
         *                 forin
         *                /     \
         *               in      body
         *              /  \
         *          init    collection
         */
         syntaxtreeindx innode=parse_addnode(p, NODE_IN, MORPHO_NIL, &start, init, cond);
         out=parse_addnode(p, NODE_FOR, MORPHO_NIL, &start, innode, body);
    } else {
        /* A traditional for loop is parsed into an equivalent while loop:
         * -> for (init; cond; inc) body;
         *
         * becomes
         *              scope
         *                   \
         *                    ;
         *                   / \
         *               init   while
         *                     /     \
         *                 cond       ; // The presence of the seq. indicates a for loop
         *                           / \
         *                       body   inc
         * */
        syntaxtreeindx loop=parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, body, final);
        syntaxtreeindx whil=parse_addnode(p, NODE_WHILE, MORPHO_NIL, &start, cond, loop);
        syntaxtreeindx seq=parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, init, whil);
        out=parse_addnode(p,NODE_SCOPE, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, seq);
    }
    
    return out;
}

/** Parses a do...while loop */
static syntaxtreeindx parse_dostatement(parser *p) {
    syntaxtreeindx body=SYNTAXTREE_UNCONNECTED, // Loop body
                   cond=SYNTAXTREE_UNCONNECTED; // Condition
    syntaxtreeindx out=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    
    body=parse_statement(p);
    
    parse_consume(p, TOKEN_WHILE, PARSE_EXPCTWHL);
    parse_consume(p, TOKEN_LEFTPAREN, COMPILE_WHILELFTPARENMISSING);
    cond=parse_expression(p);
    parse_consume(p, TOKEN_RIGHTPAREN, COMPILE_IFRGHTPARENMISSING);
    
    /* Optional statement terminator */
    if (parse_checkstatementterminator(p)) {
        parse_statementterminator(p);
    }
    
    out=parse_addnode(p, NODE_DO, MORPHO_NIL, &start, body, cond);
    
    return out;
}

/** Parses a break or continue statement */
static syntaxtreeindx parse_breakstatement(parser *p) {
    token start = p->previous;
    
    parse_statementterminator(p);
    
    return parse_addnode(p, (start.type==TOKEN_BREAK ? NODE_BREAK: NODE_CONTINUE), MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parse a return statement */
static syntaxtreeindx parse_returnstatement(parser *p) {
    token start = p->previous;
    syntaxtreeindx left = SYNTAXTREE_UNCONNECTED;
    
    if (!parse_checkstatementterminator(p)) {
        left = parse_pseudoexpression(p);
    }
    
    parse_statementterminator(p);
    
    return parse_addnode(p, NODE_RETURN, MORPHO_NIL, &start, left, SYNTAXTREE_UNCONNECTED);
}

/** Parse a try/catch statement
        try
      /          \
    body        catch block */
static syntaxtreeindx parse_trystatement(parser *p) {
    syntaxtreeindx try=SYNTAXTREE_UNCONNECTED, // Try block
                   catch=SYNTAXTREE_UNCONNECTED; // Catch dictionary
    syntaxtreeindx out=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    
    try=parse_statement(p);
    
    parse_consume(p, TOKEN_CATCH, PARSE_EXPCTCTCH);
    parse_consume(p, TOKEN_LEFTCURLYBRACKET, PARSE_CATCHLEFTCURLYMISSING);
    
    catch=parse_switch(p);
    
    /* Optional statement terminator */
    if (parse_checkstatementterminator(p)) {
        parse_statementterminator(p);
    }
    
    out=parse_addnode(p, NODE_TRY, MORPHO_NIL, &start, try, catch);
    
    return out;
}

/** Parse a breakpoint statement */
static syntaxtreeindx parse_breakpointstatement(parser *p) {
    token start = p->previous;
    
    if (parse_checkstatementterminator(p)) {
        parse_statementterminator(p);
    }
    
    return parse_addnode(p, NODE_BREAKPOINT, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Keep parsing til the end of a statement boundary. */
void parse_synchronize(parser *p) {
    while (p->current.type!=TOKEN_EOF) {
        /** Align */
        if (p->previous.type == TOKEN_SEMICOLON) return;
        switch (p->current.type) {
            case TOKEN_CLASS:
            case TOKEN_FUNCTION:
            case TOKEN_VAR:
            case TOKEN_FOR:
            case TOKEN_IF:
            case TOKEN_WHILE:
            case TOKEN_PRINT:
            case TOKEN_RETURN:
                return;
            default:
                ;
        }
        
        parse_advance(p);
    }
}

/* -------------------------------
 * The parser definition table
 * ------------------------------- */

/** @brief Parse table.
 *  Each line in the table defines the parserule(s) for a specific token type.
 *  @warning It is imperative that this table be in the same order as the tokentype enum */
#define UNUSED                         { NULL,    NULL,      PREC_NONE }
#define PREFIX(fn)                     { fn,      NULL,      PREC_NONE }
#define INFIX(fn, prec)                { NULL,    fn,        prec      }
#define MIXFIX(unaryfn, infixfn, prec) { unaryfn, infixfn,   prec      }

parserule rules[] = {
    UNUSED,                                            // TOKEN_NONE
    UNUSED,                                            // TOKEN_NEWLINE
    UNUSED,                                            // TOKEN_QUESTION

    PREFIX(parse_string),                              // TOKEN_STRING
    PREFIX(parse_interpolation),                       // TOKEN_INTERPOLATION
    PREFIX(parse_integer),                             // TOKEN_INTEGER
    PREFIX(parse_number),                              // TOKEN_NUMBER
    PREFIX(parse_symbol),                              // TOKEN_SYMBOL
    PREFIX(parse_bool),                                // TOKEN_TRUE
    PREFIX(parse_bool),                                // TOKEN_FALSE
    PREFIX(parse_nil),                                 // TOKEN_NIL
    PREFIX(parse_self),                                // TOKEN_SELF
    PREFIX(parse_super),                               // TOKEN_SUPER
    PREFIX(parse_complex),                             // TOKEN_IMAG

    MIXFIX(parse_grouping, parse_call, PREC_CALL),     // TOKEN_LEFTPAREN
    UNUSED,                                            // TOKEN_RIGHTPAREN
    MIXFIX(parse_list, parse_index, PREC_CALL),        // TOKEN_LEFTSQBRACKET
    UNUSED,                                            // TOKEN_RIGHTSQBRACKET
    PREFIX(parse_dictionary),                          // TOKEN_LEFTCURLYBRACKET
    UNUSED,                                            // TOKEN_RIGHTCURLYBRACKET

    UNUSED,                                            // TOKEN_COLON
    UNUSED,                                            // TOKEN_SEMICOLON
    UNUSED,                                            // TOKEN_COMMA
    
    INFIX(parse_binary, PREC_TERM),                    // TOKEN_PLUS
    MIXFIX(parse_unary, parse_binary, PREC_TERM),      // TOKEN_MINUS
    INFIX(parse_binary, PREC_FACTOR),                  // TOKEN_STAR
    INFIX(parse_binary, PREC_FACTOR),                  // TOKEN_SLASH
    INFIX(parse_binary, PREC_POW),                     // TOKEN_CIRCUMFLEX
    
    UNUSED,                                            // TOKEN_PLUSPLUS
    UNUSED,                                            // TOKEN_MINUSMINUS
    INFIX(parse_assignby, PREC_ASSIGN),                // TOKEN_PLUSEQ
    INFIX(parse_assignby, PREC_ASSIGN),                // TOKEN_MINUSEQ
    INFIX(parse_assignby, PREC_ASSIGN),                // TOKEN_STAREQ
    INFIX(parse_assignby, PREC_ASSIGN),                // TOKEN_SLASHEQ
    UNUSED,                                            // TOKEN_HASH
    PREFIX(parse_unary),                               // TOKEN_AT
    
    INFIX(parse_binary, PREC_CALL),                    // TOKEN_DOT
    INFIX(parse_range, PREC_RANGE),                    // TOKEN_DOTDOT
    INFIX(parse_range, PREC_RANGE),                    // TOKEN_DOTDOTDOT
    PREFIX(parse_unary),                               // TOKEN_EXCLAMATION
    UNUSED,                                            // TOKEN_AMP
    UNUSED,                                            // TOKEN_VBAR
    INFIX(parse_binary, PREC_AND),                     // TOKEN_DBLAMP
    INFIX(parse_binary, PREC_OR),                      // TOKEN_DBLVBAR
    INFIX(parse_binary, PREC_ASSIGN),                  // TOKEN_EQUAL
    INFIX(parse_binary, PREC_EQUALITY),                // TOKEN_EQ
    INFIX(parse_binary, PREC_EQUALITY),                // TOKEN_NEQ
    INFIX(parse_binary, PREC_COMPARISON),              // TOKEN_LT
    INFIX(parse_binary, PREC_COMPARISON),              // TOKEN_GT
    INFIX(parse_binary, PREC_COMPARISON),              // TOKEN_LTEQ
    INFIX(parse_binary, PREC_COMPARISON),              // TOKEN_GTEQ
    
    UNUSED,                                            // TOKEN_PRINT
    UNUSED,                                            // TOKEN_VAR
    UNUSED,                                            // TOKEN_IF
    UNUSED,                                            // TOKEN_ELSE
    UNUSED,                                            // TOKEN_IN
    UNUSED,                                            // TOKEN_WHILE
    UNUSED,                                            // TOKEN_FOR
    UNUSED,                                            // TOKEN_DO
    UNUSED,                                            // TOKEN_BREAK
    UNUSED,                                            // TOKEN_CONTINUE
    UNUSED,                                            // TOKEN_FUNCTION
    UNUSED,                                            // TOKEN_RETURN
    UNUSED,                                            // TOKEN_CLASS
    UNUSED,                                            // TOKEN_IMPORT
    UNUSED,                                            // TOKEN_AS
    UNUSED,                                            // TOKEN_IS
    UNUSED,                                            // TOKEN_TRY
    UNUSED,                                            // TOKEN_CATCH
    
    UNUSED,                                            // TOKEN_INCOMPLETE
    UNUSED,                                            // TOKEN_ERROR
    UNUSED,                                            // TOKEN_EOF
};

/** Get the rule to parse an element of type tokentype. */
static parserule *parse_getrule(parser *p, tokentype type) {
    return &rules[type];
}

/* -------------------------------
* Parser implementation functions
* ------------------------------- */

/** @brief Fills out the error record
 *  @param p        the parser
 *  @param use_prev use the previous token? [this is the more typical usage]
 *  @param id       error id
 *  @param ...      additional data for sprintf. */
static void parse_error(parser *p, bool use_prev, errorid id, ... ) {
    va_list args;
    token *tok = (use_prev ? &p->previous : &p->current);
    
    /** Only return the first error that occurs */
    if (p->err->id!=ERROR_NONE) return;
    
    va_start(args, id);
    morpho_writeerrorwithid(p->err, id, tok->line, tok->posn-tok->length, args);
    va_end(args);
}

/** @brief Advance the parser by one token
 *  @param   p the parser in use.
 *  @returns true on success, false otherwise */
static bool parse_advance(parser *p) {
    lexer *l = p->lex;
    
    p->previous=p->current;
    p->nl=false;
    
    for (;;) {
        lex(l, &p->current, p->err);
        
        /* Skip any newlines encountered */
        if (p->current.type==TOKEN_NEWLINE) {
            p->nl=true;
            continue;
        }
        
        if (p->current.type != TOKEN_ERROR) break;
        printf("UNHANDLED ERROR.\n");
    }
    
    return (p->err->cat==ERROR_NONE);
}

/** @brief Checks if the next token has the required type, otherwise generates an error.
 *  @param   p    the parser in use
 *  @param   type type to check
 *  @param   id   error id to generate if the token doesn't match
 *  @returns true on success */
static bool parse_consume(parser *p, tokentype type, errorid id) {
    if (p->current.type==type) {
        parse_advance(p);
        return true;
    }
    
    /* Raise an error */
    if (id!=ERROR_NONE) parse_error(p, true, id);
    return false;
}

/** @brief Continues parsing while tokens have a lower or equal precendece than a specified value.
 *  @param   p    the parser in use
 *  @param   precendence precedence value to keep below or equal to
 *  @returns syntaxtreeindx for the expression parsed */
static syntaxtreeindx parse_precedence(parser *p, precedence precendence) {
    parsefunction prefixrule=NULL, infixrule=NULL;
    syntaxtreeindx result;
    
    parse_advance(p);
    
    prefixrule = parse_getrule(p, p->previous.type)->prefix;
    
    if (!prefixrule) {
        parse_error(p, true, COMPILE_EXPECTEXPRESSION);
        return SYNTAXTREE_UNCONNECTED;
    }
    
    result=prefixrule(p);
    
    /* Now keep parsing while the tokens have lower precedence */
    while (precendence <= parse_getrule(p, p->current.type)->precedence) {
#ifdef MORPHO_NEWLINETERMINATORS
        /* Break if a newline is encountered before a function call */
        if (p->current.type==TOKEN_LEFTPAREN && p->nl) break;
#endif
        
        parse_advance(p);
        
        infixrule = parse_getrule(p, p->previous.type)->infix;
        if (infixrule) result=infixrule(p);
        else parse_error(p, true, 0);
    }
    
    return result;
}

/** Entry point into the parser */
bool parse(parser *p) {
    parse_advance(p);
    tokentype terminator[] = { TOKEN_EOF };
    
    p->tree->entry = parse_declarationmulti(p, 1, terminator);
    
    return (p->err->cat==ERROR_NONE);
}

/** Convenience function to parse a string into an array of values
 * @param[in] string - string to parse
 * @param[in] nmax      - maximum number of values to read
 * @param[in] v            - value array, filled out on return
 * @param[out] n          - number of values read
 * @param[out] err      - error structure filled out if an error occurs
 * @returns true if successful, false otherwise. */
bool parse_stringtovaluearray(char *string, unsigned int nmax, value *v, unsigned int *n, error *err) {
    lexer l;
    token tok;
    unsigned int k=0;
    bool minus=false;
    lex_init(&l, string, 0);
    
    do {
        if (!lex(&l, &tok, err)) return false;
        switch(tok.type) {
            case TOKEN_INTEGER: {
                long f = strtol(tok.start, NULL, 10);
                v[k]=MORPHO_INTEGER((minus ? -f : f)); k++; minus=false;
            }
                break;
            case TOKEN_NUMBER: {
                double f = strtod(tok.start, NULL);
                v[k]=MORPHO_FLOAT((minus ? -f : f)); k++; minus=false;
            }
                break;
            case TOKEN_MINUS:
                minus=true;
                break;
            case TOKEN_COMMA:
            case TOKEN_EOF:
                break;
            default:
                morpho_writeerrorwithid(err, PARSE_UNRECGNZEDTOK, ERROR_POSNUNIDENTIFIABLE, ERROR_POSNUNIDENTIFIABLE);
                return false; 
                break;
        }
    } while (tok.type!=TOKEN_EOF && k<nmax);
    
    if (n) *n=k;
    
    return true;
}
