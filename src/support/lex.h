/** @file lex.h
 *  @author T J Atherton and others (see below)
 *
 *  @brief Lexer 
*/

#ifndef lex_h
#define lex_h

#include <stdio.h>
#include "varray.h"
#include "error.h"

/** The lexer breaks an input stream into tokens, classifying them as it goes. */

/* -------------------------------------------------------
 * Tokens
 * ------------------------------------------------------- */

#define TOKEN_NONE -1

/** Token types are left as a generic int to facilitate reprogrammability in the future */
typedef int tokentype;

/** A token */
typedef struct {
    tokentype type; /** Type of the token */
    const char *start; /** Start of the token */
    unsigned int length; /** Its length */
    /* Position of the token in the source */
    int line; /** Source line */
    int posn; /** Character position of the end of the token */
} token;

/** Literal for a blank token */
#define TOKEN_BLANK ((token) {.type=TOKEN_NONE, .start=NULL, .length=0, .line=0, .posn=0} )

/* -------------------------------------------------------
 * Token definitions
 * ------------------------------------------------------- */

typedef struct {
    char *string;
    tokentype type;
} tokendefn;

DECLARE_VARRAY(tokendefn, tokendefn);

/* -------------------------------------------------------
 * Standard token types
 * ------------------------------------------------------- */

/** Enum listing standard token types. Each token will be mapped to a parserule in the parser */
enum {
    /* New line */
    TOKEN_NEWLINE,
    
    /* Question mark */
    TOKEN_QUESTION,
    
    /* Literals */
    TOKEN_STRING,
    TOKEN_INTERPOLATION,
    TOKEN_INTEGER,
    TOKEN_NUMBER,
    TOKEN_SYMBOL,
    
    /* Brackets */
    TOKEN_LEFTPAREN, TOKEN_RIGHTPAREN,
    TOKEN_LEFTSQBRACKET, TOKEN_RIGHTSQBRACKET,
    TOKEN_LEFTCURLYBRACKET, TOKEN_RIGHTCURLYBRACKET,
    
    /* Delimiters */
    TOKEN_COLON, TOKEN_SEMICOLON, TOKEN_COMMA,
    
    /* Operators */
    TOKEN_PLUS, TOKEN_MINUS, TOKEN_STAR, TOKEN_SLASH, TOKEN_CIRCUMFLEX,
    TOKEN_PLUSPLUS, TOKEN_MINUSMINUS,
    TOKEN_PLUSEQ, TOKEN_MINUSEQ, TOKEN_STAREQ, TOKEN_SLASHEQ,
    TOKEN_HASH,
    TOKEN_AT,
    
    TOKEN_DOT,
    TOKEN_DOTDOT,
    TOKEN_DOTDOTDOT,
    TOKEN_EXCLAMATION, TOKEN_AMP, TOKEN_VBAR, TOKEN_DBLAMP, TOKEN_DBLVBAR,
    TOKEN_EQUAL,
    TOKEN_EQ, TOKEN_NEQ,
    TOKEN_LT, TOKEN_GT,
    TOKEN_LTEQ, TOKEN_GTEQ,
    
    /* Keywords */
    TOKEN_TRUE, TOKEN_FALSE, TOKEN_NIL,
    TOKEN_SELF, TOKEN_SUPER, TOKEN_IMAG,
    TOKEN_PRINT, TOKEN_VAR,
    TOKEN_IF, TOKEN_ELSE, TOKEN_IN,
    TOKEN_WHILE, TOKEN_FOR, TOKEN_DO, TOKEN_BREAK, TOKEN_CONTINUE,
    TOKEN_FUNCTION, TOKEN_RETURN, TOKEN_CLASS,
    TOKEN_IMPORT, TOKEN_AS, TOKEN_IS, TOKEN_WITH,
    TOKEN_TRY, TOKEN_CATCH,
    
    /* Errors and other statuses */
    TOKEN_INCOMPLETE,
    TOKEN_ERROR,
    TOKEN_EOF
};

/* -------------------------------------------------------
 * Lexer data structure
 * ------------------------------------------------------- */

/** @brief Store the current configuration of a lexer */
typedef struct {
    const char* start; /** Starting point to lex */
    const char* current; /** Current point */
    int line; /** Line number */
    int posn; /** Character position in line */
    
    bool matchkeywords; /** Whether to match keywords or not; default is true */
    bool stringinterpolation; /** Whether to perform string interpolation */
    
    int interpolationlevel; /** Level of string interpolation */
    
    tokendefn *defns; /** Pointer to token defintions in use */
    int ndefns; /** Number of token defintions in use */
    
    varray_tokendefn defnstore; /** Used to hold custom tokens */
} lexer;

/* -------------------------------------------------------
 * Lex error messages
 * ------------------------------------------------------- */

#define LEXER_UNTERMINATEDCOMMENT       "UntrmComm"
#define LEXER_UNTERMINATEDCOMMENT_MSG   "Unterminated multiline comment '/*'."

#define LEXER_UNTERMINATEDSTRING        "UntrmStrng"
#define LEXER_UNTERMINATEDSTRING_MSG    "Unterminated string."

/* -------------------------------------------------------
 * Lex interface
 * ------------------------------------------------------- */

// Initialize and clear a lexer structure
void lex_init(lexer *l, const char *start, int line);
void lex_clear(lexer *l);

// Program lexer
void lex_settokendefns(lexer *l, tokendefn *defns);
void lex_setmatchkeywords(lexer *l, bool match);

// Obtain the next token
bool lex(lexer *l, token *tok, error *err);

// Initialization/finalization
void lexer_initialize(void);
void lexer_finalize(void);

#endif /* lex_h */
