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

typedef struct slexer lexer;

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
 * Token processing functions
 * ------------------------------------------------------- */

/** Token processing functions are called after recording it. */
typedef bool (* processtokenfn) (lexer *l, token *tok, error *err);

/* -------------------------------------------------------
 * Token definitions
 * ------------------------------------------------------- */

typedef struct {
    char *string; // String defining the token
    tokentype type; // Token type
    processtokenfn processfn; // Optional processfunction to call
} tokendefn;

DECLARE_VARRAY(tokendefn, tokendefn);

/* -------------------------------------------------------
 * Lexer data structure
 * ------------------------------------------------------- */

/** @brief Store the current configuration of a lexer */
struct slexer {
    const char* start; /** Starting point to lex */
    const char* current; /** Current point */
    int line; /** Line number */
    int posn; /** Character position in line */
    
    bool matchkeywords; /** Whether to match keywords or not; default is true */
    bool stringinterpolation; /** Whether to perform string interpolation */
    processtokenfn whitespacefn; /** Called to skip whitespace */
    processtokenfn prefn; /** Called before attempting to match the token list */
    
    tokentype eoftype; /** End of file marker */
    tokentype inttype; /** Integers */
    tokentype flttype; /** Floats */
    tokentype imagtype; /** Imaginary numbers */
    tokentype symboltype; /** Symbol numbers */
    
    int interpolationlevel; /** Level of string interpolation */
    
    tokendefn *defns; /** Pointer to token defintions in use */
    int ndefns; /** Number of token defintions in use */
    
    varray_tokendefn defnstore; /** Used to hold custom tokens */
} ;

/* -------------------------------------------------------
 * Morpho token types
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
    
    /* Other symbols */
    TOKEN_QUOTE,
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
    
    /* Shebangs at start of script */
    TOKEN_SHEBANG,
    
    /* Errors and other statuses */
    TOKEN_INCOMPLETE,
    TOKEN_EOF
};

/* -------------------------------------------------------
 * Lex error messages
 * ------------------------------------------------------- */

#define LEXER_UNRECOGNIZEDTOKEN         "UnrgnzdTkn"
#define LEXER_UNRECOGNIZEDTOKEN_MSG     "Unrecognized token."

#define LEXER_UNTERMINATEDCOMMENT       "UntrmComm"
#define LEXER_UNTERMINATEDCOMMENT_MSG   "Unterminated multiline comment '/*'."

#define LEXER_UNTERMINATEDSTRING        "UntrmStrng"
#define LEXER_UNTERMINATEDSTRING_MSG    "Unterminated string."

/* -------------------------------------------------------
 * Library functions to support customizable lexers
 * ------------------------------------------------------- */

bool lex_findtoken(lexer *l, tokendefn **defn);
bool lex_matchtoken(lexer *l, tokendefn **defn);
void lex_recordtoken(lexer *l, tokentype type, token *tok);
char lex_advance(lexer *l);
bool lex_back(lexer *l);
bool lex_isatend(lexer *l);
bool lex_isalpha(char c);
bool lex_isdigit(char c);
bool lex_isspace(char c);
char lex_peek(lexer *l);
char lex_peekahead(lexer *l, int n);
char lex_peekprevious(lexer *l);
void lex_newline(lexer *l);
bool lex_skipshebang(lexer *l);

/* -------------------------------------------------------
 * Lex interface
 * ------------------------------------------------------- */

// Initialize and clear a lexer structure
void lex_init(lexer *l, const char *start, int line);
void lex_clear(lexer *l);

// Configure lexer
void lex_settokendefns(lexer *l, tokendefn *defns);
void lex_seteof(lexer *l, tokentype eoftype);
tokentype lex_eof(lexer *l);
void lex_setnumbertype(lexer *l, tokentype inttype, tokentype flttype, tokentype imagtype);
void lex_setsymboltype(lexer *l, tokentype symboltype);
tokentype lex_symboltype(lexer *l);
void lex_setstringinterpolation(lexer *l, bool interpolation);
void lex_setmatchkeywords(lexer *l, bool match);
bool lex_matchkeywords(lexer *l);
void lex_setwhitespacefn(lexer *l, processtokenfn whitespacefn);
void lex_setprefn(lexer *l, processtokenfn prefn);

// Get information about a token
bool lex_tokeniskeyword(lexer *l, token *tok);

// Obtain the next token
bool lex(lexer *l, token *tok, error *err);

// Initialization/finalization
void lex_initialize(void);
void lex_finalize(void);

#endif /* lex_h */
