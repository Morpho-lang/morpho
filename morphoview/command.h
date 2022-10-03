/** @file command.h
 *  @author T J Atherton
 *
 *  @brief Command language for morphoview
 */

#ifndef command_h
#define command_h

#include <stdio.h>
#include <stdbool.h>
#include "scene.h"
#include "display.h"
#include "matrix3d.h"

//#define DEBUG_PARSER

/* -------------------------------------------------------
 * Tokens
 * ------------------------------------------------------- */

/** @brief The token type */
typedef enum {
    TOKEN_NONE,
    
    TOKEN_INTEGER,
    TOKEN_FLOAT,
    TOKEN_STRING,
    
    TOKEN_COLOR,
    TOKEN_SELECTCOLOR,
    TOKEN_DRAW,
    TOKEN_OBJECT,
    TOKEN_VERTICES,
    TOKEN_POINTS,
    TOKEN_LINES,
    TOKEN_FACETS,
    TOKEN_IDENTITY,
    TOKEN_MATRIX,
    TOKEN_ROTATE,
    TOKEN_SCALE,
    TOKEN_SCENE,
    TOKEN_TRANSLATE,
    TOKEN_VIEWDIRECTION,
    TOKEN_VIEWVERTICAL,
    TOKEN_WINDOW,
    TOKEN_FONT,
    TOKEN_TEXT,
    
    TOKEN_EOF
} tokentype;

/** @brief A token */
typedef struct {
    tokentype type; /** Type of the token */
    const char *start; /** Start of the token */
    unsigned int length; /** Its length */
} token;

/* -------------------------------------------------------
 * Lexer
 * ------------------------------------------------------- */

/** @brief Store the current configuration of a lexer */
typedef struct {
    const char* start; /** Starting point to lex */
    const char* current; /** Current point */
} lexer;

/* -------------------------------------------------------
 * Parser
 * ------------------------------------------------------- */

/** @brief Store the current configuration of a parser */
typedef struct {
    lexer l; 
    token current;
    token prev;
    
    scene *scene;
    display *display;
    
    mat4x4 model; /* The model matrix */
    bool modelchanged;
    
    gobject *cobject;
} parser;

/** @brief Definition of a parse function. */
typedef bool (*parsefunction) (parser *p);

/* -------------------------------------------------------
 * Prototypes
 * ------------------------------------------------------- */

bool command_getfilesize(FILE *f, size_t *s);
bool command_loadinput(const char *in, char **out);
void command_removefile(const char *in);

void command_lexinit(lexer *l, const char *start);
bool command_lex(lexer *l, token *tok);

bool command_parse(char *in);

#endif /* command_h */
