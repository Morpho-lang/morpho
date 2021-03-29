/** @file linedit.h
 *  @author T J Atherton
 *
 *  @brief A simple line editor with history, prediction and syntax highlighting
*/

#ifndef linedit_h
#define linedit_h

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>
#include <string.h>

#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>

/* **********************************************************************
 * Types
 * ********************************************************************** */

/** lineditor strings */
typedef struct slinedit_string {
    size_t capacity;
    size_t length;
    char *string;
    struct slinedit_string *next;
} linedit_string;

/** A list of strings */
typedef struct {
    int posn; /* We use this to keep track of where in the list the user is */
    linedit_string *first;
} linedit_stringlist;

/* -----------------------
 * Tokenization
 * ----------------------- */

/** lineditor tokens */
typedef struct {
    unsigned int type;
    char *start;
    size_t length;
} linedit_token;

/** @brief Tokenizer callback
 *  @param   in    - a string
 *  @param   ref   - System for storing persistent data between calls to the lexer.
 *  @param   tok   - pointer to a token structure that the caller should fill out.
 *  @details This user function is called when linedit needs to tokenize a string.
 *           The function should identify the next token in the string and fill out
 *           the following fields:
 *             tok->type   - should contain the token type. This is used e.g.
 *                           an index to the color array.
 *             tok->start  - should point to the first significant character
 *                           in the token
 *             tok->length - should contain the length of the token, in bytes
 *           The function should return true if a token was successfully processed or
 *           false otherwise.
 *
 *           Storing persistent data over a sequance of calls to implement non-CFG:
 *           On the first call, *ref is NULL. You should malloc your structure, initialize it, set *ref to point to it and return the first token.
 *           After that, *ref will point to your structure.
 *           Once linedit is done tokenizing, it will call free on your pointer.
 *           @warning: You must not malloc child structures as these will not be freed. 
 */
typedef bool (*linedit_tokenizer) (char *in, void **ref, linedit_token *tok);

/* -----------------------
 * Color
 * ----------------------- */

/** Colors */
typedef enum {
    LINEDIT_BLACK,
    LINEDIT_RED,
    LINEDIT_GREEN,
    LINEDIT_YELLOW,
    LINEDIT_BLUE,
    LINEDIT_MAGENTA,
    LINEDIT_CYAN,
    LINEDIT_WHITE,
    LINEDIT_DEFAULTCOLOR, 
} linedit_color;

typedef enum {
    LINEDIT_BOLD,
    LINEDIT_UNDERLINE,
    LINEDIT_REVERSE,
    LINEDIT_NONE
} linedit_emphasis;

/** Structure to hold all information related to syntax coloring */
typedef struct {
    linedit_tokenizer tokenizer; /* A tokenizer function */
    unsigned int ncols;          /* Number of colors provided */
    bool lexwarning; 
    linedit_color col[];         /* Array of colors, one for each
                                    token type */
} linedit_syntaxcolordata;

/* -----------------------
 * Completion
 * ----------------------- */

/** @brief Autocompletion callback
 *  @params  in          - a string
 *  @params  completion  - autocompletion structure
 *  @details This user function is called when linedit requests autocompletion
 *           of a string. The function should identify any possible suggestions
 *           and call linedit_addcompletion to add them one by one.
 *
 *           Only *remaining* characters from the suggestion should be added,
 *           e.g. for "hello" if the user has typed "he" the function should add
 *           "llo" as a suggestion.
 *
 *           The function should return true if autocompletion was successfully
 *           processed or false otherwise.
*/
typedef bool (*linedit_completer) (char *in, linedit_stringlist *completion);

/* -----------------------
 * lineditor structure
 * ----------------------- */

#define LINEDIT_DEFAULTPROMPT   ">"

/** Keep track of what the line editor is doing */
typedef enum {
    LINEDIT_DEFAULTMODE,
    LINEDIT_SELECTIONMODE,
    LINEDIT_HISTORYMODE
} lineditormode;

/** Holds all state information needed for a line editor */
typedef struct {
    lineditormode mode;      /* Current editing mode */
    int posn;                /* Position of the cursor */
    int sposn;               /* Starting point of a selection */
    int ncols;               /* Number of columns */
    linedit_string prompt;   /* The prompt */
    
    linedit_string current;  /* Current string that's being edited */
    linedit_string clipboard;  /* Copy/paste clipboard */
    
    linedit_stringlist history; /* History list */
    linedit_stringlist suggestions; /* Autocompletion suggestions */
    
    linedit_syntaxcolordata *color; /* Structure to handle syntax coloring */
    linedit_completer completer; /* Autocompletion */
} lineditor;

/* **********************************************************************
 * Public interface
 * ********************************************************************** */

/** Public interface to the line editor.
 *  @param   edit - a line editor that has been initialized with linedit_init.
 *  @returns the string input by the user, or NULL if nothing entered. */
char *linedit(lineditor *edit);

/** @brief Configures syntax coloring
 *  @param edit         Line editor to configure
 *  @param tokenizer    A function to be called that will find the next token from a string
 *  @param cols         An array of colors, one entry for each token type
 *  @param ncols        Number of entries in the color array */
void linedit_syntaxcolor(lineditor *edit, linedit_tokenizer tokenizer, linedit_color *cols, unsigned int ncols);

/** @brief Configures autocomplete
 *  @param edit         Line editor to configure
 *  @param completer    a function */
void linedit_autocomplete(lineditor *edit, linedit_completer completer);

/** @brief Adds a completion suggestion
 *  @param completion   completion data structure
 *  @param string       string to add */
void linedit_addsuggestion(linedit_stringlist *completion, char *string);

/** @brief Sets the prompt
 *  @param edit         Line editor to configure
 *  @param prompt       prompt string to use */
void linedit_setprompt(lineditor *edit, char *prompt);

/** @brief Displays a string with a given color and emphasis
 *  @param edit         Line editor in use
 *  @param string       String to display
 *  @param col          Color
 *  @param emph         Emphasis */
void linedit_displaywithstyle(lineditor *edit, char *string, linedit_color col, linedit_emphasis emph);

/** @brief Displays a string with syntax coloring
 *  @param edit         Line editor in use
 *  @param string       String to display */
void linedit_displaywithsyntaxcoloring(lineditor *edit, char *string);

/** @brief Gets the terminal width
 *  @param edit         Line editor in use
 *  @returns The width in characters */
int linedit_getwidth(lineditor *edit);

/** Initialize a line editor */
void linedit_init(lineditor *edit);

/** Finalize a line editor */
void linedit_clear(lineditor *edit);

#endif /* linedit_h */
