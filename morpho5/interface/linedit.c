/** @file linedit.c
 *  @author T J Atherton
 *
 *  @brief A line editor with history, autocomplete and syntax highlighting
 */

#include "linedit.h"

/* **********************************************************************
 * Internal types
 * ********************************************************************** */

/** Identifies the type of keypress */
typedef enum {
    UNKNOWN,
    CHARACTER,
    RETURN,
    TAB,
    DELETE,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    HOME,
    END,
    SHIFT_LEFT,
    SHIFT_RIGHT,
    CTRL,
} keytype;

/** A single keypress event obtained and processed by the terminal */
typedef struct {
    keytype type; /** Type of keypress */
    char c[5]; /** Up to four bytes of utf8 encoded unicode plus null terminator */
    int nbytes; /** Number of bytes */
} keypress;

#define LINEDIT_KEYPRESSGETCHAR(a) ((a)->c[0])

/* **********************************************************************
 * Strings
 * ********************************************************************** */

/** @brief Returns the number of bytes in the next character of a given utf8 string
    @returns number of bytes */
int linedit_utf8numberofbytes(char *string) {
    uint8_t byte = * ((uint8_t *) string);
    
    if ((byte & 0xc0) == 0x80) return 0; // In the middle of a utf8 string
    
    // Get the number of bytes from the first character
    if ((byte & 0xf8) == 0xf0) return 4;
    if ((byte & 0xf0) == 0xe0) return 3;
    if ((byte & 0xe0) == 0xc0) return 2;
    return 1;
}

/** @brief Returns a pointer to character i in a utf8 encoded string */
ssize_t linedit_utf8index(linedit_string *string, size_t i, size_t offset) {
    int advance=0;
    size_t nchars=0;
    
    for (ssize_t j=0; j+offset<=string->length; j+=advance, nchars++) {
        if (nchars==i) return j;
        advance=linedit_utf8numberofbytes(string->string+offset+j);
        if (advance==0) break; // If advance is 0, the string is corrupted; return failure
    }
    
    return -1;
}

#define linedit_MINIMUMSTRINGSIZE  8

/** Initializes a string, clearing all fields */
void linedit_stringinit(linedit_string *string) {
    string->capacity=0;
    string->length=0;
    string->next=NULL;
    string->string=NULL;
}

/** Clears a string, deallocating memory if necessary */
void linedit_stringclear(linedit_string *string) {
    if (string->string) free(string->string);
    linedit_stringinit(string);
}

/** Resizes a string
 *  @param string   - the string to grow
 *  @param size     - requested size
 *  @returns true on success, false on failure */
bool linedit_stringresize(linedit_string *string, size_t size) {
    size_t newsize=linedit_MINIMUMSTRINGSIZE;
    char *old=string->string;
    
    /* If we're increasing the size, grow by factors of 1.5 to avoid excessive calls to allocator */
    while (newsize<=size) newsize=((newsize<<1)+newsize)>>1; // mul by x1.5
        
    string->string=realloc(string->string,newsize);
    if (string->string) {
        if (!old) {
            string->string[0]='\0'; /* Make sure a new string is zero-terminated */
            string->length=0;
        }
        string->capacity=newsize;
    }
    return (string->string!=NULL);
}

/** Adds a string to a string */
void linedit_stringappend(linedit_string *string, char *c, size_t nbytes) {
    if (string->capacity<=string->length+nbytes) {
        if (!linedit_stringresize(string, string->length+nbytes+1)) return;
    }
    
    strncpy(string->string+string->length, c, nbytes);
    string->length+=nbytes;
    string->string[string->length]='\0'; /* Keep the string zero-terminated */
}

/** @brief   Inserts characters at a given position
 *  @param[in] string - string to amend
 *  @param[in] posn - insertion position as a character index
 *  @param[in] c - string to insert
 *  @param[in] n - number of bytes to insert
 *  @details If the position is after the length of the string
 *           the new characters are instead appended. */
void linedit_stringinsert(linedit_string *string, size_t posn, char *c, size_t n) {
    ssize_t offset=linedit_utf8index(string, posn, 0);
    if (offset<0) return;
    
    if (offset<string->length) {
        if (string->capacity<=string->length+n) {
            if (!linedit_stringresize(string, string->length+n+1)) return;
        }
        /* Move the remaining part of the string */
        memmove(string->string+offset+n, string->string+offset, string->length-posn+1);
        /* Copy in the text to insert */
        memmove(string->string+offset, c, n);
        string->length+=n;
    } else {
        linedit_stringappend(string, c, n);
    }
}

/** @brief   Deletes characters at a given position.
 *  @param[in] string - string to amend
 *  @param[in] posn - Delete characters as a character index
 *  @param[in] n - number of characters to delete */
void linedit_stringdelete(linedit_string *string, size_t posn, size_t n) {
    ssize_t offset=linedit_utf8index(string, posn, 0);
    ssize_t nbytes=linedit_utf8index(string, n, offset);
    if (offset<0) return;
    
    if (offset<string->length) {
        if (offset+nbytes<string->length) {
            memmove(string->string+offset, string->string+offset+nbytes, string->length-offset-nbytes+1);
        } else {
            string->string[offset]='\0';
        }
        string->length=strlen(string->string);
    }
}

/** Adds a c string to a string */
void linedit_stringaddcstring(linedit_string *string, char *s) {
    if (!s || !string) return;
    size_t size=strlen(s);
    if (string->capacity<=string->length+size+1) {
        if (!linedit_stringresize(string, string->length+size+1)) return;
    }
    
    strncpy(string->string+string->length, s, string->capacity-string->length);
    string->length+=size;
}

/** Finds the width of a string in characters */
int linedit_stringwidth(linedit_string *string) {
    int n=0;
    for (int i=0; i<string->length; ) {
        i+=linedit_utf8numberofbytes(string->string+i);
        n++;
    }
    return n;
}

/** Returns a C string from a string */
char *linedit_cstring(linedit_string *string) {
    return string->string;
}

/** Creates a new string from a C string */
linedit_string *linedit_newstring(char *string) {
    linedit_string *new = malloc(sizeof(linedit_string));
    
    if (new) {
        linedit_stringinit(new);
        linedit_stringaddcstring(new, string);
    }
    
    return new;
}

/* **********************************************************************
 * String lists
 * ********************************************************************** */

/** Adds an entry to a string list */
void linedit_stringlistadd(linedit_stringlist *list, char *string) {
    linedit_string *new=linedit_newstring(string);
    
    if (new) {
        new->next=list->first;
        list->first=new;
    }
}

/** Initializes a string list */
void linedit_stringlistinit(linedit_stringlist *list) {
    list->first=NULL;
    list->posn=0;
}

/** Frees the contents of a string list */
void linedit_stringlistclear(linedit_stringlist *list) {
    while (list->first!=NULL) {
        linedit_string *s = list->first;
        list->first=s->next;
        linedit_stringclear(s);
        free(s);
    }
    linedit_stringlistinit(list);
}

/** Removes a string from a list */
void linedit_stringlistremove(linedit_stringlist *list, linedit_string *string) {
    linedit_string *s=NULL, *prev=NULL;
    
    for (s=list->first; s!=NULL; s=s->next) {
        if (s==string) {
            if (prev) {
                prev->next=s->next;
            } else {
                list->first=s->next;
            }
            linedit_stringclear(s);
            free(s);
            return;
        }
        
        prev=s;
    }
}

/** Chooses an element of a stringlist
 * @param[in]  list the list to select from
 * @param[in]  n    entry number to select
 * @parma[out] *m   entry number actually selected
 * @returns the selected element */
linedit_string *linedit_stringlistselect(linedit_stringlist *list, unsigned int n, unsigned int *m) {
    unsigned int i=0;
    linedit_string *s=NULL;
    
    for (s=list->first; s!=NULL && s->next!=NULL; s=s->next) {
        if (i==n) break;
        i++;
    }
    
    if (m) *m=i;
    
    return s;
}

/* **********************************************************************
 * History list
 * ********************************************************************** */

/** Adds an entry to the history list */
void linedit_historyadd(lineditor *edit, char *string) {
    linedit_stringlistadd(&edit->history, string);
}

/** Frees the history list */
void linedit_historyclear(lineditor *edit) {
    linedit_stringlistclear(&edit->history);
}

/** Makes a particular history entry current */
unsigned int linedit_historyselect(lineditor *edit, unsigned int n) {
    unsigned int m=n;
    linedit_string *s=linedit_stringlistselect(&edit->history, n, &m);
    
    if (s) {
        edit->current.length=0;
        linedit_stringaddcstring(&edit->current, s->string);
    }
    
    return m;
}

/** Advances the history list */
void linedit_historyadvance(lineditor *edit, unsigned int n) {
    edit->history.posn+=n;
    edit->history.posn=linedit_historyselect(edit, edit->history.posn);
}

/* **********************************************************************
 * Autocompletion
 * ********************************************************************** */

bool lineedit_atendofline(lineditor *edit);

/** Regenerates the list of autocomplete suggestions */
void linedit_generatesuggestions(lineditor *edit) {
    if (edit->completer) {
        linedit_stringlistclear(&edit->suggestions);
        
        if (edit->current.string &&
            lineedit_atendofline(edit)) {
            (edit->completer) (edit->current.string, &edit->suggestions);
        }
    }
}

/** Check whether any suggestions are available */
bool linedit_aresuggestionsavailable(lineditor *edit) {
    return (edit->suggestions.first!=NULL);
}

/** Get the current suggestion */
char *linedit_currentsuggestion(lineditor *edit) {
    linedit_string *s=linedit_stringlistselect(&edit->suggestions, edit->suggestions.posn, NULL);
    
    if (s) return s->string;
    
    return NULL;
}

/** Advance through the suggestions */
void linedit_advancesuggestions(lineditor *edit, unsigned int n) {
    unsigned int rposn=edit->suggestions.posn+n, nposn=rposn;
    linedit_stringlistselect(&edit->suggestions, rposn, &nposn);
    edit->suggestions.posn=nposn;
    if (rposn!=edit->suggestions.posn) edit->suggestions.posn=0; /* Go back to first */
}

/* **********************************************************************
 * Terminal driver
 * ********************************************************************** */

void linedit_enablerawmode(void);
void linedit_disablerawmode(void);

typedef enum {
    LINEDIT_NOTTTY,
    LINEDIT_UNSUPPORTED,
    LINEDIT_SUPPORTED
} linedit_terminaltype;

/** @brief   Compares two c strings independently of case
 *  @param[in] str1 - } strings to compare
 *  @param[in] str2 - }
 *  @returns 0 if the strings are identical, otherwise a positive or negative number indicating their lexographic order */
int linedit_cstrcasecmp(char *str1, char *str2) {
    if (str1 == str2) return 0;
    int result=0;
    
    for (char *p1=str1, *p2=str2; result==0; p1++, p2++) {
        result=tolower(*p1)-tolower(*p2);
        if (*p1=='\0') break;
    }
    
    return result;
}

/** Checks whether the terminal is supported */
linedit_terminaltype linedit_checksupport(void) {
    /* Make sure we're a tty */
    if (!isatty(STDIN_FILENO)) {
        return LINEDIT_NOTTTY;
    }
     
    char *unsupported[]={"dumb","cons25","emacs",NULL};
    char *term = getenv("TERM");
    
    if (term == NULL) return LINEDIT_UNSUPPORTED;
    for (unsigned int i=0; unsupported[i]!=NULL; i++) {
        if (!linedit_cstrcasecmp(term, unsupported[i])) return LINEDIT_UNSUPPORTED;
    }
    
    return LINEDIT_SUPPORTED;
}

/** Holds the original terminal state */
struct termios terminit;

bool termexitregistered=false;

/** @brief Enables 'raw' mode in the terminal
 *  @details In raw mode key presses are passed directly to us rather than
 *           being buffered. */
void linedit_enablerawmode(void) {
    struct termios termraw; /* Use to set the raw state */
    if (!termexitregistered) {
        atexit(linedit_disablerawmode);
        termexitregistered=true;
    }
    
    tcgetattr(STDIN_FILENO, &terminit); /** Get the original state*/
    
    termraw=terminit;
    /* Input: Turn off: IXON   - software flow control (ctrl-s and ctrl-q)
                        ICRNL  - translate CR into NL (ctrl-m)
                        BRKINT - parity checking
                        ISTRIP - strip bit 8 of each input byte */
    termraw.c_iflag &= ~(IXON | ICRNL | BRKINT | BRKINT | ISTRIP);
    /* Output: Turn off: OPOST - output processing */
    termraw.c_oflag &= ~(OPOST);
    /* Character: CS8 Set 8 bits per byte */
    termraw.c_cflag |= (CS8);
    /* Turn off: ECHO   - causes keypresses to be printed immediately
                 ICANON - canonical mode, reads line by line
                 IEXTEN - literal (ctrl-v)
                 ISIG   - turn off signals (ctrl-c and ctrl-z) */
    termraw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &termraw);
}

/** @brief Restore terminal state to normal */
void linedit_disablerawmode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &terminit);
    printf("\r"); /** Print a carriage return to ensure we're back on the left hand side */
}

/** @brief Gets the terminal width */
void linedit_getterminalwidth(lineditor *edit) {
    struct winsize ws;
    
    edit->ncols=80;
    
    /* Try ioctl first */
    if (!(ioctl(1, TIOCGWINSZ, &ws) == -1 || ws.ws_col == 0)) {
        edit->ncols=ws.ws_col;
    } else {
        //
    }
}

/** @brief Shows visible characters l...r from a string */
void linedit_writewindow(char *string, int l, int r) {
    int i=0;
    int width=0;
    for (char *s=string; *s!='\0'; s+=width) {
        width=linedit_utf8numberofbytes(s);
        if (*s=='\r') { // Reset on a carriage return
            if (write(STDOUT_FILENO, "\r", 1)==-1) return;
            i=0;
        } else if (iscntrl(*s)) {
            if (*s=='\033') { // A terminal control character
                char *ctl=s; // First identify its length
                while (!isalpha(*ctl) && *ctl!='\0') ctl++;
                if (write(STDOUT_FILENO, s, ctl-s+1)==-1) return; // print it
                s=ctl;
            }
        } else { // Otherwise show printable characters that lie within the window
            if (i>=l && i<r) write(STDOUT_FILENO, s, width);
            i++;
        }
    }
}

/** @brief Writes a string to the terminal */
void linedit_write(char *string) {
    size_t length=strlen(string);
    if (write(STDOUT_FILENO, string, length)==-1) {
        fprintf(stderr, "Error writing to terminal.\n");
    }
}

/** @brief Writes a character to the terminal */
void linedit_writechar(char c) {
    if (write(STDOUT_FILENO, &c, 1)==-1) {
        fprintf(stderr, "Error writing to terminal.\n");
    }
}

/** Raw codes produced by the terminal */
enum keycodes {
    TAB_CODE = 9,      // Tab
    RETURN_CODE = 13,  // Enter or return
    ESC_CODE = 27,     // Escape
    DELETE_CODE = 127  // Delete
};

/** Maximum escape code size */
#define LINEDIT_CODESTRINGSIZE 24

/** Enable this macro to get reports on unhandled keypresses */
//#define LINEDIT_DEBUGKEYPRESS

/** Initializes a keypress structure */
void linedit_keypressinit(keypress *out) {
    out->type=UNKNOWN;
    for (int i=0; i<5; i++) out->c[i]='\0';
    out->nbytes=0;
}

/** @brief Read and decode a single keypress from the terminal */
bool linedit_readkey(lineditor *edit, keypress *out) {
    out->type=UNKNOWN;
    
    if (read(STDIN_FILENO, out->c, 1) == 1) {
        if (iscntrl(LINEDIT_KEYPRESSGETCHAR(out))) {
            switch (LINEDIT_KEYPRESSGETCHAR(out)) {
                case ESC_CODE:
                {   /* Escape sequences */
                    char seq[LINEDIT_CODESTRINGSIZE];
                    ssize_t ret=0;
                    
                    /* Read in the escape sequence */
                    for (unsigned int i=0; i<LINEDIT_CODESTRINGSIZE; i++) {
                        ret=read(STDIN_FILENO, &seq[i], 1);
                        if (ret<0 || isalpha(seq[i])) break;
                    }
                    
                    /** Decode the escape sequence */
                    if (seq[0]=='[') {
                        if (isdigit(seq[1])) { /* Extended seqence */
                            if (strncmp(seq, "[1;2C", 5)==0) {
                                out->type=SHIFT_RIGHT;
                            } else if (strncmp(seq, "[1;2D", 5)==0) {
                                out->type=SHIFT_LEFT;
                            } else {
#ifdef LINEDIT_DEBUGKEYPRESS
                                printf("Extended escape sequence: ");
                                for (unsigned int i=0; i<10; i++) {
                                    printf("%c", seq[i]);
                                    if (isalpha(seq[i])) break;
                                }
                                printf("\n");
#endif
                            }
                        } else {
                            switch (seq[1]) {
                                case 'A': out->type=UP; break;
                                case 'B': out->type=DOWN; break;
                                case 'C': out->type=RIGHT; break;
                                case 'D': out->type=LEFT; break;
                                default:
#ifdef LINEDIT_DEBUGKEYPRESS
                                    printf("Unhandled escape sequence: %c%c%c\r\n", seq[0], seq[1], seq[2]);
#endif
                                    break;
                            }
                        }
                    }
                }
                    break;
                case TAB_CODE:    out->type=TAB; break;
                case DELETE_CODE: out->type=DELETE; break;
                case RETURN_CODE: out->type=RETURN; break;
                default:
                    if (LINEDIT_KEYPRESSGETCHAR(out)>0 && LINEDIT_KEYPRESSGETCHAR(out)<27) { /* Ctrl+character */
                        out->type=CTRL;
                        out->c[0]+='A'-1; /* Return the character code */
                    } else {
#ifdef LINEDIT_DEBUGKEYPRESS
                        printf("Unhandled keypress: %d\r\n", LINEDIT_KEYPRESSGETCHAR(out));
#endif
                    }
            }
            
        } else {
            out->nbytes=linedit_utf8numberofbytes(out->c);
            /* Read in the unicode sequence */
            ssize_t ret=0;
            for (int i=1; i<out->nbytes; i++) {
                ret=read(STDIN_FILENO, &out->c[i], 1);
                if (ret<0) break;
            }
            out->type=CHARACTER;
#ifdef LINEDIT_DEBUGKEYPRESS
            printf("Character: %s (%i bytes)\r\n", out->c, out->nbytes);
#endif
        }
    }
    return true;
}

/** @brief Writes a control sequence to move to a given position */
void linedit_move(linedit_string *out, int posn) {
    char code[LINEDIT_CODESTRINGSIZE];
    sprintf(code, "\r\033[%uC", posn);
    linedit_stringaddcstring(out, code);
}

/** @brief Writes a control sequence to reset default text */
void linedit_setdefaulttext(linedit_string *out) {
    char code[LINEDIT_CODESTRINGSIZE];
    sprintf(code, "\033[0m");
    linedit_stringaddcstring(out, code);
}

/** @brief Writes a control sequence to set a given color */
void linedit_setcolor(linedit_string *out, linedit_color col) {
    char code[LINEDIT_CODESTRINGSIZE];
    sprintf(code, "\033[%um", (col==LINEDIT_DEFAULTCOLOR ? 0: 30+col));
    linedit_stringaddcstring(out, code);
}

/** @brief Writes a control sequence to set a given emphasis */
void linedit_setemphasis(linedit_string *out, linedit_emphasis emph) {
    char code[LINEDIT_CODESTRINGSIZE];
    switch (emph) {
        case LINEDIT_BOLD: sprintf(code, "\033[1m"); break;
        case LINEDIT_UNDERLINE: sprintf(code, "\033[4m"); break;
        case LINEDIT_REVERSE: sprintf(code, "\033[7m"); break;
        case LINEDIT_NONE: break; 
    }
    
    linedit_stringaddcstring(out, code);
}

/** @brief Writes a control sequence to erase the rest of the line */
void linedit_erasetoendofline(linedit_string *out) {
    char code[LINEDIT_CODESTRINGSIZE];
    sprintf(code, "\033[0K");
    linedit_stringaddcstring(out, code);
}

/** @brief Writes a control sequence to erase the whole line */
void linedit_eraseline(linedit_string *out) {
    char code[LINEDIT_CODESTRINGSIZE];
    sprintf(code, "\033[2K");
    linedit_stringaddcstring(out, code);
}

/* **********************************************************************
 * Interface
 * ********************************************************************** */

/** Adds a string with selection highlighting
 * @param[in] edit - active editor
 * @param[in] in - input string
 * @param[in] offset - offset of string in characters
 * @param[in] length - length of string in characters
 * @param[in] col - color
 * @param[out] out - display plus coloring information written to this string */
void linedit_addcstringwithselection(lineditor *edit, char *in, size_t offset, size_t length, linedit_color *col, linedit_string *out) {
    int lposn=-1, rposn=-1;
    
    /* If a selection is active, discover its bounds */
    if (edit->mode==LINEDIT_SELECTIONMODE && !(edit->sposn<0)) {
        lposn=(edit->sposn < edit->posn ? edit->sposn : edit->posn) - (int) offset;
        rposn = (edit->sposn < edit->posn ? edit->posn : edit->sposn) - (int) offset;
    }
    
    /* Set the color if provided */
    if (col) linedit_setcolor(out, *col);
    
    /* Is the text we're showing outside the selected region entirely? */
    if (rposn<0 || lposn>(int) length) {
        linedit_stringappend(out, in, length);
    } else {
        /* If not, add the characters one by one and insert highlighting */
        if (lposn<0) {
            linedit_setemphasis(out, LINEDIT_REVERSE);
        }
        char *c=in;
        for (int i=0; i<length; i++) {
            if (i==lposn) {
                linedit_setemphasis(out, LINEDIT_REVERSE);
            }
            if (i==rposn) {
                linedit_setdefaulttext(out);
                /* Restore the color if one is provided */
                if (col) linedit_setcolor(out, *col);
            }
            int nbytes=linedit_utf8numberofbytes(c);
            if (nbytes) linedit_stringappend(out, c, nbytes);
            else nbytes=1; // Corrupted stream
            c+=nbytes;
        }
    }
}

/** Print a string with syntax coloring */
void linedit_syntaxcolorstring(lineditor *edit, linedit_string *in, linedit_string *out) {
    linedit_tokenizer tokenizer=edit->color->tokenizer;
    linedit_color *cols=edit->color->col, col=LINEDIT_DEFAULTCOLOR;
    linedit_token tok;
    unsigned int iter=0;
    void *ref=NULL;
    
    for (char *c=in->string; c!=NULL && *c!='\0';) {
        bool success = (tokenizer) (c, &ref, &tok);
        /* Get the next token */
        if (success && tok.length>0 && tok.start>=c) {
            size_t padding=tok.start-c;
            /* If there's leading unrecognized characters, print them. */
            if (tok.start>c) {
                col=LINEDIT_DEFAULTCOLOR;
                linedit_addcstringwithselection(edit, c, c-in->string, padding, &col, out);
            }
            
            /* Set the color */
            if (tok.type<edit->color->ncols) col=cols[tok.type];
            
            /* Copy the token across */
            if (tok.length>0) {
                linedit_addcstringwithselection(edit, tok.start, tok.start-in->string, tok.length, &col, out);
            }
            
            c=tok.start+tok.length;
        } else {
            col=LINEDIT_DEFAULTCOLOR;
            linedit_addcstringwithselection(edit, c, c-in->string, in->length-(c-in->string), &col, out);
            if (ref) free(ref);
            return; 
        };
        iter++;
        if (iter>in->length) {
            if (!edit->color->lexwarning) {
                fprintf(stderr, "\n\rLinedit error: Syntax colorer appears to be stuck in an infinite loop; ensure the tokenizer returns false if it doesn't recognize a token.\n");
                edit->color->lexwarning=true;
            }
            if (ref) free(ref);
            return;
        }
    }
}

/** Print a string without syntax coloring */
void linedit_showstring(lineditor *edit, linedit_string *in, linedit_string *out) {
    linedit_addcstringwithselection(edit, in->string, 0, in->length, NULL, out);
}

/** Refreshes a single line */
void linedit_refreshline(lineditor *edit) {
    int sugglength=0;
    linedit_string output; /* Holds the output string */
    linedit_stringinit(&output);
    
    linedit_eraseline(&output);
    linedit_setdefaulttext(&output);
    
    /* Display the prompt */
    linedit_stringappend(&output, "\r", 1);
    linedit_stringaddcstring(&output, edit->prompt.string);
    
    /* Display the current line, syntax colored if available */
    if (edit->color) {
        linedit_syntaxcolorstring(edit, &edit->current, &output);
    } else {
        linedit_showstring(edit, &edit->current, &output);
    }
    
    /* Display any autocompletion suggestions */
    if (linedit_aresuggestionsavailable(edit)) {
        char *suggestion = linedit_currentsuggestion(edit);
        linedit_setemphasis(&output, LINEDIT_BOLD);
        linedit_stringaddcstring(&output, suggestion);
        sugglength=(int) strlen(suggestion);
    }
    
    linedit_setdefaulttext(&output);
    linedit_stringappend(&output, "\r", 1);
    
    /* Determine the left and right hand boundaries */
    int promptwidth=linedit_stringwidth(&edit->prompt);
    int stringwidth=linedit_stringwidth(&edit->current);
    
    int start=0, end=promptwidth+stringwidth+sugglength;
    if (end>=edit->ncols) {
        /* Are we near the start? */
        if (promptwidth+edit->posn<edit->ncols) {
            start = 0;
        } else {
            start = promptwidth+edit->posn-edit->ncols+1;
        }
        end=start+edit->ncols-1;
    }
    
    /* Move cursor to the requested position */
    linedit_move(&output, edit->posn + promptwidth - start);
    
    /* Write the output string to the display */
    linedit_writewindow(output.string, start, end);

    linedit_stringclear(&output);
}

/** Sets the current mode, setting/clearing any state dependent data  */
void linedit_setmode(lineditor *edit, lineditormode mode) {
    if (mode!=LINEDIT_HISTORYMODE) {
        if (edit->mode==LINEDIT_HISTORYMODE) {
            linedit_stringlistremove(&edit->history, edit->history.first);
        }
        edit->history.posn=0;
    }
    if (mode==LINEDIT_SELECTIONMODE) {
        if (edit->sposn<0) edit->sposn=edit->posn;
    } else {
        edit->sposn=-1;
    }
    edit->mode=mode;
}

/** Gets the current mode */
lineditormode linedit_getmode(lineditor *edit) {
    return edit->mode;
}

/** Sets the current position
 * @param edit     - the editor
 * @param posn     - position to set, or negative to move to end */
void linedit_setposition(lineditor *edit, int posn) {
    edit->posn=(posn<0 ? linedit_stringwidth(&edit->current) : posn);
}

/** Checks if we're at the end of the line */
bool lineedit_atendofline(lineditor *edit) {
    return (edit->posn==edit->current.length);
}

/** @brief Advances the position by delta
 *  @details We ensure that the current position also lies within the string. */
void linedit_advanceposition(lineditor *edit, int delta) {
    edit->posn+=delta;
    if (edit->posn<0) edit->posn=0;
    int linewidth = linedit_stringwidth(&edit->current);
    if (edit->posn>linewidth) edit->posn=linewidth;
}

/** Obtain and process a single keypress */
bool linedit_processkeypress(lineditor *edit) {
    keypress key;
    bool regeneratesuggestions=true;
    
    linedit_keypressinit(&key);
    
    if (linedit_readkey(edit, &key)) {
        switch (key.type) {
            case CHARACTER:
                linedit_setmode(edit, LINEDIT_DEFAULTMODE);
                linedit_stringinsert(&edit->current, edit->posn, key.c, key.nbytes);
                linedit_advanceposition(edit, 1);
                break;
            case DELETE:
                if (linedit_getmode(edit)==LINEDIT_SELECTIONMODE) {
                    /* Delete the selection */
                    int lposn=(edit->sposn < edit->posn ? edit->sposn : edit->posn);
                    int rposn = (edit->sposn < edit->posn ? edit->posn : edit->sposn);
                    linedit_stringdelete(&edit->current, lposn, rposn-lposn);
                    edit->posn=lposn;
                } else {
                    /* Delete a character */
                    if (edit->posn>0) {
                        linedit_stringdelete(&edit->current, edit->posn-1, 1);
                        linedit_advanceposition(edit, -1);
                    }
                }
                linedit_setmode(edit, LINEDIT_DEFAULTMODE);
                break;
            case LEFT:
                linedit_setmode(edit, LINEDIT_DEFAULTMODE);
                linedit_advanceposition(edit, -1);
                break;
            case RIGHT:
                linedit_setmode(edit, LINEDIT_DEFAULTMODE);
                linedit_advanceposition(edit, +1);
                break;
            case UP:
                if (linedit_getmode(edit)!=LINEDIT_HISTORYMODE) {
                    linedit_setmode(edit, LINEDIT_HISTORYMODE);
                    linedit_historyadd(edit, (edit->current.string ? edit->current.string : ""));
                }
                linedit_historyadvance(edit, +1);
                linedit_setposition(edit, -1);
                break;
            case DOWN:
                if (linedit_getmode(edit)==LINEDIT_HISTORYMODE) {
                    linedit_historyadvance(edit, -1);
                    linedit_setposition(edit, -1);
                } else if (linedit_aresuggestionsavailable(edit)) {
                    linedit_advancesuggestions(edit, 1);
                    regeneratesuggestions=false;
                }
                break;
            case SHIFT_LEFT:
                linedit_setmode(edit, LINEDIT_SELECTIONMODE);
                linedit_advanceposition(edit, -1);
                break;
            case SHIFT_RIGHT:
                linedit_setmode(edit, LINEDIT_SELECTIONMODE);
                linedit_advanceposition(edit, +1);
                break;
            case RETURN:
                return false;
            case TAB:
                linedit_setmode(edit, LINEDIT_DEFAULTMODE);
                /* If suggestions are available (i.e. we're at the end of the line)... */
                if (linedit_aresuggestionsavailable(edit)) {
                    char *sugg = linedit_currentsuggestion(edit);
                    if (sugg) {
                        linedit_stringaddcstring(&edit->current, sugg);
                        linedit_setposition(edit, -1);
                    }
                }
                break;
            case CTRL: /* Handle ctrl+letter combos */
                switch(LINEDIT_KEYPRESSGETCHAR(&key)) {
                    case 'A': /* Move to start of line */
                        linedit_setmode(edit, LINEDIT_DEFAULTMODE);
                        edit->posn=0;
                        break;
                    case 'B': /* Move backward */
                        linedit_setmode(edit, LINEDIT_DEFAULTMODE);
                        linedit_advanceposition(edit, -1);
                        break;
                    case 'C': /* Copy */
                        if (linedit_getmode(edit)==LINEDIT_SELECTIONMODE) {
                            int lposn=(edit->sposn < edit->posn ? edit->sposn : edit->posn);
                            int rposn = (edit->sposn < edit->posn ? edit->posn : edit->sposn);
                            size_t lindx = linedit_utf8index(&edit->current, lposn, 0);
                            size_t rindx = linedit_utf8index(&edit->current, rposn, 0);
                            linedit_stringclear(&edit->clipboard);
                            linedit_stringappend(&edit->clipboard, edit->current.string+lindx, (size_t) rindx-lindx);
                        }
                        break;
                    case 'D': /* Delete a character */
                        linedit_setmode(edit, LINEDIT_DEFAULTMODE);
                        linedit_stringdelete(&edit->current, edit->posn, 1);
                        break;
                    case 'E': /* Move to end of line */
                        linedit_setmode(edit, LINEDIT_DEFAULTMODE);
                        linedit_setposition(edit, -1);
                        break;
                    case 'F': /* Move forward */
                        linedit_setmode(edit, LINEDIT_DEFAULTMODE);
                        linedit_advanceposition(edit, +1);
                        break;
                    case 'U': /* Delete whole line */
                        linedit_setmode(edit, LINEDIT_DEFAULTMODE);
                        linedit_stringclear(&edit->current);
                        edit->posn=0;
                        break;
                    case 'V': /* Paste */
                        linedit_setmode(edit, LINEDIT_DEFAULTMODE);
                        if (edit->clipboard.length>0) {
                            linedit_stringinsert(&edit->current, edit->posn, edit->clipboard.string, edit->clipboard.length);
                            linedit_advanceposition(edit, linedit_stringwidth(&edit->clipboard));
                        }
                        break;
                    default: break;
                }
                break;
            default:
                break;
        }
    }
    
    if (regeneratesuggestions) linedit_generatesuggestions(edit);
    
    return true;
}

/* **********************************************************************
 * Interfaces
 * ********************************************************************** */

/** If we're not attached to a terminal, e.g. a pipe, simply read the
    file in. */
void linedit_noterminal(lineditor *edit) {
    int c;
    linedit_stringclear(&edit->current);
    do {
        c = fgetc(stdin);
        if (c==EOF || c=='\n') return;
        char a = (char) c;
        linedit_stringappend(&edit->current, &a, 1);
    } while (true);
}

/** If the terminal is unsupported, default to fgets with a fixed buffer */
#define LINEDIT_UNSUPPORTEDBUFFER   4096
void linedit_unsupported(lineditor *edit) {
    char buffer[LINEDIT_UNSUPPORTEDBUFFER];
    printf("%s",edit->prompt.string);
    if (fgets(buffer, LINEDIT_UNSUPPORTEDBUFFER, stdin)==buffer) {
        int length=(int) strlen(buffer);
        if (length>0) for (length--; length>=0 && iscntrl(buffer[length]); length--) {
            buffer[length]='\0'; /* Remove trailing ctrl chars */
        }
        linedit_stringaddcstring(&edit->current, buffer);
    }
}

/** Normal interface used if terminal is present */
void linedit_supported(lineditor *edit) {
    linedit_enablerawmode();

    linedit_setmode(edit, LINEDIT_DEFAULTMODE);
    linedit_getterminalwidth(edit);
    linedit_setposition(edit, 0);
    linedit_refreshline(edit);
    
    while (linedit_processkeypress(edit)) {
        linedit_refreshline(edit);
    }

    /* Remove any dangling suggestions */
    linedit_stringlistclear(&edit->suggestions);
    linedit_setmode(edit, LINEDIT_DEFAULTMODE);
    linedit_refreshline(edit);
    
    linedit_disablerawmode();
    
    if (edit->current.length>0) {
        linedit_historyadd(edit, edit->current.string);
    }
    printf("\n");
}

/* **********************************************************************
 * Public interface
 * ********************************************************************** */

/** Public interface to the line editor.
 *  @param   edit - a line editor that has been initialized with linedit_init.
 *  @returns the string input by the user, or NULL if nothing entered. */
char *linedit(lineditor *edit) {
    if (!edit) return NULL; /** Ensure we are not passed a NULL pointer */
    
    linedit_stringclear(&edit->current);
    
    switch (linedit_checksupport()) {
        case LINEDIT_NOTTTY: linedit_noterminal(edit); break;
        case LINEDIT_UNSUPPORTED: linedit_unsupported(edit); break;
        case LINEDIT_SUPPORTED: linedit_supported(edit); break;
    }
    
    return linedit_cstring(&edit->current);
}

/** @brief Configures syntax coloring
 *  @param edit         Line editor to configure
 *  @param tokenizer    A function to be called that will find the next token from a string
 *  @param cols         An array of colors, one entry for each token type
 *  @param ncols        Number of entries in the color array */
void linedit_syntaxcolor(lineditor *edit, linedit_tokenizer tokenizer, linedit_color *cols, unsigned int ncols) {
    if (!edit) return;
    if (edit->color) free(edit->color);
    
    edit->color = malloc(sizeof(linedit_syntaxcolordata)+ncols*sizeof(linedit_color));
    
    if (edit->color) {
        edit->color->tokenizer=tokenizer;
        edit->color->ncols=ncols;
        edit->color->lexwarning=false; 
        for (unsigned int i=0; i<ncols; i++) {
            edit->color->col[i]=cols[i];
        }
    }
}

/** @brief Configures autocomplete
 *  @param edit         Line editor to configure
 *  @param completer    a function */
void linedit_autocomplete(lineditor *edit, linedit_completer completer) {
    if (!edit) return;
    edit->completer=completer;
}

/** @brief Adds a completion suggestion
 *  @param completion   completion data structure
 *  @param string       string to add */
void linedit_addsuggestion(linedit_stringlist *completion, char *string) {
    linedit_stringlistadd(completion, string);
}

/** @brief Sets the prompt
 *  @param edit         Line editor to configure
 *  @param prompt       prompt string to use */
void linedit_setprompt(lineditor *edit, char *prompt) {
    if (!edit) return;
    linedit_stringclear(&edit->prompt);
    linedit_stringaddcstring(&edit->prompt, prompt);
}

/** @brief Displays a string with a given color and emphasis
 *  @param edit         Line editor in use
 *  @param string       String to display */
void linedit_displaywithstyle(lineditor *edit, char *string, linedit_color col, linedit_emphasis emph) {
    if (linedit_checksupport()==LINEDIT_SUPPORTED) {
        linedit_string out;
        linedit_stringinit(&out);
        linedit_setcolor(&out, col);
        linedit_setemphasis(&out, emph);
        linedit_stringaddcstring(&out, string);
        linedit_setdefaulttext(&out);
        
        printf("%s", out.string);
        
        linedit_stringclear(&out);
    } else {
        printf("%s", string);
    }
}

/** @brief Displays a string with syntax coloring
 *  @param edit         Line editor in use
 *  @param string       String to display
 */
void linedit_displaywithsyntaxcoloring(lineditor *edit, char *string) {
    if (linedit_checksupport()==LINEDIT_SUPPORTED) {
        linedit_string in, out;
        linedit_stringinit(&in);
        linedit_stringinit(&out);
        linedit_stringaddcstring(&in, string);
        
        linedit_syntaxcolorstring(edit, &in, &out);
        linedit_setdefaulttext(&out);
        printf("%s", out.string);
        
        linedit_stringclear(&in);
        linedit_stringclear(&out);
    } else {
        printf("%s", string);
    }
}

/** @brief Gets the terminal width
 *  @param edit         Line editor in use
 *  @returns The width in characters */
int linedit_getwidth(lineditor *edit) {
    linedit_getterminalwidth(edit);
    return edit->ncols;
}

/** Initialize a line editor */
void linedit_init(lineditor *edit) {
    if (!edit) return;
    edit->color=NULL;
    edit->ncols=0;
    linedit_stringlistinit(&edit->history);
    linedit_stringlistinit(&edit->suggestions);
    linedit_setmode(edit, LINEDIT_DEFAULTMODE);
    linedit_stringinit(&edit->current);
    linedit_stringinit(&edit->prompt);
    linedit_stringinit(&edit->clipboard);
    linedit_setprompt(edit, LINEDIT_DEFAULTPROMPT);
    edit->completer=NULL;
}

/** Finalize a line editor */
void linedit_clear(lineditor *edit) {
    if (!edit) return;
    if (edit->color) {
        free(edit->color);
        edit->color=NULL;
    }
    linedit_historyclear(edit);
    linedit_stringlistclear(&edit->suggestions);
    linedit_stringclear(&edit->current);
    linedit_stringclear(&edit->prompt);
    linedit_stringclear(&edit->clipboard);
}
