/** @file command.c
 *  @author T J Atherton
 *
 *  @brief Command language for morphoview
 */
#include <string.h>
#include <ctype.h>

#include "command.h"
#include "memory.h"
#include "varray.h"
#include "display.h"

/** Get the size of an open file
 *  @param[in] f file handle
 *  @param[out] s The file size */
bool command_getfilesize(FILE *f, size_t *s) {
    long int curr, size;
    curr=ftell(f);
    if (fseek(f, 0L, SEEK_END)!=0) return false;
    size = ftell(f);
    if (fseek(f, curr, SEEK_SET)!=0) return false;
    if (s) *s = size;
    return true;
}

/** Removes a command file (for temporary files)
 *  @param[in] in file name */
void command_removefile(const char *in) {
    size_t len = strlen(in);
    char remove[len+4];
    strcpy(remove, "rm ");
    strcpy(remove+3, in);
    int systemRet = system(remove);
    if(systemRet == -1){
        // The system method failed
        printf("Warning: the system method to remove a temporary file (command.c:34.5) has failed.");
    }
}

/** Returns the contents of a file as a string
 *  @param[in] in file name
 *  @param[out] out a string with the contents of the file. Call MORPHO_FREE on this once done.
 *  @returns bool indicating success. */
bool command_loadinput(const char *in, char **out) {
    FILE *f=NULL;
    varray_char buffer;
    
    varray_charinit(&buffer);
    
    f=fopen(in, "r");
    if (!f) {
        fprintf(stderr, "morphoview: Couldn't open input file %s.\n", in);
        goto loadinput_cleanup;
    }
    
    /* Determine the file size */
    size_t size;
    if (!command_getfilesize(f, &size)) goto loadinput_cleanup;
    
    if (size) {
        /* Size the buffer to match */
        if (!varray_charresize(&buffer, (int) size+1)) {
            fprintf(stderr, "morphoview: Couldn't allocate buffer to load input file.\n");
            goto loadinput_cleanup;
        }
        
        /* Read in the file */
        for (char *c=buffer.data; !feof(f); c=c+strlen(c)) {
            if (!fgets(c, (int) (buffer.data+buffer.capacity-c), f)) { c[0]='\0'; break; }
        }
    }
    *out = buffer.data;
    fclose(f);
    return true;
    
loadinput_cleanup:
    if (f) fclose(f);
    varray_charclear(&buffer);
    return false;
}

/* -------------------------------------------------------
 * Lexer
 * ------------------------------------------------------- */

/** @brief Records a token
 *  @param[in]  l     The lexer in use
 *  @param[in]  type  Type of token to record
 *  @param[out] tok   Token structure to fill out */
void command_lexrecordtoken(lexer *l, tokentype type, token *tok) {
    tok->type=type;
    tok->start=l->start;
    tok->length=(int) (l->current - l->start);
}

/** @brief Checks if we're at the end of the string. Doesn't advance. */
static bool command_lexisatend(lexer *l) {
    return (*(l->current) == '\0');
}

/** @brief Checks if a character is a digit. Doesn't advance. */
static bool command_lexisdigit(char c) {
    return (c>='0' && c<= '9');
}

/** @brief Checks if a character is alphanumeric or underscore.  Doesn't advance. */
/*static bool command_lexisalpha(char c) {
    return (c>='a' && c<= 'z') || (c>='A' && c<= 'Z') || (c=='_');
}*/

/** @brief Advances the lexer by one character, returning the character */
static char command_lexadvance(lexer *l) {
    char c = *(l->current);
    l->current++;
    return c;
}

/** @brief Returns the next character */
static char command_lexpeek(lexer *l) {
    return *(l->current);
}

/** @brief Returns n characters ahead. Caller should check that this is meaningfull. */
/*static char command_lexpeekahead(lexer *l, int n) {
    return *(l->current + n);
}*/

/** @brief Initialize the lexer */
void command_lexinit(lexer *l, const char *start) {
    l->start=start;
    l->current=start;
}

/** @brief Lex numbers
 *  @param[in]  l    the lexer
 *  @param[out] tok  token record to fill out
 *  @returns true on success, false if an error occurs */
static bool command_lexnumber(lexer *l, token *tok) {
    tokentype type=TOKEN_INTEGER;
    
    /* Handle initial negative sign */
    if (command_lexpeek(l) == '-') command_lexadvance(l);
    
    while (command_lexisdigit(command_lexpeek(l))) command_lexadvance(l);
    
    /* Fractional part */
    if (command_lexpeek(l) == '.') {
        type=TOKEN_FLOAT;
        command_lexadvance(l); /* Consume the '.' */
        while (command_lexisdigit(command_lexpeek(l))) command_lexadvance(l);
    }
    
    /* Exponent */
    if (command_lexpeek(l) == 'e' || command_lexpeek(l) == 'E') {
        type=TOKEN_FLOAT;
        command_lexadvance(l); /* Consume the 'e' */
        
        /* Optional sign */
        if (command_lexpeek(l) == '+' || command_lexpeek(l) == '-') command_lexadvance(l);
        
        /* Exponent digits */
        while (command_lexisdigit(command_lexpeek(l))) command_lexadvance(l);
    }
    
    command_lexrecordtoken(l, type, tok);
    
    return true;
}

/** @brief Lex strings
 *  @param[in]  l    the lexer
 *  @param[out] tok  token record to fill out
 *  @returns true on success, false if an error occurs */
static bool command_lexstring(lexer *l, token *tok) {
    while (command_lexpeek(l) != '"' && !command_lexisatend(l)) {
        command_lexadvance(l);
    }
    
    if (command_lexisatend(l)) {
        return false;
    }
    
    command_lexadvance(l); /* Closing quote */
    
    command_lexrecordtoken(l, TOKEN_STRING, tok);
    return true;
}

/** @brief Obtain the next token */
bool command_lex(lexer *l, token *tok) {
    /** Skip leading white space */
    while (isspace(command_lexpeek(l))) command_lexadvance(l);
    
    l->start = l->current;
    
    if (command_lexisatend(l)) {
        command_lexrecordtoken(l, TOKEN_EOF, tok);
        return true;
    }
    
    char c = command_lexadvance(l);
    
    if (command_lexisdigit(c) || c == '-') return command_lexnumber(l, tok);
    
    switch (c) {
        case 'c': command_lexrecordtoken(l, TOKEN_COLOR, tok); return true;
        case 'C': command_lexrecordtoken(l, TOKEN_SELECTCOLOR, tok); return true;
        case 'd': command_lexrecordtoken(l, TOKEN_DRAW, tok); return true;
        case 'o': command_lexrecordtoken(l, TOKEN_OBJECT, tok); return true;
        case 'p': command_lexrecordtoken(l, TOKEN_POINTS, tok); return true;
        case 'l': command_lexrecordtoken(l, TOKEN_LINES, tok); return true;
        case 'f': command_lexrecordtoken(l, TOKEN_FACETS, tok); return true;
        case 'F': command_lexrecordtoken(l, TOKEN_FONT, tok); return true;
        case 'i': command_lexrecordtoken(l, TOKEN_IDENTITY, tok); return true;
        case 'm': command_lexrecordtoken(l, TOKEN_MATRIX, tok); return true;
        case 'r': command_lexrecordtoken(l, TOKEN_ROTATE, tok); return true;
        case 's': command_lexrecordtoken(l, TOKEN_SCALE, tok); return true;
        case 'S': command_lexrecordtoken(l, TOKEN_SCENE, tok); return true;
        case 't': command_lexrecordtoken(l, TOKEN_TRANSLATE, tok); return true;
        case 'T': command_lexrecordtoken(l, TOKEN_TEXT, tok); return true;
        case 'v': command_lexrecordtoken(l, TOKEN_VERTICES, tok); return true;
        case 'W': command_lexrecordtoken(l, TOKEN_WINDOW, tok); return true;
        case '"': return command_lexstring(l, tok);
    }
    
    return false;
}

/* -------------------------------------------------------
 * Parser
 * ------------------------------------------------------- */

/** Initialize the parser */
void command_parseinit(parser *p, char *in) {
    command_lexinit(&p->l, in);
    p->current.type=TOKEN_NONE;
    p->prev.type=TOKEN_NONE;
    p->modelchanged=false;
}

/** Advance the parser one token */
bool command_parseisatend(parser *p) {
    return command_lexisatend(&p->l);
}

/** Advance the parser one token */
bool command_parseadvance(parser *p) {
    p->prev = p->current;
    bool success=command_lex(&p->l, &p->current);
    if (!success) fprintf(stderr, "morphoview: Unrecognized token.\n");
    return success;
}

/** Parses the current token as an integer */
bool command_parseinteger(parser *p, int *out) {
    if (p->current.type==TOKEN_INTEGER) {
        long f = strtol(p->current.start, NULL, 10);
        *out = (int) f;
        return command_parseadvance(p);
    }
    return false;
}

/** Parses the current token as a float */
bool command_parsefloat(parser *p, float *out) {
    if (p->current.type==TOKEN_INTEGER || p->current.type==TOKEN_FLOAT) {
        float f = strtof(p->current.start, NULL);
        *out = f;
        return command_parseadvance(p);
    }
    return false;
}

/** Parses the current token as a string */
bool command_parsestring(parser *p, char **out) {
    if (p->current.type==TOKEN_INTEGER || p->current.type==TOKEN_STRING) {
        int length = p->current.length-2;
        char *str = malloc(sizeof(char)*(length+1));
        if (str) {
            strncpy(str, p->current.start+1, length);
            str[length]='\0';
            *out = str;
            
            return command_parseadvance(p);
        }
    }
    return false;
}

/** Checks the current token type */
tokentype command_parsecurrenttype(parser *p) {
    return p->current.type;
}

/** Checks if the current token is an integer */
bool command_iscurrentnumerical(parser *p) {
    tokentype t=command_parsecurrenttype(p);
    return (t==TOKEN_FLOAT || t==TOKEN_INTEGER);
}

/* ---------------
 * Parse functions
 * --------------- */

#define ERRCHK(f) if (!(f)) return false;

/** Parses a color definition */
bool command_parsecolor(parser *p) {
    int id, indx=-1;
    int length=0;
    ERRCHK(command_parseinteger(p, &id));
    
#ifdef DEBUG_PARSER
    printf("Color %i ", id);
#endif
    
    while (command_iscurrentnumerical(p)) {
        float r[3];
        for (int i=0; i<3; i++) ERRCHK(command_parsefloat(p, &r[i]));
        
        /* Add to the scene's data array */
        int ret=scene_adddata(p->scene, r, 3);
        if (indx<0) indx=ret;
        
        length++;
        
#ifdef DEBUG_PARSER
        printf("%f %f %f ", r[0], r[1], r[2]);
#endif
    }
    
    if (length>0) {
        scene_addcolor(p->scene, id, length, indx);
    }
    
#ifdef DEBUG_PARSER
    printf("\n");
#endif
    
    return true;
}

/** Parses a color selection */
bool command_parseselectcolor(parser *p) {
    int id;
    
    ERRCHK(command_parseinteger(p, &id));
#ifdef DEBUG_PARSER
    printf("Select color %i\n", id);
#endif
    
    scene_adddraw(p->scene, COLOR, id, -1);
    
    return true;
}

/** Parses a draw command */
bool command_parsedraw(parser *p) {
    int id, indx = SCENE_EMPTY;
    ERRCHK(command_parseinteger(p, &id));
#ifdef DEBUG_PARSER
    printf("Draw %i\n", id);
#endif
    
    if (p->modelchanged) {
        indx=scene_adddata(p->scene, p->model, 16);
        p->modelchanged=false;
#ifdef DEBUG_PARSER
        mat3d_print4x4(p->model);
#endif
    }
    
    scene_adddraw(p->scene, OBJECT, id, indx);
    
    return true;
}

/** Parses an object */
bool command_parseobject(parser *p) {
    int id;
    ERRCHK(command_parseinteger(p, &id));
#ifdef DEBUG_PARSER
    printf("Object %i\n", id);
#endif
    
    if (p->scene) {
        p->cobject=scene_addobject(p->scene, id);
    } else {
        fprintf(stderr, "morphoview: No scene defined.\n");
        return false;
    }
    
    return true;
}

/** Parses a vertex list */
bool command_parsevertices(parser *p) {
    if (!p->scene || !p->cobject) {
        fprintf(stderr, "morphoview: No object defined.\n");
        return false;
    }
    
    char *format=NULL;
    if (command_parsestring(p, &format)) {
#ifdef DEBUG_PARSER
        printf("Vertices '%s'\n", format);
#endif
        p->cobject->vertexdata.format=format;
    }
    
    while (command_iscurrentnumerical(p)) {
        float f;
        ERRCHK(command_parsefloat(p, &f));
        
        /* Add to the scene's vertex data array */
        int ret=scene_adddata(p->scene, &f, 1);
        
        if (p->cobject->vertexdata.indx==SCENE_EMPTY) {
            p->cobject->vertexdata.indx=ret;
            p->cobject->vertexdata.length=0;
        }
        p->cobject->vertexdata.length++;
        
#ifdef DEBUG_PARSER
        printf("%f ", f);
#endif
    }
    
#ifdef DEBUG_PARSER
    printf("\n");
#endif
    
    return true;
}

/** Parses a graphics object that is a list of vertex array indices */
bool command_parseindex(parser *p) {
    if (!p->scene || !p->cobject) {
        fprintf(stderr, "morphoview: No object defined.\n");
        return false;
    }
    
    gelement el = { .type = POINTS, .indx = SCENE_EMPTY, .length = 0 };
    
    /* Remember the element type */
    if (p->prev.type==TOKEN_LINES) {
        el.type=LINES;
    } else if (p->prev.type==TOKEN_FACETS) {
        el.type=FACETS;
    }
    
#ifdef DEBUG_PARSER
    printf("Indexed list type %u\n", el.type);
#endif
    
    while (command_parsecurrenttype(p)==TOKEN_INTEGER) {
        int i;
        ERRCHK(command_parseinteger(p, &i));
        
#ifdef DEBUG_PARSER
        printf("%i ", i);
#endif
        
        /* Add to the scene's index data array */
        int ret=scene_addindex(p->scene, &i, 1);
        
        /* And remember the starting point and length */
        if (el.indx==SCENE_EMPTY) el.indx=ret;
        el.length++;
    }
#ifdef DEBUG_PARSER
    printf("\n");
#endif
    
    scene_addelement(p->cobject, &el);
    
    return true;
}

/** Parse an identity command */
bool command_parseidentity(parser *p) {
#ifdef DEBUG_PARSER
    printf("Identity\n");
#endif
    
    mat3d_identity4x4(p->model);
    p->modelchanged=true;
    return true;
}

/** Parse a matrix command */
bool command_parsematrix(parser *p) {
    mat4x4 x, m;
    for (int i=0; i<16; i++) {
        ERRCHK(command_parsefloat(p, &x[i]));
    }
    
#ifdef DEBUG_PARSER
    printf("Matrix:\n");
    mat3d_print4x4(x);
#endif
    
    mat3d_copy4x4(p->model, m);
    mat3d_mul4x4(m, x, p->model);
    
    p->modelchanged=true;
    
    return true;
}

/** Parse a rotate command */
bool command_parserotate(parser *p) {
    float phi, x[3];
    ERRCHK(command_parsefloat(p, &phi));
    for (int i=0; i<3; i++) {
        ERRCHK(command_parsefloat(p, &x[i]));
    }
#ifdef DEBUG_PARSER
    printf("Rotate %f (%f,%f,%f)\n", phi, x[0], x[1], x[2]);
#endif
    
    mat3d_rotate(p->model, x, phi, p->model);
    p->modelchanged=true;
    
    return true;
}

/** Parse a scale command */
bool command_parsescale(parser *p) {
    float s;
    ERRCHK(command_parsefloat(p, &s));
#ifdef DEBUG_PARSER
    printf("Scale %f\n", s);
#endif
    mat3d_scale(p->model, s, p->model);
    p->modelchanged=true;
    
    return true;
}

/** Parse a translate command */
bool command_parsetranslate(parser *p) {
    float x[3];
    for (int i=0; i<3; i++) {
        ERRCHK(command_parsefloat(p, &x[i]));
    }
#ifdef DEBUG_PARSER
    printf("Translate (%f,%f,%f)\n", x[0], x[1], x[2]);
#endif
    
    mat3d_translate(p->model, x, p->model);
    p->modelchanged=true;
    
    return true;
}

/** Parses a scene command */
bool command_parsescene(parser *p) {
    int id, dim;
    ERRCHK(command_parseinteger(p, &id));
    ERRCHK(command_parseinteger(p, &dim));
    
#ifdef DEBUG_PARSER
    printf("Scene id: %i dim: %i\n", id, dim);
#endif
    
    p->scene = scene_new(id, dim);
    if (p->scene) p->display=display_open(p->scene);
    
    return (p->scene!=NULL);
}

/** Parses a window command */
bool command_parsewindow(parser *p) {
    char *name;
    
    if (command_parsestring(p, &name)) {
#ifdef DEBUG_PARSER
        printf("Window '%s'\n", name);
#endif
        if (name) {
            display_setwindowtitle(p->display, name);
            free(name);
        }
        return true;
    }
    
    return false;
}

/** Parses a font definition */
bool command_parsefont(parser *p) {
    int id;
    char *file;
    float size;
    
    ERRCHK(command_parseinteger(p, &id));
    ERRCHK(command_parsestring(p, &file));
    ERRCHK(command_parsefloat(p, &size));
    
#ifdef DEBUG_PARSER
    printf("Font %i '%s' %g\n", id, file, size);
#endif
    
    return scene_addfont(p->scene, id, file, size, NULL);
}

/** Parses a text command */
bool command_parsetext(parser *p) {
    int fontid;
    char *string;
    
    ERRCHK(command_parseinteger(p, &fontid));
    ERRCHK(command_parsestring(p, &string));
    
#ifdef DEBUG_PARSER
    printf("Text %i '%s'\n", fontid, string);
#endif
    
    int matindx=SCENE_EMPTY;
    int tid=scene_addtext(p->scene, fontid, string);
    
    if (p->modelchanged) {
        matindx=scene_adddata(p->scene, p->model, 16);
        p->modelchanged=false;
#ifdef DEBUG_PARSER
        mat3d_print4x4(p->model);
#endif
    }
    
    scene_adddraw(p->scene, TEXT, tid, matindx);
    
    return true;
}

#define UNDEFINED NULL
/** The parse table defines which function handles which token type */
parsefunction parsetable[] = {
    UNDEFINED,              // TOKEN_NONE
    
    UNDEFINED,              // TOKEN_INTEGER
    UNDEFINED,              // TOKEN_FLOAT
    UNDEFINED,              // TOKEN_STRING
    
    command_parsecolor,     // TOKEN_COLOR
    command_parseselectcolor,// TOKEN_SELECTCOLOR
    command_parsedraw,      // TOKEN_DRAW
    command_parseobject,    // TOKEN_OBJECT
    command_parsevertices,  // TOKEN_VERTICES
    command_parseindex,     // TOKEN_POINTS
    command_parseindex,     // TOKEN_LINES
    command_parseindex,     // TOKEN_FACETS
    command_parseidentity,  // TOKEN_IDENTITY
    command_parsematrix,    // TOKEN_MATRIX
    command_parserotate,    // TOKEN_ROTATE
    command_parsescale,     // TOKEN_SCALE
    command_parsescene,     // TOKEN_SCENE
    command_parsetranslate, // TOKEN_TRANSLATE
    UNDEFINED,              // TOKEN_VIEWDIRECTION
    UNDEFINED,              // TOKEN_VIEWVERTICAL
    command_parsewindow,    // TOKEN_WINDOW
    command_parsefont,      // TOKEN_FONT
    command_parsetext,      // TOKEN_TEXT
    
    UNDEFINED, // TOKEN_EOF
};

/** @brief Parses a command sequence */
bool command_parse(char *in) {
    parser p;
    
    command_parseinit(&p, in);
    ERRCHK(command_parseadvance(&p));
    
    do {
        /* Lookup the current parse function */
        if (p.current.type>TOKEN_EOF) {
            fprintf(stderr, "morphoview: Inconsistent token definitions.\n");
            return false;
        }
        
        parsefunction fn = parsetable[p.current.type];
        if (fn==UNDEFINED) {
            fprintf(stderr, "morphoview: Couldn't parse token.\n");
            return false;
        }
        
        ERRCHK(command_parseadvance(&p));
        
        bool result = (*fn) (&p);
        if (!result) return false;
    } while (!command_parseisatend(&p));
    
    /** Prepare the scene for display */
    if (p.scene && p.display) {
        render_preparescene(&p.display->render, p.scene);
    }
    
    return true;
}
