/** @file cli.c
 *  @author T J Atherton
 *
 *  @brief Command line interface
*/

#include <time.h>
#include "cli.h"
#include "parse.h"
#include "file.h"

#define CLI_BUFFERSIZE 1024

/** Report an error if one has occurred. */
void cli_reporterror(error *err, vm *v) {
    if (err->cat!=ERROR_NONE) {
        printf("%sError '%s'%s", CLI_ERRORCOLOR, err->id , CLI_NORMALTEXT);
        if (ERROR_ISRUNTIMEERROR(*err)) {
            printf("%s: %s%s\n", CLI_ERRORCOLOR, err->msg, CLI_NORMALTEXT);
            morpho_stacktrace(v);
        } else {
            printf("%s [line %u char %u", CLI_ERRORCOLOR, err->line, err->posn);
            if (err->module) printf(" in module %s", err->module);
            printf("] : %s%s\n", err->msg, CLI_NORMALTEXT);
        }
    }
}

/* **********************************************************************
 * Interactive cli
 * ********************************************************************** */

const char *cli_file;

/** Define colors for different token types */
linedit_color cli_tokencolors[] = {
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_NONE
    
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_NEWLINE
    
    LINEDIT_YELLOW,                                // TOKEN_QUESTION
    
    LINEDIT_BLUE,                                  // TOKEN_STRING
    LINEDIT_BLUE,                                  // TOKEN_INTERPOLATION
    LINEDIT_BLUE,                                  // TOKEN_INTEGER
    LINEDIT_BLUE,                                  // TOKEN_NUMBER
    LINEDIT_CYAN,                                  // TOKEN_SYMBOL
    LINEDIT_MAGENTA,                               // TOKEN_TRUE
    LINEDIT_MAGENTA,                               // TOKEN_FALSE
    LINEDIT_MAGENTA,                               // TOKEN_NIL
    LINEDIT_MAGENTA,                               // TOKEN_SELF
    LINEDIT_MAGENTA,                               // TOKEN_SUPER
    
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_LEFTPAREN
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_RIGHTPAREN
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_LEFTSQBRACKET
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_RIGHTSQBRACKET
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_LEFTCURLYBRACKET
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_RIGHTCURLYBRACKET

    LINEDIT_DEFAULTCOLOR,                          // TOKEN_COLON
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_SEMICOLON
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_COMMA
    
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_PLUS
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_MINUS
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_STAR
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_SLASH
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_CIRCUMFLEX
    
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_PLUSPLUS
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_MINUSMINUS
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_PLUSEQ
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_MINUSEQ
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_STAREQ
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_SLASHEQ
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_HASH
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_AT
    
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_DOT
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_DOTDOT
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_DOTDOT
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_EXCLAMATION
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_AMP
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_VBAR
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_DBLAMP
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_DBLVBAR
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_EQUAL
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_EQ
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_NEQ
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_LT
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_GT
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_LTEQ
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_GTEQ
    
    LINEDIT_MAGENTA,                               // TOKEN_PRINT
    LINEDIT_MAGENTA,                               // TOKEN_VAR
    LINEDIT_MAGENTA,                               // TOKEN_IF
    LINEDIT_MAGENTA,                               // TOKEN_ELSE
    LINEDIT_MAGENTA,                               // TOKEN_IN
    LINEDIT_MAGENTA,                               // TOKEN_WHILE
    LINEDIT_MAGENTA,                               // TOKEN_FOR
    LINEDIT_MAGENTA,                               // TOKEN_DO
    LINEDIT_MAGENTA,                               // TOKEN_BREAK
    LINEDIT_MAGENTA,                               // TOKEN_CONTINUE
    LINEDIT_MAGENTA,                               // TOKEN_FUNCTION
    LINEDIT_MAGENTA,                               // TOKEN_RETURN
    LINEDIT_MAGENTA,                               // TOKEN_CLASS
    LINEDIT_MAGENTA,                               // TOKEN_IMPORT
    LINEDIT_MAGENTA,                               // TOKEN_AS
    
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_INCOMPLETE
    LINEDIT_DEFAULTCOLOR,                          // TOKEN_ERROR
    LINEDIT_DEFAULTCOLOR                           // TOKEN_EOF
};

/** A tokenizer for syntax coloring that leverages the parser's lexer */
bool cli_lex(char *in, void **ref, linedit_token *out) {
    lexer *l=(lexer *) *ref;
    token tok;
    error err;
    
    /* On the first call, allocate and initialize the tokenizer */
    if (!l) {
        *ref = l = malloc(sizeof(lexer));
        lex_init(l, in, 1);
    }
    
    if (lex(l, &tok, &err)) {
        out->start=(char *) tok.start;
        out->length=tok.length;
        out->type=(unsigned int) tok.type;
        return (tok.type!=TOKEN_EOF);
    }
    
    return false;
}

/** Autocomplete function */
bool cli_complete(char *in, linedit_stringlist *c) {
    size_t len=strlen(in);
    
    /* First find the last token in the input */
    char *tok = in+len;
    /* Scan backwards from end of string over alphanumeric tokens */
    while (tok>in && !isspace(*(tok-1))) tok--;
    
    /* Ensure we have at least one character */
    if (iscntrl(*tok)) return false;
    
    /* Now try to match the token against a library of words */
    len=strlen(tok);
    char *words[] = {"as", "and", "break", "class", "continue", "do", "else", "for", "false", "fn", "help", "if", "in", "import", "nil", "or", "print", "return", "true", "var", "while", "quit", "self", "super", "this", NULL};
    int success=false;
    
    for (unsigned int i=0; words[i]!=NULL; i++) {
        if ( (len<strlen(words[i])) &&
             (strncmp(tok, words[i], len)==0)) {
            linedit_addsuggestion(c, words[i]+len);
            success=true;
        }
    }
    
    return success;
}

/** Interactive help */
void cli_help (lineditor *edit, char *query, error *err, bool avail) {
    char *q=query;
    if (help_querylength(q, NULL)==0) {
        if (err->cat!=ERROR_NONE) {
            q=err->id;
	    error_clear(err);
        } else {
            q=HELP_INDEXPAGE;
        }
    }
    
    objecthelptopic *topic = help_search(q);
    if (topic) {
        help_display(edit, topic);
    } else {
        while (isspace(*q) && *q!='\0') q++;
        printf("No help found for '%s'\n", q);
    }
}

/** @brief Provide a command line interface */
void cli(clioptions opt) {
    printf("\U0001F98B morpho 0.5.0  | \U0001F44B Type 'help' or '?' for help\n");
    
    cli_file=NULL;
    
    /* Set up program and compiler */
    program *p = morpho_newprogram();
    compiler *c = morpho_newcompiler(p);
    
    bool help = help_initialize();
    
    lineditor edit;
    
    /* Set up VM */
    vm *v = morpho_newvm();
    
    /* Always enable debugging in interactive mode */
    morpho_setdebug(v, true);
    
    linedit_init(&edit);
    linedit_setprompt(&edit, CLI_PROMPT);
    linedit_syntaxcolor(&edit, cli_lex, cli_tokencolors, TOKEN_EOF);
    linedit_autocomplete(&edit, cli_complete);
    
    error err; /* Error structure that received messages from the compiler and VM */
    bool success=false; /* Keep track of whether compilation and execution was successful */
    
    /* Initialize the error struct */
    error_init(&err);
    
    /* Read-evaluate-print loop */
    for (;;) {
        char *input=NULL;
        
        while (!input) input=linedit(&edit);
        
        /* Check for CLI commands. */
        /* Let the user quit by typing 'quit'. */
        if (strncmp(input, CLI_QUIT, 4)==0) {
            break;
        } else if (strncmp(input, CLI_HELP, strlen(CLI_HELP))==0) {
            cli_help(&edit, input+strlen(CLI_HELP), &err, help); continue;
        } else if (strncmp(input, CLI_SHORT_HELP, strlen(CLI_SHORT_HELP))==0) {
            cli_help(&edit, input+strlen(CLI_SHORT_HELP), &err, help); continue;
        }
        
        /* Compile code */
        success=morpho_compile(input, c, &err);
        
        if (success) {
            /* If compilation was successful, and we're in interactive mode, execute... */
            if (opt & CLI_DISASSEMBLE) {
                morpho_disassemble(p, NULL);
            }
            if (opt & CLI_RUN) {
                success=morpho_run(v, p);
                if (!success) {
                    cli_reporterror(morpho_geterror(v), v);
                    err=*morpho_geterror(v);
                }
            }
        } else {
            /* ... otherwise just raise an error. */
            cli_reporterror(&err, v);
        }
    }
    
    linedit_clear(&edit);
    
    morpho_freeprogram(p);
    morpho_freecompiler(c);
    morpho_freevm(v);
}

/* **********************************************************************
 * Run a file
 * ********************************************************************** */

/** Loads and runs a file. */
void cli_run(const char *in, clioptions opt) {
    program *p = morpho_newprogram();
    compiler *c = morpho_newcompiler(p);
    vm *v = morpho_newvm();
    
    if (opt & CLI_DEBUG) morpho_setdebug(v, true);
    
    char *src = cli_loadsource(in);
    
    error err; /* Error structure that received messages from the compiler and VM */
    bool success=false; /* Keep track of whether compilation and execution was successful */
    
    /* Open the input file if provided */
    cli_file = in;
    file_setworkingdirectory(in);
    
    if (src) {
        /* Compile code */
        success=morpho_compile(src, c, &err);
        
        /* Run code if successful */
        if (success) {
            if (opt & CLI_DISASSEMBLE) {
                if (opt & CLI_DISASSEMBLESHOWSRC) {
                    cli_disassemblewithsrc(p, src);
                } else {
                    morpho_disassemble(p, NULL);
                }
            }
            if (opt & CLI_RUN) {
                success=morpho_run(v, p);
                if (!success) cli_reporterror(morpho_geterror(v), v);
            }
        } else {
            cli_reporterror(&err, v);
        }
    } else {
        printf("Could not open file '%s'.\n", in);
    }
    
    MORPHO_FREE(src);
    morpho_freevm(v);
    morpho_freeprogram(p);
    morpho_freecompiler(c);
}

/* **********************************************************************
 * Load source code
 * ********************************************************************** */

/** Loads a source file, returning it as a C-string. Call MORPHO_FREE on it when finished. */
char *cli_loadsource(const char *in) {
    const char *inn = (in ? in : cli_file);
    FILE *f = NULL; /* Input file */
    int size=0;
    
    varray_char buffer;
    varray_charinit(&buffer);
    
    /* Open the input file if provided */
    if (inn) f=fopen(inn, "r");
    if (inn && !f) {
        return NULL;
    }
    
    /* Determine the file size */
    fseek(f, 0L, SEEK_END);
    size = ((int) ftell(f))+1;
    rewind(f);
    
    if (size) {
        /* Size the buffer to match */
        if (!varray_charresize(&buffer, size+1)) {
            return NULL;
        }
        
        /* Read in the file */
        for (char *c=buffer.data; !feof(f); c=c+strlen(c)) {
            if (!fgets(c, (int) (buffer.data+buffer.capacity-c), f)) { c[0]='\0'; break; }
        }
        
        fclose(f);
    }
    
    return buffer.data;
}

/* **********************************************************************
 * Source listing and disassembly
 * ********************************************************************** */

/** Displays a single line of source */
static void cli_printline(lineditor *edit, int line, char *prompt, char *src, int length) {
    printf("%s %4u : ", prompt, line);
    /* Display the src line */
    char srcline[length];
    strncpy(srcline, src, length-1);
    srcline[length-1]='\0';
    linedit_displaywithsyntaxcoloring(edit, srcline);
    printf("\n");
}

/** Disassembles the program showing syntax colored lines of source */
void cli_disassemblewithsrc(program *p, char *src) {
    lineditor edit;
    linedit_init(&edit);
    linedit_syntaxcolor(&edit, cli_lex, cli_tokencolors, TOKEN_EOF);
    
    int line=1, length=0;
    for (unsigned int i=0; src[i]!='\0'; i++) {
        length++;
        if (src[i]=='\n' || src[i]=='\0') {
            cli_printline(&edit, line, ">>>", src+i-length+1, length);
            morpho_disassemble(p, &line);
            line++; length=0;
        }
    }
    
    linedit_clear(&edit);
}

/** Displays a source listing from source lines start to end */
void cli_list(const char *in, int start, int end) {
    char *src = cli_loadsource(in);
    lineditor edit;
    
    if (src) {
        linedit_init(&edit);
        linedit_syntaxcolor(&edit, cli_lex, cli_tokencolors, TOKEN_EOF);
        
        int line=1, length=0;
        for (unsigned int i=0; src[i]!='\0'; i++) {
            length++;
            if (src[i]=='\n' || src[i]=='\0') {
                if (line>=start && line <=end) cli_printline(&edit, line, "", src+i-length+1, length);
                line++;
                length=0;
            }
        }
        
        linedit_clear(&edit);
        MORPHO_FREE(src);
    }
}
