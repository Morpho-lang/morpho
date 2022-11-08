/** @file debug.c
 *  @author T J Atherton
 *
 *  @brief Debugging, dissassembly and other tools
 */

#include <stdarg.h>
#include <string.h>

#include "compile.h"
#include "vm.h"
#include "debug.h"
#include "morpho.h"

/* **********************************************************************
 * Debugging annotations
 * ********************************************************************** */

DEFINE_VARRAY(debugannotation, debugannotation);

/** Retrieve the last annotation */
debugannotation *debug_lastannotation(varray_debugannotation *list) {
    if (list->count>0) return &list->data[list->count-1];
    return NULL;
}

/** Adds an annotation to a list */
void debug_addannotation(varray_debugannotation *list, debugannotation *annotation) {
    varray_debugannotationadd(list, annotation, 1);
}

/** Removes the last annotation */
void debug_stripend(varray_debugannotation *list) {
    if (list->count>0) list->data[list->count-1].content.element.ninstr--;
}

/** Sets the current function */
void debug_setfunction(varray_debugannotation *list, objectfunction *func) {
    debugannotation ann = { .type = DEBUG_FUNCTION, .content.function.function = func};
    debug_addannotation(list, &ann);
}

/** Sets the current class */
void debug_setclass(varray_debugannotation *list, objectclass *klass) {
    debugannotation ann = { .type = DEBUG_CLASS, .content.klass.klass = klass};
    debug_addannotation(list, &ann);
}

/** Pushes an error handler onto the stack */
void debug_pusherr(varray_debugannotation *list, objectdictionary *dict) {
    debugannotation ann = { .type = DEBUG_PUSHERR, .content.errorhandler.handler = dict};
    debug_addannotation(list, &ann);
}

/** Pops an error handler from the stack */
void debug_poperr(varray_debugannotation *list) {
    debugannotation ann = { .type = DEBUG_POPERR };
    debug_addannotation(list, &ann);
}

/** Associates a register with a symbol */
void debug_setreg(varray_debugannotation *list, indx reg, value symbol) {
    if (!MORPHO_ISSTRING(symbol)) return;
    value sym = object_clonestring(symbol);
    debugannotation ann = { .type = DEBUG_REGISTER, .content.reg.reg = reg, .content.reg.symbol = sym };
    debug_addannotation(list, &ann);
}

/** Uses information from a syntaxtreenode to associate a sequence of instructions with source */
void debug_addnode(varray_debugannotation *list, syntaxtreenode *node) {
    if (!node) return; 
    debugannotation *last = debug_lastannotation(list);
    if (last && last->type==DEBUG_ELEMENT &&
        node->line==last->content.element.line &&
        node->posn==last->content.element.posn) {
        last->content.element.ninstr++;
    } else {
        debugannotation ann = { .type = DEBUG_ELEMENT, .content.element.line = node->line, .content.element.posn = node->posn, .content.element.ninstr=1 };
        debug_addannotation(list, &ann);
    }
}

/** Clear debugging list, freeing attached info */
void debug_clearannotationlist(varray_debugannotation *list) {
    for (unsigned int j=0; j<list->count; j++) {
        switch (list->data[j].type) {
            case DEBUG_REGISTER: {
                value sym=list->data[j].content.reg.symbol;
                if (MORPHO_ISOBJECT(sym)) object_free(MORPHO_GETOBJECT(sym));
            }
                break;
            default: break;
        }
    }
    varray_debugannotationclear(list);
}

/* **********************************************************************
 * Disassembler
 * ********************************************************************** */

/** Formatting rules for disassembler */
typedef struct {
    instruction op;
    char *label;
    char *display;
} assemblyrule;

/* Order is not significant here */
assemblyrule assemblyrules[] ={
    { OP_NOP, "nop", "" },
    { OP_MOV, "mov", "rA, rB" },
    { OP_LCT, "lct", "rA, cX" }, // Custom
    { OP_ADD, "add", "rA, rB, rC" },
    { OP_SUB, "sub", "rA, rB, rC" },
    { OP_MUL, "mul", "rA, rB, rC" },
    { OP_DIV, "div", "rA, rB, rC" },
    { OP_POW, "pow", "rA, rB, rC" },
    { OP_NOT, "not", "rA, rB" },
    
    { OP_EQ, "eq ", "rA, rB, rC" },
    { OP_NEQ, "neq", "rA, rB, rC" },
    { OP_LT, "lt ", "rA, rB, rC" },
    { OP_LE, "le ", "rA, rB, rC" },
    
    { OP_PRINT, "print", "rA" },
    
    { OP_B, "b", "+" },
    { OP_BIF, "bif", "rA +" },
    { OP_BIFF, "biff", "rA +" },
    
    { OP_CALL, "call", "rA, B" }, // b literal
    { OP_INVOKE, "invoke", "rA, rB, C" }, // c literal
    
    { OP_RETURN, "return", "rB" }, // c literal

    { OP_CLOSURE, "closure", "rA, pB" }, // b prototype
    
    { OP_LUP, "lup", "rA, uB" }, // b 'u'
    { OP_SUP, "sup", "uA, rB" }, // a 'u', b c|r
    
    { OP_CLOSEUP, "closeup", "rA" },
    { OP_LPR, "lpr", "rA, rB, rC" },
    { OP_SPR, "spr", "rA, rB, rC" },
    
    { OP_LIX, "lix", "rA, rB, rC" },
    { OP_SIX, "six", "rA, rB, rC" },
    
    { OP_LGL, "lgl", "rA, gX" }, //
    { OP_SGL, "sgl", "rA, gX" }, // label b with 'g'
    
    { OP_PUSHERR, "pusherr", "cX" },
    { OP_POPERR, "poperr", "+" },
    
    { OP_CAT, "cat", "rA, rB, rC" },
    { OP_BREAK, "break", "" },
    { OP_END, "end", "" },
    { 0, NULL, "" } // Null terminate the list
};

assemblyrule *debug_getassemblyrule(unsigned int op) {
    for (unsigned int i=0; assemblyrules[i].label!=NULL; i++) if (assemblyrules[i].op==op) return &assemblyrules[i];
    return NULL;
}

typedef enum { NONE, REG, CONST} debugcontents;

/** Shows the contents of a register or constant */
bool debug_showcontents(debugcontents b, int i, value *konst, value *reg) {
    value *table = NULL;
    switch (b) {
        case CONST: table=konst; break;
        case REG: table = reg; break;
        default: break;
    }
    if (!table) return false;
    printf("%s%i=", (b==CONST ? "c" : "r"), i);
    morpho_printvalue(table[i]);
    return true;
}

/** @brief Disassembles a single instruction, writing the output to the console.
 *  @param instruction The instruction to disassemble
 *  @param indx        Instruction index to display
 *  @param konst current constant table
 *  @param reg   current registers */
void debug_disassembleinstruction(instruction instruction, instructionindx indx, value *konst, value *reg) {
    unsigned int op = DECODE_OP(instruction);
    debugcontents mode=NONE, bm=NONE, cm=NONE;
    int nb=0, nc=0;
    printf("%4lu : ", indx);
    int n=0; // Number of characters displayed
    int width=25; // Width of display
    
    assemblyrule *show=debug_getassemblyrule(op);
    if (show) {
        n+=printf("%s ", show->label);
        for (char *c=show->display; *c!='\0'; c++) {
            switch (*c) {
                case 'A': n+=printf("%u", DECODE_A(instruction)); break;
                case 'B': {
                    bm=mode; nb=DECODE_B(instruction); mode=NONE;
                    n+=printf("%u", nb);
                }
                    break;
                case 'X': {
                    bm=mode; nb=DECODE_Bx(instruction); mode=NONE;
                    n+=printf("%u", nb);
                }
                    break;
                case '+': n+=printf("%i", DECODE_sBx(instruction)); break;
                case 'C': {
                    cm=mode; nc=DECODE_C(instruction);
                    n+=printf("%u", DECODE_C(instruction));
                }
                    break;
                case 'c': mode=CONST; n+=printf("%c", *c); break;
                case 'r': mode=REG; n+=printf("%c", *c); break;
                default: n+=printf("%c", *c); break;
            }
        }
        
        /* Show contents if any were produced by this instruction */
        if ((!konst && !reg) || (bm==NONE && cm==NONE)) return;
        for (int k=width-n; k>0; k--) printf(" ");
        printf("; ");
        if (debug_showcontents(bm, nb, konst, reg)) printf(" ");
        debug_showcontents(cm, nc, konst, reg);
    }
}

/** Checks if an instruction matches a label in the current error dictionary, and if so print it. */
void debug_errorlabel(varray_value *errorstack, instructionindx i) {
    objectdictionary *dict = MORPHO_GETDICTIONARY(errorstack->data[errorstack->count-1]);
    
    /* Search the current error handler to see if this line corresponds to a label */
    for (unsigned int k=0; k<dict->dict.capacity; k++) {
        value label = dict->dict.contents[k].key;
        if (!MORPHO_ISNIL(label)) {
            if (MORPHO_GETINTEGERVALUE(dict->dict.contents[k].val)==i) {
                object_print(label);
                printf(":\n");
            }
        }
    }
}

/** Disassembles a program
 *  @param code - program to disassemble
 *  @param matchline - optional line number to match */
void debug_disassemble(program *code, int *matchline) {
    instructionindx entry = program_getentry(code); // The entry point of the function
    instructionindx i=0;
    value *konst=(code->global ? code->global->konst.data : NULL);
    bool silent = matchline;
    
    varray_value errorstack;
    varray_valueinit(&errorstack);
    
    /* Loop over debugging information */
    for (unsigned int j=0; j<code->annotations.count; j++) {
        debugannotation *ann = &code->annotations.data[j];
        
        switch(ann->type) {
            case DEBUG_ELEMENT:
                {
                    if (matchline) {
                        if (ann->content.element.line<(*matchline)) {
                            i+=ann->content.element.ninstr;
                            break;
                        }
                        if (ann->content.element.line>(*matchline)) return;
                    } else if (errorstack.count>0) {
                        debug_errorlabel(&errorstack, i);
                    }
                    
                    for (unsigned int k=0; k<ann->content.element.ninstr; k++, i++) {
                        printf("%s",(i==entry ? "->" : "  "));
                        debug_disassembleinstruction(code->code.data[i], i, konst, NULL);
                        printf("\n");
                    }
                }
                break;
            case DEBUG_FUNCTION:
                {
                    objectfunction *func=ann->content.function.function;
                    konst=func->konst.data;
                    if (silent) break;
                    if (!MORPHO_ISNIL(func->name)) {
                        printf("fn ");
                        morpho_printvalue(func->name);
                        printf(":\n");
                    } else printf("\n");
                }
                break;
            case DEBUG_CLASS:
                {
                    objectclass *klass=ann->content.klass.klass;
                    if (silent) break;
                    if (klass && !MORPHO_ISNIL(klass->name)) {
                        printf("class ");
                        morpho_printvalue(klass->name);
                        printf(":\n");
                    }
                }
                break;
            case DEBUG_PUSHERR:
                {
                    objectdictionary *errdict = ann->content.errorhandler.handler;
                    varray_valuewrite(&errorstack, MORPHO_OBJECT(errdict));
                }
                break;
            case DEBUG_POPERR:
                {
                    if (errorstack.count>0) errorstack.count--;
                }
                break;
            default:
                break;
        }
    }
    
    varray_valueclear(&errorstack);
}

/** Wrapper onto debug_disassemble */
void morpho_disassemble(program *code, int *matchline) {
    debug_disassemble(code, matchline);
}

/* **********************************************************************
 * Retrieve debugging info
 * ********************************************************************** */

/** Finds debugging info asssociated with instruction at indx */
bool debug_infofromindx(program *code, instructionindx indx, int *line, int *posn, objectfunction **func, objectclass **klass) {
    objectclass *cklass=NULL;
    objectfunction *cfunc=NULL;
    instructionindx i=0;
    
    for (unsigned int j=0; j<code->annotations.count; j++) {
        debugannotation *ann = &code->annotations.data[j];
        switch (ann->type) {
            case DEBUG_ELEMENT: {
                if (i+ann->content.element.ninstr>indx) {
                    if (line) *line = ann->content.element.line;
                    if (posn) *posn = ann->content.element.posn;
                    if (func) *func = cfunc;
                    if (klass) *klass = cklass;
                    return true;
                }
                i+=ann->content.element.ninstr;
            }
                break;
            case DEBUG_FUNCTION: cfunc=ann->content.function.function; break;
            case DEBUG_CLASS: cklass=ann->content.klass.klass; break;
            default: break;
        }
    }
    
    return false;
}

/** Identifies symbols associated with registers
 * @param[in] code - a program
 * @param[in] func - the function of interest
 * @param[in] indx - (optional) instruction to stop at
 * @param[out] symbols - array of size func->negs; entries will contain associated register names on exit */
bool debug_symbolsforfunction(program *code, objectfunction *func, instructionindx *indx, value *symbols) {
    objectfunction *cfunc=code->global;
    instructionindx i=0;
    
    for (unsigned int j=0; j<func->nregs; j++) symbols[j]=MORPHO_NIL;
    
    for (unsigned int j=0; j<code->annotations.count; j++) {
        debugannotation *ann = &code->annotations.data[j];
        switch (ann->type) {
            case DEBUG_ELEMENT: {
                if (indx && i+ann->content.element.ninstr>*indx) return true;
                i+=ann->content.element.ninstr;
            }
                break;
            case DEBUG_FUNCTION: cfunc=ann->content.function.function; break;
            case DEBUG_REGISTER: {
                if (cfunc==func) {
                    symbols[ann->content.reg.reg]=ann->content.reg.symbol;
                }
            }
                break;
            default: break;
        }
    }
    
    return true;
}

/** Prints all the annotations for a program */
void debug_showannotations(varray_debugannotation *list) {
    indx ix = 0;
    printf("Showing %u annotations.\n", list->count);
    for (unsigned int j=0; j<list->count; j++) {
        printf("%u: ", j);
        debugannotation *ann = &list->data[j];
        switch (ann->type) {
            case DEBUG_CLASS:
                printf("Class: ");
                if (!ann->content.klass.klass) {
                    printf("(none)");
                } else {
                    morpho_printvalue(MORPHO_OBJECT(ann->content.klass.klass));
                }
                break;
            case DEBUG_ELEMENT:
                printf("Element: [%ti] instructions: %i line: %i posn: %i",
                       ix, ann->content.element.ninstr, ann->content.element.line, ann->content.element.posn);
                ix+=ann->content.element.ninstr;
                break;
            case DEBUG_FUNCTION:
                printf("Function: ");
                morpho_printvalue(MORPHO_OBJECT(ann->content.function.function));
                break;
            case DEBUG_MODULE:
                printf("Module: ");
                morpho_printvalue(ann->content.module.module);
                break;
            case DEBUG_PUSHERR:
                printf("Pusherr: ");
                morpho_printvalue(MORPHO_OBJECT(ann->content.errorhandler.handler));
                break;
            case DEBUG_POPERR:
                printf("Poperr: ");
                break;
            case DEBUG_REGISTER:
                printf("Register: %ti ", ann->content.reg.reg);
                morpho_printvalue(ann->content.reg.symbol);
                break;
        }
        printf("\n");
    }
}

/* **********************************************************************
 * Stack traces
 * ********************************************************************** */

/** Prints a stacktrace */
void morpho_stacktrace(vm *v) {
    for (callframe *f = (v->errfp ? v->errfp : v->fp); f!=NULL && f>=v->frame; f--) {
        instructionindx indx = f->pc-v->current->code.data;
        if (indx>0) indx--; /* Becuase the pc always points to the NEXT instr. */
        
        printf("  ");
        printf("%s", (f==v->fp ? "  in " : "from "));
        
        if (!MORPHO_ISNIL(f->function->name)) morpho_printvalue(f->function->name);
        else printf("global");
        
        int line=0;
        if (debug_infofromindx(v->current, indx, &line, NULL, NULL, NULL)) {
            printf(" at line %u", line);
        }
        
        printf("\n");
    }
}

/* **********************************************************************
 * Debugger structure
 * ********************************************************************** */

/** Clears a debugger structure */
void debugger_clear(debugger *d) {
    varray_charclear(&d->breakpoints);
}

/** Initializes a debugger structure with a specified program */
void debugger_init(debugger *d, program *p) {
    d->singlestep=false;
    varray_charinit(&d->breakpoints);
    
    int ninstructions = p->code.count;
    if (!varray_charresize(&d->breakpoints, ninstructions)) return;
    memset(d->breakpoints.data, '\0', sizeof(char)*ninstructions);
    d->breakpoints.count=ninstructions;
}

/** Sets whether single step mode is in operation */
void debugger_setsinglestep(debugger *d, bool singlestep) {
    d->singlestep=singlestep;
}

/** Are we in singlestep mode? */
bool debugger_insinglestep(debugger *d) {
    return d->singlestep;
}

/** Sets a breakpoint */
void debugger_setbreakpoint(debugger *d, instructionindx indx) {
    if (indx>d->breakpoints.count) return;
    d->breakpoints.data[indx]='b';
}

/** Clears a breakpoint */
void debugger_clearbreakpoint(debugger *d, instructionindx indx) {
    if (indx>d->breakpoints.count) return;
    d->breakpoints.data[indx]='\0';
}

/** Tests if we should break at a given point */
bool debugger_shouldbreakat(debugger *d, instructionindx indx) {
    if (indx>d->breakpoints.count) return false;
    return (d->breakpoints.data[indx]!='\0');
}

/** Should we break */
bool debug_shouldbreakatpc(vm *v, instruction *pc) {
    if (!v->debug) return false;
    if (debugger_insinglestep(v->debug)) return true;
    instructionindx iindx = pc-v->current->code.data-1;
    if (debugger_shouldbreakat(v->debug, iindx)) return true;
    return false;
}

/* **********************************************************************
 * Debugger
 * ********************************************************************** */

void morpho_runtimeerror(vm *v, errorid id, ...);

/** Shows the contents of the registers for a given frame */
void debug_showregisters(vm *v, callframe *frame) {
    unsigned int nreg=frame->function->nregs;
    value symbols[nreg];
    instructionindx cinstr=frame->pc-v->current->code.data;
    bool sym = debug_symbolsforfunction(v->current, frame->function, &cinstr, symbols);
    
    value *reg = v->stack.data + frame->roffset;
    for (unsigned int i=0; i<nreg; i++) {
        printf("  r%u: ", i);
        morpho_printvalue(reg[i]);
        if (sym && !MORPHO_ISNIL(symbols[i])) {
            printf(" (");
            morpho_printvalue(symbols[i]);
            printf(")");
        }
        printf("\n");
    }
}

/** Shows current symbols */
void debug_showsymbols(vm *v) {
    for (callframe *f=v->fp; f>=v->frame; f--) {
        printf("in %s", (f==v->frame ? "global" : ""));
        if (!MORPHO_ISNIL(f->function->name)) morpho_printvalue(f->function->name);
        printf(":\n");
        
        value symbols[f->function->nregs];
        instructionindx indx = f->pc-v->current->code.data;
        
        debug_symbolsforfunction(v->current, f->function, &indx, symbols);
        
        for (int i=0; i<f->function->nregs; i++) {
            if (!MORPHO_ISNIL(symbols[i])) {
                printf("  ");
                morpho_printvalue(symbols[i]);
                printf("=");
                morpho_printvalue(v->stack.data[f->roffset+i]);
                printf("\n");
            }
        }
        
    }
}

/** Prints a specified symbol */
void debug_printsymbol(vm *v, char *match) {
    objectstring str = MORPHO_STATICSTRING((match ? match : ""));
    value matchstr = MORPHO_OBJECT(&str);
    
    objectstring prntlabel = MORPHO_STATICSTRING(MORPHO_PRINT_METHOD);
    
    for (callframe *f=v->fp; f>=v->frame; f--) {
        value symbols[f->function->nregs];
        instructionindx indx = f->pc-v->current->code.data;
        
        debug_symbolsforfunction(v->current, f->function, &indx, symbols);
        
        for (int i=0; i<f->function->nregs; i++) {
            if (!MORPHO_ISNIL(symbols[i]) && MORPHO_ISEQUAL(symbols[i], matchstr)) {
                morpho_printvalue(symbols[i]);
                
                printf(" (in %s", (f==v->frame ? "global" : ""));
                if (!MORPHO_ISNIL(f->function->name)) morpho_printvalue(f->function->name);
                printf(") ");
                
                printf("= ");
                value val = v->stack.data[f->roffset+i];

                if (MORPHO_ISOBJECT(val)) {
                    value printmethod, out;
                    if (morpho_lookupmethod(val, MORPHO_OBJECT(&prntlabel), &printmethod)) {
                        morpho_invoke(v, val, printmethod, 0, NULL, &out);
                    }
                } else {
                    morpho_printvalue(val);
                }
                printf("\n");
            }
        }
    }
}

/** Shows the contents of the stack */
void debug_showstack(vm *v) {
    /* Determine points on the stack that correspond to different function calls. */
    ptrdiff_t fbounds[MORPHO_CALLFRAMESTACKSIZE];
    callframe *f;
    unsigned int k=0;
    for (f=v->frame; f!=v->fp; f++) {
        fbounds[k]=f->roffset;
        k++;
    }
    fbounds[k]=f->roffset;
    
    f=v->frame; k=0;
    printf("Stack contents:\n");
    for (unsigned int i=0; i<v->fp->roffset+v->fp->function->nregs-1; i++) {
        if (i==fbounds[k]) {
            printf("---");
            if (f->function) morpho_printvalue(f->function->name);
            printf("\n");
            k++; f++;
        }
        printf("  s%u: ", i);
        morpho_printvalue(v->stack.data[i]);
        printf("\n");
    }
}

/** Prints list of globals */
void morpho_globals(vm *v) {
    printf("Globals:\n");
    for (unsigned int i=0; i<v->globals.count; i++) {
        printf("  g%u: ", i);
        morpho_printvalue(v->globals.data[i]);
        printf("\n");
    }
}

#include "linedit.h"
#include "cli.h"

/** Return the previous instruction index */
instructionindx debug_previnstruction(vm *v) {
    if (v->fp->pc>v->current->code.data) return v->fp->pc-v->current->code.data-1;
    return 0;
}

/** Return the current instruction index */
instructionindx debug_currentinstruction(vm *v) {
    return v->fp->pc-v->current->code.data-1;
}

/** Source listing */
void debug_list(vm *v) {
    int line;
    if (debug_infofromindx(v->current, debug_previnstruction(v), &line, NULL, NULL, NULL)) {
        cli_list(NULL, line-5, line+5);
    }
}

/** Reads an integer parameter */
static bool debug_parseint(char *in, int *out) {
    char *input=in;
    for (input++; !isdigit(*input) && (*input!='\0'); input++);
    if (*input=='\0') return false;
    if (out) *out=atoi(input);
    return true;
}

/** Parses a symbol */
static bool debug_parsesymbol(char *in, varray_char *out) {
    char *input=in;
    while (*input!='\0' && isspace(*input)) input++; // Skip space
    
    while (*input!='\0' && (isalpha(*input) || isdigit(*input) || *input=='_')) {
        varray_charwrite(out, *input);
        input++;
    }
    
    if (out->count>0) varray_charwrite(out, '\0'); // Ensure null terminated.
    
    return (out->count>0);
}

/** Morpho debugger */
void debugger_enter(vm *v) {
    debugger *debug = v->debug;
    lineditor edit;
    instructionindx iindx = debug_currentinstruction(v);
    
    int line=0;
    objectfunction *func=NULL;
    debug_infofromindx(v->current, iindx, &line, NULL, &func, NULL);
    
    /** If we're in single step mode, only stop when we've changed line */
    if (debugger_insinglestep(debug) &&
        line==debug->currentline &&
        func==debug->currentfunc) return;
    
    linedit_init(&edit);
    linedit_setprompt(&edit, "@>");
    printf("---morpho debugger---\n");
    printf("Type '?' or 'h' for help.\n");
    printf("%s in %s", (debug->singlestep ? "Single stepping" : "Breakpoint"), ((!func) || MORPHO_ISNIL(func->name)? "global" : MORPHO_GETCSTRING(func->name)));
    if (line!=ERROR_POSNUNIDENTIFIABLE) printf(" at line %u", line);
    printf(" at instruction %ti", iindx);
    printf("\n");
    
    for (bool stop=false; !stop; ) {
        char *input = linedit(&edit);
        int k;
        
        if (input) {
            switch (input[0]) {
                case '?':
                case 'h': case 'H':// Help
                    printf("Available commands:\n");
                    printf("  [a]ddress, [b]reakpoint, [c]ontinue, [d]isassemble\n");
                    printf("  [g]lobal, [i]nfo, [l]ist, [p]rint, [q]uit,\n");
                    printf("  [r]egisters, [s]tep, [t]race, [x]clear, [=]set\n");
                    printf("  [,]garbage collect [?]/[h]elp\n");
                    // [c]lear breakpoint
                    // [d]elete breakpoint
                    // [n]ext
                    
                    break;
                case 'A': case 'a': // Address
                {
                    if (!debug_parseint(input, &k)) break;
                    
                    if (k>=0 && k<v->fp->function->nregs) {
                        value *reg = v->stack.data + v->fp->roffset;
                        if (MORPHO_ISOBJECT(reg[k])) {
                            printf("Object in register %i at %p.\n", k, (void *) MORPHO_GETOBJECT(reg[k]));
                        }
                    } else printf("Invalid register.\n");
                }
                    break;
                case 'B': case 'b': // Breakpoint
                    if (debug_parseint(input, &k)) {
                        debugger_setbreakpoint(debug, k);
                        break;
                    }
                    
                    printf("Invalid breakpoint target.\n");
                    break;
                case 'C': case 'c': // Continue
                    debugger_setsinglestep(debug, false);
                    stop=true;
                    break;
                case 'D': case 'd': // Disassemble
                    morpho_disassemble(v->current, &line);
                    break;
                case 'G': case 'g': // Globals
                {
                    if (!debug_parseint(input, &k)) {
                        morpho_globals(v);
                        break;
                    }
                    
                    if (k>=0 && k<v->globals.count) {
                        printf("global %u:", k);
                        morpho_printvalue(v->globals.data[k]);
                        printf("\n");
                    } else printf("Invalid global number.\n");
                }
                    break;
                case 'I': case 'i': // Info
                    break;
                case 'L': case 'l': // List source
                    debug_list(v);
                    break;
                case 'N': case 'n': // Next
                    printf("Not implemented...\n");
                    break;
                case 'P': case 'p': { // Print
                    varray_char symbol;
                    varray_charinit(&symbol);
                    
                    if (debug_parsesymbol(input+1, &symbol)) {
                        debug_printsymbol(v, symbol.data);
                    } else {
                        debug_showsymbols(v);
                    }
                    
                    varray_charclear(&symbol);
                }
                    break;
                case 'Q': case 'q': // Quit
                    morpho_runtimeerror(v, VM_DBGQUIT);
                    return;
                case 'R': case 'r': // Registers
                    debug_showregisters(v, v->fp);
                    break;
                case 'S': case 's': // Step
                    debugger_setsinglestep(v->debug, true);
                    v->debug->currentline=line;
                    v->debug->currentfunc=func;
                    stop=true;
                    //debug_showstack(v);
                    break;
                case 'T': case 't': // Trace
                    morpho_stacktrace(v);
                    break;
                case 'X': case 'x':
                    if (debug_parseint(input, &k)) {
                        debugger_clearbreakpoint(debug, k);
                        break;
                    }
                    printf("Invalid breakpoint target.\n");
                    break;
                case '=':
                {
                    varray_char symbol;
                    varray_charinit(&symbol);
                    //if (debug_parsesymbol(input+1, &symbol));
                    
                    printf("Not implemented...\n");
                    varray_charclear(&symbol);
                }
                    break;
                case ',':
                    vm_collectgarbage(v);
                    break;
                default:
                    printf("Unrecognized debugger command\n");
            }
        }
    }
    printf("---Resuming----------\n");
    linedit_clear(&edit);
}

/* **********************************************************************
 * Run a program with debugging active
 * ********************************************************************** */

/** Run a program with debugging
 * @param[in] v - the virtual machine to use
 * @param[in] p - program to run
 * @returns true on success, false otherwise */
bool morpho_debug(vm *v, program *p) {
    debugger debug;

    debugger_init(&debug, p);
    v->debug=&debug;
    
    bool success=morpho_run(v, p);
    
    debugger_clear(&debug);
    
    return success;
}
