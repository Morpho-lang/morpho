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

/** Associates a register with a symbol */
void debug_setreg(varray_debugannotation *list, indx reg, value symbol) {
    if (!MORPHO_ISSTRING(symbol)) return;
    value sym = object_clonestring(symbol);
    debugannotation ann = { .type = DEBUG_REGISTER, .content.reg.reg = reg, .content.reg.symbol = sym };
    debug_addannotation(list, &ann);
}

/** Uses information from a syntaxtreenode to associate a sequence of instructions with source */
void debug_addnode(varray_debugannotation *list, syntaxtreenode *node) {
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
void debug_clear(varray_debugannotation *list) {
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

assemblyrule assemblyrules[] ={
    { OP_NOP, "nop", "" },
    { OP_MOV, "mov", "rA, rB" },
    { OP_LCT, "lct", "rA, cX" }, // Custom
    { OP_ADD, "add", "rA, ?B, ?C" },
    { OP_SUB, "sub", "rA, ?B, ?C" },
    { OP_MUL, "mul", "rA, ?B, ?C" },
    { OP_DIV, "div", "rA, ?B, ?C" },
    { OP_POW, "pow", "rA, ?B, ?C" },
    { OP_NOT, "not", "rA, ?B" },
    
    { OP_EQ, "eq ", "rA, ?B, ?C" },
    { OP_NEQ, "neq", "rA, ?B, ?C" },
    { OP_LT, "lt ", "rA, ?B, ?C" },
    { OP_LE, "le ", "rA, ?B, ?C" },
    
    { OP_PRINT, "print", "?B" },
    
    { OP_B, "b", "+" },
    { OP_BIF, "bif", "F rA +" },
    
    { OP_CALL, "call", "rA, B" }, // b literal
    { OP_INVOKE, "invoke", "rA, ?B, C" }, // c literal
    
    { OP_RETURN, "return", "rA" }, // c literal

    { OP_CALL, "closure", "rA, pB" }, // b prototype
    
    { OP_LUP, "lup", "rA, uB" }, // b 'u'
    { OP_SUP, "sup", "uA, ?B" }, // a 'u', b c|r
    
    { OP_CLOSEUP, "closeup", "rA" },
    { OP_LPR, "lpr", "rA, ?B, ?C" },
    { OP_SPR, "spr", "rA, ?B, ?C" },
    { OP_LIX, "lix", "rA, ?B, ?C" },
    { OP_SIX, "six", "rA, ?B, ?C" },
    
    { OP_LGL, "lgl", "rA, gX" }, //
    { OP_SGL, "sgl", "rA, gX" }, // label b with 'g'
    
    { OP_ARRAY, "array", "rA, ?B, ?C" },
    { OP_CAT, "cat", "rA, ?B, ?C" },
    { OP_RAISE, "raise", "rA" },
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
                case 'F': n+=printf("%s", (DECODE_F(instruction) ? "t" : "f")); break;
                case '?': {
                    bool isconst=(c[1]=='B' ? DECODE_ISBCONSTANT(instruction) : DECODE_ISCCONSTANT(instruction)); // Is the next letter B or C?
                    mode=( isconst ? CONST : REG );
                    n+=printf("%s", (isconst ? "c" : "r"));
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

/** Disassembles a program
 *  @param code - program to disassemble
 *  @param matchline - optional line number to match */
void debug_disassemble(program *code, int *matchline) {
    instructionindx entry = program_getentry(code); // The entry point of the function
    instructionindx i=0;
    value *konst=(code->global ? code->global->konst.data : NULL);
    bool silent = matchline;
    
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
            default:
                break;
        }
    }
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

/* **********************************************************************
 * Stack traces
 * ********************************************************************** */

/** Prints a stacktrace */
void morpho_stacktrace(vm *v) {
    for (callframe *f = v->fp; f!=NULL && f>=v->frame; f--) {
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
void debug_showsymbol(vm *v, char *match) {
    //objectstring string = MORPHO_STATICSTRING((match ? match : ""));
    
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
    for (unsigned int i=0; i<v->stack.count; i++) {
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

/** Return the next instruction index, i.e. the one about to be executed. */
instructionindx debug_nextinstruction(vm *v) {
    return v->fp->pc-v->current->code.data;
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

/** Morpho debugger */
void debugger(vm *v) {
    lineditor edit;
    
    int line=0;
    objectfunction *func=NULL;
    debug_infofromindx(v->current, debug_nextinstruction(v), &line, NULL, &func, NULL);
    
    linedit_init(&edit);
    linedit_setprompt(&edit, "@>");
    printf("---morpho debugger---\n");
    printf("Type '?' for help.\n");
    printf("Breakpoint in %s", ((!func) || MORPHO_ISNIL(func->name)? "global" : MORPHO_GETCSTRING(func->name)) );
    if (line!=ERROR_POSNUNIDENTIFIABLE) printf(" at line %u", line);
    printf("\n");
    
    for (bool stop=false; !stop; ) {
        char *input = linedit(&edit);
        int k;
        
        if (input) {
            switch (input[0]) {
                case '?': // Help
                    printf("Available commands:\n");
                    printf("  [a]ddress, [c]ontinue, [d]isassemble, [g]lobal\n");
                    printf("  [l]ist, [q]uit, [r]egisters, [s]tack, [t]race\n");
                    printf("  [.]show symbol, [=]set, [,]garbage collect\n");
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
                case 'C': case 'c': // Continue
                    stop=true;
                    break;
                case 'D': case 'd': // Disassemble
                    morpho_disassemble(v->current, &line);
                    break;
                case 'G': case 'g': // Go
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
                case 'L': case 'l': // List source
                    debug_list(v);
                    break;
                case 'Q': case 'q': // Quit
                    morpho_runtimeerror(v, VM_DBGQUIT);
                    return;
                case 'R': case 'r': // Registers
                    debug_showregisters(v, v->fp);
                    break;
                case 'S': case 's': // Stack
                    debug_showstack(v);
                    break;
                case 'T': case 't': // Trace
                    morpho_stacktrace(v);
                    break;
                case '.':
                    debug_showsymbol(v, NULL);
                    break;
                case '=':
                    printf("Not implemented...\n");
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
