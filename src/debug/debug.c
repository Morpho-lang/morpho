/** @file debug.c
 *  @author T J Atherton
 *
 *  @brief Debugging, dissassembly and other tools
 */

#include <stdarg.h>
#include <string.h>
#include <ctype.h>

#include "compile.h"
#include "vm.h"
#include "debug.h"
#include "morpho.h"
#include "strng.h"

#include "debugannotation.h"

void morpho_runtimeerror(vm *v, errorid id, ...);

/* **********************************************************************
 * Extract debugging information using annotations
 * ********************************************************************** */

/** Finds debugging info asssociated with instruction at indx */
bool debug_infofromindx(program *code, instructionindx indx, value *module, int *line, int *posn, objectfunction **func, objectclass **klass) {
    if (module) *module=MORPHO_NIL;
    if (func) *func=code->global;
    if (klass) *klass=NULL;
    instructionindx i=0;
    
    for (unsigned int j=0; j<code->annotations.count; j++) {
        debugannotation *ann = &code->annotations.data[j];
        switch (ann->type) {
            case DEBUG_ELEMENT: {
                if (i+ann->content.element.ninstr>indx) {
                    if (line) *line = ann->content.element.line;
                    if (posn) *posn = ann->content.element.posn;
                    return true;
                }
                i+=ann->content.element.ninstr;
            }
                break;
            case DEBUG_FUNCTION: if (func) *func=ann->content.function.function; break;
            case DEBUG_CLASS: if (klass) *klass=ann->content.klass.klass; break;
            case DEBUG_MODULE: if (module) *module=ann->content.module.module; break;
            default: break;
        }
    }
    
    return false;
}

/** Finds the instruction indx corresponding to a particular line of code */
bool debug_indxfromline(program *code, int line, instructionindx *out) {
    instructionindx i=0;
    value module=MORPHO_NIL;
    
    for (unsigned int j=0; j<code->annotations.count; j++) {
        debugannotation *ann = &code->annotations.data[j];
        switch (ann->type) {
            case DEBUG_ELEMENT:
                if (ann->content.element.line==line) {
                    *out=i;
                    return true;
                }
                i+=ann->content.element.ninstr;
                break;
            case DEBUG_MODULE:
                module=ann->content.module.module;
                break;
            default: break;
        }
    }
    return false;
}

/** Finds the instruction index corresponding to the entry point of a function or method */
bool debug_indxfromfunction(program *code, value klassname, value fname, instructionindx *indx) {
    objectclass *cklass=NULL;
    objectfunction *cfunc=NULL;
    instructionindx i=0;
    
    for (unsigned int j=0; j<code->annotations.count; j++) {
        debugannotation *ann = &code->annotations.data[j];
        switch (ann->type) {
            case DEBUG_ELEMENT:
                i+=ann->content.element.ninstr;
                break;
            case DEBUG_FUNCTION:
                cfunc=ann->content.function.function;
                if (MORPHO_ISEQUAL(cfunc->name, fname) &&
                    (MORPHO_ISNIL(klassname) || MORPHO_ISEQUAL(cklass->name, klassname))) {
                    *indx=cfunc->entry;
                    return true;
                }
                break;
            case DEBUG_CLASS:
                cklass=ann->content.klass.klass;
                break;
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
    for (callframe *f = (v->errfp ? v->errfp : v->fp); f!=NULL && f>=v->frame; f--) {
        instructionindx indx = f->pc-v->current->code.data;
        if (indx>0) indx--; /* Because the pc always points to the NEXT instr. */
        
        morpho_printf(v, "  ");
        morpho_printf(v, "%s", (f==v->fp ? "  in " : "from "));
        
        if (!MORPHO_ISNIL(f->function->name)) morpho_printvalue(v, f->function->name);
        else morpho_printf(v, "global");
        
        int line=0;
        if (debug_infofromindx(v->current, indx, NULL, &line, NULL, NULL, NULL)) {
            morpho_printf(v, " at line %u", line);
        }
        
        morpho_printf(v, "\n");
    }
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
    morpho_printvalue(NULL, table[i]);
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
                morpho_printvalue(NULL, label);
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
                        morpho_printvalue(NULL, func->name);
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
                        morpho_printvalue(NULL, klass->name);
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
 * Debugger data structure
 * ********************************************************************** */

/** Initializes a debugger structure with a specified program */
void debugger_init(debugger *d, program *p) {
    d->singlestep=false;
    
    d->nbreakpoints=0;
    
    d->currentfunc=NULL;
    d->currentline=0;
    d->currentmodule=MORPHO_NIL;
    
    varray_charinit(&d->breakpoints);
    
    int ninstructions = p->code.count;
    if (!varray_charresize(&d->breakpoints, ninstructions)) return;
    memset(d->breakpoints.data, '\0', sizeof(char)*ninstructions);
    d->breakpoints.count=ninstructions;
}

/** Clears a debugger structure */
void debugger_clear(debugger *d) {
    varray_charclear(&d->breakpoints);
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
    d->nbreakpoints++;
}

/** Clears a breakpoint */
void debugger_clearbreakpoint(debugger *d, instructionindx indx) {
    if (indx>d->breakpoints.count) return;
    d->breakpoints.data[indx]='\0';
    d->nbreakpoints--;
}

/** Tests if we should break at a given point */
bool debugger_shouldbreakat(debugger *d, instructionindx indx) {
    if (indx>d->breakpoints.count) return false;
    return (d->breakpoints.data[indx]!='\0');
}

/** Tests if the debugger is in a mode that could cause breaks at arbitrary instructions */
bool debugger_isactive(debugger *d) {
    return (d->singlestep || (d->nbreakpoints>0));
}

/* **********************************************************************
 * Debugger info commands
 * ********************************************************************** */

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
