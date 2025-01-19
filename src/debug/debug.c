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
#include "gc.h"
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
bool debug_indxfromline(program *code, value file, int line, instructionindx *out) {
    instructionindx i=0;
    value module=MORPHO_NIL;
    
    for (unsigned int j=0; j<code->annotations.count; j++) {
        debugannotation *ann = &code->annotations.data[j];
        switch (ann->type) {
            case DEBUG_ELEMENT:
                if (MORPHO_ISEQUAL(file, module) &&
                    ann->content.element.line==line) {
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
    
    for (unsigned int j=0; j<code->annotations.count; j++) {
        debugannotation *ann = &code->annotations.data[j];
        switch (ann->type) {
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

/** Attempts to find a symbol.
 * @param[in] v - virtual machine to search
 * @param[in] matchstr - string to match
 * @param[out] frame - callframe matched
 * @param[out] symbol - actual symbol found
 * @param[out] val - value of the found symbol
 * @returns true on success */
bool debug_findsymbol(vm *v, value matchstr, callframe **frame, value *symbol, value **val) {
    /* Check back through callframes */
    for (callframe *f=v->fp; f>=v->frame; f--) {
        value symbols[f->function->nregs];
        instructionindx indx = f->pc-v->current->code.data;
        
        debug_symbolsforfunction(v->current, f->function, &indx, symbols);
        
        for (int i=0; i<f->function->nregs; i++) {
            if (!MORPHO_ISNIL(symbols[i]) && MORPHO_ISEQUAL(symbols[i], matchstr)) {
                if (frame) *frame = f;
                if (symbol) *symbol = symbols[i];
                if (val) *val = &v->stack.data[f->roffset+i];
                return true;
            }
        }
    }
    
    /* Otherwise is it a global? */
    for (unsigned int j=0; j<v->current->annotations.count; j++) {
        debugannotation *ann = &v->current->annotations.data[j];
        if (ann->type==DEBUG_GLOBAL &&
            MORPHO_ISEQUAL(ann->content.global.symbol, matchstr)) {
            if (frame) *frame = v->frame;
            if (symbol) *symbol = ann->content.global.symbol;
            if (val) *val = &v->globals.data[ann->content.global.gindx];
            return true;
        }
    }
    return false;
}

/** Identifies a symbol for a given global number
 * @param[in] p - program to search
 * @param[in] id - global number to find
 * @param[out] frame - callframe matched*/
bool debug_symbolforglobal(program *p, indx id, value *symbol) {
    for (unsigned int j=0; j<p->annotations.count; j++) {
        debugannotation *ann = &p->annotations.data[j];
        if (ann->type==DEBUG_GLOBAL &&
            ann->content.global.gindx==id) {
            *symbol = ann->content.global.symbol;
            return true;
        }
    }
    return false;
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
        
        value module = MORPHO_NIL;
        if (debug_infofromindx(v->current, indx, &module, &line, NULL, NULL, NULL)) {
            morpho_printf(v, " at line %u", line);
            
            if (!MORPHO_ISNIL(module)) {
                morpho_printf(v, " in module '");
                morpho_printvalue(v, module);
                morpho_printf(v, "'");
            }
        }
        
        morpho_printf(v, "\n");
    }
}

/* **********************************************************************
 * Debugger data structure
 * ********************************************************************** */

/** Initializes a debugger structure with a specified program */
void debugger_init(debugger *d, program *p) {
    d->singlestep=false;
    
    d->nbreakpoints=0;
    
    d->err=NULL;
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

/** Returns the current VM */
vm *debugger_currentvm(debugger *d) {
    return d->currentvm;
}

/** Returns the current program */
program *debugger_currentprogram(debugger *d) {
    return d->currentvm->current;
}

/** Sets the error structure that the debugger will report errors to */
void debugger_seterror(debugger *d, error *err) {
    d->err=err;
}

/** @brief Raises a debugger error
 * @param debug        the debugger
 * @param id       error id
 * @param ...      additional data for sprintf. */
void debugger_error(debugger *debug, errorid id, ... ) {
    if (!debug->err || debug->err->id!=ERROR_NONE) return; // Ensure errors are not overwritten.
    va_list args;
    va_start(args, id);
    morpho_writeerrorwithidvalist(debug->err, id, NULL, ERROR_POSNUNIDENTIFIABLE, ERROR_POSNUNIDENTIFIABLE, args);
    va_end(args);
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
bool debugger_setbreakpoint(debugger *d, instructionindx indx) {
    if (indx>d->breakpoints.count) return false;
    d->breakpoints.data[indx]='b';
    d->nbreakpoints++;
    return true;
}

/** Clears a breakpoint */
bool debugger_clearbreakpoint(debugger *d, instructionindx indx) {
    if (indx>d->breakpoints.count) return false;
    d->breakpoints.data[indx]='\0';
    d->nbreakpoints--;
    return true;
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
 * Disassembler
 * ********************************************************************** */

/** Formatting rules for disassembler */
typedef struct {
    instruction op; /** Opcode */
    char *label; /** Label to display in disasembler */
    char *display; /** Display code - rX is register, cX is constant X, gX is global X, uX is upvalue, + refers to signed B */
} assemblyrule;

/** Define disassembler by how to display each opcode */
assemblyrule assemblyrules[] ={
    { OP_NOP, "nop", "" },
    { OP_MOV, "mov", "rA, rB" },
    { OP_LCT, "lct", "rA, cX" },
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
    
    { OP_CALL, "call", "rA, B, C" }, // b, c literal
    { OP_INVOKE, "invoke", "rA, B, C" }, // b, c literal
    { OP_METHOD, "method", "rA, B, C" }, // b, c literal
    
    { OP_RETURN, "return", "?rB" }, // Return register B is A nonzero

    { OP_CLOSURE, "closure", "rA, pB" }, // b prototype
    
    { OP_LUP, "lup", "rA, uB" }, // b 'u'
    { OP_SUP, "sup", "uA, rB" }, // a 'u', b c|r
    
    { OP_CLOSEUP, "closeup", "rA" },
    { OP_LPR, "lpr", "rA, rB, rC" },
    { OP_SPR, "spr", "rA, rB, rC" },
    
    { OP_LIX,  "lix", "rA, rB, rC" },
    { OP_LIXL, "lixl", "rA, rB, rC" },
    
    { OP_SIX,  "six", "rA, rB, rC" },
    
    { OP_LGL, "lgl", "rA, gX" }, //
    { OP_SGL, "sgl", "rA, gX" }, // label b with 'g'
    
    { OP_PUSHERR, "pusherr", "cX" },
    { OP_POPERR, "poperr", "+" },
    
    { OP_CAT, "cat", "rA, rB, rC" },
    { OP_BREAK, "break", "" },
    { OP_END, "end", "" },
    { 0, NULL, "" } // Null terminate the list
};

assemblyrule *debugger_getassemblyrule(unsigned int op) {
    for (unsigned int i=0; assemblyrules[i].label!=NULL; i++) if (assemblyrules[i].op==op) return &assemblyrules[i];
    return NULL;
}

typedef enum { DBG_CONTENTS_NONE, DBG_CONTENTS_REG, DBG_CONTENTS_CONST} debugcontents;

/** Shows the contents of a register or constant */
bool debugger_showcontents(vm *v, debugcontents b, int i, value *konst, value *reg) {
    value *table = NULL;
    switch (b) {
        case DBG_CONTENTS_CONST: table=konst; break;
        case DBG_CONTENTS_REG: table = reg; break;
        default: break;
    }
    if (!table) return false;
    morpho_printf(v, "%s%i=", (b==DBG_CONTENTS_CONST ? "c" : "r"), i);
    morpho_printvalue(v, table[i]);
    return true;
}

/** @brief Disassembles a single instruction, writing the output to the console.
 *  @param v VM to use for output
 *  @param instruction The instruction to disassemble
 *  @param indx        Instruction index to display
 *  @param konst current constant table
 *  @param reg   current registers */
void debugger_disassembleinstruction(vm *v, instruction instruction, instructionindx indx, value *konst, value *reg) {
    unsigned int op = DECODE_OP(instruction);
    debugcontents mode=DBG_CONTENTS_NONE, bm=DBG_CONTENTS_NONE, cm=DBG_CONTENTS_NONE;
    int nb=0, nc=0;
    morpho_printf(v, "%4lu : ", indx);
    int n=0; // Number of characters displayed
    int width=25; // Width of display
    
    assemblyrule *show=debugger_getassemblyrule(op);
    if (show) {
        n+=morpho_printf(v, "%s ", show->label);
        for (char *c=show->display; *c!='\0'; c++) {
            switch (*c) {
                case 'A': n+=morpho_printf(v, "%u", DECODE_A(instruction)); break;
                case 'B': {
                    bm=mode; nb=DECODE_B(instruction); mode=DBG_CONTENTS_NONE;
                    n+=morpho_printf(v, "%u", nb);
                }
                    break;
                case 'X': {
                    bm=mode; nb=DECODE_Bx(instruction); mode=DBG_CONTENTS_NONE;
                    n+=morpho_printf(v, "%u", nb);
                }
                    break;
                case '+': n+=morpho_printf(v, "%i", DECODE_sBx(instruction)); break;
                case 'C': {
                    cm=mode; nc=DECODE_C(instruction);
                    n+=morpho_printf(v, "%u", DECODE_C(instruction));
                }
                    break;
                case 'c': mode=DBG_CONTENTS_CONST; n+=morpho_printf(v, "%c", *c); break;
                case 'r': mode=DBG_CONTENTS_REG; n+=morpho_printf(v, "%c", *c); break;
                case '?':
                    if (DECODE_A(instruction)==0) return;
                    break;
                default: n+=morpho_printf(v, "%c", *c); break;
            }
        }
        
        /* Show contents if any were produced by this instruction */
        if ((!konst && !reg) || (bm==DBG_CONTENTS_NONE && cm==DBG_CONTENTS_NONE)) return;
        for (int k=width-n; k>0; k--) morpho_printf(v, " ");
        morpho_printf(v, "; ");
        if (debugger_showcontents(v, bm, nb, konst, reg)) morpho_printf(v, " ");
        debugger_showcontents(v, cm, nc, konst, reg);
    }
}

/** Checks if an instruction matches a label in the current error dictionary, and if so print it. */
void debugger_errorlabel(vm *v, varray_value *errorstack, instructionindx i) {
    objectdictionary *dict = MORPHO_GETDICTIONARY(errorstack->data[errorstack->count-1]);
    
    /* Search the current error handler to see if this line corresponds to a label */
    for (unsigned int k=0; k<dict->dict.capacity; k++) {
        value label = dict->dict.contents[k].key;
        if (!MORPHO_ISNIL(label)) {
            if (MORPHO_GETINTEGERVALUE(dict->dict.contents[k].val)==i) {
                morpho_printvalue(v, label);
                morpho_printf(v, ":\n");
            }
        }
    }
}

/** Disassembles a program
 *  @param v - vm to use for output
 *  @param code - program to disassemble
 *  @param matchline - optional line number to match */
void debugger_disassemble(vm *v, program *code, int *matchline) {
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
                        debugger_errorlabel(v, &errorstack, i);
                    }
                    
                    for (unsigned int k=0; k<ann->content.element.ninstr; k++, i++) {
                        morpho_printf(v, "%s", (i==entry ? "->" : "  "));
                        debugger_disassembleinstruction(v, code->code.data[i], i, konst, NULL);
                        morpho_printf(v, "\n");
                    }
                }
                break;
            case DEBUG_FUNCTION:
                {
                    objectfunction *func=ann->content.function.function;
                    konst=func->konst.data;
                    if (silent) break;
                    if (!MORPHO_ISNIL(func->name)) {
                        morpho_printf(v, "fn ");
                        morpho_printvalue(v, func->name);
                        morpho_printf(v, ":\n");
                    } else morpho_printf(v, "\n");
                }
                break;
            case DEBUG_CLASS:
                {
                    objectclass *klass=ann->content.klass.klass;
                    if (silent) break;
                    if (klass && !MORPHO_ISNIL(klass->name)) {
                        morpho_printf(v, "class ");
                        morpho_printvalue(v, klass->name);
                        morpho_printf(v, ":\n");
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

/** Public interface to disassembler */
void morpho_disassemble(vm *v, program *code, int *matchline) {
    debugger_disassemble(NULL, code, matchline);
}

/* **********************************************************************
 * General debugger commands
 * ********************************************************************** */

void debugger_garbagecollect(debugger *debug) {
    size_t init = debug->currentvm->bound;
    vm_collectgarbage(debug->currentvm);
    morpho_printf(NULL, "Collected %ld bytes (from %zu to %zu). Next collection at %zu bytes.\n", init-debug->currentvm->bound, init, debug->currentvm->bound, debug->currentvm->nextgc);
}

void debugger_quit(debugger *debug) {
    morpho_runtimeerror(debug->currentvm, VM_DBGQUIT);
}

/* **********************************************************************
 * Breakpoints
 * ********************************************************************** */

bool debugger_breakatinstruction(debugger *debug, bool set, instructionindx indx) {
    bool success=false;
    if (set) success=debugger_setbreakpoint(debug, indx);
    else success=debugger_clearbreakpoint(debug, indx);
    
    if (!success) debugger_error(debug, DEBUGGER_INVLDINSTR);
    
    return success;
}

/** Break at a particular line */
bool debugger_breakatline(debugger *debug, bool set, value file, int line) {
    instructionindx indx;
    return (debug_indxfromline(debugger_currentprogram(debug), file, line, &indx) &&
                  debugger_breakatinstruction(debug, set, indx));
}

/** Break at a function or method */
bool debugger_breakatfunction(debugger *debug, bool set, value klass, value function) {
    instructionindx indx;
    return (debug_indxfromfunction(debugger_currentprogram(debug), klass, function, &indx) &&
                  debugger_breakatinstruction(debug, set, indx));
}

/* **********************************************************************
 * Show commands
 * ********************************************************************** */

/** Prints the location information for a given instruction */
void debugger_showlocation(debugger *debug, instructionindx indx) {
    vm *v = debugger_currentvm(debug);
    
    value module=MORPHO_NIL; // Find location information
    int line=0;
    objectfunction *fn=NULL;
    objectclass *klass=NULL;
    debug_infofromindx(debugger_currentprogram(debug), indx, &module, &line, NULL, &fn, &klass);
    
    morpho_printf(v, "in ");
    
    if (klass) {
        morpho_printvalue(v, klass->name);
        morpho_printf(v, ".");
    }
    
    if (!MORPHO_ISNIL(fn->name)) morpho_printvalue(v, fn->name);
    else if (v->current->global==fn) morpho_printf(v, "global");
    else morpho_printf(v, "anonymous fn");
    
    if (!MORPHO_ISNIL(module)) {
        morpho_printf(v, " in '");
        morpho_printvalue(v, module);
        morpho_printf(v, "'");
    }
    morpho_printf(v, " at line %i [instruction %ti]", line, indx);
}

/** Shows the address of an object */
bool debugger_showaddress(debugger *debug, indx rindx) {
    vm *v = debugger_currentvm(debug);
    bool success=false;
    if (rindx>=0 && rindx<v->fp->function->nregs) {
        value *reg = v->stack.data + debug->currentvm->fp->roffset;
        if (MORPHO_ISOBJECT(reg[rindx])) {
            morpho_printf(v, "Object in register %i at %p.\n", (int) rindx, (void *) MORPHO_GETOBJECT(reg[rindx]));
            success=true;
        } else debugger_error(debug, DEBUGGER_REGISTEROBJ, (int) rindx);
    } else debugger_error(debug, DEBUGGER_INVLDREGISTER);
    return success;
}

/** Shows active breakpoints */
void debugger_showbreakpoints(debugger *debug) {
    vm *v = debugger_currentvm(debug);
    morpho_printf(v, "Active breakpoints:\n");
    for (instructionindx i=0; i<debug->breakpoints.count; i++) {
        if (debug->breakpoints.data[i]!='\0') {
            morpho_printf(v, "  Breakpoint ");
            debugger_showlocation(debug, i);
            morpho_printf(v, "\n");
        } else if (DECODE_OP(debugger_currentprogram(debug)->code.data[i])==OP_BREAK) {
            morpho_printf(v, "  Break ");
            debugger_showlocation(debug, i);
            morpho_printf(v, "\n");
        }
    }
}

/** List a particular global */
bool debugger_showglobal(debugger *debug, indx id) {
    bool success=false;
    vm *v = debugger_currentvm(debug);
    if (id>=0 && id<v->globals.count) {
        value symbol;
        morpho_printf(v, "  g%lu:", id);
        morpho_printvalue(v, v->globals.data[id]);
        if (debug_symbolforglobal(v->current, id, &symbol)) {
            morpho_printf(v, " (");
            morpho_printvalue(v, symbol);
            morpho_printf(v, ")");
        }
        morpho_printf(v, "\n");
        success=true;
    } else debugger_error(debug, DEBUGGER_INVLDGLOBAL);
    return success;
}

/** Show all globals */
void debugger_showglobals(debugger *debug) {
    vm *v = debugger_currentvm(debug);
    morpho_printf(v, "Globals:\n");
    for (indx i=0; i<v->globals.count; i++) debugger_showglobal(debug, i);
}

/** Show the contents of all registers */
void debugger_showregisters(debugger *debug) {
    vm *v = debugger_currentvm(debug);
    callframe *frame = v->fp;
    
    unsigned int nreg=frame->function->nregs;
    value symbols[nreg];
    instructionindx cinstr=frame->pc-v->current->code.data;
    bool sym = debug_symbolsforfunction(v->current, frame->function, &cinstr, symbols);
    
    morpho_printf(v, "Register contents:\n");
    value *reg = v->stack.data + frame->roffset;
    for (unsigned int i=0; i<nreg; i++) {
        morpho_printf(v, "  r%u: ", i);
        morpho_printvalue(v, reg[i]);
        if (sym && !MORPHO_ISNIL(symbols[i])) {
            morpho_printf(v, " (");
            morpho_printvalue(v, symbols[i]);
            morpho_printf(v, ")");
        }
        morpho_printf(v, "\n");
    }
}

/** Show the contents of the stack */
void debugger_showstack(debugger *debug) {
    vm *v = debugger_currentvm(debug);
    
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
    morpho_printf(v, "Stack contents:\n");
    for (unsigned int i=0; i<v->fp->roffset+v->fp->function->nregs; i++) {
        if (i==fbounds[k]) {
            morpho_printf(v, "---");
            if (f->function) morpho_printvalue(v, f->function->name);
            morpho_printf(v, "\n");
            k++; f++;
        }
        morpho_printf(v, "  s%u: ", i);
        morpho_printvalue(v, v->stack.data[i]);
        morpho_printf(v, "\n");
    }
}

/** Show the current value of a symbol */
bool debugger_showsymbol(debugger *debug, value match) {
    vm *v = debugger_currentvm(debug);
    
    value symbol, *val=NULL;
    if (debug_findsymbol(v, match, NULL, &symbol, &val)) {
        morpho_printvalue(v, symbol);
        morpho_printf(v, " = ");
        morpho_printvalue(v, *val);
        morpho_printf(v, "\n");
    } else debugger_error(debug, DEBUGGER_FINDSYMBOL, MORPHO_GETCSTRING(match));
    
    return val;
}

/** Shows all symbols currently in view */
void debugger_showsymbols(debugger *debug) {
    vm *v = debugger_currentvm(debug);
    
    for (callframe *f=v->fp; f>=v->frame; f--) {
        morpho_printf(v, "in %s", (f==v->frame ? "global" : ""));
        if (!MORPHO_ISNIL(f->function->name)) morpho_printvalue(v, f->function->name);
        morpho_printf(v, ":\n");
        
        value symbols[f->function->nregs];
        instructionindx indx = f->pc-v->current->code.data;
        
        debug_symbolsforfunction(v->current, f->function, &indx, symbols);
        
        for (int i=0; i<f->function->nregs; i++) {
            if (!MORPHO_ISNIL(symbols[i])) {
                morpho_printf(v, "  ");
                morpho_printvalue(v, symbols[i]);
                morpho_printf(v, "=");
                morpho_printvalue(v, v->stack.data[f->roffset+i]);
                morpho_printf(v, "\n");
            }
        }
    }
}

/** Show the current value of a property */
bool debugger_showproperty(debugger *debug, value matchobj, value matchproperty) {
    bool success=false;
    vm *v = debugger_currentvm(debug);
    
    callframe *frame;
    value symbol, *instance=NULL, val;
    
    if (debug_findsymbol(v, matchobj, &frame, &symbol, &instance)) {
        if (MORPHO_ISINSTANCE(*instance)) {
            objectinstance *obj = MORPHO_GETINSTANCE(*instance);
                    
            if (objectinstance_getproperty(obj, matchproperty, &val)) {
                morpho_printvalue(v, symbol);
                morpho_printf(v, ".");
                morpho_printvalue(v, matchproperty);
                morpho_printf(v, " = ");
                morpho_printvalue(v, val);
                morpho_printf(v, "\n");
                success=true;
            } else debugger_error(debug, DEBUGGER_SYMBOLPROP, MORPHO_GETCSTRING(matchproperty));
        }
    }
    
    return success;
}

/* **********************************************************************
 * Set commands
 * ********************************************************************** */

/** Sets the contents of a register to a given value */
bool debugger_setregister(debugger *debug, indx reg, value val) {
    vm *v=debugger_currentvm(debug);
    bool success=false;
    
    if (reg>=0 && reg<v->fp->function->nregs) {
        v->stack.data[v->fp->roffset+reg]=val;
        success=true;
    }
    
    return success;
}

/** Sets a symbol to a given value */
bool debugger_setsymbol(debugger *debug, value symbol, value val) {
    value *dest=NULL;
    
    if (debug_findsymbol(debugger_currentvm(debug), symbol, NULL, NULL, &dest)) {
        *dest=val;
    } else debugger_error(debug, DEBUGGER_FINDSYMBOL, MORPHO_GETCSTRING(symbol));
    
    return (dest!=NULL);
}

/** Sets a property to a given value */
bool debugger_setproperty(debugger *debug, value symbol, value property, value val) {
    value *dest=NULL;
    bool success=false;
    
    if (debug_findsymbol(debugger_currentvm(debug), symbol, NULL, NULL, &dest)) {
        if (MORPHO_ISINSTANCE(*dest)) {
            objectinstance *obj = MORPHO_GETINSTANCE(*dest);
            
            value key = dictionary_intern(&obj->fields, property);
            success=objectinstance_setproperty(obj, key, val);
        } else debugger_error(debug, DEBUGGER_SETPROPERTY);
    } else debugger_error(debug, DEBUGGER_FINDSYMBOL, MORPHO_GETCSTRING(symbol));
    
    return success;
}

/* **********************************************************************
 * Enter the debugger (called by the VM)
 * ********************************************************************** */

/** Enters the debugger, if one is active. */
bool debugger_enter(debugger *debug, vm *v) {
    if (debug && v->debuggerfn) {
        debug->currentvm = v;
        
        // Get instruction index
        debug->iindx = vm_currentinstruction(v);
        
        // Retain previous line and function information
        int oline=debug->currentline;
        objectfunction *ofunc=debug->currentfunc;
        
        // Fetch info from annotations
        debug_infofromindx(debugger_currentprogram(debug), debug->iindx, &debug->currentmodule, &debug->currentline, NULL, &debug->currentfunc, NULL);
        
        // If we're in single step mode, only stop when we've changed line OR if a breakpoint is explicitly set
        if (debugger_insinglestep(debug) &&
            oline==debug->currentline &&
            ofunc==debug->currentfunc &&
            !debugger_shouldbreakat(debug, debug->iindx)) return false;
        
        (v->debuggerfn) (v, v->debuggerref);
    }
    return debug;
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

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

/** Intialize the debugger library */
void debugger_initialize(void) {
    morpho_defineerror(DEBUGGER_FINDSYMBOL, ERROR_DEBUGGER, DEBUGGER_FINDSYMBOL_MSG);
    morpho_defineerror(DEBUGGER_SETPROPERTY, ERROR_DEBUGGER, DEBUGGER_SETPROPERTY_MSG);
    morpho_defineerror(DEBUGGER_INVLDREGISTER, ERROR_DEBUGGER, DEBUGGER_INVLDREGISTER_MSG);
    morpho_defineerror(DEBUGGER_INVLDGLOBAL, ERROR_DEBUGGER, DEBUGGER_INVLDGLOBAL_MSG);
    morpho_defineerror(DEBUGGER_INVLDINSTR, ERROR_DEBUGGER, DEBUGGER_INVLDINSTR_MSG);
    morpho_defineerror(DEBUGGER_REGISTEROBJ, ERROR_DEBUGGER, DEBUGGER_REGISTEROBJ_MSG);
    morpho_defineerror(DEBUGGER_SYMBOLPROP, ERROR_DEBUGGER, DEBUGGER_SYMBOLPROP_MSG);
}
