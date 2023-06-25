/** @file compile.c
 *  @author T J Atherton
 *
 *  @brief Compiles raw input to Morpho instructions
 */

#include <stdarg.h>
#include <string.h>
#include "compile.h"
#include "error.h"
#include "vm.h"
#include "morpho.h"
#include "file.h"
#include "object.h"
#include "veneer.h"
#include "builtin.h"
#include "cmplx.h"
#include "optimize.h"

/** Base class for instances */
static objectclass *baseclass;

/* **********************************************************************
* Bytecode compiler
* ********************************************************************** */

/* ------------------------------------------
 * Utility functions
 * ------------------------------------------- */

static bool compiler_haserror(compiler *c) {
    return (c->err.cat!=ERROR_NONE);
}

/** @brief Fills out the error record
 * @param c        the compiler
 * @param node     the node the error occurred at
 * @param id       error id
 * @param ...      additional data for sprintf. */
static void compiler_error(compiler *c, syntaxtreenode *node, errorid id, ... ) {
    if (c->err.cat!=ERROR_NONE) return; // Ensure errors are not overwritten.
    va_list args;
    int line = (node ? node->line : ERROR_POSNUNIDENTIFIABLE);
    int posn = (node ? node->posn : ERROR_POSNUNIDENTIFIABLE);

    va_start(args, id);
    morpho_writeerrorwithidvalist(&c->err, id, line, posn, args);

    va_end(args);
}

/** Returns true if the compiler has encountered an error */
static bool compiler_checkerror(compiler *c) {
    return (c->err.cat!=ERROR_NONE); // Ensure errors are not overwritten.
}

/** @brief Catches a compiler error, resetting the errror state to none.
 * @param c        the compiler
 * @param id       error id to match
 * @returns true if the error was matched
 */
static bool compiler_catch(compiler *c, errorid id) {
    if (morpho_matcherror(&c->err, id)) {
        error_clear(&c->err);
        return true;
    }
    return false;
}

/** Gets a node given an index */
static inline syntaxtreenode *compiler_getnode(compiler *c, syntaxtreeindx indx) {
    if (indx==SYNTAXTREE_UNCONNECTED) return NULL;
    return c->tree.tree.data+indx;
}

/** Gets a node given an index */
static inline syntaxtree *compiler_getsyntaxtree(compiler *c) {
    return &c->tree;
}

/** Adds an instruction to the current program */
static instructionindx compiler_addinstruction(compiler *c, instruction instr, syntaxtreenode *node) {
    debug_addnode(&c->out->annotations, node);
    return varray_instructionwrite(&c->out->code, instr);
}

/** Gets the instruction index of the current instruction */
static instructionindx compiler_currentinstructionindex(compiler *c) {
    return (instructionindx) c->out->code.count;
}

/** Adds an instruction to the current program */
/*static instruction compiler_previousinstruction(compiler *c) {
    return c->out->code.data[c->out->code.count-1];
}*/

/** Finds the current instruction index */
/*static instructionindx compiler_currentinstruction(compiler *c) {
    return c->out->code.count;
}*/

/** Modifies the instruction at a given index */
static void compiler_setinstruction(compiler *c, instructionindx indx,  instruction instr) {
    c->out->code.data[indx]=instr;
}

/* ------------------------------------------
 * The functionstate stack
 * ------------------------------------------- */

DEFINE_VARRAY(registeralloc, registeralloc);
DEFINE_VARRAY(forwardreference, forwardreference);

/** Initializes a functionstate structure */
static void compiler_functionstateinit(functionstate *state) {
    state->func=NULL;
    state->scopedepth=0;
    state->loopdepth=0;
    state->inargs=false;
    state->nreg=0;
    state->type=FUNCTION;
    state->varg=REGISTER_UNALLOCATED;
    varray_registerallocinit(&state->registers);
    varray_forwardreferenceinit(&state->forwardref);
    varray_upvalueinit(&state->upvalues);
}

/** Clears a functionstate structure */
static void compiler_functionstateclear(functionstate *state) {
    state->func=NULL;
    state->scopedepth=0;
    state->loopdepth=0;
    state->nreg=0;
    varray_registerallocclear(&state->registers);
    varray_forwardreferenceclear(&state->forwardref);
    varray_upvalueclear(&state->upvalues);
}

/** Initializes the function stack */
void compiler_fstackinit(compiler *c) {
    for (unsigned int i=0; i<MORPHO_CALLFRAMESTACKSIZE; i++) {
        compiler_functionstateinit(&c->fstack[i]);
    }
    c->fstackp=0;
}

/** Clears the function stack */
void compiler_fstackclear(compiler *c) {
    for (unsigned int i=0; i<MORPHO_CALLFRAMESTACKSIZE; i++) {
        compiler_functionstateclear(&c->fstack[i]);
    }
    c->fstackp=0;
}

/** Gets the current functionstate structure */
static inline functionstate *compiler_currentfunctionstate(compiler *c) {
    return c->fstack+c->fstackp;
}

/** The parent functionstate */
static inline functionstate *compiler_parentfunctionstate(compiler *c) {
    if (c->fstackp==0) return NULL;
    return c->fstack+c->fstackp-1;
}

/** Detect if we're currently compiling an initializer */
static inline bool compiler_ininitializer(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);

    return FUNCTIONTYPE_ISINITIALIZER(f->type);
}

/* ------------------------------------------
 * Increment and decrement the fstack
 * ------------------------------------------- */

/** Begins a new function, advancing the fstack pointer */
static void compiler_beginfunction(compiler *c, objectfunction *func, functiontype type) {
    c->fstackp++;
    compiler_functionstateinit(&c->fstack[c->fstackp]);
    c->fstack[c->fstackp].func=func;
    c->fstack[c->fstackp].type=type;
    debug_setfunction(&c->out->annotations, func);
}

/** Sets the function register count */
static void compiler_setfunctionregistercount(compiler *c) {
    functionstate *f=&c->fstack[c->fstackp];
    if (f->nreg>f->func->nregs) f->func->nregs=f->nreg;
}

/** Ends a function, decrementing the fstack pointer  */
static void compiler_endfunction(compiler *c) {
    functionstate *f=&c->fstack[c->fstackp];
    c->prevfunction=f->func; /* Retain the function in case it needs to be bound as a method */
    compiler_setfunctionregistercount(c);
    compiler_functionstateclear(f);
    c->fstackp--;
    debug_setfunction(&c->out->annotations, c->fstack[c->fstackp].func);
}

/** Gets the current function */
static objectfunction *compiler_getcurrentfunction(compiler *c) {
    return c->fstack[c->fstackp].func;
}

/** Gets the current constant table */
static varray_value *compiler_getcurrentconstanttable(compiler *c) {
    objectfunction *f = compiler_getcurrentfunction(c);
    if (!f) {
        UNREACHABLE("find current constant table [No current function defined].");
    }

    return &f->konst;
}

/** Gets constant i from the current constant table */
static value compiler_getconstant(compiler *c, unsigned int i) {
    value ret = MORPHO_NIL;
    objectfunction *f = compiler_getcurrentfunction(c);
    if (f && i<f->konst.count) ret = f->konst.data[i];
    return ret;
}

/** Gets the most recently compiled function */
static objectfunction *compiler_getpreviousfunction(compiler *c) {
    return c->prevfunction;
}

/* ------------------------------------------
 * Argument declarations
 * ------------------------------------------- */

/** Begins arguments */
static inline void compiler_beginargs(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);
    if (f) f->inargs=true;
}

/** Ends arguments */
static inline void compiler_endargs(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);
    if (f) f->inargs=false;
}

/** Are we in an args statement? */
static inline bool compiler_inargs(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);
    return f->inargs;
}

/* ------------------------------------------
 * Class declarations
 * ------------------------------------------- */

static inline void compiler_beginclass(compiler *c, objectclass *klass) {
    /* If we're already compiling a class, retain it in a linked list */
    if (c->currentclass) klass->obj.next = (object *) klass;

    c->currentclass=klass;
    debug_setclass(&c->out->annotations, klass);
}

static inline void compiler_endclass(compiler *c) {
    /* Delink current class from list */
    objectclass *current = c->currentclass;
    c->currentclass=(objectclass *) current->obj.next;
    debug_setclass(&c->out->annotations, c->currentclass);
    current->obj.next=NULL; /* as the class is no longer part of the list */
}

/** Gets the current class */
static objectclass *compiler_getcurrentclass(compiler *c) {
    return c->currentclass;
}

/* ------------------------------------------
 * Modules
 * ------------------------------------------- */

/** Sets the current module */
static void compiler_setmodule(compiler *c, value module) {
    c->currentmodule=module;
}

/** Gets the current module */
static value compiler_getmodule(compiler *c) {
    return c->currentmodule;
}

/* ------------------------------------------
 * Loops
 * ------------------------------------------- */

/*
 9 : bif r3 f 6
10 : mov r3, r1
11 : mov r4, r0
12 : invoke r3, c4, 1            ;  c4=enumerate
13 : b  2
14 : add r0, r0, c2            ;  c2=1
15 : b  -8
16 : end
 */

/** Checks through a loop, updating any placeholders for break or continue
 * @param[in] c the current compiler
 * @param[in] start first instruction in the loop body
 * @param[in] inc position of the loop increment section (continue statements redirect here)
 * @param[in] end position of the first instruction AFTER the loop (break sections redirect here) */
static void compiler_fixloop(compiler *c, instructionindx start, instructionindx inc, instructionindx end) {
    instruction *code=c->out->code.data;
    for (instructionindx i=start; i<end; i++) {
        if (DECODE_OP(code[i])==OP_NOP) {
            if (DECODE_A(code[i])=='b') {
                code[i]=ENCODE_LONG(OP_B, REGISTER_UNALLOCATED, end-i-1);
            } else if (DECODE_A(code[i])=='c') {
                code[i]=ENCODE_LONG(OP_B, REGISTER_UNALLOCATED, inc-i-1);
            }
        }
    }
}

/** Begin a loop */
static void compiler_beginloop(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);
    f->loopdepth++;
}

/** End a loop */
static void compiler_endloop(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);
    f->loopdepth--;
}

/** Check if we are in a loop */
static bool compiler_inloop(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);
    return (f->loopdepth>0);
}

/* ------------------------------------------
 * Register allocation and deallocation
 * ------------------------------------------- */

/** Finds a free register in the current function state and claims it */
static registerindx compiler_regallocwithstate(compiler *c, functionstate *f, value symbol) {
    registeralloc r = (registeralloc) {.isallocated=true, .iscaptured=false, .scopedepth=f->scopedepth, .symbol=symbol};
    registerindx i = REGISTER_UNALLOCATED;

    if (compiler_inargs(c)) {
        /* Search backwards from the end to find the register AFTER
           the last allocated register */
        for (i=f->registers.count; i>0; i--) {
            if (f->registers.data[i-1].isallocated) break;
        }
        if (i<f->registers.count) {
            f->registers.data[i]=r;
            goto regalloc_end;
        }
    } else {
        /* Search forwards to find any unallocated register */
        for (i=0; i<f->registers.count; i++) {
            if (!f->registers.data[i].isallocated) {
                f->registers.data[i]=r;
                goto regalloc_end;
            }
        }
    }

    /* No unallocated register was found, so allocate one at the end */
    if (varray_registerallocadd(&f->registers, &r, 1)) {
        i = f->registers.count-1;
        if (f->registers.count>f->nreg) f->nreg=f->registers.count;
    }

regalloc_end:
    if (!MORPHO_ISNIL(symbol)) debug_setreg(&c->out->annotations, i, symbol);

    return i;
}

static registerindx compiler_regalloc(compiler *c, value symbol) {
    functionstate *f = compiler_currentfunctionstate(c);

    return compiler_regallocwithstate(c, f, symbol);
}

/** Sets the symbol associated with a register */
static void compiler_regsetsymbol(compiler *c, registerindx reg, value symbol) {
    functionstate *f = compiler_currentfunctionstate(c);
    if (reg<f->registers.count && f->registers.data[reg].isallocated) {
        f->registers.data[reg].symbol=symbol;
        if (!MORPHO_ISNIL(symbol)) debug_setreg(&c->out->annotations, reg, symbol);
    }
}

/** Allocates a temporary register that is guaranteed to be at the top of the stack */
static registerindx compiler_regalloctop(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);
    registerindx i = REGISTER_UNALLOCATED;
    registeralloc r = (registeralloc) {.isallocated=true, .iscaptured=false, .scopedepth=f->scopedepth, .symbol=MORPHO_NIL};

    /* Search backwards from the end to find the register AFTER
       the last allocated register */
    for (i=f->registers.count; i>0; i--) {
        if (f->registers.data[i-1].isallocated) break;
    }
    if (i<f->registers.count) {
        f->registers.data[i]=r;
        return i;
    }

    if (varray_registerallocadd(&f->registers, &r, 1)) {
        i = f->registers.count-1;
        if (f->registers.count>f->nreg) f->nreg=f->registers.count;
    }

    return i;
}

/** Claims a free register if reqreg is REGISTER_UNALLOCATED */
static registerindx compiler_regtemp(compiler *c, registerindx reqreg) {
    return (reqreg==REGISTER_UNALLOCATED ? compiler_regalloc(c, MORPHO_NIL) : reqreg);
}

/** Treis to allocate a specific register as a temporary register */
static registerindx compiler_regtempwithindx(compiler *c, registerindx reg) {
    functionstate *f = compiler_currentfunctionstate(c);
    if (reg>=f->registers.count) {
        registeralloc empty = (registeralloc) {.isallocated=false, .iscaptured=false, .scopedepth=f->scopedepth, .symbol=MORPHO_NIL};
        while (reg>=f->registers.count) {
            if (!varray_registerallocadd(&f->registers, &empty, 1)) break;
            if (f->registers.count>f->nreg) f->nreg=f->registers.count;
        }
    }

    if (reg<f->registers.count) {
        f->registers.data[reg].isallocated=true;
    }

    return reg;
}

/** Releases a register that has been previously claimed */
static void compiler_regfree(compiler *c, functionstate *f, registerindx reg) {
    if (reg<f->registers.count) {
        f->registers.data[reg].isallocated=false;
        f->registers.data[reg].scopedepth=0;
        if (!MORPHO_ISNIL(f->registers.data[reg].symbol)) {
            debug_setreg(&c->out->annotations, reg, MORPHO_NIL);
        }
        f->registers.data[reg].symbol=MORPHO_NIL;
    }
}

/** Frees a register if it is not a local */
static void compiler_regfreetemp(compiler *c, registerindx reg) {
    functionstate *f = compiler_currentfunctionstate(c);
    if (reg!=REGISTER_UNALLOCATED &&
        reg<f->registers.count &&
        f->registers.data[reg].isallocated &&
        MORPHO_ISNIL(f->registers.data[reg].symbol)) {
        compiler_regfree(c, f, reg);
    }
}

/** Frees all registers beyond and including reg  */
static void compiler_regfreetoend(compiler *c, registerindx reg) {
    functionstate *f = compiler_currentfunctionstate(c);
    if (f) for (registerindx i=reg; i<f->registers.count; i++) {
        compiler_regfree(c, f, i);
    }
}

/** @brief Releases an operand.
 *  @detail A node should call this for each operand it uses the result of.  */
static void compiler_releaseoperand(compiler *c, codeinfo info) {
    if (CODEINFO_ISREGISTER(info)) {
        compiler_regfreetemp(c, info.dest);
    }
}

/** Releases all registers at a given scope depth */
static void compiler_regfreeatscope(compiler *c, unsigned int scopedepth) {
    functionstate *f = compiler_currentfunctionstate(c);
    for (registerindx i=0; i<f->registers.count; i++) {
        if (f->registers.data[i].isallocated &&
            f->registers.data[i].scopedepth>=scopedepth) {
            compiler_regfree(c, f, i);
        }
    }
}

/** @brief Finds the register that contains symbol in a given functionstate
 *  @details Searches backwards so that the innermost scope has priority */
static registerindx compiler_findsymbol(functionstate *f, value symbol) {
    if (f) for (registerindx i=f->registers.count-1; i>=0; i--) {
        if (f->registers.data[i].isallocated) {
            if (MORPHO_ISEQUAL(f->registers.data[i].symbol, symbol)) {
                return i;
            }
        }
    }

    return REGISTER_UNALLOCATED;
}

/** @brief Finds the register that contains symbol in a given functionstate with a scopedepth of scopedepth or higher
 *  @details Searches backwards so that the innermost scope has priority */
static registerindx compiler_findsymbolwithscope(functionstate *f, value symbol, unsigned int scopedepth) {
    if (f) for (registerindx i=f->registers.count-1; i>=0; i--) {
        if (f->registers.data[i].scopedepth<scopedepth) break;
        if (f->registers.data[i].isallocated) {
            if (MORPHO_ISEQUAL(f->registers.data[i].symbol, symbol)) {
                return i;
            }
        }
    }

    return REGISTER_UNALLOCATED;
}

/** @brief Find the last allocated register
 *  @returns Index of the last allocated register, or REGISTER_UNALLOCATED if there are no registers allocated */
static registerindx compiler_regtop(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);
    for (registerindx i=f->registers.count-1; i>=0; i--) {
        if (f->registers.data[i].isallocated) return i;
    }
    return REGISTER_UNALLOCATED;
}

/** @brief Check if a register is temporary or associated with a symbol
 *  @returns True if the register is temporary, false otherwise */
static bool compiler_isregtemp(compiler *c, registerindx indx) {
    functionstate *f = compiler_currentfunctionstate(c);

    return (indx<f->registers.count && MORPHO_ISNIL(f->registers.data[indx].symbol));
}

/** @brief Check if a register is allocated
 *  @returns True if the register is allocated, false otherwise */
static bool compiler_isregalloc(compiler *c, registerindx indx) {
    functionstate *f = compiler_currentfunctionstate(c);

    return (indx<f->registers.count && f->registers.data[indx].isallocated);
}

/** @brief Get the scope level associated with a register
 *  @param[in] c        compiler
 *  @param[in] indx register to examine
 *  @param[out] scope the scope
 *  @returns True if the requested register has a meaningful scope */
static bool compiler_getregscope(compiler *c, registerindx indx, unsigned int *scope) {
    functionstate *f = compiler_currentfunctionstate(c);

    if (indx<f->registers.count && !MORPHO_ISNIL(f->registers.data[indx].symbol)) {
        if (scope) *scope=f->registers.data[indx].scopedepth;
        return true;
    }

    return false;
}

/** @brief Function to tell if a codeinfo block has returned something that is at the top of the stack
 *  @details Calls and invocations, inter alia, rely on things being put in the last available register.
 *           This function checks whether this has been achieved; if not a call to
 *           compiler_movetoregister may be made.
 *  @returns true if the codeinfo meets these requirements is satisfied, false otherwise
 */
static bool compiler_iscodeinfotop(compiler *c, codeinfo func) {
    return ( CODEINFO_ISREGISTER(func) && // We're in a register
              compiler_isregtemp(c, func.dest) && // and it's a temporary register
              func.dest==compiler_regtop(c)); // and it's the top of the stack
}

/** @brief Shows the current allocation of the registers */
static void compiler_regshow(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);

    printf("--Registers (%u in use)\n",f->nreg);
    for (unsigned int i=0; i<f->registers.count; i++) {
        registeralloc *r=f->registers.data+i;
        printf("r%u ",i);
        if (r->isallocated) {
            if (i==0 && FUNCTIONTYPE_ISMETHOD(f->type)) {
                printf("self");
            } else if (!MORPHO_ISNIL(r->symbol)) {
                morpho_printvalue(r->symbol);
            } else {
                printf("temporary");
            }
            printf(" [%u]", r->scopedepth);
            if (r->iscaptured) printf(" (captured)");
        } else {
            printf("unallocated");
        }
        printf("\n");
    }
    printf("--End registers\n");
}

/* ------------------------------------------
 * Track scope
 * ------------------------------------------- */

/** Increments the scope counter in the current functionstate */
void compiler_beginscope(compiler *c) {
    functionstate *f=compiler_currentfunctionstate(c);
    f->scopedepth++;
}

/** Decrements the scope counter in the current functionstate */
void compiler_endscope(compiler *c) {
    functionstate *f=compiler_currentfunctionstate(c);
    compiler_regfreeatscope(c, f->scopedepth);
    f->scopedepth--;
}

/** Gets the scope counter in the current functionstate */
unsigned int compiler_currentscope(compiler *c) {
    functionstate *f=compiler_currentfunctionstate(c);
    return f->scopedepth;
}

/* ------------------------------------------
 * Constants
 * ------------------------------------------- */

/** Writes a constant to the current constant table
 *  @param c        the compiler
 *  @param node     current syntaxtree node
 *  @param constant the constant to add
 *  @param usestrict whether to use a strict e
 *  @param clone    whether to clone the constant if it's not already present
 *                  (typically this is set to copy strings from the syntax tree) */
static registerindx compiler_addconstant(compiler *c, syntaxtreenode *node, value constant, bool usestrict, bool clone) {
    varray_value *konst = compiler_getcurrentconstanttable(c);
    if (!konst) return REGISTER_UNALLOCATED;

    registerindx out=REGISTER_UNALLOCATED;
    unsigned int prev=0;

    if (konst) {
        /* Was a similar previous constant already added? */
        if (usestrict) {
            if (varray_valuefindsame(konst, constant, &prev)) out=(registerindx) prev;
        } else {
            if (varray_valuefind(konst, constant, &prev)) out=(registerindx) prev;
        }

        /* No, so create a new one */
        if (out==REGISTER_UNALLOCATED) {
            if (konst->count>=MORPHO_MAXCONSTANTS) {
                compiler_error(c, node, COMPILE_TOOMANYCONSTANTS);
                return REGISTER_UNALLOCATED;
            } else {
                value add = constant;
                if (clone && MORPHO_ISOBJECT(add)) {
                    /* If clone is set, we should try to clone the contents if the thing is an object. */
                    if (MORPHO_ISSTRING(add)) {
                        add=object_clonestring(add);
                    } else if (MORPHO_ISCOMPLEX(add)) {
                        add=object_clonecomplexvalue(add);
                    } else {
                        UNREACHABLE("Erroneously being asked to clone a non-string non-complex constant.");
                    }
                }

                bool success=varray_valueadd(konst, &add, 1);
                out=konst->count-1;
                if (!success) compiler_error(c, node, ERROR_ALLOCATIONFAILED);

                /* If the object is a constant we make sure its bound to the program */
                if (MORPHO_ISOBJECT(constant)) {
                    /* Bind the object to the program */
                    program_bindobject(c->out, MORPHO_GETOBJECT(add));
                }
            }
        }
    }

    return out;
}

/** Write a symbol to the constant table, performing interning.
 * @param c        the compiler
 * @param node     current syntaxtree node
 *  @param symbol the constant to add */
static registerindx compiler_addsymbol(compiler *c, syntaxtreenode *node, value symbol) {
    /* Intern the symbol */
    value add=program_internsymbol(c->out, symbol);

    return compiler_addconstant(c, node, add, true, false);
}

/** Finds a builtin function and loads it into a register at the top of the stack
 *  @param c the compiler
 *  @param node current syntax tree node
 *  @param name the symbol to lookup
 *  @param req requested register
 *  @details This function is oriented to setting up a function call, so req is checked whether it's at the top of the stack. */
codeinfo compiler_findbuiltin(compiler *c, syntaxtreenode *node, char *name, registerindx req) {
    objectstring symbol = MORPHO_STATICSTRING(name);
    codeinfo ret=CODEINFO_EMPTY;
    registerindx rfn=req;

    /* Find the function */
    value fn=builtin_findfunction(MORPHO_OBJECT(&symbol));
    if (MORPHO_ISNIL(fn)) {
        UNREACHABLE("Compiler couldn't locate builtin function.");
    }

    registerindx cfn=compiler_addconstant(c, node, fn, false, false);
    /* Ensure output register is at top of stack */
    if (rfn==REGISTER_UNALLOCATED || rfn<compiler_regtop(c)) {
        rfn=compiler_regalloctop(c);
    }
    compiler_addinstruction(c, ENCODE_LONG(OP_LCT, rfn, cfn), node);

    ret.returntype=REGISTER;
    ret.ninstructions=1;
    ret.dest=rfn;

    return ret;
}

/* ------------------------------------------
 * Local variables
 * ------------------------------------------- */

/** @brief Creates a new local variable
 *  @param c      the current compiler
 *  @param symbol symbol for the variable
 *  @returns an allocated register for the variable */
static registerindx compiler_addlocal(compiler *c, syntaxtreenode *node, value symbol) {
    unsigned int current = compiler_currentscope(c);
    registerindx out=compiler_findsymbolwithscope(compiler_currentfunctionstate(c), symbol, current);
    if (out==REGISTER_UNALLOCATED) {
        out = compiler_regalloc(c, symbol);
    } else {
        unsigned int scope;
        if (compiler_getregscope(c, out, &scope)) {
            compiler_error(c, node, COMPILE_VARALREADYDECLARED);
        } else {
            UNREACHABLE("Local variable allocated incorrectly.");
        }
    }

    return out;
}

/** @brief Gets the register associated with a local variable
 *  @param c      the current compiler
 *  @param symbol symbol for the variable
 *  @returns an allocated register for the variable, or REGISTER_UNALLOCATED if the variable already exists */
static registerindx compiler_getlocal(compiler *c, value symbol) {
    functionstate *f = compiler_currentfunctionstate(c);
    return compiler_findsymbol(f, symbol);
}

/** @brief Determines is a register is a local variable
 *  @param c      the current compiler
 *  @param reg    Register to examine */
/*static bool compiler_islocal(compiler *c, registerindx reg) {
    functionstate *f = compiler_currentfunctionstate(c);

    if (f && reg!=REGISTER_UNALLOCATED) {
        if (f->registers.data[reg].isallocated &&
            !MORPHO_ISNIL(f->registers.data[reg].symbol)) return true;
    }

    return false;
}*/

/** @brief Moves the results of a codeinfo block into a register
 *  @details includes constants, upvalues etc.
 *  @param   c      the current compiler
 *  @param   node   current syntaxtreenode
 *  @param   info   a codeinfo struct
 *  @param   reg    destination register, or REGISTER_UNALLOCATED to allocate a new one
 *  @returns Number of instructions generated */
static codeinfo compiler_movetoregister(compiler *c, syntaxtreenode *node, codeinfo info, registerindx reg) {
    codeinfo out = info;
    out.ninstructions=0;

    if (CODEINFO_ISCONSTANT(info)) {
        out.returntype=REGISTER;
        out.dest=compiler_regtemp(c, reg);
        compiler_addinstruction(c, ENCODE_LONG(OP_LCT, out.dest, info.dest), node);

        out.ninstructions++;
    } else if (CODEINFO_ISUPVALUE(info)) {
        /* Move upvalues */
        out.dest=compiler_regtemp(c, reg);
        out.returntype=REGISTER;
        compiler_addinstruction(c, ENCODE_DOUBLE(OP_LUP, out.dest, info.dest), node);
        out.ninstructions++;
    } else if (CODEINFO_ISGLOBAL(info)) {
        /* Move upvalues */
        out.dest=compiler_regtemp(c, reg);
        out.returntype=REGISTER;
        compiler_addinstruction(c, ENCODE_LONG(OP_LGL, out.dest, info.dest), node);
        out.ninstructions++;
    } else {
        /* Move between registers */
        if (reg==REGISTER_UNALLOCATED) {
            out.dest=compiler_regtemp(c, reg);
        } else {
            out.dest=reg;
        }

        if (out.dest!=info.dest) {
            compiler_addinstruction(c, ENCODE_DOUBLE(OP_MOV, out.dest, info.dest), node);
            out.ninstructions++;
        }
    }

    return out;
}

/** Write a symbol to the constant table, performing interning and checking the result fits into a register definition
 * @param c        the compiler
 * @param node     current syntaxtree node
 *  @param symbol the constant to add */
codeinfo compiler_addsymbolwithsizecheck(compiler *c, syntaxtreenode *node, value symbol) {
    codeinfo out = CODEINFO(CONSTANT, 0, 0);
    out.dest = compiler_addsymbol(c, node, symbol);
    //if (out.dest>MORPHO_MAXREGISTERS) {
        out = compiler_movetoregister(c, node, out, REGISTER_UNALLOCATED);
    //}
    return out;
}

/* ------------------------------------------
 * Optional and variadic args
 * ------------------------------------------- */

DEFINE_VARRAY(optionalparam, optionalparam);

/** Adds  an optional argument */
static inline void compiler_addoptionalarg(compiler *c, syntaxtreenode *node, value symbol, value def) {
    functionstate *f = compiler_currentfunctionstate(c);

    if (f) {
        value sym=program_internsymbol(c->out, symbol);
        registerindx reg = compiler_addlocal(c, node, sym);
        registerindx val = compiler_addconstant(c, node, def, false, true);

        optionalparam param = {.symbol=sym, .def=val, .reg=reg};

        varray_optionalparamwrite(&f->func->opt, param);
    }
}

/** Adds  a variadic parameter */
static inline void compiler_addvariadicarg(compiler *c, syntaxtreenode *node, value symbol) {
    functionstate *f = compiler_currentfunctionstate(c);

    if (f) {
        if (object_functionhasvargs(f->func)) {
            compiler_error(c, node, COMPILE_MLTVARPRMTR);
            return;
        }

        value sym=program_internsymbol(c->out, symbol);
        registerindx reg = compiler_addlocal(c, node, sym);

        object_functionsetvarg(f->func, reg-1);
    }
}

/** Check if the current function has variadic parameters */
bool compiler_hasvariadicarg(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);
    return object_functionhasvargs(f->func);
}

/* ------------------------------------------
 * Global variables
 * ------------------------------------------- */

/** Should we use global variables or registers?  */
static bool compiler_checkglobal(compiler *c) {
#ifdef MORPHO_NOGLOBALS
    return false;
#else
    return ((c->fstackp==0) && (c->fstack[0].scopedepth==0));
#endif
}

/** Finds a global symbol, optionally searching successively through parent compilers */
static globalindx compiler_getglobal(compiler *c, value symbol, bool recurse) {
    for (compiler *cc=c; cc!=NULL; cc=cc->parent) {
        value indx;
        if (dictionary_get(&cc->globals, symbol, &indx)) {
            if (MORPHO_ISINTEGER(indx)) return (globalindx) MORPHO_GETINTEGERVALUE(indx);
            else UNREACHABLE("Unknown type in global table.");
        }
        if (!recurse) break;
    }

    return GLOBAL_UNALLOCATED;
}

/** Adds a global variable to */
static globalindx compiler_addglobal(compiler *c, syntaxtreenode *node, value symbol) {
    globalindx indx=compiler_getglobal(c, symbol, false);

    if (indx==GLOBAL_UNALLOCATED) {
        if (dictionary_insert(&c->globals, object_clonestring(symbol), MORPHO_INTEGER(c->out->nglobals))) {
            indx=c->out->nglobals;
            c->out->nglobals++;
            debug_setglobal(&c->out->annotations, indx, symbol);
        }
    }

    return indx;
}

/* Moves the result of a calculation to an global variable */
static codeinfo compiler_movetoglobal(compiler *c, syntaxtreenode *node, codeinfo in, globalindx slot) {
    codeinfo use = in;
    codeinfo out = CODEINFO_EMPTY;
    bool tmp=false;

    if (!(CODEINFO_ISREGISTER(in))) {
        use=compiler_movetoregister(c, node, in, REGISTER_UNALLOCATED);
        out.ninstructions+=use.ninstructions;
        tmp=true;
    }

    compiler_addinstruction(c, ENCODE_LONG(OP_SGL, use.dest, slot) , node);
    out.ninstructions++;

    if (tmp) {
        compiler_releaseoperand(c, use);
    }

    return out;
}


static codeinfo compiler_addvariable(compiler *c, syntaxtreenode *node, value symbol) {
    codeinfo out=CODEINFO_EMPTY;

    if (compiler_checkglobal(c)) {
        out.dest=compiler_addglobal(c, node, symbol);
        out.returntype=GLOBAL;
    } else {
        out.dest=compiler_addlocal(c, node, symbol);
        out.returntype=REGISTER;
    }

    return out;
}

/* ------------------------------------------
 * Upvalues
 * ------------------------------------------- */

/** Adds an upvalue to a functionstate */
registerindx compiler_addupvalue(functionstate *f, bool islocal, indx ix) {
    upvalue v = (upvalue) { .islocal = islocal, .reg = ix};

    /* Does this upvalue already exist? */
    for (registerindx i=0; i<f->upvalues.count; i++) {
        upvalue *up = &f->upvalues.data[i];
        if (up->islocal==islocal &&
            up->reg==ix) {
            return i;
        }
    }

    /* If not, add it */
    varray_upvalueadd(&f->upvalues, &v, 1);
    return (registerindx) f->upvalues.count-1;
}

/** @brief Determines whether a symbol refers to something outside its scope
    @param c      the compiler
    @param symbol symbol to resolve
    @returns the index of the upvalue, or REGISTER_UNALLOCATED if not found */
static registerindx compiler_resolveupvalue(compiler *c, value symbol) {
    registerindx indx=REGISTER_UNALLOCATED;
    functionstate *found=NULL;

    for (functionstate *f = c->fstack+c->fstackp-1; f>=c->fstack; f--) {
        /* Try to find the symbol */
        indx = compiler_findsymbol(f, symbol);
        if (indx!=REGISTER_UNALLOCATED) {
            /* Mark that this register must be captured as an upvalue */
            f->registers.data[indx].iscaptured=true;
            found=f;
            break;
        }
    }

    /* Now walk up the functionstate stack adding in the upvalues */
    if (found) for (functionstate *f = found; f<c->fstack+c->fstackp; f++) {
        indx=compiler_addupvalue(f, f==found, indx);
    }

    return indx;
}

/* Moves the result of a calculation to an upvalue */
static codeinfo compiler_movetoupvalue(compiler *c, syntaxtreenode *node, codeinfo in, registerindx slot) {
    codeinfo use = in;
    codeinfo out = CODEINFO_EMPTY;
    bool tmp=false;

    if (!CODEINFO_ISREGISTER(in)) {
        use=compiler_movetoregister(c, node, in, REGISTER_UNALLOCATED);
        out.ninstructions+=use.ninstructions;
        tmp=true;
    }

    compiler_addinstruction(c, ENCODE_DOUBLE(OP_SUP, slot, use.dest), node);
    out.ninstructions++;

    if (tmp) {
        compiler_releaseoperand(c, use);
    }

    return out;
}

/** Creates code to generate a closure for the current environment. */
indx compiler_closure(compiler *c, syntaxtreenode *node, registerindx reg) {
    functionstate *f=compiler_currentfunctionstate(c);
    objectfunction *func = f->func;
    indx ix=REGISTER_UNALLOCATED;

    if (f->upvalues.count>0) {
        object_functionaddprototype(func, &f->upvalues, &ix);
    }
    return ix;
}

/* ------------------------------------------
 * Helper function to move from a register
 * ------------------------------------------- */

/* Moves the result of a calculation from a register to another destination */
codeinfo compiler_movefromregister(compiler *c, syntaxtreenode *node, codeinfo dest, registerindx reg) {
    codeinfo in=CODEINFO(REGISTER, reg, 0), out=CODEINFO_EMPTY;

    if (CODEINFO_ISUPVALUE(dest)) {
        out=compiler_movetoupvalue(c, node, in, dest.dest);
    } else if (CODEINFO_ISGLOBAL(dest)) {
        out=compiler_movetoglobal(c, node, in, dest.dest);
    } else if (CODEINFO_ISCONSTANT(dest)) {
        UNREACHABLE("Cannot move to a constant.");
    } else {
        out=compiler_movetoregister(c, node, in, dest.dest);
    }

    return out;
}

/* ------------------------------------------
 * Self
 * ------------------------------------------- */

/** @brief Resolves self by walking back along the functionstate stack
    @param c      the compiler
    @returns the index of self */
static registerindx compiler_resolveself(compiler *c) {
    registerindx indx=REGISTER_UNALLOCATED;
    functionstate *found=NULL;

    for (functionstate *f = c->fstack+c->fstackp-1; f>=c->fstack; f--) {
        /* If this is a method, we can capture self from it */
        if (FUNCTIONTYPE_ISMETHOD(f->type)) {
            indx=0;
            f->registers.data[indx].iscaptured=true;
            found=f;
            break;
        }
    }

    /* Now walk up the functionstate stack adding in the upvalues */
    if (found) for (functionstate *f = found; f<c->fstack+c->fstackp; f++) {
        indx=compiler_addupvalue(f, f==found, indx);
    }

    return indx;
}

/* ------------------------------------------
 * Forward references
 * ------------------------------------------- */

/** Adds a forward reference
 * @param[in] c            the compiler
 * @param[in] symbol symbol corresponding to forward reference */
static codeinfo compiler_addforwardreference(compiler *c, syntaxtreenode *node, value symbol) {
    codeinfo ret = CODEINFO_EMPTY;
    functionstate *fparent = compiler_parentfunctionstate(c);

    if (!fparent) { // If in global scope, generate symbol not defined
        char *label = MORPHO_GETCSTRING(node->content);
        compiler_error(c, node, COMPILE_SYMBOLNOTDEFINED, label);
        return ret;
    }

    forwardreference ref = { .symbol = symbol,
                             .node = node,
                             .returntype=REGISTER,
                             .dest=compiler_regallocwithstate(c, fparent, symbol),
                             .scopedepth = fparent->scopedepth
    };

    ret.returntype = UPVALUE;
    ret.dest=compiler_addupvalue(fparent, true, ref.dest);

    varray_forwardreferencewrite(&fparent->forwardref, ref);

    return ret;
}

/** Checks if a function definition resolves a forward reference */
static bool compiler_resolveforwardreference(compiler *c, objectfunction *func, codeinfo *out) {
    bool success=false;
    functionstate *f = compiler_currentfunctionstate(c);

    for (unsigned int i=0; i<f->forwardref.count; i++) {
        forwardreference *ref = f->forwardref.data+i;
        if (MORPHO_ISEQUAL(func->name, ref->symbol) &&
            ref->scopedepth==compiler_currentscope(c)) {
            out->returntype=ref->returntype;
            out->dest=ref->dest;
            ref->symbol=MORPHO_NIL; // Forward reference was successfully resolved
            success=true;
        }
    }

    return success;
}

/** Check for outstanding forward references  */
bool compiler_checkoutstandingforwardreference(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);

    for (unsigned int i=0; i<f->forwardref.count; i++) {
        forwardreference *ref = f->forwardref.data+i;
        if (!MORPHO_ISNIL(ref->symbol)) {
            compiler_error(c, ref->node, COMPILE_FORWARDREF, MORPHO_GETCSTRING(ref->symbol));
            return false;
        }
    }
    return true;
}

/* ------------------------------------------
 * Compiler node implementation functions
 * ------------------------------------------- */

static codeinfo compiler_constant(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_list(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_dictionary(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_index(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_negate(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_not(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_binary(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_property(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_grouping(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_sequence(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_interpolation(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_range(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_scope(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_print(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_if(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_while(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_for(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_do(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_break(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_try(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_logical(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_declaration(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_function(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_arglist(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_call(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_invoke(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_return(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_class(compiler *c, syntaxtreenode *node, registerindx out);
static codeinfo compiler_symbol(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_self(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_super(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_assign(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_import(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_breakpoint(compiler *c, syntaxtreenode *node, registerindx out);

static codeinfo compiler_nodetobytecode(compiler *c, syntaxtreeindx indx, registerindx out);

/* ------------------------------------------
 * Compiler definition table
 * ------------------------------------------- */

#define NODE_NORULE NULL
#define NODE_UNDEFINED { NODE_NORULE }

/** Compiler definition table. This specifies a compiler function that handles each node type */
compilenoderule noderules[] = {
    NODE_UNDEFINED,                  // NODE_BASE

    { compiler_constant      },      // NODE_NIL,
    { compiler_constant      },      // NODE_BOOL,
    { compiler_constant      },      // NODE_FLOAT
    { compiler_constant      },      // NODE_INTEGER
    { compiler_constant      },      // NODE_STRING
    { compiler_symbol        },      // NODE_SYMBOL
    { compiler_self          },      // NODE_SELF
    { compiler_super         },      // NODE_SUPER
    { compiler_constant      },      // NODE_IMAG
    NODE_UNDEFINED,                  // NODE_LEAF

    { compiler_negate        },      // NODE_NEGATE
    { compiler_not           },      // NODE_NOT
    NODE_UNDEFINED,                  // NODE_UNARY

    { compiler_binary        },      // NODE_ADD
    { compiler_binary        },      // NODE_SUBTRACT
    { compiler_binary        },      // NODE_MULTIPLY
    { compiler_binary        },      // NODE_DIVIDE
    { compiler_binary        },      // NODE_POW

    { compiler_assign        },      // NODE_ASSIGN
    { compiler_binary        },      // NODE_EQ
    { compiler_binary        },      // NODE_NEQ
    { compiler_binary        },      // NODE_LT
    { compiler_binary        },      // NODE_GT
    { compiler_binary        },      // NODE_LTEQ
    { compiler_binary        },      // NODE_GTEQ

    { compiler_logical       },      // NODE_AND
    { compiler_logical       },      // NODE_OR

    { compiler_property      },      // NODE_DOT

    { compiler_range         },      // NODE_RANGE

    NODE_UNDEFINED,                  // NODE_OPERATOR

    { compiler_print         },      // NODE_PRINT
    { compiler_declaration   },      // NODE_DECLARATION
    { compiler_function      },      // NODE_FUNCTION
    { NODE_NORULE            },      // NODE_METHOD
    { compiler_class         },      // NODE_CLASS
    { compiler_return        },      // NODE_RETURN
    { compiler_if            },      // NODE_IF
    { NODE_NORULE            },      // NODE_ELSE
    { compiler_while         },      // NODE_WHILE
    { compiler_for           },      // NODE_FOR
    { compiler_do            },      // NODE_DO
    { NODE_NORULE            },      // NODE_IN
    { compiler_break         },      // NODE_BREAK
    { compiler_break         },      // NODE_CONTINUE
    { compiler_try           },      // NODE_TRY

    NODE_UNDEFINED,                  // NODE_STATEMENT

    { compiler_grouping      },      // NODE_GROUPING
    { compiler_sequence      },      // NODE_SEQUENCE
    { compiler_dictionary    },      // NODE_DICTIONARY
    NODE_UNDEFINED,                  // NODE_DICTENTRY
    { compiler_interpolation },      // NODE_INTERPOLATION
    { compiler_arglist       },      // NODE_ARGLIST
    { compiler_scope         },      // NODE_SCOPE
    { compiler_call          },      // NODE_CALL
    { compiler_index         },      // NODE_INDEX
    { compiler_list          },      // NODE_LIST
    { compiler_import        },      // NODE_IMPORT
    { compiler_breakpoint    }       // NODE_BREAKPOINT
};

/** Get the associated node rule for a given node type */
static compilenoderule *compiler_getrule(syntaxtreenodetype type) {
    return &noderules[type];
}

/** Compile a constant */
static codeinfo compiler_constant(compiler *c, syntaxtreenode *node, registerindx reqout) {
    registerindx indx = compiler_addconstant(c, node, node->content, (node->type==NODE_FLOAT ? true : false), true);
    return CODEINFO(CONSTANT, indx, 0);
}

/** Compiles a list */
static codeinfo compiler_list(compiler *c, syntaxtreenode *node, registerindx reqout) {
    syntaxtreenodetype dictentrytype[] = { NODE_ARGLIST };
    varray_syntaxtreeindx entries;

    /* Set up a call to the List() function */
    codeinfo out = compiler_findbuiltin(c, node, LIST_CLASSNAME, reqout);

    varray_syntaxtreeindxinit(&entries);
    if (node->right!=SYNTAXTREE_UNCONNECTED) syntaxtree_flatten(compiler_getsyntaxtree(c), node->right, 1, dictentrytype, &entries);

    /* Now loop over the nodes */
    unsigned int nargs=0;
    for (int i=0; i<entries.count; i++) {
        syntaxtreenode *entry=compiler_getnode(c, entries.data[i]);

        registerindx reg=compiler_regalloctop(c);
        codeinfo val = compiler_nodetobytecode(c, entries.data[i], reg);
        out.ninstructions+=val.ninstructions;
        if (!(CODEINFO_ISREGISTER(val) && (val.dest==reg))) {
            compiler_releaseoperand(c, val);
            compiler_regtempwithindx(c, reg);
            val=compiler_movetoregister(c, entry, val, reg);
            out.ninstructions+=val.ninstructions;
        }
        nargs++;
    }

    varray_syntaxtreeindxclear(&entries);

    /* Make the function call */
    compiler_addinstruction(c, ENCODE_DOUBLE(OP_CALL, out.dest, nargs), node);
    out.ninstructions++;
    compiler_regfreetoend(c, out.dest+1);

    return out;
}

/** Compiles a dictionary */
static codeinfo compiler_dictionary(compiler *c, syntaxtreenode *node, registerindx reqout) {
    syntaxtreenodetype dictentrytype[] = { NODE_DICTIONARY, NODE_DICTENTRY };
    varray_syntaxtreeindx entries;

    /* Set up a call to the Dictionary() function */
    codeinfo out = compiler_findbuiltin(c, node, DICTIONARY_CLASSNAME, reqout);

    varray_syntaxtreeindxinit(&entries);
    /* Flatten all the child nodes; these end up as a sequence: key, val, key, val, ... */
    if (node->left!=SYNTAXTREE_UNCONNECTED) syntaxtree_flatten(compiler_getsyntaxtree(c), node->left, 2, dictentrytype, &entries);
    if (node->right!=SYNTAXTREE_UNCONNECTED) syntaxtree_flatten(compiler_getsyntaxtree(c), node->right, 2, dictentrytype, &entries);

    /* Now loop over the nodes */
    unsigned int nargs=0;
    for (int i=0; i<entries.count; i+=2) {
        /* Loop over key/val pairs */
        for (int j=0; j<2; j++) {
            syntaxtreenode *entry=compiler_getnode(c, entries.data[i+j]);

            registerindx reg=compiler_regalloctop(c);
            codeinfo val = compiler_nodetobytecode(c, entries.data[i+j], reg);
            out.ninstructions+=val.ninstructions;
            if (!(CODEINFO_ISREGISTER(val) && (val.dest==reg))) {
                compiler_releaseoperand(c, val);
                val=compiler_movetoregister(c, entry, val, reg);
                out.ninstructions+=val.ninstructions;
            }
            nargs++;
        }
    }
    varray_syntaxtreeindxclear(&entries);

    /* Make the function call */
    compiler_addinstruction(c, ENCODE_DOUBLE(OP_CALL, out.dest, nargs), node);
    out.ninstructions++;
    compiler_regfreetoend(c, out.dest+1);

    return out;
}

/** Compiles a range */
static codeinfo compiler_range(compiler *c, syntaxtreenode *node, registerindx reqout) {
    syntaxtreeindx s[3]={ SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED};

    /* Determine whether we have start..end or start..end:step */
    syntaxtreenode *left=compiler_getnode(c, node->left);
    if (left && left->type==NODE_RANGE) {
        s[0]=left->left; s[1]=left->right; s[2]=node->right;
    } else {
        s[0]=node->left; s[1]=node->right;
    }

    /* Set up a call to the Range() function */
    codeinfo rng = compiler_findbuiltin(c, node, RANGE_CLASSNAME, reqout);

    /* Construct the arguments */
    unsigned int n;
    for (n=0; n<3; n++) {
        if (s[n]!=SYNTAXTREE_UNCONNECTED) {
            registerindx rarg=compiler_regalloctop(c);
            codeinfo data=compiler_nodetobytecode(c, s[n], rarg);
            rng.ninstructions+=data.ninstructions;
            if (!(CODEINFO_ISREGISTER(data) && (data.dest==rarg))) {
                compiler_releaseoperand(c, data);
                data=compiler_movetoregister(c, node, data, rarg);
                rng.ninstructions+=data.ninstructions;
            }
        } else {
            break;
        }
    }

    /* Make the function call */
    compiler_addinstruction(c, ENCODE_DOUBLE(OP_CALL, rng.dest, n), node);
    rng.ninstructions++;
    compiler_regfreetoend(c, rng.dest+1);

    return rng;
}

/* @brief Compiles a index node
 * @param[in] c        - the compiler
 * @param[in] indxnode - the root syntaxtreenode (should be of type NODE_INDEX)
 * @param[out] start   - first register used
 * @param[out] end     - last register used
 * @returns number of instructions */
static codeinfo compiler_compileindexlist(compiler *c, syntaxtreenode *indxnode, registerindx *start, registerindx *end) {
    registerindx istart=compiler_regtop(c), iend;

    compiler_beginargs(c);
    codeinfo right = compiler_nodetobytecode(c, indxnode->right, REGISTER_UNALLOCATED);
    compiler_endargs(c);
    iend=compiler_regtop(c);

    if (iend==istart) compiler_error(c, indxnode, COMPILE_VARBLANKINDEX);

    if (start) *start = istart+1;
    if (end) *end = iend;

    return right;
}

/** Compiles a lookup of an indexed variable */
static codeinfo compiler_index(compiler *c, syntaxtreenode *node, registerindx reqout) {
    registerindx start, end;

    /* Compile the index selector */
    codeinfo left = compiler_nodetobytecode(c, node->left, reqout);
    unsigned int ninstructions=left.ninstructions;

    if (!CODEINFO_ISREGISTER(left)) {
        left=compiler_movetoregister(c, node, left, REGISTER_UNALLOCATED);
        ninstructions+=left.ninstructions;
    }

    /* Compile indices */
    codeinfo out = compiler_compileindexlist(c, node, &start, &end);
    ninstructions+=out.ninstructions;

    /* Compile instruction */
    compiler_addinstruction(c, ENCODE(OP_LIX, left.dest, start, end), node);
    ninstructions++;

    /* Free anything we're done with */
    compiler_releaseoperand(c, left);
    compiler_regfreetoend(c, start+1);

    return CODEINFO(REGISTER, start, ninstructions);
}

/** Compile negation. Note that this is compiled as A=(0-B) */
static codeinfo compiler_negate(compiler *c, syntaxtreenode *node, registerindx reqout) {
    syntaxtreenode *operand = compiler_getnode(c, node->left);
    codeinfo out;

    if (operand->type==NODE_FLOAT) {
        registerindx neg = compiler_addconstant(c, node, MORPHO_FLOAT(-MORPHO_GETFLOATVALUE(operand->content)), true, false);
        out = CODEINFO(CONSTANT, neg, 0);
    } else {
        out = compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
        unsigned int ninstructions=out.ninstructions;

        if (!CODEINFO_ISREGISTER(out)) {
            /* Ensure we're working with a register */
            out=compiler_movetoregister(c, node, out, REGISTER_UNALLOCATED);
            ninstructions+=out.ninstructions;
        }

        registerindx zero = compiler_addconstant(c, node, MORPHO_INTEGER(0), false, false);
        codeinfo zeroinfo = CODEINFO(CONSTANT, zero, 0);
        zeroinfo=compiler_movetoregister(c, node, zeroinfo, REGISTER_UNALLOCATED);
        ninstructions+=zeroinfo.ninstructions;

        registerindx rout = compiler_regtemp(c, reqout);

        compiler_addinstruction(c, ENCODE(OP_SUB, rout, zeroinfo.dest, out.dest), node);
        ninstructions++;
        compiler_releaseoperand(c, out);
        compiler_releaseoperand(c, zeroinfo);
        out = CODEINFO(REGISTER, rout, ninstructions);
    }

    return out;
}

/** Compile not operator */
static codeinfo compiler_not(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo left = compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
    unsigned int ninstructions=left.ninstructions;
    registerindx out = compiler_regtemp(c, reqout);

    if (!CODEINFO_ISREGISTER(left)) {
        /* Ensure we're working with a register or a constant */
        left=compiler_movetoregister(c, node, left, REGISTER_UNALLOCATED);
        ninstructions+=left.ninstructions;
    }

    compiler_addinstruction(c, ENCODE_DOUBLE(OP_NOT, out, left.dest), node);
    ninstructions++;
    compiler_releaseoperand(c, left);

    return CODEINFO(REGISTER, out, ninstructions);
}

/** Compile arithmetic operators */
static codeinfo compiler_binary(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo left = compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
    unsigned int ninstructions=left.ninstructions;
    if (!(CODEINFO_ISREGISTER(left))) {
        /* Ensure we're working with a register */
        left=compiler_movetoregister(c, node, left, REGISTER_UNALLOCATED);
        ninstructions+=left.ninstructions;
    }

    codeinfo right = compiler_nodetobytecode(c, node->right, REGISTER_UNALLOCATED);
    ninstructions+=right.ninstructions;
    if (!(CODEINFO_ISREGISTER(right))) {
        /* Ensure we're working with a register  */
        right=compiler_movetoregister(c, node, right, REGISTER_UNALLOCATED);
        ninstructions+=right.ninstructions;
    }

    registerindx out = compiler_regtemp(c, reqout);

    opcode op=OP_NOP;

    switch (node->type) {
        case NODE_ADD: op=OP_ADD; break;
        case NODE_SUBTRACT: op=OP_SUB; break;
        case NODE_MULTIPLY: op=OP_MUL; break;
        case NODE_DIVIDE: op=OP_DIV; break;
        case NODE_POW: op=OP_POW; break;
        case NODE_EQ: op=OP_EQ; break;
        case NODE_NEQ: op=OP_NEQ; break;
        case NODE_LT: op=OP_LT; break;
        case NODE_LTEQ: op=OP_LE; break;
        case NODE_GT:
            {   /* a>b is equivalent to b<a */
                codeinfo swap = right; right=left; left = swap;
                op=OP_LT;
            }
            break;
        case NODE_GTEQ:
            {   /* a>=b is equivalent to b<=a */
                codeinfo swap = right; right=left; left = swap;
                op=OP_LE;
            }
            break;
        default:
            UNREACHABLE("in compiling binary instruction [check bytecode compiler table]");
    }

    compiler_addinstruction(c, ENCODE(op, out, left.dest, right.dest), node);
    ninstructions++;
    compiler_releaseoperand(c, left);
    compiler_releaseoperand(c, right);

    return CODEINFO(REGISTER, out, ninstructions);
}

/** Compiles a group (used to modify precedence) */
static codeinfo compiler_grouping(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo out = CODEINFO_EMPTY;

    if (node->left!=SYNTAXTREE_UNCONNECTED)
        out=compiler_nodetobytecode(c, node->left, reqout);

    return out;
}

/** Compiles a sequence of nodes */
static codeinfo compiler_sequence(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo left = CODEINFO_EMPTY;
    codeinfo right = CODEINFO_EMPTY;
    bool inargs=compiler_inargs(c);

    if (node->left!=SYNTAXTREE_UNCONNECTED)
        left=compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
    if (!inargs) compiler_releaseoperand(c, left);

    if (node->right!=SYNTAXTREE_UNCONNECTED)
        right=compiler_nodetobytecode(c, node->right, REGISTER_UNALLOCATED);
    if (!inargs) compiler_releaseoperand(c, right);

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, left.ninstructions+right.ninstructions);
}

/** Compiles a string interpolation */
static codeinfo compiler_interpolation(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo exp;
    registerindx start = REGISTER_UNALLOCATED;
    unsigned int ninstructions=0;
    registerindx r = REGISTER_UNALLOCATED;

    for (syntaxtreenode *current = node;
         current!=NULL;
         current=compiler_getnode(c, current->right)) {

        /* Add the contents of the node as a constant and move them into a register at the top of the stack */
        registerindx cindx = compiler_addconstant(c, current, current->content, false, true);
        r=compiler_regalloctop(c);
        codeinfo string=compiler_movetoregister(c, current, CODEINFO(CONSTANT, cindx, 0), r);
        ninstructions+=string.ninstructions;
        if (start==REGISTER_UNALLOCATED) start = string.dest;

        /* Now compile the interpolation expression */
        if (current->left!=SYNTAXTREE_UNCONNECTED) {
            r = compiler_regalloctop(c);
            exp=compiler_nodetobytecode(c, current->left, r);
            ninstructions+=exp.ninstructions;

            /* Make sure this goes into the correct register */
            if (!(exp.returntype==REGISTER && exp.dest==r)) {
                compiler_regtempwithindx(c, r);
                exp=compiler_movetoregister(c, current, exp, r);
                ninstructions+=exp.ninstructions;
            }
            /* Free any registers above the result */
            compiler_regfreetoend(c, r+1);
        }
    }

    compiler_addinstruction(c, ENCODE(OP_CAT, (reqout!=REGISTER_UNALLOCATED ? reqout : start), start, r), node);
    ninstructions++;

    /* Free all the registers used, including start if it wasn't the destination for the output */
    if (start!=REGISTER_UNALLOCATED) compiler_regfreetoend(c, start + (reqout!=REGISTER_UNALLOCATED ? 0: 1));

    return CODEINFO(REGISTER, (reqout!=REGISTER_UNALLOCATED ? reqout : start), ninstructions);
}

/** Inserts instructions to close upvalues */
static codeinfo compiler_closeupvaluesforscope(compiler *c, syntaxtreenode *node) {
    functionstate *f = compiler_currentfunctionstate(c);
    codeinfo out=CODEINFO_EMPTY;
    indx closereg=VM_MAXIMUMREGISTERNUMBER; /* Keep track of the lowest register number to close */
    bool closed=false; /* Do we need to close anything? */

    for (unsigned int i=0; i<f->upvalues.count; i++) {
        if (f->upvalues.data[i].islocal) {
            indx reg = f->upvalues.data[i].reg;

            if (f->registers.data[reg].scopedepth>=f->scopedepth) {
                if (reg<closereg) closereg=reg; closed=true;
            }
        }
    }

    if (closed) {
        compiler_addinstruction(c, ENCODE_SINGLE(OP_CLOSEUP, (registerindx) closereg), node);
        out.ninstructions++;
    }
    return out;
}

/** Compiles a scope node. */
static codeinfo compiler_scope(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo out=CODEINFO_EMPTY, up;
    compiler_beginscope(c);
    if (node->right!=REGISTER_UNALLOCATED) {
        out=compiler_nodetobytecode(c, node->right, reqout);
    }
    up=compiler_closeupvaluesforscope(c, node);
    out.ninstructions+=up.ninstructions;
    compiler_endscope(c);

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, out.ninstructions);
}

/** Compile print statements */
static codeinfo compiler_print(compiler *c, syntaxtreenode *node, registerindx reqout) {
    if (node->left==REGISTER_UNALLOCATED) return CODEINFO_EMPTY;

    codeinfo left=compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
    unsigned int ninstructions=left.ninstructions;

    if (!CODEINFO_ISREGISTER(left)) {
        left=compiler_movetoregister(c, node, left, REGISTER_UNALLOCATED);
        ninstructions+=left.ninstructions;
    } else if (c->err.cat==ERROR_NONE &&  left.dest==REGISTER_UNALLOCATED) {
        UNREACHABLE("print was passed an invalid operand");
    }

    compiler_addinstruction(c, ENCODE_SINGLE(OP_PRINT, left.dest), node);
    ninstructions++;
    compiler_releaseoperand(c, left);

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, ninstructions);
}

/** @brief   Compile if statements
 *  @details If statements come in two flavors, those with else clauses and those
 *           without. They are encoded in the syntax tree as follows:
 *
 *           <pre>
 *                   IF                        IF
 *                  /  \                      /  \
 *              cond    then statement    cond    THEN
 *                                               /    \
 *                                 then statement      else statement
 *           </pre>
 *
 *           and are compiled to bytecode that looks like
 *
 *           <pre>
 *              test   rtst, rB, rC
 *              bif    rtst, fail:    ; branch if condition isn't met
 *              .. then statement...
 *              b      end            ; generated if an else statement is present
 *           fail:
 *              .. else statement (if present)
 *           end:
 *           </pre>
 */
static codeinfo compiler_if(compiler *c, syntaxtreenode *node, registerindx reqout) {
    unsigned int ninstructions=0;
    bool unreachable=false;

    /* The left node is the condition; compile it already */
    codeinfo cond = compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
    ninstructions+=cond.ninstructions;

    if (CODEINFO_ISCONSTANT(cond) &&
        MORPHO_ISFALSE(compiler_getconstant(c, cond.dest)) ) unreachable=true;

    /* And make sure it's in a register */
    if (!CODEINFO_ISREGISTER(cond)) {
       cond=compiler_movetoregister(c, node, cond, REGISTER_UNALLOCATED);
       ninstructions+=cond.ninstructions; /* Keep track of instructions */
    }

    compiler_releaseoperand(c, cond);

    /* The right node may be a then node or just a regular statement */
    syntaxtreenode *right = compiler_getnode(c, node->right);

    /* Hold the position of the then and else statements */
    codeinfo then=CODEINFO_EMPTY, els=CODEINFO_EMPTY;

    /* Remember where the if conditional branch is located */
    instructionindx ifindx=REGISTER_UNALLOCATED, elsindx=REGISTER_UNALLOCATED;

    /* Keep track of the number of instructions */
    unsigned int nextra=0;

    /* Generate empty instruction to contain the conditional branch */
    ifindx=compiler_addinstruction(c, ENCODE_BYTE(OP_NOP), node);
    ninstructions++;

    if (right->type==NODE_THEN) {
        /* If the right node is a THEN node, the then/else statements are located off it. */
        if (!unreachable) {
            then = compiler_nodetobytecode(c, right->left, REGISTER_UNALLOCATED);
        }
        ninstructions+=then.ninstructions;

        /* Create a blank instruction */
        elsindx=compiler_addinstruction(c, ENCODE_BYTE(OP_NOP), right);
        ninstructions++;
        nextra++; /* Keep track of this extra instruction for the original branch */

        /* Now compile the els clause */
        els = compiler_nodetobytecode(c, right->right, REGISTER_UNALLOCATED);
        ninstructions+=els.ninstructions;
    } else {
        /* Otherwise, the then statement is just the right operand */
        if (!unreachable) {
            then = compiler_nodetobytecode(c, node->right, REGISTER_UNALLOCATED);
        }
        ninstructions+=then.ninstructions;
    }

    /* Now generate the conditional branch over the then clause */
    compiler_setinstruction(c, ifindx, ENCODE_LONG(OP_BIFF, cond.dest, then.ninstructions+nextra));

    /* If necessary generate the unconditional branch over the else clause */
    if (right->type==NODE_THEN) {
        compiler_setinstruction(c, elsindx, ENCODE_LONG(OP_B, REGISTER_UNALLOCATED, els.ninstructions));
    }
    compiler_releaseoperand(c, then);
    compiler_releaseoperand(c, els);

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, ninstructions);
}

/** @brief   Compiles a while statement
 *  @details While statements are encoded as follows:
 *
 *           <pre>
 *                   WHILE
 *                  /     \
 *              cond       body
 *           </pre>
 *
 *           and are compiled to bytecode that looks like
 *
 *           <pre>
 *           start:
 *               test   rtst, rB, rC
 *               bif    rtst, end:    ; branch if condition isn't met
 *               .. body ..
 *               b      start
 *           end:
 *           </pre>
*/
static codeinfo compiler_while(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo cond = CODEINFO_EMPTY,
             body = CODEINFO_EMPTY,
             inc = CODEINFO_EMPTY;
    unsigned int ninstructions=0;
    instructionindx condindx=REGISTER_UNALLOCATED; /* Where is the condition located */
    instructionindx startindx=compiler_currentinstructionindex(c);
    instructionindx nextindx=startindx; /* Where should continue jump to? */

    /* The left node is the condition; compile it already */
    if (node->left!=SYNTAXTREE_UNCONNECTED) {
        cond=compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
        ninstructions+=cond.ninstructions; /* Keep track of instructions */

        /* And make sure it's in a register */
        if (!CODEINFO_ISREGISTER(cond)) {
            cond=compiler_movetoregister(c, node, cond, REGISTER_UNALLOCATED);
            ninstructions+=cond.ninstructions; /* Keep track of instructions */
        }

        /* Generate empty instruction to contain the conditional branch */
        condindx=compiler_addinstruction(c, ENCODE_BYTE(OP_NOP), node);
        nextindx=condindx;
        ninstructions++;

        compiler_releaseoperand(c, cond);
    }

    compiler_beginloop(c);

    if (node->right!=SYNTAXTREE_UNCONNECTED) {
        syntaxtreenode *bodynode=compiler_getnode(c, node->right);

        /* Check if we're in a for loop */
        if (bodynode->type!=NODE_SEQUENCE) {
            body = compiler_nodetobytecode(c, node->right, REGISTER_UNALLOCATED);
            ninstructions+=body.ninstructions;
            compiler_releaseoperand(c, body);
        } else {
            if (bodynode->left!=SYNTAXTREE_UNCONNECTED) {
                body = compiler_nodetobytecode(c, bodynode->left, REGISTER_UNALLOCATED);
                compiler_releaseoperand(c, body);
            }
            nextindx = compiler_currentinstructionindex(c);
            if (bodynode->right!=SYNTAXTREE_UNCONNECTED) {
                inc = compiler_nodetobytecode(c, bodynode->right, REGISTER_UNALLOCATED);
                compiler_releaseoperand(c, inc);
            }

            body.ninstructions+=inc.ninstructions; // Used for the branch back
            ninstructions+=body.ninstructions; // Track all the instructions
        }
    }

    compiler_endloop(c);

    /* Compile the unconditional branch back to the test instruction */
    instructionindx end=compiler_addinstruction(c, ENCODE_LONG(OP_B, REGISTER_UNALLOCATED, -ninstructions-1), node);
    ninstructions++;

    compiler_fixloop(c, startindx, nextindx, end+1);

    /* If we did have a condition... */
    if (node->left!=SYNTAXTREE_UNCONNECTED) {
        /* And generate the conditional branch at the start of the loop.
           The extra 1 is to skip the loop instruction */
        compiler_setinstruction(c, condindx, ENCODE_LONG(OP_BIFF, cond.dest, body.ninstructions+1));
    }

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, ninstructions);
}

/** @brief Compiles a for .. in loop
 * @details For..in loops enable looping over the contents of a collection without knowing how many elements are present
 *                 forin
 *                /     \
 *               in      body
 *              /   \
 *            init     collection
 * This works by successively calling enumerate() on the collections, first with no arguments to get the bound, then
 * with the integer counter.
 *
 * The body of the loop is then evaluated with the value set up as a local variable.
 *
 * Register allocation
 *  +0 - loop counter
 *  +1 - maximum value of loop counter
 *  +2 - value from the collection
 */
static codeinfo compiler_for(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo body;
    unsigned int ninstructions=0;
    syntaxtreenode *innode=NULL, *initnode=NULL, *indxnode=NULL, *collnode=NULL;
    instructionindx condindx=REGISTER_UNALLOCATED; /* Where is the condition located */

    compiler_beginscope(c);

    /* Collect the syntaxtree nodes */
    innode=compiler_getnode(c, node->left);
    if (innode) {
        initnode=compiler_getnode(c, innode->left);

        if (initnode->type==NODE_SEQUENCE) { // Unpack a variable, indx sequence
            indxnode=compiler_getnode(c, initnode->right);
            initnode=compiler_getnode(c, initnode->left);
        }

        if (initnode->type==NODE_DECLARATION) initnode=compiler_getnode(c, initnode->left);
        if (indxnode && indxnode->type==NODE_DECLARATION) indxnode=compiler_getnode(c, indxnode->left);

        collnode=compiler_getnode(c, innode->right);
    }

    /* Allocate a register for the (usually hidden) counter */
    registerindx rcount=compiler_regalloc(c, MORPHO_NIL);
    registerindx cnil = compiler_addconstant(c, node, MORPHO_INTEGER(0), false, false);
    compiler_addinstruction(c, ENCODE_LONG(OP_LCT, rcount, cnil), node);
    ninstructions++;

    /* Find the collection symbol */
    codeinfo coll=compiler_nodetobytecode(c, innode->right, REGISTER_UNALLOCATED);
    ninstructions+=coll.ninstructions;

    /* Now obtain the maximum value for the counter by invoking enumerate on the collection */
    codeinfo method=compiler_addsymbolwithsizecheck(c, node, enumerateselector);
    ninstructions+=method.ninstructions;

    registerindx rmax=compiler_regalloctop(c);
    registerindx rmone=compiler_regalloctop(c);
    registerindx cmone = compiler_addconstant(c, node, MORPHO_INTEGER(-1), false, false);
    codeinfo mv=compiler_movetoregister(c, collnode, coll, rmax);
    ninstructions+=mv.ninstructions;

    compiler_addinstruction(c, ENCODE_LONG(OP_LCT, rmone, cmone), node);
    compiler_addinstruction(c, ENCODE(OP_INVOKE, rmax, method.dest, 1), collnode);
    ninstructions+=2;
    compiler_regfreetemp(c, rmone);

    /* The test instruction */
    registerindx rcond=compiler_regtemp(c, REGISTER_UNALLOCATED);
    instructionindx tst=compiler_addinstruction(c, ENCODE(OP_LT, rcond, rcount, rmax), node);
    condindx=compiler_addinstruction(c, ENCODE_BYTE(OP_NOP), node);
    ninstructions+=2;
    compiler_regfreetemp(c, rcond);

    /* Call enumerate again to retrieve the value */
    registerindx rval=compiler_regalloctop(c);
    mv=compiler_movetoregister(c, collnode, coll, rval);
    ninstructions+=mv.ninstructions;

    registerindx rarg=compiler_regalloctop(c);
    compiler_addinstruction(c, ENCODE_DOUBLE(OP_MOV, rarg, rcount), node);
    compiler_addinstruction(c, ENCODE(OP_INVOKE, rval, method.dest, 1), collnode);
    ninstructions+=2;

    compiler_regsetsymbol(c, rval, initnode->content);
    if (indxnode) compiler_regsetsymbol(c, rcount, indxnode->content);

    compiler_beginloop(c);

    /* Compile the body */
    if (node->right==SYNTAXTREE_UNCONNECTED) {
        compiler_error(c, node, COMPILE_MSSNGLOOPBDY);
    } else {
        body=compiler_nodetobytecode(c, node->right, REGISTER_UNALLOCATED);
        ninstructions+=body.ninstructions;
        compiler_releaseoperand(c, body);
    }

    compiler_endloop(c);

    /* Increment the counter */
    instructionindx inc=compiler_currentinstructionindex(c);

    registerindx cone = compiler_addconstant(c, node, MORPHO_INTEGER(1), false, false);
    codeinfo oneinfo = CODEINFO(CONSTANT, cone, 0);
    oneinfo = compiler_movetoregister(c, node, oneinfo, REGISTER_UNALLOCATED);
    ninstructions+=oneinfo.ninstructions;

    instructionindx add=compiler_addinstruction(c, ENCODE(OP_ADD, rcount, rcount, oneinfo.dest), node);
    ninstructions++;

    /* Compile the unconditional branch back to the test instruction */
    instructionindx end=compiler_addinstruction(c, ENCODE_LONG(OP_B, REGISTER_UNALLOCATED, -(add-tst)-2), node);
    ninstructions++;

    /* Go back and generate the condition instruction */
    compiler_setinstruction(c, condindx, ENCODE_LONG(OP_BIFF, rcond, (add-tst) ));

    compiler_fixloop(c, tst, inc, end+1);

    if (CODEINFO_ISREGISTER(method)) compiler_regfreetemp(c, method.dest);

    compiler_endscope(c);

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, ninstructions);
}

/** @brief   Compiles a do...while statement
 *  @details While statements are encoded as follows:
 *
 *           <pre>
 *                   DO
 *                  /     \
 *              body        cond
 *           </pre>
 *
 *           and are compiled to bytecode that looks like
 *
 *           <pre>
 *           start:
 *               .. body ..
 *           test   rtst, rB, rC
 *               bif    rtst, end:    ; branch if condition isn't met
 *           end:
 *           </pre>
*/
static codeinfo compiler_do(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo cond = CODEINFO_EMPTY,
             body = CODEINFO_EMPTY;
    unsigned int ninstructions=0;
    instructionindx condindx=REGISTER_UNALLOCATED; /* Where is the condition located */
    instructionindx startindx=compiler_currentinstructionindex(c);
    instructionindx nextindx=REGISTER_UNALLOCATED; /* Where should continue jump to? */

    compiler_beginloop(c);

    /* Compile the body */
    if (node->left!=SYNTAXTREE_UNCONNECTED) {
        body = compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
        ninstructions+=body.ninstructions;
        compiler_releaseoperand(c, body);
    }

    compiler_endloop(c);

    nextindx=compiler_currentinstructionindex(c);

    /* The left node is the condition; compile it already */
    if (node->right!=SYNTAXTREE_UNCONNECTED) {
        cond=compiler_nodetobytecode(c, node->right, REGISTER_UNALLOCATED);
        ninstructions+=cond.ninstructions; /* Keep track of instructions */

        /* And make sure it's in a register */
        if (!CODEINFO_ISREGISTER(cond)) {
            cond=compiler_movetoregister(c, node, cond, REGISTER_UNALLOCATED);
            ninstructions+=cond.ninstructions; /* Keep track of instructions */
        }

        /* Generate empty instruction to contain the conditional branch */
        condindx=compiler_addinstruction(c, ENCODE_LONG(OP_BIF, cond.dest, -ninstructions-1), node);
        ninstructions++;

        compiler_releaseoperand(c, cond);
    }

    instructionindx end=compiler_currentinstructionindex(c);

    compiler_fixloop(c, startindx, nextindx, end);

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, ninstructions);
}


/** @brief Compiles a break or continue statement.
 *  @details Break and continue statements are inserted as NOP instructions with the a register set to a marker.
 *  */
static codeinfo compiler_break(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo out = CODEINFO_EMPTY;

    if (compiler_inloop(c)) {
        compiler_addinstruction(c, ENCODE(OP_NOP, (node->type==NODE_BREAK ? 'b' : 'c'), 0, 0), node);
        out.ninstructions++;
    } else compiler_error(c, node, (node->type==NODE_BREAK ? COMPILE_BRKOTSDLP : COMPILE_CNTOTSDLP));

    return out;
}

/** @brief Compiles a try/catch block.
 *  @details Break and continue statements are inserted as NOP instructions with the a register set to a marker.
 *  */
static codeinfo compiler_try(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo out = CODEINFO_EMPTY;

    objectdictionary *cdict = object_newdictionary();
    if (!cdict) { compiler_error(c, node, ERROR_ALLOCATIONFAILED); return out; }

    registerindx cdictindx = compiler_addconstant(c, node, MORPHO_OBJECT(cdict), false, false);

    compiler_addinstruction(c, ENCODE_LONG(OP_PUSHERR, 0, cdictindx), node);
    out.ninstructions++;

    debug_pusherr(&c->out->annotations, cdict);

    /* Compile the body */
    if (node->left!=SYNTAXTREE_UNCONNECTED) {
        codeinfo body = compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
        out.ninstructions+=body.ninstructions;
        compiler_releaseoperand(c, body);
    }

    instructionindx popindx = compiler_addinstruction(c, ENCODE_BYTE(OP_NOP), node);
    out.ninstructions++;

    /* Compile the catch dictionary */
    varray_syntaxtreeindx switchnodes;
    varray_syntaxtreeindx labelnodes;
    varray_syntaxtreeindxinit(&switchnodes);
    varray_syntaxtreeindxinit(&labelnodes);

    syntaxtreenodetype match[] = { NODE_DICTIONARY };
    syntaxtree_flatten(compiler_getsyntaxtree(c), node->right, 1, match, &switchnodes);

    for (unsigned int i=0; i<switchnodes.count; i++) {
        syntaxtreenode *entry = compiler_getnode(c, switchnodes.data[i]);
        instructionindx entryindx = compiler_currentinstructionindex(c);

        syntaxtreenode *body=compiler_getnode(c, entry->right);

        if (body) {
            codeinfo entrybody = compiler_nodetobytecode(c, entry->right, REGISTER_UNALLOCATED);
            out.ninstructions+=entrybody.ninstructions;
            compiler_releaseoperand(c, entrybody);
        }

        // Add a break instruction after each entry body except for the last
        if (i!=switchnodes.count-1) {
            compiler_addinstruction(c, ENCODE(OP_NOP, 'b', 0, 0), node);
            out.ninstructions++;
        }

        /* Now flatten the label nodes */
        labelnodes.count=0;
        syntaxtreenodetype labelmatch[] = { NODE_SEQUENCE };
        syntaxtree_flatten(compiler_getsyntaxtree(c), entry->left, 1, labelmatch, &labelnodes);

        for (unsigned int j=0; j<labelnodes.count; j++) {
            syntaxtreenode *label=compiler_getnode(c, labelnodes.data[j]);

            if (label->type!=NODE_STRING) {
                compiler_error(c, label, COMPILE_INVLDLBL);
                break;
            }

            registerindx labelsymbol=compiler_addsymbol(c, entry, label->content);
            value symbolkey = compiler_getconstant(c, labelsymbol);

            dictionary_insert(&cdict->dict, symbolkey, MORPHO_INTEGER(entryindx));
        }
    }

    instructionindx endindx = compiler_currentinstructionindex(c);

    /* Fix the poperr instruction that jumps around the switch block */
    compiler_setinstruction(c, popindx, ENCODE_LONG(OP_POPERR, 0, endindx-popindx-1));
    /* Fix the nop instructions in the switch block to jump to the end of block */
    compiler_fixloop(c, popindx, popindx, endindx);

    varray_syntaxtreeindxclear(&switchnodes);
    varray_syntaxtreeindxclear(&labelnodes);

    debug_poperr(&c->out->annotations);

    return out;
}

/** @brief   Compiles logical operators */
static codeinfo compiler_logical(compiler *c, syntaxtreenode *node, registerindx reqout) {
    /* An AND operator must branch if the first operand is false,
       an OR  operator must branch if the first operator is true */
    bool biffflag = (node->type==NODE_AND ? true : false); // Generate a BIFF instruction

    registerindx out = compiler_regtemp(c, reqout);
    instructionindx condindx=0; /* Where is the condition located */
    unsigned int linstructions=0, rinstructions=0; /* Size of code for both operands */

    /* Generate code to get the left hand expression */
    codeinfo left = compiler_nodetobytecode(c, node->left, out);
    linstructions+=left.ninstructions;
    if (!(CODEINFO_ISREGISTER(left) && left.dest==out)) {
        compiler_releaseoperand(c, left);
        compiler_regtempwithindx(c, out);
        left=compiler_movetoregister(c, node, left, out);
        linstructions+=left.ninstructions;
    }

    /* Generate empty instruction to contain the conditional branch */
    condindx=compiler_addinstruction(c, ENCODE_BYTE(OP_NOP), node);
    linstructions++;

    /* Now evaluate right operand */
    codeinfo right = compiler_nodetobytecode(c, node->right, out);
    rinstructions+=right.ninstructions;
    if (!(CODEINFO_ISREGISTER(right) && right.dest==out)) {
        compiler_releaseoperand(c, right);
        compiler_regtempwithindx(c, out);
        right=compiler_movetoregister(c, node, right, out);
        rinstructions+=right.ninstructions;
    }

    /* Generate the branch instruction */
    compiler_setinstruction(c, condindx, ENCODE_LONG((biffflag ? OP_BIFF : OP_BIF), out, rinstructions));

    return CODEINFO(REGISTER, out, linstructions+rinstructions);
}

/** Compile declarations */
static codeinfo compiler_declaration(compiler *c, syntaxtreenode *node, registerindx reqout) {
    syntaxtreenode *varnode = compiler_getnode(c, node->left);
    syntaxtreenode *lftnode = NULL, *indxnode = NULL;
    codeinfo right;
    value var=MORPHO_NIL;
    registerindx reg;
    unsigned int ninstructions = 0;

    /* Find the symbol */
    if (varnode) {
        if (varnode->type==NODE_SYMBOL) {
            var = varnode->content;
        } else if (varnode->type==NODE_INDEX) {
            lftnode=compiler_getnode(c, varnode->left);
            if (lftnode && lftnode->type==NODE_SYMBOL) {
                indxnode=varnode;
                varnode=lftnode;
                var = varnode->content;
            } else {
                UNREACHABLE("Unexpected node type in variable declaration");
            }
        }
    }

    if (!MORPHO_ISNIL(var)) {
        /* Create the variable */
        codeinfo vloc = compiler_addvariable(c, varnode, var);
        codeinfo array = CODEINFO_EMPTY;

        if (vloc.returntype==REGISTER) {
            reg=vloc.dest; /* The variable was assigned to a register, so we can use it directly */
        } else {
            /* The variable was assigned somewhere else, so use that instead */
            reg=compiler_regtemp(c, REGISTER_UNALLOCATED);
        }

        /* If this is an array, we must create it */
        if (indxnode) {
            /* Set up a call to the Array() function */
            array=compiler_findbuiltin(c, node, ARRAY_CLASSNAME, reqout);
            ninstructions+=array.ninstructions;

            // Dimensions
            registerindx istart=REGISTER_UNALLOCATED, iend=REGISTER_UNALLOCATED;
            codeinfo indxinfo=compiler_compileindexlist(c, indxnode, &istart, &iend);
            ninstructions+=indxinfo.ninstructions;

            // Initializer
            if (node->right!=SYNTAXTREE_UNCONNECTED) {
                iend=compiler_regalloctop(c);

                right = compiler_nodetobytecode(c, node->right, iend);
                ninstructions+=right.ninstructions;

                right=compiler_movetoregister(c, node, right, iend); // Ensure in register
                ninstructions+=right.ninstructions;
            }

            // Call Array()
            compiler_addinstruction(c, ENCODE_DOUBLE(OP_CALL, array.dest, iend-istart+1), node);
            ninstructions++;

            compiler_regfreetoend(c, istart);

            if (vloc.returntype==REGISTER && array.dest!=vloc.dest) { // Move to correct register
                codeinfo move=compiler_movetoregister(c, node, array, vloc.dest);
                ninstructions+=move.ninstructions;
            } else reg=array.dest;

        } else if (node->right!=SYNTAXTREE_UNCONNECTED) { /* Not an array, but has an initializer */
            right = compiler_nodetobytecode(c, node->right, reg);
            ninstructions+=right.ninstructions;

            /* Ensure operand is in the desired register  */
            right=compiler_movetoregister(c, node, right, reg);
            ninstructions+=right.ninstructions;

            compiler_releaseoperand(c, right);
        } else { /* Otherwise, we should zero out the register */
            registerindx cnil = compiler_addconstant(c, node, MORPHO_NIL, false, false);
            compiler_addinstruction(c, ENCODE_LONG(OP_LCT, reg, cnil), node);
            ninstructions++;
        }

        if (vloc.returntype!=REGISTER) {
            codeinfo mv=compiler_movefromregister(c, node, vloc, reg);
            ninstructions+=mv.ninstructions;

            compiler_regfreetemp(c, reg);
        }
    }

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, ninstructions);
}

/** Compiles an parameter declaration */
static void compiler_functionparameters(compiler *c, syntaxtreeindx indx) {
    syntaxtreenode *node = compiler_getnode(c, indx);
    if (!node) return;

    switch(node->type) {
        case NODE_SYMBOL:
        {
            if (!compiler_hasvariadicarg(c)) {
                compiler_addlocal(c, node, node->content);
            } else compiler_error(c, node, COMPILE_VARPRMLST);
        }
            break;
        case NODE_ASSIGN:
        {
            syntaxtreenode *name=compiler_getnode(c, node->left);
            syntaxtreenode *def=compiler_getnode(c, node->right);

            if (SYNTAXTREE_ISLEAF(def->type)) {
                compiler_addoptionalarg(c, node, name->content, def->content);
            } else compiler_error(c, def, COMPILE_OPTPRMDFLT);

            break;
        }
        case NODE_ARGLIST:
        {
            compiler_functionparameters(c, node->left);
            compiler_functionparameters(c, node->right);
        }
            break;
        case NODE_RANGE:
        {
            syntaxtreenode *name=compiler_getnode(c, node->right);
            compiler_addvariadicarg(c, node, name->content);
            break;
        }
        default:
            compiler_error(c, node, COMPILE_ARGSNOTSYMBOLS);
            break;
    }
}

value _selfsymbol;

/** Compiles a function declaration */
static codeinfo compiler_function(compiler *c, syntaxtreenode *node, registerindx reqout) {
    syntaxtreeindx body=node->right; /* Function body */
    codeinfo bodyinfo = CODEINFO_EMPTY; /* Code info generated by the body */
    indx closure = REGISTER_UNALLOCATED;
    registerindx kindx=REGISTER_UNALLOCATED;
    registerindx reg=REGISTER_UNALLOCATED; /* Register where the function is stored */
    instructionindx bindx;
    unsigned int ninstructions=0;
    bool ismethod = (c->currentmethod==node);
    bool isanonymous = MORPHO_ISNIL(node->content);
    bool isinitializer = false;

    objectstring initlabel = MORPHO_STATICSTRING(MORPHO_INITIALIZER_METHOD);

    if (!isanonymous) isinitializer=MORPHO_ISEQUAL(MORPHO_OBJECT(&initlabel), node->content);

    /* We preface the function code with a branch;
       for now simply create a blank instruction and store the indx */
    bindx=compiler_addinstruction(c, ENCODE_BYTE(OP_NOP), node);

    objectfunction *func = object_newfunction(bindx+1, node->content, compiler_getcurrentfunction(c), 0);
    
    /* Record the class is a method */
    if (ismethod) func->klass=compiler_getcurrentclass(c);
    
    /* Add the function as a constant */
    kindx=compiler_addconstant(c, node, MORPHO_OBJECT(func), false, false);

    /* Begin the new function definition, finding whether the current function
       is a regular function or a method declaration by looking at currentmethod */
    functiontype ftype = (ismethod ? METHOD : FUNCTION);
    if (ismethod &&
        MORPHO_ISSTRING(node->content) &&
        isinitializer ) ftype=INITIALIZER;
    compiler_beginfunction(c, func, ftype);

    /* The function has a reference to itself in r0, which may be 'self' if we're in a method */
    value r0symbol=node->content;
    if (ismethod) r0symbol=_selfsymbol;
    compiler_regalloc(c, r0symbol);

    /* -- Compile the parameters -- */
    compiler_functionparameters(c, node->left);

    func->nargs=compiler_regtop(c);

    /* Check we don't have too many arguments */
    if (func->nargs>MORPHO_MAXARGS) {
        compiler_error(c, node, COMPILE_TOOMANYPARAMS);
        return CODEINFO_EMPTY;
    }

    /* -- Compile the body -- */
    if (body!=REGISTER_UNALLOCATED) bodyinfo=compiler_nodetobytecode(c, body, REGISTER_UNALLOCATED);
    ninstructions+=bodyinfo.ninstructions;

    /* Add a return instruction if necessary */
    //if (DECODE_OP(compiler_previousinstruction(c))!=OP_RETURN) { // 8/11/21 -> fix for final return in if
    if (true) {
        /* Methods automatically return self unless another argument is specified */

#ifndef MORPHO_LOXCOMPATIBILITY
        if (ismethod)
#else
        if (ismethod && isinitializer)
#endif
        {
            compiler_addinstruction(c, ENCODE_DOUBLE(OP_RETURN, 1, 0), node); /* Add a return */
        } else {
            compiler_addinstruction(c, ENCODE_BYTE(OP_RETURN), node); /* Add a return */
        }

        ninstructions++;
    }

    /* Verify if we have any outstanding forward references */
    compiler_checkoutstandingforwardreference(c);

    /* Correct the branch instruction before the function definition code */
    compiler_setinstruction(c, bindx, ENCODE_LONG(OP_B, REGISTER_UNALLOCATED, ninstructions));
    ninstructions++;

    /* Restore the old function */
    compiler_endfunction(c);

    if (!ismethod) {
        /* Generate a closure prototype if necessary */
        closure=compiler_closure(c, node, REGISTER_UNALLOCATED);

        /* Allocate a variable to refer to the function definition */
        codeinfo fvar=CODEINFO_EMPTY;
        if (isanonymous) {
            fvar.dest=compiler_regtemp(c, reqout);
            fvar.returntype=REGISTER;
        } else {
            /* Check if this resolves a forward reference */
            if (!compiler_resolveforwardreference(c, func, &fvar)) {
                fvar=compiler_addvariable(c, node, node->content);
            }
        }
        reg=fvar.dest;

        /* If it's not in a register, allocate a temporary register */
        if (fvar.returntype!=REGISTER) reg=compiler_regtemp(c, REGISTER_UNALLOCATED);

        /* Move function into register */
        compiler_addinstruction(c, ENCODE_LONG(OP_LCT, reg, kindx), node);
        ninstructions++;

        /* Wrap in a closure if necessary */
        if (closure!=REGISTER_UNALLOCATED) {
            compiler_addinstruction(c, ENCODE_DOUBLE(OP_CLOSURE, reg, (registerindx) closure), node);
            ninstructions++;
        }

        /* If the variable wasn't a local one, move to the correct place */
        if (fvar.returntype!=REGISTER) {
            codeinfo mv=compiler_movefromregister(c, node, fvar, reg);
            ninstructions+=mv.ninstructions;
            compiler_regfreetemp(c, reg);
        }
    }

    return CODEINFO(REGISTER, (isanonymous ? reg : REGISTER_UNALLOCATED), ninstructions);
}

/** Compiles a list of arguments, flattening child nodes */
static codeinfo compiler_arglist(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo arginfo;
    unsigned int ninstructions=0;

    varray_syntaxtreeindx argnodes;
    varray_syntaxtreeindxinit(&argnodes);
    /* Collapse all these types of nodes */
    syntaxtreenodetype match[] = {node->type};
    /* Flatten both left and right nodes */
    syntaxtree_flatten(compiler_getsyntaxtree(c), node->left, 1, match, &argnodes);
    syntaxtree_flatten(compiler_getsyntaxtree(c), node->right, 1, match, &argnodes);

    for (unsigned int i=0; i<argnodes.count; i++) {
        /* Claim a new register */
        registerindx reg=compiler_regalloctop(c);

        syntaxtreenode *arg = compiler_getnode(c, argnodes.data[i]);
        if (arg->type==NODE_ASSIGN) {
            syntaxtreenode *symbol = compiler_getnode(c, arg->left);

            /* Intern the symbol and add to the constant table */
            value s=program_internsymbol(c->out, symbol->content);
            registerindx sym=compiler_addconstant(c, arg, s, true, false);
            /* For the call, move the interned symbol to a register */
            codeinfo info=CODEINFO(CONSTANT, sym, 0);
            info=compiler_movetoregister(c, arg, info, reg);
            ninstructions+=info.ninstructions;
            reg=compiler_regalloctop(c);

            arginfo=compiler_nodetobytecode(c, arg->right, reg);
        } else {
            arginfo=compiler_nodetobytecode(c, argnodes.data[i], reg);
        }
        ninstructions+=arginfo.ninstructions;

        /* If the child node didn't put it in the right place, move to the register */
        if (!(CODEINFO_ISREGISTER(arginfo) && arginfo.dest==reg)) {
            compiler_releaseoperand(c, arginfo);
            compiler_regtempwithindx(c, reg);
            arginfo=compiler_movetoregister(c, node, arginfo, reg);
            ninstructions+=arginfo.ninstructions;
        }

        if (!compiler_haserror(c) && compiler_regtop(c)!=reg) {
            compiler_regshow(c);
            UNREACHABLE("Incorrectly freed registers in compiling argument list.");
        }
    }

    varray_syntaxtreeindxclear(&argnodes);

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, ninstructions);
}

/** Is this a method invocation? */
static bool compiler_isinvocation(compiler *c, syntaxtreenode *call) {
    bool isinvocation=false;
    syntaxtreenode *selector, *method;
    /* Get the selector node */
    selector=compiler_getnode(c, call->left);
    if (selector->type==NODE_DOT) {
        /* Check that the method is a symbol */
        method=compiler_getnode(c, selector->right);
        if (method->type==NODE_SYMBOL) isinvocation=true;
    }
    return isinvocation;
}

/** Compiles a function call */
static codeinfo compiler_call(compiler *c, syntaxtreenode *node, registerindx reqout) {
    unsigned int ninstructions=0;
    if (compiler_haserror(c)) return CODEINFO_EMPTY;

    if (compiler_isinvocation(c, node)) {
        return compiler_invoke(c, node, reqout);
    }
    registerindx top=compiler_regtop(c);

    compiler_beginargs(c);

    /* Compile the function selector */
    syntaxtreenode *selnode=compiler_getnode(c, node->left);
    codeinfo func = compiler_nodetobytecode(c, node->left, (reqout<top ? REGISTER_UNALLOCATED : reqout));

    // Detect possible forward reference
    if (selnode->type==NODE_SYMBOL && compiler_catch(c, COMPILE_SYMBOLNOTDEFINED)) {
        syntaxtreenode *symbol=compiler_getnode(c, node->left);
        func=compiler_addforwardreference(c, symbol, symbol->content);
    }
    ninstructions+=func.ninstructions;

    /* Move selector into a temporary register unless we already have one
       that's at the top of the stack */
    if (!compiler_iscodeinfotop(c, func)) {
        registerindx otop = compiler_regalloctop(c);
        func=compiler_movetoregister(c, node, func, otop);
        ninstructions+=func.ninstructions;
    }

    /* Compile the arguments */
    codeinfo args = CODEINFO_EMPTY;
    if (node->right!=SYNTAXTREE_UNCONNECTED) args=compiler_nodetobytecode(c, node->right, REGISTER_UNALLOCATED);
    ninstructions+=args.ninstructions;

    /* Remember the last argument */
    registerindx lastarg=compiler_regtop(c);

    /* Check we don't have too many arguments */
    if (lastarg-func.dest>MORPHO_MAXARGS) {
        compiler_error(c, node, COMPILE_TOOMANYARGS);
        return CODEINFO_EMPTY;
    }

    compiler_endargs(c);

    /* Generate the call instruction */
    compiler_addinstruction(c, ENCODE_DOUBLE(OP_CALL, func.dest, lastarg-func.dest), node);
    ninstructions++;

    /* Free all the registers used for the call */
    compiler_regfreetoend(c, func.dest+1);

    /* Move the result to the requested register */
    if (reqout!=REGISTER_UNALLOCATED && func.dest!=reqout) {
        compiler_addinstruction(c, ENCODE_DOUBLE(OP_MOV, reqout, func.dest), node);
        ninstructions++;
        compiler_regfreetemp(c, func.dest);
        func.dest=reqout;
    }

    return CODEINFO(REGISTER, func.dest, ninstructions);
}

#include <stdint.h>

/** Compiles a method invocation */
static codeinfo compiler_invoke(compiler *c, syntaxtreenode *node, registerindx reqout) {
    unsigned int ninstructions=0;
    codeinfo object;

    /* Get the selector node */
    syntaxtreenode *selector=compiler_getnode(c, node->left);

    compiler_beginargs(c);

    syntaxtreenode *methodnode=compiler_getnode(c, selector->right);
    codeinfo method=compiler_addsymbolwithsizecheck(c, methodnode, methodnode->content);
    ninstructions+=method.ninstructions;

    registerindx top=compiler_regtop(c);

    /* Fetch the object */
    object=compiler_nodetobytecode(c, selector->left, (reqout<top ? REGISTER_UNALLOCATED : reqout));
    ninstructions+=object.ninstructions;

    /* Move object into a temporary register unless we already have one
       that's at the top of the stack */
    if (!compiler_iscodeinfotop(c, object)) {
        registerindx otop = compiler_regalloctop(c);
        object=compiler_movetoregister(c, node, object, otop);
        ninstructions+=object.ninstructions;
    }

    /* Compile the arguments */
    codeinfo args = CODEINFO_EMPTY;
    if (node->right!=SYNTAXTREE_UNCONNECTED) args=compiler_nodetobytecode(c, node->right, REGISTER_UNALLOCATED);
    ninstructions+=args.ninstructions;

    /* Remember the last argument */
    registerindx lastarg=compiler_regtop(c);

    /* Check we don't have too many arguments */
    if (lastarg-object.dest>MORPHO_MAXARGS) {
        compiler_error(c, node, COMPILE_TOOMANYARGS);
        return CODEINFO_EMPTY;
    }

    compiler_endargs(c);

    /* Generate the call instruction */
    compiler_addinstruction(c, ENCODE(OP_INVOKE, object.dest, method.dest, lastarg-object.dest), node);
    ninstructions++;

    /* Free all the registers used for the call */
    compiler_regfreetoend(c, object.dest+1);

    /* Move the result to the requested register */
    if (reqout!=REGISTER_UNALLOCATED && object.dest!=reqout) {
        compiler_addinstruction(c, ENCODE_DOUBLE(OP_MOV, reqout, object.dest), node);
        ninstructions++;
        compiler_regfreetemp(c, object.dest);
        object.dest=reqout;
    }

    if (CODEINFO_ISREGISTER(method)) compiler_regfreetemp(c, method.dest);

    return CODEINFO(REGISTER, object.dest, ninstructions);
}

/** Compile a return statement */
static codeinfo compiler_return(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo left = CODEINFO_EMPTY;
    unsigned int ninstructions=0;
    bool isinitializer = compiler_ininitializer(c);

    if (node->left!=SYNTAXTREE_UNCONNECTED) {
        if (isinitializer) {
            compiler_error(c, node, COMPILE_RETURNININITIALIZER);
        } else {
            left=compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
            ninstructions=left.ninstructions;

            /* Ensure we're working with a register */
            if (!(CODEINFO_ISREGISTER(left))) {
                left=compiler_movetoregister(c, node, left, REGISTER_UNALLOCATED);
                ninstructions+=left.ninstructions;
            }

            compiler_addinstruction(c, ENCODE_DOUBLE(OP_RETURN, 1,  left.dest), node);
            ninstructions++;
        }
        compiler_releaseoperand(c, left);
    } else {
        /* Methods return self unless a return value is specified */
        if (compiler_getcurrentclass(c)) {
            compiler_addinstruction(c, ENCODE_DOUBLE(OP_RETURN, 1, 0), node); /* Add a return */
        } else {
            compiler_addinstruction(c, ENCODE_DOUBLE(OP_RETURN, 0, 0), node);
        }
        ninstructions++;
    }

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, ninstructions);
}

/** Compiles a list of method declarations. */
static codeinfo compiler_method(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo out;
    unsigned int ninstructions=0;
    objectclass *klass=compiler_getcurrentclass(c);

    switch (node->type) {
        case NODE_FUNCTION:
            {
                /* Store the current method so that compiler_function can recognize that
                   it is in a method definition */
                c->currentmethod=node;

                /* Compile the method declaration */
                out=compiler_function(c, node, reqout);
                ninstructions+=out.ninstructions;

                /* Insert the compiled function into the method dictionary, making sure the method name is interned */
                objectfunction *method = compiler_getpreviousfunction(c);
                if (method) {
                    value symbol = program_internsymbol(c->out, node->content);
                    dictionary_insert(&klass->methods, symbol, MORPHO_OBJECT(method));
                }
            }
            break;
        case NODE_SEQUENCE:
            {
                syntaxtreenode *child=NULL;
                if (node->left!=SYNTAXTREE_UNCONNECTED) {
                    child=compiler_getnode(c, node->left);
                    out=compiler_method(c, child, reqout);
                    ninstructions+=out.ninstructions;
                }
                if (node->right!=SYNTAXTREE_UNCONNECTED) {
                    child=compiler_getnode(c, node->right);
                    out=compiler_method(c, child, reqout);
                    ninstructions+=out.ninstructions;
                }
            }
            break;
        default:
            UNREACHABLE("Incorrect node type found in class declaration");
            break;
    }

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, ninstructions);
}

/** Finds a class in the constant table of a function */
objectclass *compiler_findclass(objectfunction *f, value name) {
    /* Search the constant table */
    for (unsigned int i=0; i<f->konst.count; i++) {
        if (MORPHO_ISCLASS(f->konst.data[i])) {
            objectclass *k = MORPHO_GETCLASS(f->konst.data[i]);

            if (morpho_comparevalue(k->name, name)==0) {
                return k;
            }
        }
    }

    /* If we don't find it, try searching the parent */
    if (f->parent!=NULL) return compiler_findclass(f->parent, name);

    return NULL;
}


/** Compiles a class declaration */
static codeinfo compiler_class(compiler *c, syntaxtreenode *node, registerindx reqout) {
    unsigned int ninstructions=0;
    registerindx kindx;
    codeinfo mout;

    if (compiler_getcurrentclass(c)) {
        compiler_error(c, node, COMPILE_NSTDCLSS);
        return CODEINFO_EMPTY;
    }

    objectclass *klass=object_newclass(node->content);
    compiler_beginclass(c, klass);

    /* Store the object class as a constant */
    kindx=compiler_addconstant(c, node, MORPHO_OBJECT(klass), false, false);
    
    /* Is there a superclass and/or mixins? */
    if (node->left!=SYNTAXTREE_UNCONNECTED) {
        syntaxtreenodetype dictentrytype[] = { NODE_SEQUENCE };
        varray_syntaxtreeindx entries;
        varray_syntaxtreeindxinit(&entries);
        
        syntaxtree_flatten(compiler_getsyntaxtree(c), node->left, 1, dictentrytype, &entries);
        
        for (int i=entries.count-1; i>=0; i--) { // Loop over super and mixins in reverse order
                                                 // As super will be LAST in this list
            syntaxtreenode *snode = syntaxtree_nodefromindx(compiler_getsyntaxtree(c), entries.data[i]) ;
            
            if (snode->type==NODE_SYMBOL) {
                objectclass *superclass=compiler_findclass(c->out->global, snode->content);
                
                if (superclass) {
                    if (superclass!=klass) {
                        if (!klass->superclass) klass->superclass=superclass; // Only the first class is the super class, all others are mixins.
                        dictionary_copy(&superclass->methods, &klass->methods);
                    } else {
                        compiler_error(c, snode, COMPILE_CLASSINHERITSELF);
                    }
                } else {
                    compiler_error(c, snode, COMPILE_SUPERCLASSNOTFOUND, MORPHO_GETCSTRING( snode->content));
                }
            } else {
                UNREACHABLE("Superclass node should be a symbol.");
            }
        }
        
        varray_syntaxtreeindxclear(&entries);
        
        /*if (snode->type==NODE_SYMBOL) {
            objectclass *superclass=compiler_findclass(c->out->global, snode->content);

            if (superclass) {
                if (superclass!=klass) {
                    klass->superclass=superclass;
                    dictionary_copy(&superclass->methods, &klass->methods);
                } else {
                    compiler_error(c, snode, COMPILE_CLASSINHERITSELF);
                }
            } else {
                compiler_error(c, snode, COMPILE_SUPERCLASSNOTFOUND, MORPHO_GETCSTRING( snode->content));
            }
        } else {
            UNREACHABLE("Superclass node should be a symbol.");
        }*/
    } else {
        klass->superclass=baseclass;
        if (baseclass) dictionary_copy(&baseclass->methods, &klass->methods);
    }

    /* Compile method declarations */
    if (node->right!=SYNTAXTREE_UNCONNECTED) {
        syntaxtreenode *child = compiler_getnode(c, node->right);
        mout=compiler_method(c, child, reqout);
        ninstructions+=mout.ninstructions;
    }

    /* End class definition */
    compiler_endclass(c);

    /* Allocate a variable to refer to the class definition */
    codeinfo cvar=compiler_addvariable(c, node, node->content);
    registerindx reg=cvar.dest;

    /* If it's not in a register, allocate a temporary register */
    if (cvar.returntype!=REGISTER) reg=compiler_regtemp(c, REGISTER_UNALLOCATED);

    /* Move function into register */
    compiler_addinstruction(c, ENCODE_LONG(OP_LCT, reg, kindx), node);
    ninstructions++;

    /* If the variable wasn't a local one, move to the correct place */
    if (cvar.returntype!=REGISTER) {
        codeinfo mv=compiler_movefromregister(c, node, cvar, reg);
        ninstructions+=mv.ninstructions;
        compiler_regfreetemp(c, reg);
    }

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, ninstructions);
}

/** Compile a reference to self */
static codeinfo compiler_self(compiler *c, syntaxtreenode *node, registerindx reqout) {
    objectclass *klass = compiler_getcurrentclass(c);

    /* If we're in a method, self is available in r0 */
    codeinfo ret=CODEINFO(REGISTER, 0, 0);

    /* Check that we're in a class definition */
    if (!klass) {
        compiler_error(c, node, COMPILE_SELFOUTSIDECLASS);
    }

    /* If we're inside a function embedded in a method, we need to capture self as an upvalue */
    functionstate *fstate = compiler_currentfunctionstate(c);
    if (fstate->type==FUNCTION) {
        ret.dest = compiler_resolveself(c);
        if (ret.dest!=REGISTER_UNALLOCATED) {
            ret.returntype=UPVALUE;
        }
    }

    return ret;
}

/** Compile a reference to super */
static codeinfo compiler_super(compiler *c, syntaxtreenode *node, registerindx reqout) {
    objectclass *klass = compiler_getcurrentclass(c);
    codeinfo ret=CODEINFO_EMPTY;

    /* Check that we're in a class definition */
    if (klass) {
        if (klass->superclass) {
           ret.returntype=CONSTANT;
           ret.dest=compiler_addconstant(c, node, MORPHO_OBJECT(klass->superclass), false, false);
        } else {
            compiler_error(c, node, COMPILE_NOSUPER);
        }
    } else {
        compiler_error(c, node, COMPILE_SUPEROUTSIDECLASS);
    }

    return ret;
}

/** Lookup a symbol */
static codeinfo compiler_symbol(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo ret=CODEINFO_EMPTY;

    /* Is it a local variable? */
    ret.dest=compiler_getlocal(c, node->content);
    if (ret.dest!=REGISTER_UNALLOCATED) return ret;

    /* Is it an upvalue? */
    ret.dest = compiler_resolveupvalue(c, node->content);
    if (ret.dest!=REGISTER_UNALLOCATED) {
        ret.returntype=UPVALUE;
        return ret;
    }

    /* Is it a global variable */
    ret.dest=compiler_getglobal(c, node->content, true);
    if (ret.dest!=REGISTER_UNALLOCATED) {
        ret.returntype=GLOBAL;
        return ret;
    }

    /* Is it a builtin function or class? */
    value binf = builtin_findfunction(node->content);
    if (MORPHO_ISNIL(binf)) binf = builtin_findclass(node->content);

    if (!MORPHO_ISNIL(binf)) {
        /* It is; so add it to the constant table */
        ret.returntype=CONSTANT;
        ret.dest=compiler_addconstant(c, node, binf, false, false);
        return ret;
    }

    /* Is it a class? */
    objectclass *klass = compiler_findclass(compiler_getcurrentfunction(c), node->content);
    if (klass) {
        /* It is; so add it to the constant table */
        ret.returntype=CONSTANT;
        ret.dest=compiler_addconstant(c, node, MORPHO_OBJECT(klass), false, false);
        return ret;
    }

    char *label = MORPHO_GETCSTRING(node->content);
    compiler_error(c, node, COMPILE_SYMBOLNOTDEFINED, label);

    return ret;
}

static codeinfo compiler_movetoproperty(compiler *c, syntaxtreenode *node, codeinfo in, syntaxtreenode *obj);

/** Assign to a symbol */
static codeinfo compiler_assign(compiler *c, syntaxtreenode *node, registerindx reqout) {
    syntaxtreenode *varnode = compiler_getnode(c, node->left);
    syntaxtreenode *indxnode = NULL;
    codeinfo ret, right=CODEINFO_EMPTY;
    value var=MORPHO_NIL;
    registerindx reg=REGISTER_UNALLOCATED, istart=0, iend=0, tmp=REGISTER_UNALLOCATED;
    enum { ASSIGN_VAR, ASSIGN_UPVALUE, ASSIGN_OBJ, ASSIGN_GLBL, ASSIGN_INDEX, ASSIGN_UPINDEX } mode=ASSIGN_VAR;
    unsigned int ninstructions = 0;

    /* Find the symbol or check if it's an object */
    if (varnode) {
        if (varnode->type==NODE_SYMBOL) {
            var = varnode->content;
        } else if (varnode->type==NODE_DOT) {
            mode=ASSIGN_OBJ; /* Or object */
        } else if (varnode->type==NODE_INDEX) {
            mode=ASSIGN_INDEX;
            indxnode=varnode;
            varnode=compiler_getnode(c, varnode->left);
            var = varnode->content;

            if (varnode->type==NODE_DOT) {
                codeinfo mv=compiler_nodetobytecode(c, indxnode->left, reg);
                ninstructions+=mv.ninstructions;
                reg=mv.dest;
            }
        }
    }

    if (!MORPHO_ISNIL(var) || mode==ASSIGN_OBJ || mode==ASSIGN_INDEX) {
        if (mode!=ASSIGN_OBJ) {
            /* Find the local variable and get the assigned register */
            if (reg==REGISTER_UNALLOCATED) reg=compiler_getlocal(c, var);

            /* Perhaps it's an upvalue? */
            if (reg==REGISTER_UNALLOCATED) {
                reg=compiler_resolveupvalue(c, var);
                if (reg!=REGISTER_UNALLOCATED) mode=(mode==ASSIGN_INDEX ? ASSIGN_UPINDEX : ASSIGN_UPVALUE);
            }

            /* .. or a global? */
            if (reg==REGISTER_UNALLOCATED) {
                reg=compiler_getglobal(c, var, true);
                if (reg!=REGISTER_UNALLOCATED) {
                    if (indxnode) {
                        /* If an indexed global, move the global into a register */
                        tmp=compiler_regalloctop(c);
                        codeinfo mv=compiler_movetoregister(c, node, CODEINFO(GLOBAL, reg, 0), tmp);
                        ninstructions+=mv.ninstructions;
                        reg=tmp;
                        mode=ASSIGN_INDEX;
                    } else mode=ASSIGN_GLBL;
                }
            }

            /* Still couldn't resolve it, so generate an error */
            if (reg==REGISTER_UNALLOCATED) {
                compiler_error(c, node, COMPILE_SYMBOLNOTDEFINED, (MORPHO_ISSTRING(var) ? MORPHO_GETCSTRING(var) : ""));
                return CODEINFO_EMPTY;
            }
        }

        if (indxnode) {
            right=compiler_compileindexlist(c, indxnode, &istart, &iend);
            ninstructions+=right.ninstructions;
        }

        /* Evaluate the rhs */
        right = compiler_nodetobytecode(c, node->right, (mode!=ASSIGN_VAR ? REGISTER_UNALLOCATED : reg));
        ninstructions+=right.ninstructions;

        switch (mode) {
            case ASSIGN_VAR:
                /* Move to a register */
                ret=compiler_movetoregister(c, node, right, reg);
                ninstructions+=ret.ninstructions;
                break;
            case ASSIGN_UPVALUE:
                ret=compiler_movetoupvalue(c, node, right, reg);
                ninstructions+=ret.ninstructions;
                break;
            case ASSIGN_OBJ:
                ret=compiler_movetoproperty(c, node, right, varnode);
                ninstructions+=ret.ninstructions;
                break;
            case ASSIGN_GLBL:
                ret=compiler_movetoglobal(c, node, right, reg);
                ninstructions+=ret.ninstructions;
                break;
            case ASSIGN_INDEX:
            {
                /* Make sure the rhs is in the register after the last index */
                if (!(CODEINFO_ISREGISTER(right) && right.dest==iend+1)) {
                    registerindx last=compiler_regtempwithindx(c, iend+1);
                    compiler_releaseoperand(c, right);
                    right=compiler_movetoregister(c, node, right, last);
                    ninstructions+=right.ninstructions;
                }
                compiler_regfreetoend(c, right.dest+1);
                if (right.dest!=iend+1) {
                    UNREACHABLE("Failed register allocation in compiling SIX instruction.");
                }
                compiler_addinstruction(c, ENCODE(OP_SIX, reg, istart, right.dest), node);
                ninstructions++;
            }
                break;
            case ASSIGN_UPINDEX:
                UNREACHABLE("Assign to indexed upvalue not implemented.");
                break;
        }
    } else {
        compiler_error(c, node, COMPILE_INVALIDASSIGNMENT);
        return CODEINFO_EMPTY;
    }

    if (tmp!=REGISTER_UNALLOCATED) compiler_regfreetemp(c, tmp);

    /* Make sure the correct number of instructions is returned */
    ret=right;
    ret.ninstructions=ninstructions;

    return ret;
}

/* Compiles property lookup */
static codeinfo compiler_property(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo left = CODEINFO_EMPTY, prop = CODEINFO_EMPTY;
    registerindx out = compiler_regtemp(c, reqout);

    /* The left hand side should evaluate to the object in question */
    left = compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
    unsigned int ninstructions=left.ninstructions;

    if (!(CODEINFO_ISREGISTER(left))) {
        /* Ensure we're working with a register */
        left=compiler_movetoregister(c, node, left, REGISTER_UNALLOCATED);
        ninstructions+=left.ninstructions;
    }

    /* The right hand side should be a method name */
    syntaxtreenode *selector = compiler_getnode(c, node->right);
    if (selector->type==NODE_SYMBOL) {
        prop=compiler_addsymbolwithsizecheck(c, selector, selector->content);
        ninstructions+=prop.ninstructions;
    } else {
        compiler_error(c, selector, COMPILE_PROPERTYNAMERQD);
    }

    if (out !=REGISTER_UNALLOCATED) {
        compiler_addinstruction(c, ENCODE(OP_LPR, out, left.dest, prop.dest), node);
        ninstructions++;
        compiler_releaseoperand(c, left);
        if (CODEINFO_ISREGISTER(prop)) compiler_releaseoperand(c, prop);
    }

    return CODEINFO(REGISTER, out, ninstructions);
}

/* Moves the result of a calculation to an object property */
static codeinfo compiler_movetoproperty(compiler *c, syntaxtreenode *node, codeinfo in, syntaxtreenode *obj) {
    codeinfo prop = CODEINFO_EMPTY;

    /* The left hand side of obj should evaluate to the object in question */
    codeinfo left = compiler_nodetobytecode(c, obj->left, REGISTER_UNALLOCATED);
    unsigned int ninstructions=left.ninstructions;

    if (!(CODEINFO_ISREGISTER(left) || CODEINFO_ISSHORTCONSTANT(left))) {
        /* Ensure we're working with a register or a constant */
        left=compiler_movetoregister(c, node, left, REGISTER_UNALLOCATED);
        ninstructions+=left.ninstructions;
    }

    /* The right hand side of obj should be a property name */
    syntaxtreenode *selector = compiler_getnode(c, obj->right);
    if (selector->type==NODE_SYMBOL) {
        prop=compiler_addsymbolwithsizecheck(c, selector, selector->content);
        ninstructions+=prop.ninstructions;
    } else {
        compiler_error(c, selector, COMPILE_PROPERTYNAMERQD);
        return CODEINFO_EMPTY;
    }

    codeinfo store = in;
    if (!CODEINFO_ISREGISTER(in)) {
        /* Ensure we're working with a register */
        store=compiler_movetoregister(c, node, store, REGISTER_UNALLOCATED);
        ninstructions+=store.ninstructions;
        compiler_releaseoperand(c, store);
    }

    compiler_addinstruction(c, ENCODE(OP_SPR, left.dest, prop.dest, store.dest), node);
    ninstructions++;

    if (CODEINFO_ISREGISTER(prop)) compiler_releaseoperand(c, prop);

    return CODEINFO(CODEINFO_ISCONSTANT(in), in.dest, ninstructions);
}

/** Compiles a node to bytecode */
static codeinfo compiler_nodetobytecode(compiler *c, syntaxtreeindx indx, registerindx reqout) {
    syntaxtreenode *node = NULL;
    codeinfo ret = CODEINFO_EMPTY;

    if (indx!=SYNTAXTREE_UNCONNECTED) {
        node=compiler_getnode(c, indx);
    } else {
        UNREACHABLE("compiling an unexpectedly blank node [run with debugger]");
    }

    if (!node) return CODEINFO_EMPTY;

#ifdef MORPHO_DEBUG_DISPLAYREGISTERALLOCATION
    compiler_regshow(c);
#endif

    compiler_nodefn compilenodefn = compiler_getrule(node->type)->nodefn;

    if (compilenodefn!=NULL) {
        ret = (*compilenodefn) (c, node, reqout);
        if (CODEINFO_ISREGISTER(ret) && ret.dest!=REGISTER_UNALLOCATED &&!compiler_isregalloc(c, ret.dest)) {
            UNREACHABLE("compiler node returned an unallocated register");
        }
    } else {
        UNREACHABLE("unhandled syntax tree node type [Check bytecode compiler definition table]");
    }

    return ret;
}

/** Compiles the current syntax tree to bytecode */
static bool compiler_tobytecode(compiler *c, program *out) {
    if (c->tree.tree.count>0 && c->tree.entry!=SYNTAXTREE_UNCONNECTED) {
        codeinfo info=compiler_nodetobytecode(c, c->tree.entry, REGISTER_UNALLOCATED);
        compiler_releaseoperand(c, info);
    }
    compiler_checkoutstandingforwardreference(c);
    if (c->tree.tree.count==0) {
        compiler_addinstruction(c, ENCODE_BYTE(OP_END), NULL);
    } else if (c->tree.entry>=0) {
        compiler_addinstruction(c, ENCODE_BYTE(OP_END), syntaxtree_nodefromindx(&c->tree, c->tree.entry));
    }

    return true;
}

/* **********************************************************************
* Modules
* ********************************************************************** */

/** Strip an 'end' instruction at the end of the program */
void compiler_stripend(compiler *c) {
    program *out = c->out;
    instructionindx last = out->code.count; /* End of old code */

    if (last>0 && out->code.data[last-1] == ENCODE_BYTE(OP_END)) {
        out->code.count--;
        debug_stripend(&out->annotations);
    }
}

/** Copies the globals across from one compiler to another.
 * @param[in] src source dictionary
 * @param[in] dest destination dictionary
 * @param[in] compare (optional) a dictionary to check the contents against; globals are only copied if they also appear in compare */
void compiler_copyglobals(compiler *src, compiler *dest, dictionary *compare) {
    for (unsigned int i=0; i<src->globals.capacity; i++) {
        if (!MORPHO_ISNIL(src->globals.contents[i].key)) {
            if (compare && !dictionary_get(compare, src->globals.contents[i].key, NULL)) continue;

            value key = src->globals.contents[i].key;

            if (!dictionary_get(&dest->globals, key, NULL)) {
                key=object_clonestring(key);
            }

            dictionary_insert(&dest->globals, key, src->globals.contents[i].val);
        }
    }
}

/** Searches for a module with given name, returns the file name for inclusion. */
bool compiler_findmodule(char *name, varray_char *fname) {
    char *ext[] = { MORPHO_EXTENSION, "" };
    value out=MORPHO_NIL;
    bool success=morpho_findresource(MORPHO_MODULEDIR, name, ext, true, &out);
    
    if (success) {
        fname->count=0;
        if (MORPHO_ISSTRING(out)) {
            varray_charadd(fname, MORPHO_GETCSTRING(out), (int) MORPHO_GETSTRINGLENGTH(out));
            varray_charwrite(fname, '\0');
        }
        morpho_freeobject(out);
    }
    
    return success;
}

/** Import a module */
static codeinfo compiler_import(compiler *c, syntaxtreenode *node, registerindx reqout) {
    varray_char filename;
    syntaxtreenode *module = compiler_getnode(c, node->left);
    syntaxtreenode *qual = compiler_getnode(c, node->right);
    dictionary fordict;
    char *fname=NULL;
    unsigned int start=0, end=0;
    FILE *f = NULL;

    dictionary_init(&fordict);
    varray_charinit(&filename);

    if (compiler_checkerror(c)) return CODEINFO_EMPTY;

    if (qual) {
        if (qual->type==NODE_FOR) {
            /* Convert list of symbols following for into a dictionary */
            while (qual!=NULL) {
                syntaxtreenode *l = compiler_getnode(c, qual->right);
                if (l && l->type==NODE_SYMBOL) {
                    dictionary_insert(&fordict, l->content, MORPHO_NIL);
                } else UNREACHABLE("Import encountered non symbolic in for clause.");
                qual=compiler_getnode(c, qual->left);
            }
        } else UNREACHABLE("AS not implemented.");
    }

    if (module) {
        if (module->type==NODE_SYMBOL) {
            if (morpho_loadextension(MORPHO_GETCSTRING(module->content))) {
            } else if (compiler_findmodule(MORPHO_GETCSTRING(module->content), &filename)) {
                fname=filename.data;
            } else {
                compiler_error(c, module, COMPILE_MODULENOTFOUND, MORPHO_GETCSTRING(module->content));
            }
        } else if (module->type==NODE_STRING) {
            fname=MORPHO_GETCSTRING(module->content);
        }

        compiler *root = c;
        while (root->parent!=NULL) root=root->parent;

        if (fname) {
            objectstring chkmodname = MORPHO_STATICSTRING(fname);
            if (dictionary_get(&root->modules, MORPHO_OBJECT(&chkmodname), NULL)) {
                goto compiler_import_cleanup;
            }
        }

        if (fname) f=file_openrelative(fname, "r");
        else goto compiler_import_cleanup;

        if (f) {
            value modname=object_stringfromcstring(fname, strlen(fname));
            dictionary_insert(&root->modules, modname, MORPHO_NIL);

            /* Read in source */
            varray_char src;
            varray_charinit(&src);
            if (!file_readintovarray(f, &src)) {
                compiler_error(c, module, COMPILE_IMPORTFLD, fname);
                goto compiler_import_cleanup;
            }

            /* Remember the initial position of the code */
            start=c->out->code.count;

            /* Set up the compiler */
            compiler cc;
            compiler_init(src.data, c->out, &cc);
            compiler_setmodule(&cc, modname);
            debug_setmodule(&c->out->annotations, modname);
            cc.parent=c; /* Ensures global variables can be found */

            morpho_compile(src.data, &cc, false, &c->err);

            if (ERROR_SUCCEEDED(c->err)) {
                compiler_stripend(c);
                compiler_copyglobals(&cc, c, (fordict.count>0 ? &fordict : NULL));
            } else {
                c->err.module = MORPHO_GETCSTRING(modname);
            }
            
            debug_setmodule(&c->out->annotations, compiler_getmodule(c));
            
            compiler_clear(&cc);

            end=c->out->code.count;

            varray_charclear(&src);
        } else compiler_error(c, module, COMPILE_FILENOTFOUND, fname);
    }

compiler_import_cleanup:
    if (f) fclose(f);
    varray_charclear(&filename);
    dictionary_clear(&fordict);

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, end-start);
}

/** Compile a breakpoint */
static codeinfo compiler_breakpoint(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo info = CODEINFO_EMPTY;

    compiler_addinstruction(c, ENCODE_BYTE(OP_BREAK), node);

    if (node->left!=SYNTAXTREE_UNCONNECTED) {
        info=compiler_nodetobytecode(c, node->left, reqout);
    }

    info.ninstructions++;

    return info;
}

/* **********************************************************************
* Compiler initialization/finalization
* ********************************************************************** */

/** @brief Initialize a compiler
 *  @param[in]  source  source code to compile
 *  @param[in]  out     destination program to compile to
 *  @param[out] c       compiler structure is filled out */
void compiler_init(const char *source, program *out, compiler *c) {
    lex_init(&c->lex, source, 1);
    error_init(&c->err);
    syntaxtree_init(&c->tree);
    parse_init(&c->parse, &c->lex, &c->err, &c->tree);
    compiler_fstackinit(c);
    dictionary_init(&c->globals);
    dictionary_init(&c->modules);
    if (out) c->fstack[0].func=out->global; /* The global pseudofunction */
    c->out = out;
    c->prevfunction = NULL;
    c->currentclass = NULL;
    c->currentmethod = NULL;
    c->currentmodule = MORPHO_NIL;
    c->parent = NULL;
}

/** @brief Clear attached data structures from a compiler
 *  @param[in]  c        compiler to clear */
void compiler_clear(compiler *c) {
    compiler_fstackclear(c);
    syntaxtree_clear(&c->tree);
    dictionary_freecontents(&c->globals, true, false);
    dictionary_clear(&c->globals);
    dictionary_freecontents(&c->modules, true, false);
    dictionary_clear(&c->modules);
}

/* **********************************************************************
* Interfaces
* ********************************************************************** */

/** Interface to the compiler
 * @param[in]  in    A string to compile
 * @param[in]  c     The compiler
 * @param[in]  opt   Whether or not to invoke the optimizer
 * @param[out] err   Pointer to error block on failure
 * @returns    A bool indicating success or failure
 */
bool morpho_compile(char *in, compiler *c, bool opt, error *err) {
    program *out = c->out;
    bool success = false;

    error_clear(&c->err);
    error_clear(err);

    /* Remove any previous END instruction */
    compiler_stripend(c);
    instructionindx last = out->code.count; /* End of old code */

    /* Initialize lexer */
    lex_init(&c->lex, in, 1); /* Count lines from 1. */

    parse(&c->parse);
    if (!ERROR_SUCCEEDED(c->err)) {
        *err = c->err;
    } else {
#ifdef MORPHO_DEBUG_DISPLAYSYNTAXTREE
        syntaxtree_print(&c->tree);
#endif
#ifdef MORPHO_DEBUG_FILLGLOBALCONSTANTTABLE
        if (out->global->konst.count<255) {
            for (unsigned int i=0; i<256; i++) compiler_addconstant(c, NULL, MORPHO_INTEGER(i), false, false);
        }
#endif

        compiler_tobytecode(c, out);
        if (ERROR_SUCCEEDED(c->err)) {
            compiler_setfunctionregistercount(c);
            program_setentry(out, last);
            success=true;
        } else {
            *err = c->err;
        }
    }

    if (success && opt) {
        optimize(c->out);
    }

    return success;
}

/* **********************************************************************
 * Public interfaces
 * ********************************************************************** */

/** Creates a new compiler */
compiler *morpho_newcompiler(program *out) {
    compiler *new = MORPHO_MALLOC(sizeof(compiler));

    if (new) {
        compiler_init("", out, new);
    }

    return new;
}

/** Frees a compiler */
void morpho_freecompiler(compiler *c) {
    compiler_clear(c);
    MORPHO_FREE(c);
}

/* **********************************************************************
* Initialization/Finalization
* ********************************************************************** */

void morpho_setbaseclass(value klss) {
    if (MORPHO_ISCLASS(klss)) {
        baseclass=MORPHO_GETCLASS(klss);
    }
}

/** Initializes the compiler */
void compile_initialize(void) {
    _selfsymbol=builtin_internsymbolascstring("self");

    /* Lex errors */
    morpho_defineerror(COMPILE_UNTERMINATEDCOMMENT, ERROR_LEX, COMPILE_UNTERMINATEDCOMMENT_MSG);
    morpho_defineerror(COMPILE_UNTERMINATEDSTRING, ERROR_LEX, COMPILE_UNTERMINATEDSTRING_MSG);

    /* Parse errors */
    morpho_defineerror(COMPILE_INCOMPLETEEXPRESSION, ERROR_PARSE, COMPILE_INCOMPLETEEXPRESSION_MSG);
    morpho_defineerror(COMPILE_MISSINGPARENTHESIS, ERROR_PARSE, COMPILE_MISSINGPARENTHESIS_MSG);
    morpho_defineerror(COMPILE_EXPECTEXPRESSION, ERROR_PARSE, COMPILE_EXPECTEXPRESSION_MSG);
    morpho_defineerror(COMPILE_MISSINGSEMICOLON, ERROR_PARSE, COMPILE_MISSINGSEMICOLON_MSG);
    morpho_defineerror(COMPILE_MISSINGSEMICOLONEXP, ERROR_PARSE, COMPILE_MISSINGSEMICOLONEXP_MSG);
    morpho_defineerror(COMPILE_MISSINGSEMICOLONVAR, ERROR_PARSE, COMPILE_MISSINGSEMICOLONVAR_MSG);
    morpho_defineerror(COMPILE_VAREXPECTED, ERROR_PARSE, COMPILE_VAREXPECTED_MSG);
    morpho_defineerror(COMPILE_BLOCKTERMINATOREXP, ERROR_PARSE, COMPILE_BLOCKTERMINATOREXP_MSG);
    morpho_defineerror(COMPILE_IFLFTPARENMISSING, ERROR_PARSE, COMPILE_IFLFTPARENMISSING_MSG);
    morpho_defineerror(COMPILE_IFRGHTPARENMISSING, ERROR_PARSE, COMPILE_IFRGHTPARENMISSING_MSG);
    morpho_defineerror(COMPILE_WHILELFTPARENMISSING, ERROR_PARSE, COMPILE_WHILELFTPARENMISSING_MSG);
    morpho_defineerror(COMPILE_FORLFTPARENMISSING, ERROR_PARSE, COMPILE_FORLFTPARENMISSING_MSG);
    morpho_defineerror(COMPILE_FORSEMICOLONMISSING, ERROR_PARSE, COMPILE_FORSEMICOLONMISSING_MSG);
    morpho_defineerror(COMPILE_FORRGHTPARENMISSING, ERROR_PARSE, COMPILE_FORRGHTPARENMISSING_MSG);
    morpho_defineerror(COMPILE_FNNAMEMISSING, ERROR_PARSE, COMPILE_FNNAMEMISSING_MSG);
    morpho_defineerror(COMPILE_FNLEFTPARENMISSING, ERROR_PARSE, COMPILE_FNLEFTPARENMISSING_MSG);
    morpho_defineerror(COMPILE_FNRGHTPARENMISSING, ERROR_PARSE, COMPILE_FNRGHTPARENMISSING_MSG);
    morpho_defineerror(COMPILE_FNLEFTCURLYMISSING, ERROR_PARSE, COMPILE_FNLEFTCURLYMISSING_MSG);
    morpho_defineerror(COMPILE_CALLRGHTPARENMISSING, ERROR_PARSE, COMPILE_CALLRGHTPARENMISSING_MSG);
    morpho_defineerror(COMPILE_EXPECTCLASSNAME, ERROR_PARSE, COMPILE_EXPECTCLASSNAME_MSG);
    morpho_defineerror(COMPILE_CLASSLEFTCURLYMISSING, ERROR_PARSE, COMPILE_CLASSLEFTCURLYMISSING_MSG);
    morpho_defineerror(COMPILE_CLASSRGHTCURLYMISSING, ERROR_PARSE, COMPILE_CLASSRGHTCURLYMISSING_MSG);
    morpho_defineerror(COMPILE_EXPECTDOTAFTERSUPER, ERROR_PARSE, COMPILE_EXPECTDOTAFTERSUPER_MSG);
    morpho_defineerror(COMPILE_INCOMPLETESTRINGINT, ERROR_PARSE, COMPILE_INCOMPLETESTRINGINT_MSG);
    morpho_defineerror(COMPILE_VARBLANKINDEX, ERROR_COMPILE, COMPILE_VARBLANKINDEX_MSG);
    morpho_defineerror(COMPILE_IMPORTMISSINGNAME, ERROR_PARSE, COMPILE_IMPORTMISSINGNAME_MSG);
    morpho_defineerror(COMPILE_IMPORTUNEXPCTDTOK, ERROR_PARSE, COMPILE_IMPORTUNEXPCTDTOK_MSG);
    morpho_defineerror(COMPILE_IMPORTASSYMBL, ERROR_PARSE, COMPILE_IMPORTASSYMBL_MSG);
    morpho_defineerror(COMPILE_IMPORTFORSYMBL, ERROR_PARSE, COMPILE_IMPORTFORSYMBL_MSG);

    morpho_defineerror(PARSE_UNRECGNZEDTOK, ERROR_PARSE, PARSE_UNRECGNZEDTOK_MSG);
    morpho_defineerror(PARSE_DCTSPRTR, ERROR_PARSE, PARSE_DCTSPRTR_MSG);
    morpho_defineerror(PARSE_SWTCHSPRTR, ERROR_PARSE, PARSE_SWTCHSPRTR_MSG);
    morpho_defineerror(PARSE_DCTENTRYSPRTR, ERROR_PARSE, PARSE_DCTENTRYSPRTR_MSG);
    morpho_defineerror(PARSE_EXPCTWHL, ERROR_PARSE, PARSE_EXPCTWHL_MSG);
    morpho_defineerror(PARSE_EXPCTCTCH, ERROR_PARSE, PARSE_EXPCTCTCH_MSG);
    morpho_defineerror(PARSE_ONEVARPR, ERROR_PARSE, PARSE_ONEVARPR_MSG);
    morpho_defineerror(PARSE_CATCHLEFTCURLYMISSING, ERROR_PARSE, PARSE_CATCHLEFTCURLYMISSING_MSG);

    /* Compile errors */
    morpho_defineerror(COMPILE_SYMBOLNOTDEFINED, ERROR_COMPILE, COMPILE_SYMBOLNOTDEFINED_MSG);
    morpho_defineerror(COMPILE_TOOMANYCONSTANTS, ERROR_COMPILE, COMPILE_TOOMANYCONSTANTS_MSG);
    morpho_defineerror(COMPILE_ARGSNOTSYMBOLS, ERROR_COMPILE, COMPILE_ARGSNOTSYMBOLS_MSG);
    morpho_defineerror(COMPILE_PROPERTYNAMERQD, ERROR_COMPILE, COMPILE_PROPERTYNAMERQD_MSG);
    morpho_defineerror(COMPILE_SELFOUTSIDECLASS, ERROR_COMPILE, COMPILE_SELFOUTSIDECLASS_MSG);
    morpho_defineerror(COMPILE_RETURNININITIALIZER, ERROR_COMPILE, COMPILE_RETURNININITIALIZER_MSG);
    morpho_defineerror(COMPILE_EXPECTSUPER, ERROR_COMPILE, COMPILE_EXPECTSUPER_MSG);
    morpho_defineerror(COMPILE_SUPERCLASSNOTFOUND, ERROR_COMPILE, COMPILE_SUPERCLASSNOTFOUND_MSG);
    morpho_defineerror(COMPILE_SUPEROUTSIDECLASS, ERROR_COMPILE, COMPILE_SUPEROUTSIDECLASS_MSG);
    morpho_defineerror(COMPILE_NOSUPER, ERROR_COMPILE, COMPILE_NOSUPER_MSG);
    morpho_defineerror(COMPILE_INVALIDASSIGNMENT, ERROR_COMPILE, COMPILE_INVALIDASSIGNMENT_MSG);
    morpho_defineerror(COMPILE_CLASSINHERITSELF, ERROR_COMPILE, COMPILE_CLASSINHERITSELF_MSG);
    morpho_defineerror(COMPILE_TOOMANYARGS, ERROR_COMPILE, COMPILE_TOOMANYARGS_MSG);
    morpho_defineerror(COMPILE_TOOMANYPARAMS, ERROR_COMPILE, COMPILE_TOOMANYPARAMS_MSG);
    morpho_defineerror(COMPILE_ISOLATEDSUPER, ERROR_COMPILE, COMPILE_ISOLATEDSUPER_MSG);
    morpho_defineerror(COMPILE_VARALREADYDECLARED, ERROR_COMPILE, COMPILE_VARALREADYDECLARED_MSG);
    morpho_defineerror(COMPILE_FILENOTFOUND, ERROR_COMPILE, COMPILE_FILENOTFOUND_MSG);
    morpho_defineerror(COMPILE_MODULENOTFOUND, ERROR_COMPILE, COMPILE_MODULENOTFOUND_MSG);
    morpho_defineerror(COMPILE_IMPORTFLD, ERROR_COMPILE, COMPILE_IMPORTFLD_MSG);
    morpho_defineerror(COMPILE_BRKOTSDLP, ERROR_COMPILE, COMPILE_BRKOTSDLP_MSG);
    morpho_defineerror(COMPILE_CNTOTSDLP, ERROR_COMPILE, COMPILE_CNTOTSDLP_MSG);
    morpho_defineerror(COMPILE_OPTPRMDFLT, ERROR_COMPILE, COMPILE_OPTPRMDFLT_MSG);
    morpho_defineerror(COMPILE_FORWARDREF, ERROR_COMPILE, COMPILE_FORWARDREF_MSG);
    morpho_defineerror(COMPILE_MLTVARPRMTR, ERROR_COMPILE, COMPILE_MLTVARPRMTR_MSG);
    morpho_defineerror(COMPILE_MSSNGLOOPBDY, ERROR_COMPILE, COMPILE_MSSNGLOOPBDY_MSG);
    morpho_defineerror(COMPILE_NSTDCLSS, ERROR_COMPILE, COMPILE_NSTDCLSS_MSG);
    morpho_defineerror(COMPILE_VARPRMLST, ERROR_COMPILE, COMPILE_VARPRMLST_MSG);
    morpho_defineerror(COMPILE_INVLDLBL, ERROR_COMPILE, COMPILE_INVLDLBL_MSG);
}

/** Finalizes the compiler */
void compile_finalize(void) {
    optimize_finalize();
}
