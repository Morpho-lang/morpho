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
#include "classes.h"
#include "file.h"
#include "resources.h"
#include "extensions.h"

/** Base class for instances */
static objectclass *baseclass;

static optimizerfn *optimizer;

/* **********************************************************************
* Bytecode compiler
* ********************************************************************** */

/* ------------------------------------------
 * Utility functions
 * ------------------------------------------- */

/** Checks if the compiler is in an error state */
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

    char *file = (MORPHO_ISSTRING(c->currentmodule) ? MORPHO_GETCSTRING(c->currentmodule) : NULL);
    
    va_start(args, id);
    morpho_writeerrorwithidvalist(&c->err, id, file, line, posn, args);
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
    debugannotation_addnode(&c->out->annotations, node);
    return varray_instructionwrite(&c->out->code, instr);
}

/** Gets the instruction index of the current instruction */
static instructionindx compiler_currentinstructionindex(compiler *c) {
    return (instructionindx) c->out->code.count;
}

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
    varray_functionrefinit(&state->functionref);
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
    varray_functionrefclear(&state->functionref);
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
void compiler_beginfunction(compiler *c, objectfunction *func, functiontype type) {
    c->fstackp++;
    compiler_functionstateinit(&c->fstack[c->fstackp]);
    c->fstack[c->fstackp].func=func;
    c->fstack[c->fstackp].type=type;
    debugannotation_setfunction(&c->out->annotations, func);
}

/** Sets the function register count */
void compiler_setfunctionregistercount(compiler *c) {
    functionstate *f=&c->fstack[c->fstackp];
    if (f->nreg>f->func->nregs) f->func->nregs=f->nreg;
}

/** Ends a function, decrementing the fstack pointer  */
void compiler_endfunction(compiler *c) {
    functionstate *f=&c->fstack[c->fstackp];
    c->prevfunction=f->func; /* Retain the function in case it needs to be bound as a method */
    compiler_setfunctionregistercount(c);
    compiler_functionstateclear(f);
    c->fstackp--;
    debugannotation_setfunction(&c->out->annotations, c->fstack[c->fstackp].func);
}

/** Gets the current function */
objectfunction *compiler_getcurrentfunction(compiler *c) {
    return c->fstack[c->fstackp].func;
}

/** Gets the current constant table */
varray_value *compiler_getcurrentconstanttable(compiler *c) {
    objectfunction *f = compiler_getcurrentfunction(c);
    if (!f) {
        UNREACHABLE("find current constant table [No current function defined].");
    }

    return &f->konst;
}

/** Gets constant i from the current constant table */
value compiler_getconstant(compiler *c, unsigned int i) {
    value ret = MORPHO_NIL;
    objectfunction *f = compiler_getcurrentfunction(c);
    if (f && i<f->konst.count) ret = f->konst.data[i];
    return ret;
}

/** Gets the most recently compiled function */
objectfunction *compiler_getpreviousfunction(compiler *c) {
    return c->prevfunction;
}

/* ------------------------------------------
 * Types
 * ------------------------------------------- */

value _closuretype;

/* ------------------------------------------
 * Argument declarations
 * ------------------------------------------- */

/** Begins arguments */
static inline void compiler_beginargs(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);
    if (f) {
        f->inargs=true;
    }
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
    debugannotation_setclass(&c->out->annotations, klass);
}

static inline void compiler_endclass(compiler *c) {
    /* Delink current class from list */
    objectclass *current = c->currentclass;
    c->currentclass=(objectclass *) current->obj.next;
    debugannotation_setclass(&c->out->annotations, c->currentclass);
    current->obj.next=NULL; /* as the class is no longer part of the list */
}

/** Gets the current class */
static objectclass *compiler_getcurrentclass(compiler *c) {
    return c->currentclass;
}

/** Adds an objectclass to the compilers dictionary of classes */
void compiler_addclass(compiler *c, objectclass *klass) {
    klass->uid = program_addclass(c->out, MORPHO_OBJECT(klass));
    
    dictionary_insert(&c->classes, klass->name, MORPHO_OBJECT(klass));
}

/** Finds a class in the compiler's dictionary of classes */
objectclass *compiler_findclass(compiler *c, value name) {
    value val;
    if (dictionary_get(&c->classes, name, &val) &&
        MORPHO_ISCLASS(val)) return MORPHO_GETCLASS(val);
    
    if (c->parent) return compiler_findclass(c->parent, name);
    
    return NULL;
}

/** Adds a class to a class's parent list, and also links the class into the parent's child list */
void compiler_addparent(compiler *c, objectclass *klass, objectclass *parent) {
    varray_valuewrite(&klass->parents, MORPHO_OBJECT(parent));
    varray_valuewrite(&parent->children, MORPHO_OBJECT(klass));
}

/* ------------------------------------------
 * Types
 * ------------------------------------------- */

/** Identifies a type from a label */
bool compiler_findtype(compiler *c, value label, value *out) {
    value type=MORPHO_NIL;
    
    objectclass *clss=compiler_findclass(c, label); // A class we defined
    if (clss) {
        type = MORPHO_OBJECT(clss);
    } else type = builtin_findclass(label); // Or a built in one
    
    if (!MORPHO_ISNIL(type)) *out = type;
    
    return (!MORPHO_ISNIL(type));
}

/** Identify a type from a label */
bool compiler_findtypefromcstring(compiler *c, char *label, value *out) {
    objectstring str = MORPHO_STATICSTRING(label);
    return compiler_findtype(c, MORPHO_OBJECT(&str), out);
}

/** Identifies a type from a value */
bool compiler_typefromvalue(compiler *c, value v, value *out) {
    return metafunction_typefromvalue(v, out);
}

/** Recursively searches the parents list of classes to see if the type 'match' is present */
bool compiler_findtypeinparent(compiler *c, objectclass *type, value match) {
    for (int i=0; i<type->parents.count; i++) {
        if (MORPHO_ISEQUAL(type->parents.data[i], match) ||
            compiler_findtypeinparent(c, MORPHO_GETCLASS(type->parents.data[i]), match)) return true;
    }
    return false;
}

/** Checks if type "match" matches a given type "type"  */
bool compiler_checktype(compiler *c, value type, value match) {
    if (MORPHO_ISNIL(type) || // If type is unset, we always match
        MORPHO_ISEQUAL(type, match)) return true; // Or if the types are the same
    
    // Also match if 'match' inherits from 'type'
    if (MORPHO_ISCLASS(match)) return compiler_findtypeinparent(c, MORPHO_GETCLASS(match), type);
    
    return false;
}

/** Determines the type associated with a constant */
bool compiler_getconstanttype(compiler *c, unsigned int i, value *type) {
    value val = compiler_getconstant(c, i);
    return compiler_typefromvalue(c, val, type);
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
    registeralloc r = REGISTERALLOC_EMPTY(f->scopedepth, symbol);
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
    if (!MORPHO_ISNIL(symbol)) debugannotation_setreg(&c->out->annotations, i, symbol);

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
        if (!MORPHO_ISNIL(symbol)) debugannotation_setreg(&c->out->annotations, reg, symbol);
    }
}

/** Allocates a temporary register that is guaranteed to be at the top of the stack */
static registerindx compiler_regalloctop(compiler *c) {
    functionstate *f = compiler_currentfunctionstate(c);
    registerindx i = REGISTER_UNALLOCATED;
    registeralloc r = REGISTERALLOC_EMPTY(f->scopedepth, MORPHO_NIL);

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
        registeralloc empty = REGISTERALLOC_EMPTY(f->scopedepth, MORPHO_NIL);
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
        f->registers.data[reg].isoptionalarg=false;
        f->registers.data[reg].scopedepth=0;
        if (!MORPHO_ISNIL(f->registers.data[reg].symbol)) {
            debugannotation_setreg(&c->out->annotations, reg, MORPHO_NIL);
        }
        f->registers.data[reg].symbol=MORPHO_NIL;
        f->registers.data[reg].type=MORPHO_NIL;
        f->registers.data[reg].currenttype=MORPHO_NIL;
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

/** Sets the type associated with a register */
void compiler_regsettype(compiler *c, registerindx reg, value type) {
    functionstate *f = compiler_currentfunctionstate(c);
    if (reg<0 || reg>=f->registers.count) return;
    f->registers.data[reg].type=type;
}

/** Gets the current type of a register */
bool compiler_regtype(compiler *c, registerindx reg, value *type) {
    functionstate *f = compiler_currentfunctionstate(c);
    if (reg>=f->registers.count) return false;
    *type = f->registers.data[reg].type;
    return true;
}

/** Raises a type violation error */
void compiler_typeviolation(compiler *c, syntaxtreenode *node, value type, value badtype, value symbol) {
    char *tname="(unknown)", *bname="(unknown)";
    char *sym="";

    if (MORPHO_ISCLASS(type)) tname=MORPHO_GETCSTRING(MORPHO_GETCLASS(type)->name);
    if (MORPHO_ISCLASS(badtype)) bname=MORPHO_GETCSTRING(MORPHO_GETCLASS(badtype)->name);
    if (MORPHO_ISSTRING(symbol)) sym = MORPHO_GETCSTRING(symbol);
    
    compiler_error(c, node, COMPILE_TYPEVIOLATION, bname, tname, sym);
}

/** Sets the current type of a register. Raises a type violation error if this is not compatible with the required type  */
bool compiler_regsetcurrenttype(compiler *c, syntaxtreenode *node, registerindx reg, value type) {
    functionstate *f = compiler_currentfunctionstate(c);
    if (reg>=f->registers.count) return false;
    
    if (compiler_checktype(c, f->registers.data[reg].type, type)) {
        f->registers.data[reg].currenttype=type;
        return true;
    }
    
    compiler_typeviolation(c, node, f->registers.data[reg].type, type, f->registers.data[reg].symbol);
    
    return false;
}

/** Gets the current type of a register */
bool compiler_regcurrenttype(compiler *c, registerindx reg, value *type) {
    functionstate *f = compiler_currentfunctionstate(c);
    if (reg>=f->registers.count) return false;
    *type = f->registers.data[reg].currenttype;
    return true;
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

/** @brief Sets that a register contains an optional argument */
void compiler_regsetoptionalarg(compiler *c, registerindx reg) {
    functionstate *f = compiler_currentfunctionstate(c);
    if (reg<0 || reg>=f->registers.count) return;
    f->registers.data[reg].isoptionalarg=true;
}

/** @brief Checks if a register contains an optional argument */
static bool compiler_isregoptionalarg(compiler *c, registerindx indx) {
    functionstate *f = compiler_currentfunctionstate(c);

    return (indx<f->registers.count && f->registers.data[indx].isoptionalarg);
}

/** Gets the number of args from the most recent argument specifier*/
void compiler_regcountargs(compiler *c, registerindx start, registerindx end, int *nposn, int *nopt) {
    functionstate *f = compiler_currentfunctionstate(c);
    int np=0, nop=0;
    if (f) {
        for (registerindx r=start; r<=end; r++) {
            if (compiler_isregoptionalarg(c, r)) nop++;
            else np++;
        }
        
        *nposn=np;
        *nopt=nop/2;
    }
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
                morpho_printvalue(NULL, r->symbol);
            } else {
                printf("temporary");
            }
            printf(" [%u]", r->scopedepth);
            if (r->iscaptured) printf(" (captured)");
            if (r->isoptionalarg) printf(" (optarg)");
        } else {
            printf("unallocated");
        }
        if (!MORPHO_ISNIL(r->type)) {
            printf(" [");
            morpho_printvalue(NULL, r->type);
            printf("]");
        }
        if (!MORPHO_ISNIL(r->currenttype)) {
            printf(" contains: ");
            morpho_printvalue(NULL, r->currenttype);
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

void compiler_functionreffreeatscope(compiler *c, unsigned int scope);

/** Decrements the scope counter in the current functionstate */
void compiler_endscope(compiler *c) {
    functionstate *f=compiler_currentfunctionstate(c);
    compiler_regfreeatscope(c, f->scopedepth);
    compiler_functionreffreeatscope(c, f->scopedepth);
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

                /* If the constant is an object and we cloned it, make sure it's bound to the program */
                if (clone && MORPHO_ISOBJECT(add)) {
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
    value type = MORPHO_NIL;
    codeinfo out = info;
    out.ninstructions=0;

    if (CODEINFO_ISCONSTANT(info)) {
        out.returntype=REGISTER;
        out.dest=compiler_regtemp(c, reg);
        
        if (compiler_getconstanttype(c, info.dest, &type)) {
            compiler_regsetcurrenttype(c, node, out.dest, type);
        }
        
        compiler_addinstruction(c, ENCODE_LONG(OP_LCT, out.dest, info.dest), node);
        out.ninstructions++;
    } else if (CODEINFO_ISUPVALUE(info)) {
        /* Move upvalues */
        out.dest=compiler_regtemp(c, reg);
        out.returntype=REGISTER;
        compiler_addinstruction(c, ENCODE_DOUBLE(OP_LUP, out.dest, info.dest), node);
        out.ninstructions++;
    } else if (CODEINFO_ISGLOBAL(info)) {
        /* Move globals */
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
            if (compiler_regcurrenttype(c, info.dest, &type)) compiler_regsetcurrenttype(c, node, out.dest, type);
            
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
    out = compiler_movetoregister(c, node, out, REGISTER_UNALLOCATED);
    return out;
}

/* ------------------------------------------
 * Optional and variadic args
 * ------------------------------------------- */

DEFINE_VARRAY(optionalparam, optionalparam);

/** Adds  a variadic parameter */
static inline registerindx compiler_addpositionalarg(compiler *c, syntaxtreenode *node, value symbol) {
    functionstate *f = compiler_currentfunctionstate(c);

    if (f) {
        if (!function_hasvargs(f->func)) {
            f->func->nargs++;
            value sym=program_internsymbol(c->out, symbol);
            return compiler_addlocal(c, node, sym);
        } else compiler_error(c, node, COMPILE_VARPRMLST);
    }
    
    return REGISTER_UNALLOCATED;
}

/** Adds  an optional argument */
static inline void compiler_addoptionalarg(compiler *c, syntaxtreenode *node, value symbol, value def) {
    functionstate *f = compiler_currentfunctionstate(c);

    if (f) {
        f->func->nopt++;
        
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
        if (function_hasvargs(f->func)) {
            compiler_error(c, node, COMPILE_MLTVARPRMTR);
            return;
        }

        value sym=program_internsymbol(c->out, symbol);
        registerindx reg = compiler_addlocal(c, node, sym);

        function_setvarg(f->func, reg-1);
    }
}

/* ------------------------------------------
 * Global variables
 * ------------------------------------------- */

/** Should we use global variables or registers?  */
bool compiler_checkglobal(compiler *c) {
    return ((c->fstackp==0) && (c->fstack[0].scopedepth==0));
}

/** Finds a global symbol, optionally searching successively through parent compilers */
globalindx compiler_findglobal(compiler *c, value symbol, bool recurse) {
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
globalindx compiler_addglobal(compiler *c, syntaxtreenode *node, value symbol) {
    globalindx indx=compiler_findglobal(c, symbol, false);

    if (indx==GLOBAL_UNALLOCATED) {
        indx = program_addglobal(c->out, symbol);
        value key = program_internsymbol(c->out, symbol);
        
        if (dictionary_insert(&c->globals, key, MORPHO_INTEGER(indx))) {
            debugannotation_setglobal(&c->out->annotations, indx, symbol);
        }
    }

    return indx;
}

/** Sets the type of a global variable */
void compiler_setglobaltype(compiler *c, globalindx indx, value type) {
    program_globalsettype(c->out, indx, type);
}

/** Checks if the type match satisfies the type of the global variable indx */
bool compiler_checkglobaltype(compiler *c, syntaxtreenode *node, globalindx indx, value match) {
    value type=MORPHO_NIL;
    if (!program_globaltype(c->out, indx, &type)) return false;
    
    bool success=compiler_checktype(c, type, match);
    
    if (!success) {
        value symbol=MORPHO_NIL;
        program_globalsymbol(c->out, indx, &symbol);
        compiler_typeviolation(c, node, type, match, symbol);
    }
    
    return success;
}

/** Shows all currently allocated globals */
void compiler_globalshow(compiler *c) {
    int nglobals = program_countglobals(c->out);
    printf("--Globals (%u in use)\n", nglobals);
    for (unsigned int i=0; i<nglobals; i++) {
        globalinfo *r=&c->out->globals.data[i];
        printf("g%u ",i);
        if (!MORPHO_ISNIL(r->symbol)) morpho_printvalue(NULL, r->symbol);
        if (!MORPHO_ISNIL(r->type)) {
            printf(" [");
            morpho_printvalue(NULL, r->type);
            printf("]");
        }
        printf("\n");
    }
    printf("--End globals\n");
}

/* Moves the result of a calculation to an global variable */
codeinfo compiler_movetoglobal(compiler *c, syntaxtreenode *node, codeinfo in, globalindx slot) {
    codeinfo use = in;
    codeinfo out = CODEINFO_EMPTY;
    bool tmp=false;

    if (!(CODEINFO_ISREGISTER(in))) {
        use=compiler_movetoregister(c, node, in, REGISTER_UNALLOCATED);
        out.ninstructions+=use.ninstructions;
        tmp=true;
    }

    value type=MORPHO_NIL;
    if (compiler_regcurrenttype(c, in.dest, &type)) {
        if (!compiler_checkglobaltype(c, node, slot, type)) goto compiler_movetoglobal_cleanup;
    }
    
    compiler_addinstruction(c, ENCODE_LONG(OP_SGL, use.dest, slot) , node);
    out.ninstructions++;

compiler_movetoglobal_cleanup:
    
    if (tmp) compiler_releaseoperand(c, use);
    return out;
}

codeinfo compiler_addvariable(compiler *c, syntaxtreenode *node, value symbol) {
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

/** Propagates upvalues up the functionstate stack.
    @param c    the compiler
    @param start    the initial functionstate
    @param sindx    starting index
    @returns register index at the top of the function state */
registerindx compiler_propagateupvalues(compiler *c, functionstate *start, registerindx sindx) {
    registerindx indx=sindx;
    for (functionstate *f = start; f<c->fstack+c->fstackp; f++) {
        indx=compiler_addupvalue(f, f==start, indx);
    }
    return indx;
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
    if (found) indx=compiler_propagateupvalues(c, found, indx);

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
 * Manage the functionref stack
 * ------------------------------------------- */

DEFINE_VARRAY(functionref, functionref)

/** Adds a reference to a function in the current functionstate */
int compiler_addfunctionref(compiler *c, objectfunction *func) {
    functionstate *f=compiler_currentfunctionstate(c);
    functionref ref = { .function = func, .symbol = func->name, .scopedepth = f->scopedepth};
    return varray_functionrefwrite(&f->functionref, ref);
}

/** Removes functions visible at a given scope level */
void compiler_functionreffreeatscope(compiler *c, unsigned int scope) {
    functionstate *f=compiler_currentfunctionstate(c);
    
    while (f->functionref.count>0 && f->functionref.data[f->functionref.count-1].scopedepth>=scope) f->functionref.count--;
}

void _addmatchingfunctionref(compiler *c, value symbol, value fn, value *out) {
    value in = *out;
    if (MORPHO_ISNIL(in)) {
        // If the function has a signature, will need to wrap in a metafunction
        if (MORPHO_ISFUNCTION(fn) && function_hastypedparameters(MORPHO_GETFUNCTION(fn))) {
            if (metafunction_wrap(symbol, fn, out)) {
                program_bindobject(c->out, MORPHO_GETOBJECT(*out));
            }
        } else *out=fn;
    } else if (MORPHO_ISFUNCTION(in)) {
        if (metafunction_wrap(symbol, in, out)) { metafunction_add(MORPHO_GETMETAFUNCTION(*out), fn);
            program_bindobject(c->out, MORPHO_GETOBJECT(*out));
        }
    } else if (MORPHO_ISMETAFUNCTION(in)) {
        metafunction_add(MORPHO_GETMETAFUNCTION(in), fn);
    }
}

/** Finds an existing metafunction in the current context that matches a given set of implementations */
bool compiler_findmetafunction(compiler *c, value symbol, int n, value *fns, codeinfo *out) {
    functionstate *f=compiler_currentfunctionstate(c);
    
    for (int i=0; i<f->func->konst.count; i++) {
        value v = f->func->konst.data[i];
        if (MORPHO_ISMETAFUNCTION(v) &&
            MORPHO_ISEQUAL(MORPHO_GETMETAFUNCTION(v)->name, symbol) &&
            metafunction_matchset(MORPHO_GETMETAFUNCTION(v), n, fns)) {
            *out = CODEINFO(CONSTANT, i, 0);
            return true;
        }
    }
    
    return false;
}

/** Finds a closure, looking back up the functionstate stack and propagating upvalues as necessary */
void _findclosure(compiler *c, objectfunction *closure, codeinfo *out) {
    functionstate *fc = compiler_currentfunctionstate(c);
    
    if (fc->func==closure->parent) {
        out->returntype=REGISTER;
        out->dest = (registerindx) closure->creg;
    } else {
        out->returntype=UPVALUE;
        
        for (functionstate *f=fc; f>=c->fstack; f--) {
            if (f->func==closure->parent) {
                f->registers.data[closure->creg].iscaptured=true;
                out->dest=compiler_propagateupvalues(c, f, closure->creg);
                return;
            }
        }

        UNREACHABLE("Couldn't locate parent of closure.");
    }
}

/** Compile a metafunction constructor by setting up a call to the Metafunction() constructor */
codeinfo compiler_metafunction(compiler *c, syntaxtreenode *node, int n, value *fns) {
    codeinfo out = compiler_findbuiltin(c, node, METAFUNCTION_CLASSNAME, REGISTER_UNALLOCATED);

    for (int i=0; i<n; i++) { // Loop over the implementations
        registerindx reg=compiler_regalloctop(c);
        codeinfo val = CODEINFO(CONSTANT, REGISTER_UNALLOCATED, 0);
        
        if (MORPHO_ISFUNCTION(fns[i]) && function_isclosure(MORPHO_GETFUNCTION(fns[i]))) {
            _findclosure(c, MORPHO_GETFUNCTION(fns[i]), &val);
        } else {
            val.dest = compiler_addconstant(c, node, fns[i], true, false);
        }
        
        val=compiler_movetoregister(c, node, val, reg);
        out.ninstructions+=val.ninstructions;
    }

    /* Make the function call */
    compiler_addinstruction(c, ENCODE_DOUBLE(OP_CALL, out.dest, n), node);
    out.ninstructions++;
    compiler_regfreetoend(c, out.dest+1);
    
    return out;
}

/** Checks for duplicate function refs */
static bool _checkduplicateref(varray_functionref *refs, functionref *match) {
    for (int i=0; i<refs->count; i++) {
        functionref *r = &refs->data[i];
        // Match functionrefs with the same signature, but only if they're
        // in a different scope or parent 
        if (signature_isequal(&r->function->sig, &match->function->sig) &&
            (r->function->parent!=match->function->parent ||
             r->scopedepth!=match->scopedepth)) return true;
    }
    return false;
}

/** Collects function implementations that match a given symbol */
static void _findfunctionref(compiler *c, value symbol, bool *hasclosure, varray_value *out) {
    bool closure=false;
    
    varray_functionref refs;
    varray_functionrefinit(&refs);
    
    functionstate *fc = compiler_currentfunctionstate(c);
    for (functionstate *f=fc; f>=c->fstack; f--) { // Go backwards to prioritize recent def'ns
        for (int i=f->functionref.count-1; i>=0; i--) { // Go backwards
            functionref *ref=&f->functionref.data[i];
            if (MORPHO_ISEQUAL(ref->symbol, symbol) &&
                !_checkduplicateref(&refs, ref)) {
                closure |= function_isclosure(ref->function);
                varray_functionrefadd(&refs, ref, 1);
            }
        }
    }
    
    // Return the collected implementations
    for (int i=0; i<refs.count; i++) {
        varray_valuewrite(out, MORPHO_OBJECT(refs.data[i].function));
    }
    
    varray_functionrefclear(&refs);
    
    *hasclosure=closure;
}

/** Determines whether a symbol refers to one (or more) functions. If so, returns either a single function or a metafunction as appropriate. */
bool compiler_resolvefunctionref(compiler *c, syntaxtreenode *node, value symbol, codeinfo *out) {
    value outfn=MORPHO_NIL;
    bool closure=false; // Set if one of the references contains a closure.
    
    varray_value fns;
    varray_valueinit(&fns);
    
    _findfunctionref(c, symbol, &closure, &fns);
 
    if (!fns.count) return false; // No need to clear an empty varray
    
    if (closure) {
        if (fns.count>1) { // If the list contains a closure, we must build the MF at runtime
            *out = compiler_metafunction(c, node, fns.count, fns.data);
        } else { // If just one closure, no need to build a metafunction
            _findclosure(c, MORPHO_GETFUNCTION(fns.data[0]), out);
            out->ninstructions=0;
        }
    } else if (!compiler_findmetafunction(c, symbol, fns.count, fns.data, out)) {
        // If a suitable MF doesn't exist in the constant table, we should build one
        for (int i=0; i<fns.count; i++) {
            _addmatchingfunctionref(c, symbol, fns.data[i], &outfn);
        }
        
        if (MORPHO_ISMETAFUNCTION(outfn)) {
            metafunction_compile(MORPHO_GETMETAFUNCTION(outfn), &c->err);
        }
        
        out->returntype=CONSTANT;
        out->dest=compiler_addconstant(c, node, outfn, true, false);
        out->ninstructions=0;
    }
    
    varray_valueclear(&fns);
    
    return true;
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
    if (found) indx = compiler_propagateupvalues(c, found, indx);

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

/** Checks if a symbol resolves a forward reference */
static bool compiler_resolveforwardreference(compiler *c, value symbol, codeinfo *out) {
    bool success=false;
    functionstate *f = compiler_currentfunctionstate(c);

    for (unsigned int i=0; i<f->forwardref.count; i++) {
        forwardreference *ref = f->forwardref.data+i;
        if (MORPHO_ISEQUAL(symbol, ref->symbol) &&
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
 * Namespaces
 * ------------------------------------------- */

/** Adds a namespace to the compiler */
namespc *compiler_addnamespace(compiler *c, value symbol) {
    namespc *new=MORPHO_MALLOC(sizeof(namespc));
    
    if (new) {
        new->label=symbol;
        dictionary_init(&new->symbols);
        dictionary_init(&new->classes);
        
        new->next=c->namespaces; // Link namespace to compiler
        c->namespaces=new;
    }
    
    return new;
}

/** Checks if a given label corresponds to a namespace */
namespc *compiler_isnamespace(compiler *c, value label) {
    for (namespc *spc=c->namespaces; spc!=NULL; spc=spc->next) {
        if (MORPHO_ISEQUAL(spc->label, label)) return spc;
    }
    return NULL;
}

/** Attempts to locate a symbol given a namespace label */
bool compiler_findsymbolwithnamespace(compiler *c, syntaxtreenode *node, value label, value symbol, value *out) {
    bool success=false;
    namespc *spc = compiler_isnamespace(c, label);
    
    // Now try to find the symbol
    if (spc) {
        success=dictionary_get(&spc->symbols, symbol, out);
        
        if (!success) compiler_error(c, node, COMPILE_SYMBOLNOTDEFINEDNMSPC, MORPHO_GETCSTRING(symbol), MORPHO_GETCSTRING(label));
    }
    
    return success;
}

/** Attempts to locate a class given a namespace label */
bool compiler_findclasswithnamespace(compiler *c, syntaxtreenode *node, value label, value symbol, value *out) {
    bool success=false;
    namespc *spc = compiler_isnamespace(c, label);
    
    // Now try to find the symbol
    if (spc) {
        success=dictionary_get(&spc->classes, symbol, out);
        
        if (!success) compiler_error(c, node, COMPILE_SYMBOLNOTDEFINEDNMSPC, MORPHO_GETCSTRING(symbol), MORPHO_GETCSTRING(label));
    }
    
    return success;
}

/** Clears the namespace list, freeing attached data */
void compiler_clearnamespacelist(compiler *c) {
    namespc *next=NULL;
    
    for (namespc *spc=c->namespaces; spc!=NULL; spc=next) {
        next=spc->next;
        
        dictionary_clear(&spc->symbols);
        dictionary_clear(&spc->classes);
        MORPHO_FREE(spc);
    }
    
    c->namespaces=NULL;
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
static codeinfo compiler_ternary(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_property(compiler *c, syntaxtreenode *node, registerindx reqout);
static codeinfo compiler_dot(compiler *c, syntaxtreenode *node, registerindx reqout);
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
    
    { compiler_ternary       },      // NODE_TERNARY

    { compiler_dot           },      // NODE_DOT

    { compiler_range         },      // NODE_RANGE
    { compiler_range         },      // NODE_INCLUSIVERANGE

    NODE_UNDEFINED,                  // NODE_OPERATOR

    { compiler_print         },      // NODE_PRINT
    { compiler_declaration   },      // NODE_DECLARATION
    { compiler_declaration   },      // NODE_TYPE
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
    { compiler_list          },      // NODE_TUPLE
    { compiler_import        },      // NODE_IMPORT
    NODE_UNDEFINED,                  // NODE_AS
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

/** Compiles a list or tuple */
static codeinfo compiler_list(compiler *c, syntaxtreenode *node, registerindx reqout) {
    syntaxtreenodetype dictentrytype[] = { NODE_ARGLIST };
    varray_syntaxtreeindx entries;

    /* Set up a call to the List() function */
    char *classname = LIST_CLASSNAME;
    if (node->type==NODE_TUPLE) classname = TUPLE_CLASSNAME;
    codeinfo out = compiler_findbuiltin(c, node, classname, reqout);
    
    value listtype=MORPHO_NIL; /* Set the type associated with the register */
    if (compiler_findtypefromcstring(c, classname, &listtype)) {
        if (!compiler_regsetcurrenttype(c, node, out.dest, listtype)) return CODEINFO_EMPTY;
    }

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

    value dicttype=MORPHO_NIL; /* Set the type associated with the register */
    if (compiler_findtypefromcstring(c, DICTIONARY_CLASSNAME, &dicttype)) {
        if (!compiler_regsetcurrenttype(c, node, out.dest, dicttype)) return CODEINFO_EMPTY;
    }
    
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
    bool inclusive = node->type==NODE_INCLUSIVERANGE;

    /* Determine whether we have start..end or start..end:step */
    syntaxtreenode *left=compiler_getnode(c, node->left);
    if (left && (left->type==NODE_RANGE || left->type==NODE_INCLUSIVERANGE)) {
        s[0]=left->left; s[1]=left->right; s[2]=node->right;
        inclusive = left->type==NODE_INCLUSIVERANGE;
    } else {
        s[0]=node->left; s[1]=node->right;
    }

    /* Set up a call to the Range() function */
    codeinfo rng = compiler_findbuiltin(c, node, (inclusive ? RANGE_INCLUSIVE_CONSTRUCTOR: RANGE_CLASSNAME), reqout);
    
    value rngtype=MORPHO_NIL; /* Set the type associated with the register */
    if (compiler_findtypefromcstring(c, RANGE_CLASSNAME, &rngtype)) {
        if (!compiler_regsetcurrenttype(c, node, rng.dest, rngtype)) return CODEINFO_EMPTY;
    }

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
 * @param[out] out     - codeinfo struct with number of instructions used
 * @returns true on success */
codeinfo compiler_compileindexlist(compiler *c, syntaxtreenode *indxnode, registerindx *start, registerindx *end) {
    registerindx istart=compiler_regtop(c), iend;

    if (indxnode->right==SYNTAXTREE_UNCONNECTED) {
        compiler_error(c, indxnode, COMPILE_MSSNGINDX);
        return CODEINFO_EMPTY;
    }
    
    compiler_beginargs(c);
    codeinfo right = compiler_nodetobytecode(c, indxnode->right, REGISTER_UNALLOCATED);
    compiler_endargs(c);
    iend=compiler_regtop(c);

    if (iend==istart) compiler_error(c, indxnode, PARSE_VARBLANKINDEX);

    if (start) *start = istart+1;
    if (end) *end = iend;

    return right;
}

/** Compiles a lookup of an indexed variable */
static codeinfo compiler_index(compiler *c, syntaxtreenode *node, registerindx reqout) {
    registerindx start, end;

    /* Compile the index selector */
    codeinfo left = compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
    unsigned int ninstructions=left.ninstructions;

    if (!CODEINFO_ISREGISTER(left)) {
        left=compiler_movetoregister(c, node, left, REGISTER_UNALLOCATED);
        ninstructions+=left.ninstructions;
    }

    /* Compile indices */
    codeinfo out = compiler_compileindexlist(c, node, &start, &end);
    if (compiler_haserror(c)) return CODEINFO_EMPTY;
    ninstructions+=out.ninstructions;

    /* Compile instruction */
    compiler_addinstruction(c, ENCODE(OP_LIX, left.dest, start, end), node);
    ninstructions++;
    
    /* Free anything we're done with */
    compiler_releaseoperand(c, left);
    compiler_regfreetoend(c, start+1);
    
    codeinfo iout = CODEINFO(REGISTER, start, ninstructions);
    
    if (reqout>=0 &&
        start!=reqout) {
        compiler_regfreetemp(c, start);
        iout = compiler_movetoregister(c, node, iout, reqout);
        ninstructions+=iout.ninstructions;
    }
    
    iout.ninstructions=ninstructions;

    return iout;
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

    if (compiler_haserror(c)) return CODEINFO_EMPTY;
    
    compiler_addinstruction(c, ENCODE(op, out, left.dest, right.dest), node);
    ninstructions++;
    compiler_releaseoperand(c, left);
    compiler_releaseoperand(c, right);

    return CODEINFO(REGISTER, out, ninstructions);
}

/** @brief Compiles the ternary operator
 *  @details Ternary operators are encoded in the syntax tree
 *
 *           <pre>
 *                 TERNARY
 *                  /  \
 *              cond    SEQUENCE
 *                       /    \
 *            true outcome    false outcome
 *           </pre> 
 *   and are compiled to:
 *              <cond>
 *              bif    <cond>, fail:  ; branch if condition isn't met
 *              .. true outcome
 *              b      end            ; generated if an else statement is present
 *           fail:
 *              .. false outcome
 *           end:
 */
static codeinfo compiler_ternary(compiler *c, syntaxtreenode *node, registerindx reqout) {
    unsigned int ninstructions=0;
    
    // Compile the condition
    codeinfo cond = compiler_nodetobytecode(c, node->left, REGISTER_UNALLOCATED);
    ninstructions+=cond.ninstructions;
    
    // And make sure it's in a register */
    if (!CODEINFO_ISREGISTER(cond)) {
       cond=compiler_movetoregister(c, node, cond, REGISTER_UNALLOCATED);
       ninstructions+=cond.ninstructions; /* Keep track of instructions */
    }

    // Generate empty instruction to contain the conditional branch
    instructionindx cbrnchindx=compiler_addinstruction(c, ENCODE_BYTE(OP_NOP), node);
    ninstructions++;
    
    // We're now done with the result of the condition
    compiler_releaseoperand(c, cond);
    
    // Claim an output register
    registerindx rout=compiler_regtemp(c, reqout);
    
    // Get the possible outcomes
    syntaxtreenode *outcomes = compiler_getnode(c, node->right);
    
    // Compile true outcome
    codeinfo left = compiler_nodetobytecode(c, outcomes->left, rout);
    ninstructions+=left.ninstructions;
    
    // Ensure the result is in the output register
    left = compiler_movetoregister(c, outcomes, left, rout);
    ninstructions+=left.ninstructions;
    
    // Generate empty instruction to branch to the end
    instructionindx brnchendindx=compiler_addinstruction(c, ENCODE_BYTE(OP_NOP), node);
    ninstructions++;
    
    // Compile false outcome
    codeinfo right = compiler_nodetobytecode(c, outcomes->right, rout);
    ninstructions+=right.ninstructions;
    
    // Ensure the result is in the output register
    right = compiler_movetoregister(c, outcomes, right, rout);
    ninstructions+=right.ninstructions;
    
    instructionindx end=compiler_currentinstructionindex(c);
    
    // Patch up branches
    compiler_setinstruction(c, cbrnchindx, ENCODE_LONG(OP_BIFF, cond.dest, brnchendindx-cbrnchindx));
    compiler_setinstruction(c, brnchendindx, ENCODE_LONG(OP_B, REGISTER_UNALLOCATED, end-brnchendindx-1));
    
    return CODEINFO(REGISTER, rout, ninstructions);
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
 * with the integer counter as in the below code
 *
 * Register allocation
 *  |  rObj  | rIndx  | | rMax  | rEnum |  rVal   | rTmp   |  ...
 *
 *  Find maximum value of counter
 *   lct   rEnum,  c             ; "enumerate"
 *   mov   rVal,   rObj          ;
 *   lct   rTmp, c               ; -1
 *   invoke rEnum, 1, 0
 *   mov   rMax, rVal

 *  loopstart:
 *   lt    rTmp,  rIndx,  rMax
 *   biff  rTmp,  <loopend>

 *   mov   rVal, rObj
 *   mov   rTmp, rIndx
 *   invoke rEnum, 1, 0

 *   ... Loop body ...

 *   lct   rTmp, c               ; 1
 *   add   rIndx, rIndx, rTmp    ;

 *   b     <loopstart>
 *  loopend:
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

    // Register allocation for the loop
    // |  rObj  | rIndx  || rMax  | rEnum |  rVal   | rTmp   |  ...

    // Fetch the collection object
    codeinfo coll=compiler_nodetobytecode(c, innode->right, REGISTER_UNALLOCATED);
    ninstructions+=coll.ninstructions;
    if (!CODEINFO_ISREGISTER(coll)) {
        coll=compiler_movetoregister(c, collnode, coll, REGISTER_UNALLOCATED);
        ninstructions+=coll.ninstructions;
    }
    registerindx rObj = coll.dest;
    
    // Initialize the index variable
    registerindx rIndx=compiler_regalloc(c, MORPHO_NIL);
    if (indxnode) compiler_regsetsymbol(c, rIndx, indxnode->content);
    int cNil = compiler_addconstant(c, node, MORPHO_INTEGER(0), false, false);
    compiler_addinstruction(c, ENCODE_LONG(OP_LCT, rIndx, cNil), node);
    ninstructions++;
    
    /* Obtain the maximum value of rIndx by invoking enumerate */
    
    // Allocate register to contain maximum value of the counter
    registerindx rMax=compiler_regalloctop(c);
    
    // Initialize enumerate selector
    registerindx rEnum=compiler_regalloctop(c);
    int cEnum = compiler_addconstant(c, node, enumerateselector, false, false);
    compiler_addinstruction(c, ENCODE_LONG(OP_LCT, rEnum, cEnum), node);
    ninstructions++;

    // Place the object into the register after rEnum
    registerindx rVal=compiler_regalloctop(c);
    compiler_regsetsymbol(c, rVal, initnode->content);
    compiler_addinstruction(c, ENCODE_LONG(OP_MOV, rVal, rObj), node);
    ninstructions++;

    // Parameter is -1 to query size of collection
    registerindx rTmp=compiler_regalloctop(c);
    registerindx cNegOne = compiler_addconstant(c, node, MORPHO_INTEGER(-1), false, false);
    compiler_addinstruction(c, ENCODE_LONG(OP_LCT, rTmp, cNegOne), node);
    ninstructions++;
    
    compiler_addinstruction(c, ENCODE(OP_INVOKE, rEnum, 1, 0), collnode);
    ninstructions++;
    
    // Store the maximum value
    compiler_addinstruction(c, ENCODE_DOUBLE(OP_MOV, rMax, rVal), collnode);
    ninstructions++;
    
    /* Test index against the maximum value */
    instructionindx tst=compiler_addinstruction(c, ENCODE(OP_LT, rTmp, rIndx, rMax), node);
    condindx=compiler_addinstruction(c, ENCODE_BYTE(OP_NOP), node); // Placeholder for branch
    ninstructions+=2;
    
    /* Load enumerated value */
    compiler_addinstruction(c, ENCODE_DOUBLE(OP_MOV, rVal, rObj), node);
    compiler_addinstruction(c, ENCODE_DOUBLE(OP_MOV, rTmp, rIndx), node);
    compiler_addinstruction(c, ENCODE(OP_INVOKE, rEnum, 1, 0), collnode);
    ninstructions+=3;
    
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

    int cOne = compiler_addconstant(c, node, MORPHO_INTEGER(1), false, false);
    compiler_addinstruction(c, ENCODE_LONG(OP_LCT, rTmp, cOne), node);
    instructionindx add=compiler_addinstruction(c, ENCODE(OP_ADD, rIndx, rIndx, rTmp), node);
    ninstructions+=2;
    
    /* Compile the unconditional branch back to the test instruction */
    instructionindx end=compiler_addinstruction(c, ENCODE_LONG(OP_B, REGISTER_UNALLOCATED, -(add-tst)-2), node);
    ninstructions++;

    /* Go back and generate the condition instruction */
    compiler_setinstruction(c, condindx, ENCODE_LONG(OP_BIFF, rTmp, (add-tst) ));

    compiler_fixloop(c, tst, inc, end+1);

    compiler_regfreetemp(c, rObj);
    compiler_regfreetemp(c, rIndx);
    compiler_regfreetoend(c, rMax);

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
        compiler_addinstruction(c, ENCODE_LONG(OP_BIF, cond.dest, -ninstructions-1), node);
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

/** Checks through a catch block, fixing any references
 * @param[in] c the current compiler
 * @param[in] start first instruction in the loop body
 * @param[in] inc position of the loop increment section (continue statements redirect here)
 * @param[in] end position of the first instruction AFTER the loop (break sections redirect here) */
static void compiler_fixcatch(compiler *c, instructionindx start, instructionindx inc, instructionindx end) {
    instruction *code=c->out->code.data;
    for (instructionindx i=start; i<end; i++) {
        if (DECODE_OP(code[i])==OP_NOP &&
            (DECODE_A(code[i])=='s' || DECODE_A(code[i])=='b')) {
            code[i]=ENCODE_LONG(OP_B, REGISTER_UNALLOCATED, end-i-1);
        }
    }
}

/** @brief Compiles a try/catch block. */
static codeinfo compiler_try(compiler *c, syntaxtreenode *node, registerindx reqout) {
    codeinfo out = CODEINFO_EMPTY;

    objectdictionary *cdict = object_newdictionary();
    if (!cdict) { compiler_error(c, node, ERROR_ALLOCATIONFAILED); return out; }

    program_bindobject(c->out, (object *) cdict);
    
    registerindx cdictindx = compiler_addconstant(c, node, MORPHO_OBJECT(cdict), false, false);

    compiler_addinstruction(c, ENCODE_LONG(OP_PUSHERR, 0, cdictindx), node);
    out.ninstructions++;

    debugannotation_pusherr(&c->out->annotations, cdict);

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

        // Add an effective 'break' instruction after each entry body except for the last
        if (i!=switchnodes.count-1) {
            compiler_addinstruction(c, ENCODE(OP_NOP, 's', 0, 0), node);
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
    /* Fix any nop instructions in the switch block to jump to the end of block */
    compiler_fixcatch(c, popindx, popindx, endindx);

    varray_syntaxtreeindxclear(&switchnodes);
    varray_syntaxtreeindxclear(&labelnodes);

    debugannotation_poperr(&c->out->annotations);

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
    syntaxtreenode *decnode = node;
    syntaxtreenode *typenode = NULL;
    
    if (node->type==NODE_TYPE) {
        typenode=compiler_getnode(c, node->left);
        decnode=compiler_getnode(c, node->right);
    }
    
    syntaxtreenode *varnode = NULL;
    syntaxtreenode *lftnode = NULL, *indxnode = NULL;
    codeinfo right=CODEINFO_EMPTY;
    value var=MORPHO_NIL, type=MORPHO_NIL;
    registerindx reg;
    unsigned int ninstructions = 0;
    
    varnode=compiler_getnode(c, decnode->left);
    
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

        if (typenode &&
            compiler_findtype(c, typenode->content, &type)) {
            compiler_regsettype(c, reg, type);
            if (vloc.returntype==GLOBAL) compiler_setglobaltype(c, vloc.dest, type);
        }
        
        /* If this is an array, we must create it */
        if (indxnode) {
            /* Set up a call to the Array() function */
            array=compiler_findbuiltin(c, decnode, ARRAY_CLASSNAME, reqout);
            ninstructions+=array.ninstructions;

            // Dimensions
            registerindx istart=REGISTER_UNALLOCATED, iend=REGISTER_UNALLOCATED;
            codeinfo indxinfo=compiler_compileindexlist(c, indxnode, &istart, &iend);
            ninstructions+=indxinfo.ninstructions;

            // Initializer
            if (decnode->right!=SYNTAXTREE_UNCONNECTED) {
                iend=compiler_regalloctop(c);

                right = compiler_nodetobytecode(c, decnode->right, iend);
                ninstructions+=right.ninstructions;

                right=compiler_movetoregister(c, decnode, right, iend); // Ensure in register
                ninstructions+=right.ninstructions;
            }

            // Call Array()
            compiler_addinstruction(c, ENCODE_DOUBLE(OP_CALL, array.dest, iend-istart+1), node);
            ninstructions++;

            compiler_regfreetoend(c, istart);

            if (vloc.returntype==REGISTER && array.dest!=vloc.dest) { // Move to correct register
                codeinfo move=compiler_movetoregister(c, decnode, array, vloc.dest);
                ninstructions+=move.ninstructions;
            } else reg=array.dest;

        } else if (decnode->right!=SYNTAXTREE_UNCONNECTED) { /* Not an array, but has an initializer */
            right = compiler_nodetobytecode(c, decnode->right, reg);
            ninstructions+=right.ninstructions;

            /* Ensure operand is in the desired register  */
            right=compiler_movetoregister(c, decnode, right, reg);
            ninstructions+=right.ninstructions;
        } else { /* Otherwise, we should zero out the register */
            registerindx cnil = compiler_addconstant(c, decnode, MORPHO_NIL, false, false);
            compiler_addinstruction(c, ENCODE_LONG(OP_LCT, reg, cnil), node);
            ninstructions++;
        }

        if (vloc.returntype!=REGISTER) {
            codeinfo mv=compiler_movefromregister(c, decnode, vloc, reg);
            ninstructions+=mv.ninstructions;

            compiler_regfreetemp(c, reg);
        }
        
        compiler_releaseoperand(c, right);
    }

    return CODEINFO(REGISTER, REGISTER_UNALLOCATED, ninstructions);
}

/** Compiles an parameter declaration */
static registerindx compiler_functionparameters(compiler *c, syntaxtreeindx indx) {
    syntaxtreenode *node = compiler_getnode(c, indx);
    if (!node) return REGISTER_UNALLOCATED;

    switch(node->type) {
        case NODE_SYMBOL:
            return compiler_addpositionalarg(c, node, node->content);
            break;
        case NODE_TYPE:
        {
            value type=MORPHO_NIL;
            syntaxtreenode *typenode = compiler_getnode(c, node->left);
            if (!typenode) UNREACHABLE("Incorrectly formed type node.");
            if (typenode->type==NODE_DOT) {
                syntaxtreenode *nsnode = compiler_getnode(c, typenode->left);
                syntaxtreenode *labelnode = compiler_getnode(c, typenode->right);
                
                if (!(nsnode &&
                    labelnode &&
                    MORPHO_ISSTRING(nsnode->content) &&
                    MORPHO_ISSTRING(labelnode->content))) UNREACHABLE("Incorrectly formed type namespace node.");
                
                if (!compiler_isnamespace(c, nsnode->content)) {
                    compiler_error(c, nsnode, COMPILE_UNKNWNNMSPC, MORPHO_GETCSTRING(nsnode->content));
                    return REGISTER_UNALLOCATED;
                }
                
                if (!compiler_findclasswithnamespace(c, typenode, nsnode->content, labelnode->content, &type)) {
                    compiler_error(c, typenode, COMPILE_SYMBOLNOTDEFINEDNMSPC, MORPHO_GETCSTRING(nsnode->content), MORPHO_GETCSTRING(labelnode->content));
                    return REGISTER_UNALLOCATED;
                }
                    
            } else if (MORPHO_ISSTRING(typenode->content)) {
                if (!compiler_findtype(c, typenode->content, &type)) {
                    compiler_error(c, node, COMPILE_UNKNWNTYPE, MORPHO_GETCSTRING(typenode->content));
                    return REGISTER_UNALLOCATED;
                }
            } else UNREACHABLE("Type node should have string label.");
            
            registerindx reg = compiler_functionparameters(c, node->right);
            compiler_regsettype(c, reg, type);
            compiler_regsetcurrenttype(c, node, reg, type);
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
    
    return REGISTER_UNALLOCATED;;
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
    if (!func) { compiler_error(c, node, ERROR_ALLOCATIONFAILED); return CODEINFO_EMPTY; }
    
    program_bindobject(c->out, (object *) func);
    
    /* Record the class is a method */
    if (ismethod) func->klass=compiler_getcurrentclass(c);
    
    /* Add the function as a constant */
    kindx=compiler_addconstant(c, node, MORPHO_OBJECT(func), false, false);
    
    /* Keep a reference to the function */
    if (!ismethod) compiler_addfunctionref(c, func);

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
    
    value signature[func->nargs+1];
    for (int i=0; i<func->nargs; i++) compiler_regtype(c, i+1, &signature[i]);
    function_setsignature(func, signature);
    signature_setvarg(&func->sig, function_hasvargs(func));

    /* Check we don't have too many arguments */
    if (func->nargs+func->nopt>MORPHO_MAXARGS) {
        compiler_error(c, node, COMPILE_TOOMANYPARAMS);
        return CODEINFO_EMPTY;
    }

    /* -- Compile the body -- */
    if (body!=REGISTER_UNALLOCATED) bodyinfo=compiler_nodetobytecode(c, body, REGISTER_UNALLOCATED);
    ninstructions+=bodyinfo.ninstructions;

    /* Add a return instruction if necessary */
    if (ismethod) { // Methods automatically return self unless another argument is specified
        compiler_addinstruction(c, ENCODE_DOUBLE(OP_RETURN, 1, 0), node); /* Add a return */
    } else {
        compiler_addinstruction(c, ENCODE_BYTE(OP_RETURN), node); /* Add a return */
    }
    ninstructions++;

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

        /* Allocate a variable to refer to the function definition, but only in global
           context */
        /* TODO: Do we need to do this now functionstates capture function info? */
        codeinfo fvar=CODEINFO_EMPTY;
        fvar.dest=compiler_regtemp(c, reqout);
        fvar.returntype=REGISTER;
        
        if (!isanonymous) {
            if (!compiler_resolveforwardreference(c, func->name, &fvar) &&
                compiler_checkglobal(c)) {
                compiler_regfreetemp(c, fvar.dest);
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
            // Save the register where the closure is to be found
            compiler_regsetsymbol(c, reg, func->name);
            compiler_regsettype(c, reg, _closuretype);
            function_setclosure(func, reg);
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
        bool isOptional=false;
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
            compiler_regsetoptionalarg(c, info.dest);
            
            reg=compiler_regalloctop(c);
            arginfo=compiler_nodetobytecode(c, arg->right, reg);
            if (CODEINFO_ISREGISTER(arginfo)) compiler_regsetoptionalarg(c, arginfo.dest);
            
            isOptional=true;
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
            if (isOptional) compiler_regsetoptionalarg(c, arginfo.dest);
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
    syntaxtreenode *selector, *target, *method;
    /* Get the selector node */
    selector=compiler_getnode(c, call->left);
    if (selector->type==NODE_DOT) {
        /* Check that if the target is a namespace */
        target=compiler_getnode(c, selector->left);
        if (target->type==NODE_SYMBOL &&
            compiler_isnamespace(c, target->content)) {
            return false;
        }
        
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
    
    // Check if the call is a constructor
    syntaxtreenode *selnode=compiler_getnode(c, node->left);
    
    value rtype=MORPHO_NIL;
    if (selnode->type==NODE_SYMBOL) { // A regular call from a symbol
        compiler_findtype(c, selnode->content, &rtype);
    } else if (selnode->type==NODE_DOT) { // An constructor in a namespace?
        syntaxtreenode *nsnode = compiler_getnode(c, selnode->left);
        syntaxtreenode *snode = compiler_getnode(c, selnode->right);
        if (nsnode && snode &&
            compiler_isnamespace(c, nsnode->content)) {
            compiler_findclasswithnamespace(c, snode, nsnode->content, snode->content, &rtype);
            compiler_catch(c, COMPILE_SYMBOLNOTDEFINEDNMSPC); // We don't care if it wasn't there
        }
    }
    
    // Compile the selector
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
    int nposn=0, nopt=0;
    compiler_regcountargs(c, func.dest+1, lastarg, &nposn, &nopt);
    compiler_addinstruction(c, ENCODE(OP_CALL, func.dest, nposn, nopt), node);
    ninstructions++;

    /* Free all the registers used for the call */
    compiler_regfreetoend(c, func.dest+1);
    
    /* Set the current type of the register */
    compiler_regsetcurrenttype(c, selnode, func.dest, rtype);

    /* Move the result to the requested register */
    if (reqout!=REGISTER_UNALLOCATED && func.dest!=reqout) {
        codeinfo mv = compiler_movetoregister(c, node, func, reqout);
        ninstructions+=mv.ninstructions;
        compiler_regfreetemp(c, func.dest);
        func.dest=reqout;
    }

    return CODEINFO(REGISTER, func.dest, ninstructions);
}

#include <stdint.h>

/** Compiles a method invocation */
static codeinfo compiler_invoke(compiler *c, syntaxtreenode *node, registerindx reqout) {
    unsigned int ninstructions=0;
    codeinfo object=CODEINFO_EMPTY;

    /* Get the selector node */
    syntaxtreenode *selector=compiler_getnode(c, node->left);

    compiler_beginargs(c);
    
    registerindx rSel = compiler_regalloctop(c);
    registerindx rObj = compiler_regalloctop(c);

    syntaxtreenode *methodnode=compiler_getnode(c, selector->right);
    codeinfo cSel = CODEINFO(CONSTANT, 0, 0);
    cSel.dest = compiler_addsymbol(c, methodnode, methodnode->content);
    codeinfo method=compiler_movetoregister(c, methodnode, cSel, rSel);
    ninstructions+=method.ninstructions;

    /* Fetch the object. We patch to ensure that builtin classes are prioritized over constructor functions. */
    syntaxtreenode *objectnode=compiler_getnode(c, selector->left);
    if (objectnode->type==NODE_SYMBOL) {
        value klass=builtin_findclass(objectnode->content);
        if (MORPHO_ISCLASS(klass)) {
            registerindx kindx = compiler_addconstant(c, objectnode, klass, true, false);
            object=CODEINFO(CONSTANT, kindx, 0);
        }
    }
    
    // Otherwise just fetch the object normally
    if (object.returntype==REGISTER && object.dest==REGISTER_UNALLOCATED) { 
      object=compiler_nodetobytecode(c, selector->left, rObj);
    }

    ninstructions+=object.ninstructions;
    if (object.returntype==REGISTER && object.dest!=rObj) {
        compiler_regfreetemp(c, object.dest);
        compiler_regtempwithindx(c, rObj); // Ensure rObj remains allocated
    }
    object=compiler_movetoregister(c, selector, object, rObj);
    ninstructions+=object.ninstructions;
    
    /* Compile the arguments */
    codeinfo args = CODEINFO_EMPTY;
    if (node->right!=SYNTAXTREE_UNCONNECTED) args=compiler_nodetobytecode(c, node->right, REGISTER_UNALLOCATED);
    ninstructions+=args.ninstructions;

    /* Remember the last argument */
    registerindx lastarg=compiler_regtop(c);

    /* Check we don't have too many arguments */
    if (lastarg-rSel>MORPHO_MAXARGS) {
        compiler_error(c, node, COMPILE_TOOMANYARGS);
        return CODEINFO_EMPTY;
    }

    compiler_endargs(c);

    /* Generate the call instruction */
    int nposn=0, nopt=0;
    compiler_regcountargs(c, object.dest+1, lastarg, &nposn, &nopt);
    compiler_addinstruction(c, ENCODE(OP_INVOKE, rSel, nposn, nopt), node);
    ninstructions++;

    /* Free all the registers used for the call */
    compiler_regfreetemp(c, rSel);
    compiler_regfreetoend(c, rObj+1);

    /* Move the result to the requested register */
    if (reqout!=REGISTER_UNALLOCATED && object.dest!=reqout) {
        compiler_addinstruction(c, ENCODE_DOUBLE(OP_MOV, reqout, rObj), node);
        ninstructions++;
        compiler_regfreetemp(c, rObj);
        object.dest=reqout;
    }

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

/** Overrides or adds to an existing method implementation */
void compiler_overridemethod(compiler *c, syntaxtreenode *node, objectfunction *method, value prev) {
    value symbol = method->name;
    objectclass *klass=compiler_getcurrentclass(c);
    
    if (MORPHO_ISMETAFUNCTION(prev)) {
        objectmetafunction *f = MORPHO_GETMETAFUNCTION(prev);
        if (f->klass!=klass) {
            f=metafunction_clone(f);
            if (f) program_bindobject(c->out, (object *) f);
        }
        
        if (f) {
            metafunction_setclass(f, klass);
            dictionary_insert(&klass->methods, symbol, MORPHO_OBJECT(f));
            
            for (int i=0; i<f->fns.count; i++) { // Check if this overrides
                signature *sig = metafunction_getsignature(f->fns.data[i]);
                if (sig && signature_isequal(sig, &method->sig)) {
                    // TODO: Should check for duplicate implementation here
                    f->fns.data[i] = MORPHO_OBJECT(method);
                    return;
                }
            }
            
            metafunction_add(f, MORPHO_OBJECT(method));
        }
        
    } else if (MORPHO_ISFUNCTION(prev)) {
        objectfunction *prevmethod = MORPHO_GETFUNCTION(prev);
        
        if (signature_isequal(&prevmethod->sig, &method->sig)) { // Does the method overshadow an old one?
            if (prevmethod->klass!=klass) { // If so, is the old one in the parent or ancestor class?
                dictionary_insert(&klass->methods, symbol, MORPHO_OBJECT(method));
            } else { // It's a redefinition
                compiler_error(c, node, COMPILE_CLSSDPLCTIMPL, MORPHO_GETCSTRING(symbol), MORPHO_GETCSTRING(klass->name));
            }
        } else { // It doesn't override the old definition so wrap in a metafunction
            objectmetafunction *f = object_newmetafunction(symbol);
            
            if (f) {
                metafunction_add(f, prev);
                metafunction_add(f, MORPHO_OBJECT(method));
                metafunction_setclass(f, klass);
                dictionary_insert(&klass->methods, symbol, MORPHO_OBJECT(f));
                program_bindobject(c->out, (object *) f);
            }
        }
    } else if (MORPHO_ISBUILTINFUNCTION(prev)) { // A builtin function can only come from a parent class, so overwrite it
        dictionary_insert(&klass->methods, symbol, MORPHO_OBJECT(method));
    }
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
                    value omethod = MORPHO_OBJECT(method);
                    value symbol = program_internsymbol(c->out, node->content),
                          prev=MORPHO_NIL;
                    
                    if (dictionary_get(&klass->methods, symbol, &prev)) {
                        compiler_overridemethod(c, node, method, prev); // Override or create a metafunction
                    } else dictionary_insert(&klass->methods, symbol, omethod); // Just insert
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

/** Compiles a class declaration */
static codeinfo compiler_class(compiler *c, syntaxtreenode *node, registerindx reqout) {
    unsigned int ninstructions=0;
    registerindx kindx;
    codeinfo mout=CODEINFO_EMPTY;

    if (compiler_getcurrentclass(c)) {
        compiler_error(c, node, COMPILE_NSTDCLSS);
        return CODEINFO_EMPTY;
    }

    objectclass *klass=object_newclass(node->content);
    if (!klass) { compiler_error(c, node, ERROR_ALLOCATIONFAILED); return CODEINFO_EMPTY; }
    
    compiler_beginclass(c, klass);

    /** Store the object class as a constant */
    kindx=compiler_addconstant(c, node, MORPHO_OBJECT(klass), false, false);
    
    /** Add the class to the class table */
    if (ERROR_SUCCEEDED(c->err)) compiler_addclass(c, klass);
    
    /* Is there a superclass and/or mixins? */
    if (node->left!=SYNTAXTREE_UNCONNECTED) {
        syntaxtreenodetype dictentrytype[] = { NODE_SEQUENCE };
        varray_syntaxtreeindx entries;
        varray_syntaxtreeindxinit(&entries);
        
        syntaxtree_flatten(compiler_getsyntaxtree(c), node->left, 1, dictentrytype, &entries);
        
        for (int i=entries.count-1; i>=0; i--) { // Loop over super and mixins in reverse order
                                                 // As super will be LAST in this list
            syntaxtreenode *snode = syntaxtree_nodefromindx(compiler_getsyntaxtree(c), entries.data[i]) ;
            
            objectclass *superclass=NULL;
            value classlabel=MORPHO_NIL;
            
            if (snode->type==NODE_SYMBOL) {
                classlabel=snode->content;
                superclass=compiler_findclass(c, classlabel);
            } else if (snode->type==NODE_DOT) {
                syntaxtreenode *left = compiler_getnode(c, snode->left),
                               *right = compiler_getnode(c, snode->right);
                
                if (left->type!=NODE_SYMBOL || right->type!=NODE_SYMBOL) UNREACHABLE("Superclass or mixin namespace node should have symbols");
                
                classlabel=right->content;
                
                value klass=MORPHO_NIL;
                compiler_findclasswithnamespace(c, snode, left->content, classlabel, &klass);
                
                if (MORPHO_ISCLASS(klass)) superclass=MORPHO_GETCLASS(klass);
            } else {
                UNREACHABLE("Superclass or mixin node should be a symbol.");
            }
                
            if (superclass) {
                if (superclass!=klass) {
                    if (!klass->superclass) klass->superclass=superclass; // Only the first class is the super class, all others are mixins.
                    compiler_addparent(c, klass, superclass);
                    dictionary_copy(&superclass->methods, &klass->methods);
                } else {
                    compiler_error(c, snode, COMPILE_CLASSINHERITSELF);
                }
            } else {
                if (MORPHO_ISSTRING(classlabel)) {
                    compiler_error(c, snode, COMPILE_SUPERCLASSNOTFOUND, MORPHO_GETCSTRING(classlabel));
                } else UNREACHABLE("No class label available");
            }
        }
        
        varray_syntaxtreeindxclear(&entries);
    } else {
        klass->superclass=baseclass;
        if (baseclass) dictionary_copy(&baseclass->methods, &klass->methods);
    }
    
    /* Now compute the class linearization */
    if (!class_linearize(klass)) {
        compiler_error(c, node, COMPILE_CLSSLNRZ, MORPHO_GETCSTRING(klass->name));
    }

    /* Compile method declarations */
    if (node->right!=SYNTAXTREE_UNCONNECTED) {
        syntaxtreenode *child = compiler_getnode(c, node->right);
        mout=compiler_method(c, child, reqout);
        ninstructions+=mout.ninstructions;
    }

    /* End class definition */
    compiler_endclass(c);

    compiler_checkoutstandingforwardreference(c);
    
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
    
    /* Bind the klass to the program to be freed on exit */
    if (klass) program_bindobject(c->out, (object *) klass);
    
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
    value type;
    
    /* Is it a local variable? */
    ret.dest=compiler_getlocal(c, node->content);
    if (ret.dest!=REGISTER_UNALLOCATED &&
        compiler_regtype(c, ret.dest, &type) && // If it's a closure it should be resolved later
        !MORPHO_ISEQUAL(type, _closuretype)) {
        return ret;
    }

    /* Is it a reference to a function? */
    if (compiler_resolvefunctionref(c, node, node->content, &ret)) {
        return ret;
    }
    
    /* Is it an upvalue? */
    ret.dest = compiler_resolveupvalue(c, node->content);
    if (ret.dest!=REGISTER_UNALLOCATED) {
        ret.returntype=UPVALUE;
        return ret;
    }
    
    /* Is it a global variable */
    ret.dest=compiler_findglobal(c, node->content, true);
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
    objectclass *klass = compiler_findclass(c, node->content);
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

            if (varnode->type==NODE_DOT || varnode->type==NODE_SELF) {
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
                reg=compiler_findglobal(c, var, true);
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

/** Compiles the dot operator, which may be property lookup or a method call */
static codeinfo compiler_dot(compiler *c, syntaxtreenode *node, registerindx reqout) {
    syntaxtreenode *left = compiler_getnode(c, node->left),
                   *right = compiler_getnode(c, node->right);
    value out=MORPHO_NIL;
    
    if (left->type==NODE_SYMBOL && 
        right->type==NODE_SYMBOL &&
        compiler_findsymbolwithnamespace(c, node, left->content, right->content, &out)) {
        
        if (MORPHO_ISINTEGER(out)) {
            return CODEINFO(GLOBAL, MORPHO_GETINTEGERVALUE(out), 0);
        } else if (MORPHO_ISFUNCTION(out) ||
                   MORPHO_ISMETAFUNCTION(out) ||
                   MORPHO_ISBUILTINFUNCTION(out) ||
                   MORPHO_ISCLASS(out)) {
            registerindx indx = compiler_addconstant(c, node, out, true, false);
            return CODEINFO(CONSTANT, indx, 0);
        } else {
            UNREACHABLE("Namespace dictionary contains noninteger value");
        }
    }
    
    return compiler_property(c, node, reqout);
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
        debugannotation_stripend(&out->annotations);
    }
}

/** Copies the globals across from one compiler to another. The globals dictionary maps keys to global numbers
 * @param[in] src source dictionary
 * @param[in] dest destination dictionary
 * @param[in] compare (optional) a dictionary to check the contents against; globals are only copied if they also appear in compare */
void compiler_copysymbols(dictionary *src, dictionary *dest, dictionary *compare) {
    for (unsigned int i=0; i<src->capacity; i++) {
        value key = src->contents[i].key;
        if (!MORPHO_ISNIL(key)) {
            if (compare && !dictionary_get(compare, key, NULL)) continue;
            
            if (MORPHO_ISSTRING(key) &&
                MORPHO_GETCSTRING(key)[0]=='_') continue;

            dictionary_insert(dest, key, src->contents[i].val);
        }
    }
}

/** Copies the global function ref into the destination compiler's current function ref */
void compiler_copyfunctionref(compiler *src, compiler *dest, dictionary *fordict) {
    functionstate *in=compiler_currentfunctionstate(src);
    functionstate *out=compiler_currentfunctionstate(dest);
    
    if (fordict) {
        for (int i=0; i<in->functionref.count; i++) {
            functionref *ref=&in->functionref.data[i];
            if (!dictionary_get(fordict, ref->function->name, NULL)) continue;
            
            varray_functionrefwrite(&out->functionref, in->functionref.data[i]);
        }
    } else varray_functionrefadd(&out->functionref, in->functionref.data, in->functionref.count);
}

/** Copies the global function ref into the designated namespace, checking whether the functions are present in the dictionary, and creating metafunctions where necessary */
void compiler_copyfunctionreftonamespace(compiler *src, namespc *dest, dictionary *fordict) {
    functionstate *f=compiler_currentfunctionstate(src);
    
    dictionary symbols;
    dictionary_init(&symbols);
    
    for (int i=0; i<f->functionref.count; i++) {
        functionref *ref=&f->functionref.data[i];
        // Skip if not in the fordict
        if (fordict && !dictionary_get(fordict, ref->function->name, NULL)) continue;
        
        value fn=MORPHO_OBJECT(ref->function);
        if (dictionary_get(&symbols, ref->function->name, &fn)) {
            // If the function already exists, wrap in a metafunction
            _addmatchingfunctionref(src, ref->function->name, MORPHO_OBJECT(ref->function), &fn);
        }
        dictionary_insert(&symbols, ref->function->name, fn);
    }
    
    compiler_copysymbols(&symbols, &dest->symbols, NULL);
    dictionary_clear(&symbols);
}

/** Searches for a module with given name, returns the file name for inclusion. */
bool compiler_findmodule(char *name, varray_char *fname) {
    value out=MORPHO_NIL;
    bool success=morpho_findresource(MORPHO_RESOURCE_MODULE, name, &out);
    
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
    namespc *nmspace=NULL;
    char *fname=NULL;
    unsigned int start=0, end=0;
    FILE *f = NULL;

    dictionary_init(&fordict);
    varray_charinit(&filename);

    if (compiler_checkerror(c)) return CODEINFO_EMPTY;

    while (qual) {
        if (qual->type==NODE_FOR) {
            syntaxtreenode *l = compiler_getnode(c, qual->left);
            if (l && l->type==NODE_SYMBOL) {
                dictionary_insert(&fordict, l->content, MORPHO_NIL);
            } else UNREACHABLE("Incorrect syntax tree structure in FOR node.");
        } else if (qual->type==NODE_AS) {
            syntaxtreenode *l = compiler_getnode(c, qual->left);
            if (l && l->type==NODE_SYMBOL) {
                nmspace=compiler_addnamespace(c, l->content);
                
                if (!nmspace) { compiler_error(c, node, ERROR_ALLOCATIONFAILED); return CODEINFO_EMPTY; }
            } else UNREACHABLE("Incorrect syntax tree structure in AS node.");
        } else UNREACHABLE("Unexpected node type.");
        qual=compiler_getnode(c, qual->right);
    }

    if (module) {
        if (module->type==NODE_SYMBOL) {
            dictionary *fndict, *clssdict;
            
            if (extension_load(MORPHO_GETCSTRING(module->content), &fndict, &clssdict)) {
                compiler_copysymbols(clssdict, (nmspace ? &nmspace->symbols: builtin_getclasstable()), (fordict.count>0 ? &fordict : NULL));
                compiler_copysymbols(fndict, (nmspace ? &nmspace->symbols: builtin_getfunctiontable()), (fordict.count>0 ? &fordict : NULL));
                
                if (nmspace) { // Copy classes into the namespace's class table
                    compiler_copysymbols(clssdict, &nmspace->classes, (fordict.count>0 ? &fordict : NULL));
                }
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

        // Check if the module was previously imported
        if (fname) {
            objectstring chkmodname = MORPHO_STATICSTRING(fname);
            value symboldict=MORPHO_NIL;
            
            if (dictionary_get(&root->modules, MORPHO_OBJECT(&chkmodname), &symboldict)) {
                // If so, copy its symbols into the compiler
                compiler_copysymbols(MORPHO_GETDICTIONARYSTRUCT(symboldict), (nmspace ? &nmspace->symbols: &c->globals), (fordict.count>0 ? &fordict : NULL));
                
                goto compiler_import_cleanup;
            }
        }

        if (fname) f=file_openrelative(fname, "r");
        else goto compiler_import_cleanup;

        if (f) {
            value modname=object_stringfromcstring(fname, strlen(fname));
            value symboldict=MORPHO_NIL;

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
            debugannotation_setmodule(&c->out->annotations, modname);
            cc.parent=c; /* Ensures global variables can be found */

            morpho_compile(src.data, &cc, false, &c->err);

            if (ERROR_SUCCEEDED(c->err)) {
                compiler_stripend(c);
                compiler_copysymbols(&cc.globals, (nmspace ? &nmspace->symbols: &c->globals), (fordict.count>0 ? &fordict : NULL));
                if (nmspace) { // If we're in a namespace, copy the class table into that
                    compiler_copysymbols(&cc.classes, &nmspace->classes, (fordict.count>0 ? &fordict : NULL));
                    compiler_copyfunctionreftonamespace(&cc, nmspace, (fordict.count>0 ? &fordict : NULL));
                } else { // Otherwise just put it into the parent compiler's class table
                    compiler_copysymbols(&cc.classes, &c->classes, (fordict.count>0 ? &fordict : NULL));
                    compiler_copyfunctionref(&cc, c, (fordict.count>0 ? &fordict : NULL));
                }
                
                objectdictionary *dict = object_newdictionary(); // Preserve all symbols for further imports
                if (dict) {
                    compiler_copysymbols(&cc.globals, &dict->dict, NULL);
                    symboldict = MORPHO_OBJECT(dict);
                }
                
            } else {
                c->err.file = (cc.err.file ? cc.err.file : MORPHO_GETCSTRING(modname));
            }
            
            debugannotation_setmodule(&c->out->annotations, compiler_getmodule(c));
            
            end=c->out->code.count;
            
            compiler_clear(&cc);
            varray_charclear(&src);
            
            dictionary_insert(&root->modules, modname, symboldict);
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
    dictionary_init(&c->classes);
    dictionary_init(&c->modules);
    if (out) c->fstack[0].func=out->global; /* The global pseudofunction */
    c->out = out;
    c->prevfunction = NULL;
    c->currentclass = NULL;
    c->currentmethod = NULL;
    c->namespaces = NULL; 
    c->currentmodule = MORPHO_NIL;
    c->parent = NULL;
    c->line = 1; // Count from 1
}

/** @brief Clear attached data structures from a compiler
 *  @param[in]  c        compiler to clear */
void compiler_clear(compiler *c) {
    lex_clear(&c->lex);
    parse_clear(&c->parse);
    compiler_fstackclear(c);
    syntaxtree_clear(&c->tree);
    compiler_clearnamespacelist(c);
    dictionary_clear(&c->globals); // Keys are bound to the program
    dictionary_freecontents(&c->modules, true, true);
    dictionary_clear(&c->modules);
    dictionary_clear(&c->classes);
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
    lex_init(&c->lex, in, c->line); /* Count lines from 1. */

    if ((!morpho_parse(&c->parse)) || !ERROR_SUCCEEDED(c->err)) {
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

    if (success) {
        if (opt && optimizer) (*optimizer) (c->out);
        
        c->line=c->lex.line+1; // Update the line counter if compilation was a success; assumes a new line every time morpho_compile is called.
    }

    return success;
}

/* **********************************************************************
 * Public interfaces
 * ********************************************************************** */

/** Creates a new compiler */
compiler *morpho_newcompiler(program *out) {
    compiler *new = MORPHO_MALLOC(sizeof(compiler));

    if (new) compiler_init("", out, new);

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
    if (MORPHO_ISCLASS(klss)) baseclass=MORPHO_GETCLASS(klss);
}

void morpho_setoptimizer(optimizerfn *opt) {
    optimizer = opt;
}

/** Initializes the compiler */
void compile_initialize(void) {
    _selfsymbol=builtin_internsymbolascstring("self");
    
    /** Types we need to refer to */
    _closuretype = MORPHO_OBJECT(object_getveneerclass(OBJECT_CLOSURE));
    
    optimizer = NULL;

    /* Compile errors */
    morpho_defineerror(COMPILE_SYMBOLNOTDEFINED, ERROR_COMPILE, COMPILE_SYMBOLNOTDEFINED_MSG);
    morpho_defineerror(COMPILE_SYMBOLNOTDEFINEDNMSPC, ERROR_COMPILE, COMPILE_SYMBOLNOTDEFINEDNMSPC_MSG);
    morpho_defineerror(COMPILE_TOOMANYCONSTANTS, ERROR_COMPILE, COMPILE_TOOMANYCONSTANTS_MSG);
    morpho_defineerror(COMPILE_ARGSNOTSYMBOLS, ERROR_COMPILE, COMPILE_ARGSNOTSYMBOLS_MSG);
    morpho_defineerror(COMPILE_PROPERTYNAMERQD, ERROR_COMPILE, COMPILE_PROPERTYNAMERQD_MSG);
    morpho_defineerror(COMPILE_SELFOUTSIDECLASS, ERROR_COMPILE, COMPILE_SELFOUTSIDECLASS_MSG);
    morpho_defineerror(COMPILE_RETURNININITIALIZER, ERROR_COMPILE, COMPILE_RETURNININITIALIZER_MSG);
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
    morpho_defineerror(COMPILE_MSSNGINDX, ERROR_COMPILE, COMPILE_MSSNGINDX_MSG);
    morpho_defineerror(COMPILE_TYPEVIOLATION, ERROR_COMPILE, COMPILE_TYPEVIOLATION_MSG);
    morpho_defineerror(COMPILE_UNKNWNTYPE, ERROR_COMPILE, COMPILE_UNKNWNTYPE_MSG);
    morpho_defineerror(COMPILE_UNKNWNNMSPC, ERROR_COMPILE, COMPILE_UNKNWNNMSPC_MSG);
    morpho_defineerror(COMPILE_UNKNWNTYPENMSPC, ERROR_COMPILE, COMPILE_UNKNWNTYPENMSPC_MSG);
    morpho_defineerror(COMPILE_CLSSLNRZ, ERROR_COMPILE, COMPILE_CLSSLNRZ_MSG);
    morpho_defineerror(COMPILE_CLSSDPLCTIMPL, ERROR_COMPILE, COMPILE_CLSSDPLCTIMPL_MSG);
    
    morpho_addfinalizefn(compile_finalize);
}

/** Finalizes the compiler */
void compile_finalize(void) {
}
