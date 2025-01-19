/** @file vm.c
 *  @author T J Atherton
 *
 *  @brief Morpho virtual machine
 */

#include <stdarg.h>
#include <time.h>
#include "vm.h"
#include "gc.h"
#include "compile.h"
#include "morpho.h"
#include "classes.h"
#include "debug.h"
#include "profile.h"
#include "resources.h"
#include "extensions.h"

value initselector = MORPHO_NIL;
value indexselector = MORPHO_NIL;
value setindexselector = MORPHO_NIL;
value addselector = MORPHO_NIL;
value addrselector = MORPHO_NIL;
value subselector = MORPHO_NIL;
value subrselector = MORPHO_NIL;
value mulselector = MORPHO_NIL;
value mulrselector = MORPHO_NIL;
value divselector = MORPHO_NIL;
value divrselector = MORPHO_NIL;
value powselector = MORPHO_NIL;
value powrselector = MORPHO_NIL;
value printselector = MORPHO_NIL;
value enumerateselector = MORPHO_NIL;
value countselector = MORPHO_NIL;
value cloneselector = MORPHO_NIL;

#ifdef MORPHO_DEBUG_GCSIZETRACKING
dictionary sizecheck;
#endif

/* **********************************************************************
* VM objects
* ********************************************************************** */

vm *globalvm=NULL;

/** Initializes a virtual machine */
static void vm_init(vm *v) {
    globalvm=v;
    v->current=NULL;
    v->instructions=NULL;
    v->objects=NULL;
    v->openupvalues=NULL;
    v->fp=NULL;
    v->fpmax=&v->frame[MORPHO_CALLFRAMESTACKSIZE-1]; // Last valid value of v->fp
    v->ehp=NULL;
    v->bound=0;
    v->nextgc=MORPHO_GCINITIAL;
    v->debug=NULL;
    vm_graylistinit(&v->gray);
    varray_valueinit(&v->stack);
    varray_valueinit(&v->tlvars);
    varray_valueinit(&v->globals);
    varray_valueinit(&v->retain);
    varray_valueresize(&v->stack, MORPHO_STACKINITIALSIZE);
    varray_charinit(&v->buffer);
    error_init(&v->err);
    v->errfp=NULL;
#ifdef MORPHO_PROFILER
    v->profiler=NULL;
    v->status=VM_RUNNING;
#endif
    v->parent=NULL;
    varray_vminit(&v->subkernels);
    
    v->printfn=NULL;
    v->printref=NULL;
    v->inputfn=NULL;
    v->inputref=NULL;
    v->warningfn=NULL;
    v->warningref=NULL;
    v->debuggerfn=NULL;
    v->debuggerref=NULL;
}

/** Clears a virtual machine */
static void vm_clear(vm *v) {
    varray_valueclear(&v->stack);
    varray_valueclear(&v->globals);
    varray_valueclear(&v->tlvars);
    varray_valueclear(&v->retain);
    vm_graylistclear(&v->gray);
    varray_charclear(&v->buffer);
    vm_freeobjects(v);
    
    for (int i=0; i<v->subkernels.count; i++) {
        vm *subkernel = v->subkernels.data[i];
        subkernel->globals.capacity=0; // Globals duplicates the parent vm so ignore
        subkernel->globals.data=NULL;
        vm_clear(subkernel);
        MORPHO_FREE(subkernel);
    }
    varray_vmclear(&v->subkernels);
}

/** Prepares a vm to run program p */
bool vm_start(vm *v, program *p) {
    /* Set the current program */
    v->current=p;

    /* Clear current error state */
    error_clear(&v->err);
    v->errfp=NULL;

    /* Set up the callframe stack */
    v->fp=v->frame; /* Set the frame pointer to the bottom of the stack */
    v->fp->function=p->global;
    v->fp->closure=NULL;
    v->fp->roffset=0;
    v->fp->nopt=0;
    
#ifdef MORPHO_PROFILER
    v->fp->inbuiltinfunction=NULL;
#endif
    
    /* Set instruction base */
    v->instructions = p->code.data;
    if (!v->instructions) return false;

    /* Set up the constant table */
    varray_value *konsttable=object_functiongetconstanttable(p->global);
    if (!konsttable) return false;
    v->konst = konsttable->data;
    
    return true;
}

/** Frees all objects bound to a virtual machine */
void vm_freeobjects(vm *v) {
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    long k = 0;
    morpho_printf(v, "--- Freeing objects bound to VM ---\n");
#endif
    object *next=NULL;
    for (object *e=v->objects; e!=NULL; e=next) {
        next = e->next;
        object_free(e);
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
        k++;
#endif
    }

#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    morpho_printf(v, "--- Freed %li objects bound to VM ---\n", k);
#endif
}

/* **********************************************************************
* Binding and unbinding objects to the VM
* ********************************************************************** */

/** Unbinds an object from a VM. */
void vm_unbindobject(vm *v, value obj) {
    object *ob=MORPHO_GETOBJECT(obj);
    
    if (v->objects==ob) {
        v->objects=ob->next;
    } else {
        for (object *e=v->objects; e!=NULL; e=e->next) {
            if (e->next==ob) { e->next=ob->next; break; }
        }
    }
    // Correct estimate of bound size.
    if (MORPHO_ISGARBAGECOLLECTED(obj)) {
        v->bound-=object_size(ob);
        ob->status=OBJECT_ISUNMANAGED;
    }
}

/** @brief Binds an object to a Virtual Machine.
 *  @details Any object created during execution should be bound to a VM; this object is then managed by the garbage collector.
 *  @param v      the virtual machine
 *  @param obj    object to bind */
static void vm_bindobject(vm *v, value obj) {
    object *ob = MORPHO_GETOBJECT(obj);
    ob->status=OBJECT_ISUNMARKED;
    ob->next=v->objects;
    v->objects=ob;
    size_t size=object_size(ob);
#ifdef MORPHO_DEBUG_GCSIZETRACKING
    dictionary_insert(&sizecheck, obj, MORPHO_INTEGER(size));
#endif

    v->bound+=size;

#ifdef MORPHO_DEBUG_STRESSGARBAGECOLLECTOR
    vm_collectgarbage(v);
#else
    if (v->bound>v->nextgc) vm_collectgarbage(v);
#endif
}

/** @brief Binds an object to a Virtual Machine without garbage collection.
 *  @details Any object created during execution should be bound to a VM; this object is then managed by the garbage collector.
 *  @param v      the virtual machine
 *  @param obj    object to bind
 *  @warning: This should only be used in circumstances where the internal state of the VM is not consistent (i.e. calling the GC could cause a sigsev) */
static void vm_bindobjectwithoutcollect(vm *v, value obj) {
    object *ob = MORPHO_GETOBJECT(obj);
    ob->status=OBJECT_ISUNMARKED;
    ob->next=v->objects;
    v->objects=ob;
    size_t size=object_size(ob);
#ifdef MORPHO_DEBUG_GCSIZETRACKING
    dictionary_insert(&sizecheck, obj, MORPHO_INTEGER(size));
#endif

    v->bound+=size;
}

/* **********************************************************************
* Virtual machine
* ********************************************************************** */

/** @brief Raises a runtime error
 * @param v        the virtual machine
 * @param id       error id
 * @param ...      additional data for sprintf. */
void vm_runtimeerror(vm *v, ptrdiff_t iindx, errorid id, ...) {
    va_list args;
    int line=ERROR_POSNUNIDENTIFIABLE, posn=ERROR_POSNUNIDENTIFIABLE;
    value module=MORPHO_NIL;
    debug_infofromindx(v->current, iindx, &module, &line, &posn, NULL, NULL);
    
    va_start(args, id);
    morpho_writeerrorwithidvalist(&v->err, id, NULL, line, posn, args);
    va_end(args);
}

/** @brief Raises a BadOp error
 * @param v        the virtual machine
 * @param id       error id
 * @param op       the opertion that went bad in (human readable)
 * @param left     the left hand side of the bad operation
 * @param right    the right hand side of the bad operation */
void vm_throwOpError(vm *v, ptrdiff_t iindx, errorid id, char* op, value left, value right){
    varray_char left_buffer;
    varray_char right_buffer;
    varray_charinit(&left_buffer);
    varray_charinit(&right_buffer);
    morpho_printtobuffer(v, left, &left_buffer);
    morpho_printtobuffer(v, right, &right_buffer);
    varray_charresize(&left_buffer,left_buffer.count);
    varray_charresize(&right_buffer,right_buffer.count);

    // ensure the the rest of the alocated data is empty
    for (int i = left_buffer.count; i<left_buffer.capacity; i++ ){
        varray_charwrite(&left_buffer,'\0');
    }

    for (int i = right_buffer.count; i<right_buffer.capacity; i++ ){
        varray_charwrite(&right_buffer,'\0');
    }

    vm_runtimeerror(v,iindx,id,op,left_buffer.data,right_buffer.data);
    varray_charclear(&left_buffer);
    varray_charclear(&right_buffer);
    }


/** @brief Captures an upvalue
 *  @param v        the virtual machine
 *  @param reg      register to capture
 *  @returns an objectupvalue */
static inline objectupvalue *vm_captureupvalue(vm *v, value *reg) {
    objectupvalue *prev = NULL;
    objectupvalue *up = v->openupvalues;
    objectupvalue *new = NULL;

    /* Is there an existing open upvalue that points to the same location? */
    for (;up!=NULL && up->location>reg;up=up->next) {
        prev=up;
    }

    /* If so, return it */
    if (up != NULL && up->location==reg) return up;

    /* If not create a new one */
    new=object_newupvalue(reg);

    if (new) {
        /* And link it into the list */
        new->next=up;
        if (prev) {
            prev->next=new;
        } else {
            v->openupvalues=new;
        }
        vm_bindobject(v, MORPHO_OBJECT(new));
    }

    return new;
}

/** @brief Closes upvalues that refer beyond a specified register
 *  @param v        the virtual machine
 *  @param reg      register to capture */
static inline void vm_closeupvalues(vm *v, value *reg) {
    while (v->openupvalues!=NULL && v->openupvalues->location>=reg) {
        objectupvalue *up = v->openupvalues;

        up->closed=*up->location; /* Store closed value */
        up->location=&up->closed; /* Point to closed value */
        v->openupvalues=up->next; /* Delink from openupvalues list */
        up->next=NULL;
    }
}

/** Check if we should break at a given pc */
bool vm_shouldbreakatpc(vm *v, instruction *pc) {
    if (!v->debug) return false;
    if (debugger_insinglestep(v->debug)) return true;
    instructionindx iindx = pc-v->current->code.data-1;
    if (debugger_shouldbreakat(v->debug, iindx)) return true;
    return false;
}

/** Return the previous instruction index */
instructionindx vm_previnstruction(vm *v) {
    if (v->fp->pc>v->current->code.data) return v->fp->pc-v->current->code.data-1;
    return 0;
}

/** Return the current instruction index */
instructionindx vm_currentinstruction(vm *v) {
    return v->fp->pc-v->current->code.data-1;
}

/** Return the current debugger */
debugger *vm_getdebugger(vm *v) {
    return v->debug;
}

/** @brief Expands the stack by a specified amount
 *  @param v        the virtual machine
 *  @param reg      the current register base
 *  @param n        Number of stack spaces to expand by */
static inline void vm_expandstack(vm *v, value **reg, unsigned int n) {
    if (v->stack.count+n>v->stack.capacity) {
        /* Calculate new size */
        unsigned int newsize=MORPHO_STACKGROWTHFACTOR*v->stack.capacity;
        if (newsize<morpho_powerof2ceiling(n)) newsize=morpho_powerof2ceiling(n);

        /* Preserve the offset of the old stack pointer into the stack */
        ptrdiff_t roffset=*reg-v->stack.data;

        varray_ptrdiff diff;
        varray_ptrdiffinit(&diff);

        /* Preserve open upvalue offsets */
        for (objectupvalue *u=v->openupvalues; u!=NULL; u=u->next) {
            ptrdiff_t p=u->location-v->stack.data;
            varray_ptrdiffadd(&diff, &p, 1);
        }

        /* Resize the stack */
        varray_valueresize(&v->stack, newsize);

        /* Recalculate upvalues */
        unsigned int k=0;
        for (objectupvalue *u=v->openupvalues; u!=NULL; u=u->next) {
            u->location=v->stack.data+diff.data[k];
            k++;
        }

        /* Free our varray of ptrdiffs */
        varray_ptrdiffclear(&diff);

        /* Correct the stack pointer */
        *reg = v->stack.data+roffset;
    }
    v->stack.count+=n;
}

/** Process optional args */
bool vm_optargs(vm *v, ptrdiff_t iindx, objectfunction *func, unsigned int nopt, value *args, value *outreg) {
    unsigned int nfopt = func->opt.count;
    
    // Copy across default values
    for (unsigned int i=0; i<nfopt; i++) {
        outreg[i]=func->konst.data[func->opt.data[i].def];
    }

    for (unsigned int i=0; i<nopt; i++) {
        value key = args[2*i];
        // TODO: Is a dictionary lookup faster here?
        unsigned int k=0;
        for (; k<func->nopt; k++) if (MORPHO_ISSAME(func->opt.data[k].symbol, key)) break;
        if (k>=func->nopt) { // If we didn't find a match, we're done with optional arguments
            if (MORPHO_ISSTRING(key)) {
                vm_runtimeerror(v, iindx, VM_UNKNWNOPTARG, MORPHO_GETCSTRING(key));
                return false;
            }
            break;
        }
        outreg[k]=args[2*i+1];
    }
    return true;
}

/** Process variadic and optional arguments
 * @param[in] v          - the VM
 * @param[in] iindx - instruction index (used to raise errors if need be)
 * @param[in] func    - function being called
 * @param[in] regcall - Register for the call
 * @param[in] nargs  - number of arguments being called with
 * @param[in] args - arguments used
 * @param[in] reg       - register base
 * @param[in] newreg - new register base
 */
static inline bool vm_vargs(vm *v, ptrdiff_t iindx, objectfunction *func, unsigned int nargs, value *args, value *newreg) {
    int nfixed = func->nargs, // No. of fixed params
        rVarg = nfixed+1, // Position of first optional parameter in output
        nposn=nargs; // No. of positional arguments this function was called with

    if (func->varg>=0) {
        if (nposn<nfixed) {
            vm_runtimeerror(v, iindx, VM_INVALIDARGS, nfixed-1, nposn);
            return false;
        }

        objectlist *new = object_newlist(nposn-nfixed, args+nfixed);
        if (new) {
            newreg[rVarg] = MORPHO_OBJECT(new);
            vm_bindobjectwithoutcollect(v, newreg[rVarg]);
        }
    } else if (nposn!=nfixed) { // Verify number of fixed args is correct
        vm_runtimeerror(v, iindx, VM_INVALIDARGS, nfixed, nposn);
        return false;
    }

    return true;
}


/** @brief Performs a function call
 *  @details A function call involves:
 *           1. Saving the program counter, register index and stacksize to the callframe stack;
 *           2. Advancing the frame pointer;
 *           3. Extracting the function from a closure if necessary;
 *           4. Expanding the stack if necessary
 *             Copy arguments to appropriate registers
 *           5. Loading the constant table from the function definition
 *           6. Shifting the register base
 *           7. Moving the program counter to the function
 * @param[in]  v                         The virtual machine
 * @param[in]  fn                       Function to call
 * @param[in]  regcall            rshift becomes r0 in the new call frame
 * @param[in]  nargs                number of positional arguments
 * @param[in]  nopt                  number of optional arguments
 * @param[in]  args                  pointer to the list of args
 * @param[out] pc                       program counter, updated
 * @param[out] reg                     register/stack pointer, updated */
static inline bool vm_call(vm *v, value fn, unsigned int regcall, unsigned int nargs, unsigned int nopt, value *args, instruction **pc, value **reg) {
    objectfunction *func = MORPHO_GETFUNCTION(fn);
    bool argsonstack=true;
    ptrdiff_t aoffset=0;

    value *arglist=args;
    if (arglist) {
        /** Determine whether the arguments provided are on the stack or not */
        argsonstack=(v->stack.data && arglist>v->stack.data && arglist<v->stack.data+v->stack.capacity);
    } else { /** Otherwise use the registers after regcall */
        arglist=(*reg)+regcall+1;
    }
    
    if (argsonstack) aoffset=arglist-v->stack.data; /** If args on stack retain where they are */
    
    /* In the old frame... */
    v->fp->pc=*pc; /* Save the program counter */
    v->fp->stackcount=v->fp->function->nregs+(unsigned int) v->fp->roffset; /* Store the stacksize */
    v->fp->returnreg=regcall; /* Store the return register */
    unsigned int oldnregs = v->fp->function->nregs; /* Get the old number of registers */

    if (v->fp==v->fpmax) { // Detect stack overflow
        vm_runtimeerror(v, (*pc) - v->instructions, VM_STCKOVFLW);
        return false;
    }
    v->fp++; /* Advance frame pointer */
    v->fp->pc=*pc; /* We will also store the program counter in the new frame;
                      this will be used to detect whether the VM should return on OP_RETURN */
#ifdef MORPHO_PROFILER
    v->fp->inbuiltinfunction=NULL;
#endif

    if (MORPHO_ISCLOSURE(fn)) {
        objectclosure *closure=MORPHO_GETCLOSURE(fn); /* Closure object in use */
        func=closure->func;
        v->fp->closure=closure;
    } else {
        v->fp->closure=NULL;
    }

    v->fp->ret=false; /* Interpreter should not return from this frame */
    v->fp->function=func; /* Store the function */

    /* Do we need to expand the stack? */
    if (v->stack.count+func->nregs>v->stack.capacity) {
        vm_expandstack(v, reg, func->nregs); /* Expand the stack */
    } else {
        v->stack.count+=func->nregs;
    }

    v->konst = func->konst.data; /* Load the constant table */
    *reg += oldnregs; /* Shift the register frame */
    v->fp->roffset=*reg-v->stack.data; /* Store the register index */
    
    /* Copy arguments into new register window */
    if (argsonstack) {
        arglist = v->stack.data+aoffset; // If they were on the stack, the pointer may be invalid so update it.
        (*reg)[0] = arglist[-1]; // Copy the caller into r0
    } else {
        (*reg)[0] = fn;
    }
    
    for (unsigned int i=0; i<nargs; i++) (*reg)[i+1] = arglist[i];

    int nvarg=0;
    if (func->varg>=0) {
        if (!vm_vargs(v, (*pc) - v->instructions, func, nargs, arglist, *reg)) return false;
        nvarg=1;
    } else if (func->nargs!=nargs) {
        vm_runtimeerror(v, (*pc) - v->instructions, VM_INVALIDARGS, func->nargs, nargs);
        return false;
    }
    
    /* Handle optional args */
    if (func->opt.count>0) {
        if (!vm_optargs(v, (*pc) - v->instructions, func, nopt, arglist+nargs, (*reg)+func->nargs+nvarg+1)) return false;
    } else if (nopt>0) {
        vm_runtimeerror(v, (*pc) - v->instructions, VM_NOOPTARG);
        return false;
    }

    /* Zero out registers beyond args up to the top of the stack
       This has to be fast: memset was too slow. Zero seems to be faster than MORPHO_NIL */
    for (value *r = *reg + func->nregs-1; r > *reg + func->nargs + func->nopt + nvarg; r--) *r = MORPHO_INTEGER(0);

    *pc=v->instructions+func->entry; /* Jump to the function */
    return true;
}

/** Invokes a method on a given object by name */
static inline bool vm_invoke(vm *v, value obj, value method, int nargs, value *args, value *out) {
    if (MORPHO_ISINSTANCE(obj)) {
        /* Look up the method */
        objectinstance *instance=MORPHO_GETINSTANCE(obj);
        value fn=MORPHO_NIL;
        if (dictionary_getintern(&instance->klass->methods, method, &fn)) {
            return morpho_invoke(v, obj, fn, nargs, args, out);
        }
    } else if (MORPHO_ISCLASS(obj)) {
        objectclass *klass=MORPHO_GETCLASS(obj);
        value fn=MORPHO_NIL;
        if (dictionary_getintern(&klass->methods, method, &fn)) {
            return morpho_invoke(v, obj, fn, nargs, args, out);
        }
    } else if (MORPHO_ISOBJECT(obj)) {
        /* If it's an object, it may have a veneer class */
        objectclass *klass = object_getveneerclass(MORPHO_GETOBJECTTYPE(obj));
        if (klass) {
            value ifunc;
            if (dictionary_getintern(&klass->methods, method, &ifunc)) {
                if (MORPHO_ISBUILTINFUNCTION(ifunc)) {
                    value sargs[nargs+1];
                    sargs[0]=obj;
                    for (unsigned int i=0; i<nargs; i++) sargs[i+1]=args[i];
                    *out = (MORPHO_GETBUILTINFUNCTION(ifunc)->function) (v, nargs, sargs);
                    return true;
                }
            }
        }
    }
    return false;
}

/** Recovers the number of optional arguments */
int vm_getoptionalargs(vm *v) {
    return v->fp->nopt;
}

/** @brief   Executes a sequence of code
 *  @param   v       The virtual machine to use
 *  @param   rstart  Starting register pointer
 *  @param   istart  Instruction to begin at
 *  @returns A morpho error */
bool morpho_interpret(vm *v, value *rstart, instructionindx istart) {
    /* Set the register pointer to the bottom of the stack */
    value *reg = rstart;

    /* Set the program counter to start */
    instruction *pc=v->instructions+istart; /* Pointer to the next instruction to be executed */

    /* Temporary variables to */
    int op=OP_NOP, a, b, c; /* Opcode and operands a, b, c */
    instruction bc; /* The current bytecode */
    value left, right;

#ifdef MORPHO_DEBUG_PRINT_INSTRUCTIONS
#define MORPHO_DISASSEMBLE_INSRUCTION(bc,pc,k,r) { morpho_printf(v, "  ");  debugger_disassembleinstruction(v, bc, pc-1, k, r); morpho_printf(v, "\n"); }
#else
#define MORPHO_DISASSEMBLE_INSRUCTION(bc,pc,k,r);
#endif

#ifdef MORPHO_OPCODE_USAGE
    unsigned long opcount[OP_END+1];
    unsigned long opopcount[OP_END+1][OP_END+1];
    for (unsigned int i=0; i<OP_END+1; i++) {
        opcount[i]=0;
        for (unsigned int j=0; j<OP_END+1; j++) { opopcount[i][j]=0; }
    }
    #define OPCODECNT(p) { opcount[p]++; }
    #define OPOPCODECNT(p, bc) { opopcount[op][DECODE_OP(bc)]++; }
#else
    #define OPCODECNT(p)
    #define OPOPCODECNT(p, bc)
#endif

#define ENTERDEBUGGER() { \
    v->fp->pc=pc; \
    v->fp->roffset=reg-v->stack.data; \
    debugger_enter(v->debug, v); \
}

/** Define the interpreter loop. Computed gotos or regular switch statements can be used here. */
#ifdef MORPHO_COMPUTED_GOTO
    /** The dispatch table, containing the entry points for each opcode */
    static void* dispatchtable[] = {
      #define OPCODE(name) &&code_##name,
      #include "opcodes.h"
      #undef OPCODE
    };
    
    static void* debugdispatchtable[] = { // Backup copy of the dispatch table
      #define OPCODE(name) &&code_##name,
      #include "opcodes.h"
      #undef OPCODE
    };
    
    /** The interpret loop begins by dispatching an instruction */
    #define INTERPRET_LOOP    DISPATCH();

    /** Create a label corresponding to each opcode */
    #define CASE_CODE(name)   code_##name
    
    int nopcodes;
    for (nopcodes=0; dispatchtable[nopcodes]!=&&code_END; ) nopcodes++;
    
    #define DEBUG_ENABLE() { if (v->debug) for (int i=0; i<nopcodes; i++) dispatchtable[i]=&&code_BREAK; }
    #define DEBUG_DISABLE() { if (v->debug) for (int i=0; i<nopcodes; i++) dispatchtable[i]=debugdispatchtable[i]; }
    
    /** Dispatch here means fetch the next instruction, decode and jump */
    #define FETCHANDDECODE()                                                 \
        {                                                                    \
            bc=*pc++;                                                        \
            OPOPCODECNT(pp, bc)                                              \
            op=DECODE_OP(bc);                                                \
            OPCODECNT(op)                                                    \
            MORPHO_DISASSEMBLE_INSRUCTION(bc,pc-v->instructions,v->konst, reg)     \
        }
    
    #define DISPATCH()                                                       \
        do {                                                                 \
            FETCHANDDECODE()                                                 \
            goto *dispatchtable[op];                                         \
        } while(false);
    
#else
    /** Every iteration of the interpret loop we fetch, decode and switch */
    #define INTERPRET_LOOP                                                   \
        loop:                                                                \
        bc=*pc++;                                                            \
        OPOPCODECNT(pp, bc)                                                  \
        op=DECODE_OP(bc);                                                    \
        OPCODECNT(op)                                                        \
        MORPHO_DISASSEMBLE_INSRUCTION(bc,pc-v->instructions,v->konst, reg)   \
        if (vm_shouldbreakatpc(v, pc)) ENTERDEBUGGER();                   \
        switch (op)

    /** Each opcode generates a case statement */
    #define CASE_CODE(name)  case OP_##name

    /** Dispatch means return to the beginning of the loop */
    #define DISPATCH() goto loop;
#endif

/** Macros for error checking */
#define ERROR(id) { vm_runtimeerror(v, pc-v->instructions, id); goto vm_error; }
#define VERROR(id, ...) { vm_runtimeerror(v, pc-v->instructions, id, __VA_ARGS__); goto vm_error; }
#define OPERROR(op){vm_throwOpError(v,pc-v->instructions,VM_INVLDOP,op,left,right); goto vm_error; }
#define ERRORCHK() if (v->err.cat!=ERROR_NONE) goto vm_error;
    
/** Macro to redirect an opcode to a method call on an object */
#define OPREDIRECT(leftselector, rightselector, regout) \
    if (MORPHO_ISOBJECT(left)) { \
        if (vm_invoke(v, left, leftselector, 1, &right, &reg[regout])) { \
            ERRORCHK(); \
            if (!MORPHO_ISNIL(reg[a])) DISPATCH(); \
        } \
    } \
    if (MORPHO_ISOBJECT(right)) { \
        if (vm_invoke(v, right, rightselector, 1, &left, &reg[regout])) { \
            ERRORCHK(); \
            DISPATCH(); \
        } \
    }
    
    INTERPRET_LOOP
    {
        CASE_CODE(NOP):
            DISPATCH();

        CASE_CODE(MOV):
            a=DECODE_A(bc); b=DECODE_B(bc);
            reg[a] = reg[b];
            DISPATCH();

        CASE_CODE(LCT):
            a=DECODE_A(bc); b=DECODE_Bx(bc);
            reg[a] = v->konst[b];
            DISPATCH();

        CASE_CODE(ADD):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[b];
            right = reg[c];

            if (MORPHO_ISFLOAT(left)) {
                if (MORPHO_ISFLOAT(right)) {
                    reg[a] = MORPHO_FLOAT( MORPHO_GETFLOATVALUE(left) + MORPHO_GETFLOATVALUE(right));
                    DISPATCH();
                } else if (MORPHO_ISINTEGER(right)) {
                    reg[a] = MORPHO_FLOAT( MORPHO_GETFLOATVALUE(left) + (double) MORPHO_GETINTEGERVALUE(right));
                    DISPATCH();
                }
            } else if (MORPHO_ISINTEGER(left)) {
                if (MORPHO_ISFLOAT(right)) {
                    reg[a] = MORPHO_FLOAT( (double) MORPHO_GETINTEGERVALUE(left) + MORPHO_GETFLOATVALUE(right));
                    DISPATCH();
                } else if (MORPHO_ISINTEGER(right)) {
                    reg[a] = MORPHO_INTEGER( MORPHO_GETINTEGERVALUE(left) + MORPHO_GETINTEGERVALUE(right));
                    DISPATCH();
                }
            } else if (MORPHO_ISSTRING(left) && MORPHO_ISSTRING(right)) {
                reg[a] = object_concatenatestring(left, right);
                if (!MORPHO_ISNIL(reg[a])) {
                    vm_bindobject(v, reg[a]);
                    DISPATCH();
                } else {
                    ERROR(VM_CNCTFLD);
                    DISPATCH();
                }
            }

            OPREDIRECT(addselector, addrselector, a);
            
            OPERROR("Add");
            DISPATCH();

        CASE_CODE(SUB):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[b];
            right = reg[c];

            if (MORPHO_ISFLOAT(left)) {
                if (MORPHO_ISFLOAT(right)) {
                    reg[a] = MORPHO_FLOAT( MORPHO_GETFLOATVALUE(left) - MORPHO_GETFLOATVALUE(right));
                    DISPATCH();
                } else if (MORPHO_ISINTEGER(right)) {
                    reg[a] = MORPHO_FLOAT( MORPHO_GETFLOATVALUE(left) - (double) MORPHO_GETINTEGERVALUE(right));
                    DISPATCH();
                }
            } else if (MORPHO_ISINTEGER(left)) {
                if (MORPHO_ISFLOAT(right)) {
                    reg[a] = MORPHO_FLOAT( (double) MORPHO_GETINTEGERVALUE(left) - MORPHO_GETFLOATVALUE(right));
                    DISPATCH();
                } else if (MORPHO_ISINTEGER(right)) {
                    reg[a] = MORPHO_INTEGER( MORPHO_GETINTEGERVALUE(left) - MORPHO_GETINTEGERVALUE(right));
                    DISPATCH();
                }
            }
        
            OPREDIRECT(subselector, subrselector, a);

            OPERROR("Subtract")
            DISPATCH();

        CASE_CODE(MUL):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[b];
            right = reg[c];

            if (MORPHO_ISFLOAT(left)) {
                if (MORPHO_ISFLOAT(right)) {
                    reg[a] = MORPHO_FLOAT( MORPHO_GETFLOATVALUE(left) * MORPHO_GETFLOATVALUE(right));
                    DISPATCH();
                } else if (MORPHO_ISINTEGER(right)) {
                    reg[a] = MORPHO_FLOAT( MORPHO_GETFLOATVALUE(left) * (double) MORPHO_GETINTEGERVALUE(right));
                    DISPATCH();
                }
            } else if (MORPHO_ISINTEGER(left)) {
                if (MORPHO_ISFLOAT(right)) {
                    reg[a] = MORPHO_FLOAT( (double) MORPHO_GETINTEGERVALUE(left) * MORPHO_GETFLOATVALUE(right));
                    DISPATCH();
                } else if (MORPHO_ISINTEGER(right)) {
                    reg[a] = MORPHO_INTEGER( MORPHO_GETINTEGERVALUE(left) * MORPHO_GETINTEGERVALUE(right));
                    DISPATCH();
                }
            }

            OPREDIRECT(mulselector, mulrselector, a);

            OPERROR("Multiply")
            DISPATCH();

        CASE_CODE(DIV):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[b];
            right = reg[c];

            if (MORPHO_ISFLOAT(left)) {
                if (MORPHO_ISFLOAT(right)) {
                    reg[a] = MORPHO_FLOAT( MORPHO_GETFLOATVALUE(left) / MORPHO_GETFLOATVALUE(right));
                    DISPATCH();
                } else if (MORPHO_ISINTEGER(right)) {
                    reg[a] = MORPHO_FLOAT( MORPHO_GETFLOATVALUE(left) / (double) MORPHO_GETINTEGERVALUE(right));
                    DISPATCH();
                }
            } else if (MORPHO_ISINTEGER(left)) {
                if (MORPHO_ISFLOAT(right)) {
                    reg[a] = MORPHO_FLOAT( (double) MORPHO_GETINTEGERVALUE(left) / MORPHO_GETFLOATVALUE(right));
                    DISPATCH();
                } else if (MORPHO_ISINTEGER(right)) {
                    reg[a] = MORPHO_FLOAT( (double) MORPHO_GETINTEGERVALUE(left) / (double) MORPHO_GETINTEGERVALUE(right));
                    DISPATCH();
                }
            }
        
            OPREDIRECT(divselector, divrselector, a);

            OPERROR("Divide");
            DISPATCH();

        CASE_CODE(POW):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[b];
            right = reg[c];

            if (MORPHO_ISFLOAT(left)) {
                if (MORPHO_ISFLOAT(right)) {
                    reg[a] = MORPHO_FLOAT( pow(MORPHO_GETFLOATVALUE(left), MORPHO_GETFLOATVALUE(right)) );
                    DISPATCH();
                } else if (MORPHO_ISINTEGER(right)) {
                    reg[a] = MORPHO_FLOAT( pow(MORPHO_GETFLOATVALUE(left), (double) MORPHO_GETINTEGERVALUE(right)) );
                    DISPATCH();
                }
            } else if (MORPHO_ISINTEGER(left)) {
                if (MORPHO_ISFLOAT(right)) {
                    reg[a] = MORPHO_FLOAT( pow((double) MORPHO_GETINTEGERVALUE(left), MORPHO_GETFLOATVALUE(right)) );
                    DISPATCH();
                } else if (MORPHO_ISINTEGER(right)) {
                    reg[a] = MORPHO_FLOAT( pow((double) MORPHO_GETINTEGERVALUE(left), (double) MORPHO_GETINTEGERVALUE(right)) );
                    DISPATCH();
                }
            }

            OPREDIRECT(powselector, powrselector, a);

            OPERROR("Exponentiate")
            DISPATCH();


        CASE_CODE(NOT):
            a=DECODE_A(bc); b=DECODE_B(bc);
            left = reg[b];

            if (MORPHO_ISBOOL(left)) {
                reg[a] = MORPHO_BOOL(!MORPHO_GETBOOLVALUE(left));
            } else {
                reg[a] = MORPHO_BOOL(MORPHO_ISNIL(left));
            }
            DISPATCH();

        CASE_CODE(EQ):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[b];
            right = reg[c];

            reg[a] = (morpho_extendedcomparevalue(left, right)==0 ? MORPHO_BOOL(true) : MORPHO_BOOL(false));
            DISPATCH();

        CASE_CODE(NEQ):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[b];
            right = reg[c];

            reg[a] = (morpho_extendedcomparevalue(left, right)!=0 ? MORPHO_BOOL(true) : MORPHO_BOOL(false));
            DISPATCH();

        CASE_CODE(LT):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[b];
            right = reg[c];

            if ( !( (MORPHO_ISFLOAT(left) || MORPHO_ISINTEGER(left)) &&
                   (MORPHO_ISFLOAT(right) || MORPHO_ISINTEGER(right)) ) ) {
                OPERROR("Compare");
            }

            reg[a] = (morpho_extendedcomparevalue(left, right)>0 ? MORPHO_BOOL(true) : MORPHO_BOOL(false));
            DISPATCH();

        CASE_CODE(LE):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[b];
            right = reg[c];

            if ( !( (MORPHO_ISFLOAT(left) || MORPHO_ISINTEGER(left)) &&
                   (MORPHO_ISFLOAT(right) || MORPHO_ISINTEGER(right)) ) ) {
                OPERROR("Compare");
            }

            reg[a] = (morpho_extendedcomparevalue(left, right)>=0 ? MORPHO_BOOL(true) : MORPHO_BOOL(false));
            DISPATCH();

        CASE_CODE(B):
            b=DECODE_sBx(bc);
            pc+=b;
            DISPATCH();

        CASE_CODE(BIF):
            a=DECODE_A(bc);
            left=reg[a];

            if (MORPHO_ISTRUE(left)) pc+=DECODE_sBx(bc);
            DISPATCH();

        CASE_CODE(BIFF):
            a=DECODE_A(bc);
            left=reg[a];

            if (MORPHO_ISFALSE(left)) pc+=DECODE_sBx(bc);
            DISPATCH();

        CASE_CODE(CALL):
            a=DECODE_A(bc);
            left=reg[a];
            b=DECODE_B(bc); // Number of positional arguments
            c=DECODE_C(bc); // Number of optional arguments
        
callfunction: // Jump here if an instruction becomes a call
            if (MORPHO_ISINVOCATION(left)) {
                /* An method invocation */
                objectinvocation *inv = MORPHO_GETINVOCATION(left);
                left=inv->method;
                reg[a]=inv->receiver;
            }
        
            if (MORPHO_ISMETAFUNCTION(left) &&
                !metafunction_resolve(MORPHO_GETMETAFUNCTION(left), b, reg+a+1,                 &v->err, &left)) {
                ERRORCHK();
                ERROR(VM_MLTPLDSPTCHFLD);
            }

            if (MORPHO_ISFUNCTION(left) || MORPHO_ISCLOSURE(left)) {
                if (!vm_call(v, left, a, b, c, NULL, &pc, &reg)) goto vm_error;

            } else if (MORPHO_ISBUILTINFUNCTION(left)) {
                /* Save program counter in the old callframe */
                v->fp->pc=pc;
                v->fp->nopt=c;

                objectbuiltinfunction *f = MORPHO_GETBUILTINFUNCTION(left);

#ifdef MORPHO_PROFILER
                v->fp->inbuiltinfunction=f;
#endif
                value ret = (f->function) (v, b, reg+a);
#ifdef MORPHO_PROFILER
                v->fp->inbuiltinfunction=NULL;
#endif
                ERRORCHK();
                reg=v->stack.data+v->fp->roffset; /* Ensure register pointer is correct */
                reg[a]=ret;

            } else if (MORPHO_ISCLASS(left)) {
                /* A function call on a class instantiates it */
                objectclass *klass = MORPHO_GETCLASS(left);
                objectinstance *instance = object_newinstance(klass);
                if (instance) {
                    reg[a] = MORPHO_OBJECT(instance);
                    vm_bindobject(v, reg[a]);

                    /* Call the initializer if class provides one */
                    value ifunc;
                    if (dictionary_getintern(&klass->methods, initselector, &ifunc)) {
                        if (MORPHO_ISMETAFUNCTION(ifunc) &&
                            !metafunction_resolve(MORPHO_GETMETAFUNCTION(ifunc), b, reg+a+1, &v->err, &ifunc)) {
                            ERRORCHK();
                            ERROR(VM_MLTPLDSPTCHFLD);
                        }
                        
                        /* If so, call it */
                        if (MORPHO_ISFUNCTION(ifunc)) {
                            if (!vm_call(v, ifunc, a, b, c, NULL, &pc, &reg)) goto vm_error;
                        } else if (MORPHO_ISBUILTINFUNCTION(ifunc)) {
                            v->fp->nopt=c;
#ifdef MORPHO_PROFILER
                            v->fp->inbuiltinfunction=MORPHO_GETBUILTINFUNCTION(ifunc);
#endif
                            (MORPHO_GETBUILTINFUNCTION(ifunc)->function) (v, b, reg+a);
#ifdef MORPHO_PROFILER
                            v->fp->inbuiltinfunction=NULL;
#endif
                            ERRORCHK();
                        }
                    } else {
                        if (b>0) {
                            VERROR(VM_NOINITIALIZER, MORPHO_GETCSTRING(klass->name));
                        }
                    }
                } else {
                    ERROR(VM_INSTANTIATEFAILED);
                }
            } else {
                ERROR(VM_UNCALLABLE);
            }
            DISPATCH();
        
        CASE_CODE(METHOD): // Direct method call
            a=DECODE_A(bc);
            b=DECODE_B(bc);
            c=DECODE_C(bc);
        
            right=reg[a]; // The method
            left=reg[a+1]; // The object
        
            if (MORPHO_ISMETAFUNCTION(right)) {
                if (!metafunction_resolve(MORPHO_GETMETAFUNCTION(right), b, reg+a+2, &v->err, &right)) {
                    ERRORCHK();
                    ERROR(VM_MLTPLDSPTCHFLD);
                }
            }
        
            if (MORPHO_ISFUNCTION(right)) {
                if (!vm_call(v, right, a+1, b, c, NULL, &pc, &reg)) goto vm_error;
            } else if (MORPHO_ISBUILTINFUNCTION(right)) {
                v->fp->nopt=c;
    #ifdef MORPHO_PROFILER
                v->fp->inbuiltinfunction=MORPHO_GETBUILTINFUNCTION(right);
    #endif
                value ret = (MORPHO_GETBUILTINFUNCTION(right)->function) (v, b, reg+a+1);
                reg=v->fp->roffset+v->stack.data; /* Restore registers */
                reg[a+1] = ret;
    #ifdef MORPHO_PROFILER
                v->fp->inbuiltinfunction=NULL;
    #endif
                ERRORCHK();
            }
        
        DISPATCH();

        CASE_CODE(INVOKE):
            a=DECODE_A(bc);
            b=DECODE_B(bc);
            c=DECODE_C(bc);
            right=reg[a]; // The method
            left=reg[a+1]; // The object

            if (MORPHO_ISINSTANCE(left)) {
                objectinstance *instance = MORPHO_GETINSTANCE(left);
                value ifunc;

                /* Check if we have this method */
                if (dictionary_getintern(&instance->klass->methods, right, &ifunc)) {

                    if (MORPHO_ISMETAFUNCTION(ifunc)) {
                        if (!metafunction_resolve(MORPHO_GETMETAFUNCTION(ifunc), b, reg+a+2, &v->err, &ifunc)) {
                            ERRORCHK();
                            ERROR(VM_MLTPLDSPTCHFLD);
                        }
                    }
                    
                    /* If so, call it */
                    if (MORPHO_ISFUNCTION(ifunc)) {
                        if (!vm_call(v, ifunc, a+1, b, c, NULL, &pc, &reg)) goto vm_error;
                    } else if (MORPHO_ISBUILTINFUNCTION(ifunc)) {
                        v->fp->nopt=c;
#ifdef MORPHO_PROFILER
                        v->fp->inbuiltinfunction=MORPHO_GETBUILTINFUNCTION(ifunc);
#endif
                        value ret = (MORPHO_GETBUILTINFUNCTION(ifunc)->function) (v, b, reg+a+1);
                        reg=v->fp->roffset+v->stack.data; /* Restore registers */
                        reg[a+1] = ret;
#ifdef MORPHO_PROFILER
                        v->fp->inbuiltinfunction=NULL;
#endif
                        ERRORCHK();
                    }
                } else if (dictionary_getintern(&instance->fields, right, &left)) {
                    
                    /* Otherwise, if it's a property, try to call it */
                    if (morpho_iscallable(left)) {
                        a+=1;
                        reg[a]=left; // Make sure the function is in r0
                        goto callfunction; // Transmute into a call instruction
                    } else {
                        ERROR(VM_UNCALLABLE);
                    }
                } else {
                    /* Otherwise, raise an error */
                    char *p = (MORPHO_ISSTRING(right) ? MORPHO_GETCSTRING(right) : "");
                    VERROR(VM_OBJECTLACKSPROPERTY, p);
                }
            } else if (MORPHO_ISCLASS(left)) {
                objectclass *klass = MORPHO_GETCLASS(left);
                value ifunc;

                if (dictionary_getintern(&klass->methods, right, &ifunc)) {
                    /* If we're not in the global context, invoke the method on self which is in r0 */
                    if (v->fp>v->frame) reg[a+1]=reg[0]; /* Copy self into r[a+1] and call */

                    if (MORPHO_ISMETAFUNCTION(ifunc)) {
                        if (!metafunction_resolve(MORPHO_GETMETAFUNCTION(ifunc), b, reg+a+2, &v->err, &ifunc)) {
                            ERRORCHK();
                            ERROR(VM_MLTPLDSPTCHFLD);
                        }
                    }
                    
                    if (MORPHO_ISFUNCTION(ifunc)) {
                        if (!vm_call(v, ifunc, a+1, b, c, NULL, &pc, &reg)) goto vm_error;
                    } else if (MORPHO_ISBUILTINFUNCTION(ifunc)) {
                        v->fp->nopt=c;
#ifdef MORPHO_PROFILER
                        v->fp->inbuiltinfunction=MORPHO_GETBUILTINFUNCTION(ifunc);
#endif
                        value ret = (MORPHO_GETBUILTINFUNCTION(ifunc)->function) (v, b, reg+a+1);
                        reg=v->fp->roffset+v->stack.data; /* Restore registers */
                        reg[a+1] = ret;
#ifdef MORPHO_PROFILER
                        v->fp->inbuiltinfunction=NULL;
#endif
                        ERRORCHK();
                    }
                } else {
                    /* Otherwise, raise an error */
                    char *p = (MORPHO_ISSTRING(right) ? MORPHO_GETCSTRING(right) : "");
                    VERROR(VM_CLASSLACKSPROPERTY, p);
                }
            } else {
                /* Check if the operand has a veneer class */
                objectclass *klass;
                if (MORPHO_ISOBJECT(left)) klass = object_getveneerclass(MORPHO_GETOBJECTTYPE(left));
                else klass = value_getveneerclass(left);
                
                if (klass) {
                    value ifunc;
                    if (dictionary_getintern(&klass->methods, right, &ifunc)) {
                        if (MORPHO_ISMETAFUNCTION(ifunc)) {
                            if (!metafunction_resolve(MORPHO_GETMETAFUNCTION(ifunc), b,  reg+a+2, &v->err, &ifunc)) {
                                ERRORCHK();
                                ERROR(VM_MLTPLDSPTCHFLD);
                            }
                        }
                        
                        if (MORPHO_ISBUILTINFUNCTION(ifunc)) {
                            v->fp->nopt=c;
#ifdef MORPHO_PROFILER
                            v->fp->inbuiltinfunction=MORPHO_GETBUILTINFUNCTION(ifunc);
#endif
                            value ret = (MORPHO_GETBUILTINFUNCTION(ifunc)->function) (v, b, reg+a+1);
                            reg=v->fp->roffset+v->stack.data; /* Restore registers */
                            reg[a+1] = ret;
#ifdef MORPHO_PROFILER
                            v->fp->inbuiltinfunction=NULL;
#endif
                            ERRORCHK();
                        }
                    } else {
                        char *p = (MORPHO_ISSTRING(right) ? MORPHO_GETCSTRING(right) : "");
                        VERROR(VM_CLASSLACKSPROPERTY, p);
                    }
                } else {
                    ERROR(VM_NOTANINSTANCE);
                }
            }

            DISPATCH();

        CASE_CODE(RETURN):
            a=DECODE_A(bc);

            if (v->openupvalues) { /* Close upvalues */
                vm_closeupvalues(v, reg);
            }
        
            if (v->ehp) { /* Remove any error handlers from this call frame */
                while (v->ehp->fp==v->fp &&
                       v->ehp>=v->errorhandlers) v->ehp--;
                if (v->ehp<v->errorhandlers) v->ehp=NULL; // If the stack is empty rest to NULL
            }

            value retvalue;

            if (a>0) {
                b=DECODE_B(bc);
                retvalue = reg[b];
            } else {
                retvalue = MORPHO_NIL; /* No return value; returns nil */
            }

            if (v->fp>v->frame) {
                bool shouldreturn = (v->fp->ret);
                // value *or = reg + v->fp->function->nargs;
                v->fp--;
                v->konst=v->fp->function->konst.data; /* Restore the constant table */
                reg=v->fp->roffset+v->stack.data; /* Restore registers */
                v->stack.count=v->fp->stackcount; /* Restore the stack size */

                reg[v->fp->returnreg]=retvalue; /* Copy the return value */
                // Clear registers
                // for (value *r = reg + v->fp->function->nregs-1; r > or; r--) *r = MORPHO_INTEGER(0);

                pc=v->fp->pc; /* Jump back */
                if (shouldreturn) return true;
                DISPATCH();
            } else {
                ERROR(VM_GLBLRTRN);
            }

        CASE_CODE(CLOSURE):
        {
            a=DECODE_A(bc);
            b=DECODE_B(bc);
            objectclosure *closure = object_newclosure(v->fp->function, MORPHO_GETFUNCTION(reg[a]), (indx) b);
            /* Now capture or copy upvalues from this frame */
            if (closure) {
                for (unsigned int i=0; i<closure->nupvalues; i++) {
                    upvalue *up = &v->fp->function->prototype.data[b].data[i];
                    if (up->islocal) {
                        closure->upvalues[i]=vm_captureupvalue(v, &reg[up->reg]);
                    } else {
                        if (v->fp->closure) closure->upvalues[i]=v->fp->closure->upvalues[up->reg];
                    }
                }

                reg[a] = MORPHO_OBJECT(closure);
                vm_bindobject(v, MORPHO_OBJECT(closure));
            }
        }
            DISPATCH();

        CASE_CODE(LUP):
            a=DECODE_A(bc);
            b=DECODE_B(bc);
            if (v->fp->closure && v->fp->closure->upvalues[b]) {
                reg[a]=*v->fp->closure->upvalues[b]->location;
            } else {
                UNREACHABLE("Closure unavailable");
            }
            DISPATCH();

        CASE_CODE(SUP):
            a=DECODE_A(bc);
            b=DECODE_B(bc);
            right = reg[b];
            if (v->fp->closure && v->fp->closure->upvalues[a]) {
                *v->fp->closure->upvalues[a]->location=right;
            } else {
                UNREACHABLE("Closure unavailable");
            }
            DISPATCH();

        CASE_CODE(LGL):
            a=DECODE_A(bc);
            b=DECODE_Bx(bc);
            reg[a]=v->globals.data[b];

            DISPATCH();

        CASE_CODE(SGL):
            a=DECODE_A(bc);
            b=DECODE_Bx(bc);
            v->globals.data[b]=reg[a];
            DISPATCH();

        CASE_CODE(CLOSEUP):
            a=DECODE_A(bc);
            vm_closeupvalues(v, &reg[a]);
            DISPATCH();

        CASE_CODE(LPR): /* Load property */
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[b];
            right = reg[c];

            if (MORPHO_ISINSTANCE(left)) {
                objectinstance *instance = MORPHO_GETINSTANCE(left);
                /* Is there a property with this id? */
                if (dictionary_getintern(&instance->fields, right, &reg[a])) {
                } else if (dictionary_getintern(&instance->klass->methods, right, &reg[a])) {
                    /* ... or a method? */
                    objectinvocation *bound=object_newinvocation(left, reg[a]);
                    if (bound) {
                        /* Bind into the VM */
                        reg[a]=MORPHO_OBJECT(bound);
                        vm_bindobject(v, reg[a]);
                    }
                } else if (dictionary_get(&instance->fields, right, &reg[a])) {
                } else {
                    /* Otherwise, raise an error */
                    char *p = (MORPHO_ISSTRING(right) ? MORPHO_GETCSTRING(right) : "");
                    VERROR(VM_OBJECTLACKSPROPERTY, p);
                }
            } else if (MORPHO_ISCLASS(left)) {
                /* If it's a class, we lookup the method and create the invocation */
                objectclass *klass = MORPHO_GETCLASS(left);
                if (klass && dictionary_get(&klass->methods, right, &reg[a])) {
                    objectinvocation *bound=object_newinvocation(left, reg[a]);
                    if (bound) {
                        /* Bind into the VM */
                        reg[a]=MORPHO_OBJECT(bound);
                        vm_bindobject(v, reg[a]);
                    }
                } else {
                    /* Otherwise, raise an error */
                    char *p = (MORPHO_ISSTRING(right) ? MORPHO_GETCSTRING(right) : "");
                    VERROR(VM_CLASSLACKSPROPERTY, p);
                }
            } else {
                /* Check for veneer class */
                objectclass *klass;
                if (MORPHO_ISOBJECT(left)) klass = object_getveneerclass(MORPHO_GETOBJECTTYPE(left));
                else klass = value_getveneerclass(left);
                
                if (klass) {
                    value ifunc;
                    if (dictionary_get(&klass->methods, right, &ifunc)) {
                        objectinvocation *bound=object_newinvocation(left, ifunc);
                        if (bound) {
                            /* Bind into the VM */
                            reg[a]=MORPHO_OBJECT(bound);
                            vm_bindobject(v, reg[a]);
                        }
                    } else {
                        char *p = (MORPHO_ISSTRING(right) ? MORPHO_GETCSTRING(right) : "");
                        VERROR(VM_CLASSLACKSPROPERTY, p);
                    }
                } else {
                    ERROR(VM_NOTANOBJECT);
                }
            }
            DISPATCH();

        CASE_CODE(SPR):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[a];
            right = reg[c];

            if (MORPHO_ISINSTANCE(left)) {
                objectinstance *instance = MORPHO_GETINSTANCE(left);
                left = reg[b];
                dictionary_insertintern(&instance->fields, left, right);
            } else {
                ERROR(VM_NOTANOBJECT);
            }

            DISPATCH();

        CASE_CODE(LIX):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[a];

            if (MORPHO_ISARRAY(left)) {
                unsigned int ndim = c-b+1;
                unsigned int indx[ndim];
                if (array_valuelisttoindices(ndim, &reg[b], indx)){
                    objectarrayerror err=array_getelement(MORPHO_GETARRAY(left), ndim, indx, &reg[b]);
                    if (err!=ARRAY_OK) ERROR( array_error(err) );
                } else {
                    value newval = MORPHO_NIL;
                    objectarrayerror err = getslice(&left,&array_slicedim,&array_sliceconstructor,\
                                                    &array_slicecopy,ndim,&reg[b],&newval);
                    if (err!=ARRAY_OK) ERROR(array_error(err));

                    if (!MORPHO_ISNIL(newval)) {
                        reg[b] = newval;
                        vm_bindobject(v, reg[b]);
                    } else  ERROR(VM_NONNUMINDX);
                }
            } else {
                if (!vm_invoke(v, left, indexselector, c-b+1, &reg[b], &reg[b])) {
                    ERROR(VM_NOTINDEXABLE);
                }
                ERRORCHK();
            }

            DISPATCH();
        
        CASE_CODE(LIXL):
        {
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            
            /* Variant 1: */
             value args[2] = { reg[b], reg[c] };
             reg[a] = List_getindex(v, 1, args);
             ERRORCHK();
            
            /* Variant 2: Potentially even faster 
            objectlist *list = MORPHO_GETLIST(reg[b]);
            int i=MORPHO_GETINTEGERVALUE(reg[c]);
            if (i>list->val.count) ERROR(VM_OUTOFBOUNDS);
            reg[a]=list->val.data[i];*/
            
            DISPATCH();
        }

        CASE_CODE(SIX):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[a];

            if (MORPHO_ISARRAY(left)) {
                unsigned int ndim = c-b;
                unsigned int indx[ndim];
                if (!array_valuelisttoindices(ndim, &reg[b], indx)) ERROR(VM_NONNUMINDX);
                objectarrayerror err=array_setelement(MORPHO_GETARRAY(left), ndim, indx, reg[c]);
                if (err!=ARRAY_OK) ERROR( array_error(err) );
            } else {
                if (!vm_invoke(v, left, setindexselector, c-b+1, &reg[b], &right)) {
                    ERROR(VM_NOTINDEXABLE);
                }
                ERRORCHK();
            }

            DISPATCH();

        CASE_CODE(PUSHERR):
            b=DECODE_Bx(bc);
            if (v->ehp && v->ehp>=v->errorhandlers+MORPHO_ERRORHANDLERSTACKSIZE-1) {
                ERROR(VM_ERRSTCKOVFLW);
            }
            if (!v->ehp) v->ehp=v->errorhandlers; else v->ehp++; // Add new error handler to the error stack
            v->ehp->fp=v->fp; // Store the current frame pointer
            v->ehp->dict=v->konst[b]; // Store the error handler dictionary from the constant table
            DISPATCH();

        CASE_CODE(POPERR):
            b=DECODE_sBx(bc); // Optional branch
            pc+=b;            // Advance program counter
            v->ehp--;         // Pull error handler off error stack
            if (v->ehp<v->errorhandlers) v->ehp=NULL; // If the stack is empty rest to NULL
            DISPATCH();

        CASE_CODE(CAT):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            reg[a]=morpho_concatenate(v, c-b+1, reg+b);
            vm_bindobject(v, reg[a]);
            DISPATCH();

        CASE_CODE(PRINT):
            a=DECODE_A(bc);
            left=reg[a];
            if (!vm_invoke(v, left, printselector, 0, NULL, &right)) {
                morpho_printvalue(v, left);
            }
            morpho_printf(v, "\n");
            DISPATCH();

        CASE_CODE(BREAK):
            if (v->debug) {
                if (vm_shouldbreakatpc(v, pc) ||
                    op==OP_BREAK) {
                    ENTERDEBUGGER();
                    ERRORCHK();
                }
                
#ifdef MORPHO_COMPUTED_GOTO
                // If using computed gotos, all instructions are routed through OP_BREAK
                // when the debugger is active. When this is happening we must perform a regular
                // dispatch after we've checked whether to enter the debugger
                bool debugdispatchactive = (dispatchtable[0]==&&code_BREAK);
                
                if (debugger_isactive(v->debug)) { // Check if singlestep or breakpoints are active  
                    if (!debugdispatchactive) DEBUG_ENABLE()
                } else if (debugdispatchactive) DEBUG_DISABLE()
                
                if (op==OP_BREAK) DISPATCH(); // Perform a regular dispatch if we stopped at OP_BREAK
                
                // If the debug dispatch table was active, must dispatch to execute the instruction
                if (debugdispatchactive) goto *debugdispatchtable[op];
#endif
            }
            DISPATCH();

        CASE_CODE(END):
            #ifdef MORPHO_OPCODE_USAGE
            {
                char *opname[] = {
                #define OPCODE(name) #name,
                #include "opcodes.h"
                    ""
                };
                #undef OPCODE
                for (unsigned int i=0; i<OP_END; i++) {
                    morpho_printf(v, "%s:\t\t%lu\n", opname[i], opcount[i]);
                }

                morpho_printf(v, ",");
                for (unsigned int i=0; i<OP_END; i++) morpho_printf(v, "%s, ", opname[i]);
                morpho_printf(v, "\n");

                for (unsigned int i=0; i<OP_END; i++) {
                    morpho_printf(v, "%s, ", opname[i]);
                    for (unsigned int j=0; j<OP_END; j++) {
                        morpho_printf(v, "%lu ", opopcount[i][j]);
                        if (j<OP_END-1) morpho_printf(v, ",");
                    }
                    morpho_printf(v, "\n");
                }
            }
            #endif
            return true;
    }

vm_error:
    {
        objectstring erridstring=MORPHO_STATICSTRING(v->err.id);
        value errid = MORPHO_OBJECT(&erridstring);

        /* Find the most recent callframe that requires us to return */
        callframe *retfp=NULL;
        for (retfp=v->fp; retfp>v->frame && !retfp->ret; retfp--);

        /* Search down the error stack for an error handler that can handle the error  */
        for (errorhandler *eh=v->ehp; eh && eh>=v->errorhandlers; eh--) {
            /* Abort if we pass an intermediate frame that requires us to return */
            if (eh->fp<retfp) {
                v->ehp=eh; // Pop off all earlier error handlers
                break;
            }

            if (MORPHO_ISDICTIONARY(eh->dict)) {
                value branchto = MORPHO_NIL;
                objectdictionary *dict = MORPHO_GETDICTIONARY(eh->dict);
                if (dictionary_get(&dict->dict, errid, &branchto)) {
                    error_clear(&v->err);

                    // Jump to the error handler
                    v->fp=eh->fp;
                    v->konst=v->fp->function->konst.data;
                    pc=v->instructions+MORPHO_GETINTEGERVALUE(branchto);
                    reg=v->stack.data+v->fp->roffset;

                    if (v->openupvalues) { /* Close any upvalues */
                        vm_closeupvalues(v, reg+v->fp->function->nregs);
                    }

                    v->ehp=eh-1; // Unwind the error handler stack
                    if (v->ehp<v->errorhandlers) v->ehp=NULL;
                    DISPATCH()
                }
            }
        }

        /* The error was not caught; unwind the stack to the point where we have to return  */
        if (!v->errfp) {
            v->errfp=v->fp; // Record frame pointer for stacktrace
            v->errfp->pc=pc;
        }

        v->fp=retfp-1;

    }

#undef INTERPRET_LOOP
#undef CASE_CODE
#undef DISPATCH

    //v->fp->pc=pc;

    return false;
}

/* **********************************************************************
* VM public interfaces
* ********************************************************************** */

/** Creates a new virtual machine */
vm *morpho_newvm(void) {
    vm *new = MORPHO_MALLOC(sizeof(vm));

    if (new) vm_init(new);

    return new;
}

/** Frees a virtual machine */
void morpho_freevm(vm *v) {
    vm_clear(v);
    MORPHO_FREE(v);
}

/** Configure input callback function */
void morpho_setinputfn(vm *v, morphoinputfn inputfn, void *ref) {
    v->inputfn=inputfn;
    v->inputref=ref;
}

/** Configure print callback function */
void morpho_setprintfn(vm *v, morphoprintfn printfn, void *ref) {
    v->printfn=printfn;
    v->printref=ref;
}

/** Configure warning callback function */
void morpho_setwarningfn(vm *v, morphowarningfn warningfn, void *ref) {
    v->warningfn=warningfn;
    v->warningref=ref;
}

/** Configure debugger callback function */
void morpho_setdebuggerfn(vm *v, morphodebuggerfn debuggerfn, void *ref) {
    v->debuggerfn=debuggerfn;
    v->debuggerref=ref;
}

/** Returns a VM's error block */
error *morpho_geterror(vm *v) {
    return &v->err;
}

/** @brief Raises an error described by an error structure
 * @param v        the virtual machine
 * @param err    error to raise */
void morpho_error(vm *v, error *err) {
    if (err->cat!=ERROR_WARNING) { // Errors of category ERROR_WARNING are always raised as warnings
        v->err = *err;
    } else morpho_warning(v, err);
}

/** @brief Raises an warning described by an error structure  */
void morpho_warning(vm *v, error *err) {
    if (v->warningfn) {
        (v->warningfn) (v, v->warningref, err);
    } else {
        fprintf(stderr, "Warning [%s]: %s\n", err->id, err->msg);
    }
}

/** @brief Raise a runtime error given an id
 * @param v        the virtual machine
 * @param id       error id
 * @param ...      additional data for sprintf. */
void morpho_runtimeerror(vm *v, errorid id, ...) {
    va_list args;
    error err;
    error_init(&err);

    va_start(args, id);
    morpho_writeerrorwithidvalist(&err, id, NULL, ERROR_POSNUNIDENTIFIABLE, ERROR_POSNUNIDENTIFIABLE, args);
    va_end(args);
    
    morpho_error(v, &err);
    error_clear(&err);
}

/** @brief Raise a runtime warning given an id  */
void morpho_runtimewarning(vm *v, errorid id, ...) {
    va_list args;
    error err;
    error_init(&err);

    va_start(args, id);
    morpho_writeerrorwithidvalist(&err, id, NULL, ERROR_POSNUNIDENTIFIABLE, ERROR_POSNUNIDENTIFIABLE, args);
    va_end(args);
    
    morpho_warning(v, &err);
    error_clear(&err);
}

/** @brief Binds a set of objects to a Virtual Machine; public interface.
 *  @details Any object created during execution should be bound to a VM; this object is then managed by the garbage collector.
 *  @param v      the virtual machine
 *  @param obj    objects to bind */
void morpho_bindobjects(vm *v, int nobj, value *obj) {
    /* Now bind the new objects in. */
    for (unsigned int i=0; i<nobj; i++) {
        object *ob = MORPHO_GETOBJECT(obj[i]);
        if (MORPHO_ISOBJECT(obj[i]) && ob->status<OBJECT_ISUNMARKED) {
            ob->status=OBJECT_ISUNMARKED;
            ob->next=v->objects;
            v->objects=ob;
            size_t size=object_size(ob);
            v->bound+=size;
#ifdef MORPHO_DEBUG_GCSIZETRACKING
            dictionary_insert(&sizecheck, obj[i], MORPHO_INTEGER(size));
#endif
        }
    }

    /* Check if size triggers garbage collection */
#ifndef MORPHO_DEBUG_STRESSGARBAGECOLLECTOR
    if (v->bound>v->nextgc)
#endif
    {
        int handle=morpho_retainobjects(v, nobj, obj);
        
        vm_collectgarbage(v);
        
        morpho_releaseobjects(v, handle);
    }
}

/** @brief   Convenience function to wrap a single object into a value and bind to the VM
 *  @param   v VM to use
 *  @param   out Object to wrap
 *  @returns object wrapped in a value, or MORPHO_NIL if obj is NULL */
value morpho_wrapandbind(vm *v, object *obj) {
    value out = MORPHO_NIL;
    if (obj) {
        out=MORPHO_OBJECT(obj);
        morpho_bindobjects(v, 1, &out);
    }
    return out;
}

/** @brief Temporarily retain objects across multiple reentrant calls to the VM.
 *  @param v      the virtual machine
 *  @param nobj  number of objects to retain
 *  @param obj    objects to retain
 *  @returns an integer handle to pass to releaseobjects */
int morpho_retainobjects(vm *v, int nobj, value *obj) {
    int gcount=v->retain.count;
    varray_valueadd(&v->retain, obj, nobj);
    return gcount;
}

/** @brief Release objects temporarily retained by the VM.
 *  @param v      the virtual machine
 *  @param handle a handle returned by morpho_retainobjects. */
void morpho_releaseobjects(vm *v, int handle) {
    if (handle>=0) v->retain.count=handle;
}

/** @brief Inform the VM that the size of an object has changed
 *  @param v      the virtual machine
 *  @param obj  the object to resize
 *  @param oldsize old size
 *  @param newsize new size
 */
void morpho_resizeobject(vm *v, object *obj, size_t oldsize, size_t newsize) {
#ifdef MORPHO_DEBUG_GCSIZETRACKING
    dictionary_insert(&sizecheck, MORPHO_OBJECT(obj), MORPHO_INTEGER(newsize));
#endif
    if (!MORPHO_ISGARBAGECOLLECTED(MORPHO_OBJECT(obj))) return;
    v->bound-=oldsize;
    v->bound+=newsize;
}


/** @brief Checks if an object is managed by the garbage collector
 *  @param obj  the object to check
 *  @returns true if it is managed, false otherwise 
 */
bool morpho_ismanagedobject(object *obj) {
    return (obj->status==OBJECT_ISUNMARKED || obj->status==OBJECT_ISMARKED);
}

/** Runs a program
 * @param[in] v - the virtual machine to use
 * @param[in] p - program to run
 * @returns true on success, false if an error occurred */
bool morpho_run(vm *v, program *p) {
    if (!vm_start(v, p)) return false;
    
    /* Initialize global variables */
    int oldsize = v->globals.count;
    int nglobals = program_countglobals(p);
    varray_valueresize(&v->globals, nglobals);
    v->globals.count=nglobals;
    for (int i=oldsize; i<nglobals; i++) v->globals.data[i]=MORPHO_NIL; /* Zero out globals */

    /* and initially set the register pointer to the bottom of the stack */
    value *reg = v->stack.data;
    
    /* Expand and clear the stack if necessary */
    if (v->fp->function->nregs>v->stack.count) {
        unsigned int oldcount=v->stack.count;
        vm_expandstack(v, &reg, v->fp->function->nregs-v->stack.count);
        for (unsigned int i=oldcount; i<v->stack.count; i++) v->stack.data[i]=MORPHO_NIL;
    }

    instructionindx start = program_getentry(p);

    int success = morpho_interpret(v, reg, start);

    if (!success &&
        morpho_matcherror(morpho_geterror(v), VM_EXIT)) {
        success=true;
        error_clear(morpho_geterror(v));
    }
    
    return success;
}

/* Call a morpho function from C code */
bool morpho_call(vm *v, value f, int nargs, value *args, value *ret) {
    bool success=false;
    value fn=f;
    value r0=f;

    if (MORPHO_ISINVOCATION(fn)) {
        /* An method invocation */
        objectinvocation *inv = MORPHO_GETINVOCATION(f);
        fn=inv->method;
        r0=inv->receiver;
    }
    
    if (MORPHO_ISMETAFUNCTION(fn) &&
        !metafunction_resolve(MORPHO_GETMETAFUNCTION(fn), nargs, args, &v->err, &fn)) {
        if (!morpho_checkerror(&v->err)) morpho_runtimeerror(v, VM_MLTPLDSPTCHFLD);
        return false;
    }
    
    if (MORPHO_ISBUILTINFUNCTION(fn)) {
        objectbuiltinfunction *f = MORPHO_GETBUILTINFUNCTION(fn);

        /* Copy arguments across to comply with call standard */
        value xargs[nargs+1];
        xargs[0]=r0;
        for (unsigned int i=0; i<nargs; i++) xargs[i+1]=args[i];
        v->fp->nopt=0; // TODO: Extend morpho call to support optional args.
        
#ifdef MORPHO_PROFILER
        v->fp->inbuiltinfunction=f;
#endif
        *ret=(f->function) (v, nargs, xargs);
#ifdef MORPHO_PROFILER
        v->fp->inbuiltinfunction=NULL;
#endif
        success=true;
    } else if (MORPHO_ISFUNCTION(fn) || MORPHO_ISCLOSURE(fn)) {
        value *reg=v->stack.data+v->fp->roffset;
        instruction *pc=v->fp->pc;

        /* Set up the function call, advancing the frame pointer and expanding the stack if necessary */
        if (vm_call(v, fn, v->fp->function->nregs, nargs, 0, args, &pc, &reg)) {
            /* Now ensure the function (or self) is in r0 */
            reg[0]=r0;

            /* Set return to true in this callframe */
            v->fp->ret=true;

            /* Keep track of the stack in case it is reallocated */
            value *stackbase=v->stack.data;
            ptrdiff_t roffset=reg-v->stack.data;

            success=morpho_interpret(v, reg, pc-v->instructions);

            /* Restore reg if stack has expanded */
            if (v->stack.data!=stackbase) reg=v->stack.data+roffset;

            if (success) *ret=reg[0]; /* Return value */
        }
    } else if (MORPHO_ISCLASS(fn)) {
        /* A function call on a class instantiates it */
        objectclass *klass = MORPHO_GETCLASS(fn);
        objectinstance *instance = object_newinstance(klass);
        if (instance) {
            value obj = MORPHO_OBJECT(instance);

            /* Call the initializer if the class provides one */
            value ifunc;
            if (morpho_lookupmethod(obj, initselector, &ifunc)) {
                success=morpho_invoke(v, obj, ifunc, nargs, args, ret);
            } else if (nargs==0) {
                success=true;
            } else morpho_runtimeerror(v, VM_NOINITIALIZER, MORPHO_GETCSTRING(klass->name));
            
            if (success) {
                vm_bindobject(v, obj);
                *ret = obj;
            } else morpho_freeobject(obj);
        } else morpho_runtimeerror(v, VM_INSTANTIATEFAILED);
    }

    return success;
}

/** Find the class associated with a value */
objectclass *morpho_lookupclass(value v) {
    objectclass *klass=NULL;
    if (MORPHO_ISINSTANCE(v)) klass=MORPHO_GETINSTANCE(v)->klass;
    else if (MORPHO_ISCLASS(v)) klass=MORPHO_GETCLASS(v);
    else if (MORPHO_ISOBJECT(v)) klass=object_getveneerclass(MORPHO_GETOBJECTTYPE(v));
    else klass=value_getveneerclass(v);
    return klass;
}

/** Finds a method */
bool morpho_lookupmethod(value obj, value label, value *method) {
    objectclass *klass = morpho_lookupclass(obj);
    if (klass) return dictionary_get(&klass->methods, label, method);

    return false;
}

/** Invoke a method on an object.
 @param[in] v - the virtual machine
 @param[in] obj - object to call on
 @param[in] method - method to invoke. NOTE lookup this first with morpho_lookupmethod if you just have a string
 @param[in] nargs - number of arguments
 @param[in] args - the arguments
 @param[out] ret - result of call
 @returns true on success, false otherwise */
bool morpho_invoke(vm *v, value obj, value method, int nargs, value *args, value *ret) {
    objectinvocation inv;
    object_init((object *) &inv, OBJECT_INVOCATION);
    inv.receiver=obj;
    inv.method=method;

    return morpho_call(v, MORPHO_OBJECT(&inv), nargs, args, ret);
}

/* **********************************************************************
* I/O
* ********************************************************************** */

/** Prints a formatted string to the VM's output channel */
int morpho_printf(vm *v, char *format, ...) {
    int nchars=0;
    
    va_list args;
    
    if (v && v->printfn) {
        for (;;) {
            va_start(args, format);
            nchars = vsnprintf(v->buffer.data, v->buffer.capacity, format, args);
            va_end(args);
            
            if (nchars+1<=v->buffer.capacity) break;
            
            if (!varray_charresize(&v->buffer, nchars+1)) return 0;
        }
        
        v->buffer.count=nchars;
    
        (v->printfn) (v, v->printref, v->buffer.data);
    } else { // If no VM or no print function available, resort to regular printf
        va_start(args, format);
        nchars=vprintf(format, args);
        va_end(args); 
    }
    
    return nchars;
}

/** Reads a line of text from the VM's input channel; returns the number of bytes read */
int morpho_readline(vm *v, varray_char *buffer) {
    int start=buffer->count;
    if (v && v->inputfn) {
        (v->inputfn) (v, v->inputref, MORPHO_INPUT_LINE, buffer);
    } else { // If no VM or no input function available, resort to regular fgetc
        do {
            int c = fgetc(stdin);
            if (c==EOF || c=='\n') break;
            varray_charwrite(buffer, (char) c);
        } while (true);
    }
    return buffer->count-start;
}

/* **********************************************************************
* Subkernels
* ********************************************************************** */

DEFINE_VARRAY(vm, struct svm *)

/** Obtain subkernels from the VM for use in a thread */
bool vm_subkernels(vm *v, int nkernels, vm **subkernels) {
    int nk=0;
    
    /* Check for unused subkernels */
    for (int i=0; i<v->subkernels.count; i++) {
        vm *kernel=v->subkernels.data[i];
        if (!kernel->parent) { // Check whether subkernel is unused
            subkernels[nk]=kernel;
            kernel->parent=v;
            nk++;
        }
    }
    
    /* Create any additional kernels that need to be made */
    for (int i=nk; i<nkernels; i++) {
        vm *new = morpho_newvm();
        if (!new) return false;
        if (!varray_vmadd(&v->subkernels, &new, 1)) return false;
        vm_start(new, v->current);
        new->globals.count=v->globals.count;
        new->globals.data=v->globals.data;
        new->parent=v;
        subkernels[i]=new;
    }
    
    return true;
}

/** Release a subkernels from the VM for use in a thread */
void vm_releasesubkernel(vm *subkernel) {
    vm *v = subkernel->parent;
    if (!v) return;
    
    /** Transfer objects from subkernel to kernel */
    if (subkernel->objects) {
        object *obj;
    
        for (obj=subkernel->objects; obj!=NULL; obj=obj->next) {
            if (obj->next==NULL) break;
        }
        
        /* Add the subkernel's objects to the parent */
        obj->next=v->objects;
        v->objects=subkernel->objects;
        
        /* Include this in the bound list */
        v->bound+=subkernel->bound;
        
        /* Remove from the subkernel */
        subkernel->objects=NULL;
        subkernel->bound=0;
    }
    
    /** Check if the subkernel is in an error state */
    if (!ERROR_SUCCEEDED(subkernel->err) &&
        ERROR_SUCCEEDED(v->err)) {
        v->err=subkernel->err;
    }
    
    subkernel->parent=NULL;
}

/** Clean out attached objects from a subkernel */
void vm_cleansubkernel(vm *subkernel) {
    object *next=NULL;
    for (object *obj=subkernel->objects; obj!=NULL; obj=next) {
        next=obj->next;
        object_free(obj);
    }
    subkernel->objects=NULL;
    subkernel->bound=0;
}

/* **********************************************************************
* Thread local storage
* ********************************************************************** */

int ntlvars=0;

/** Adds a thread local variable, returning the handle */
int vm_addtlvar(void) {
    int out = ntlvars;
    ntlvars++;
    return out;
}

/** Initialize threadlocal variables for a vm */
bool vm_inittlvars(vm *v) {
    if (v->tlvars.capacity<ntlvars) {
        if (!varray_valueresize(&v->tlvars, ntlvars)) return false;
        v->tlvars.count=ntlvars;
        for (int i=0; i<ntlvars; i++) v->tlvars.data[i]=MORPHO_NIL;
    }
    return true;
}

/** Sets the value of a thread local variable */
bool vm_settlvar(vm *v, int handle, value val) {
    bool success=false;
    if (handle<ntlvars &&
        vm_inittlvars(v)) {
        v->tlvars.data[handle]=val;
        success=true;
    }
    return success;
}

/** Gets the value of a thread local variable */
bool vm_gettlvar(vm *v, int handle, value *out) {
    bool success=false;
    if (handle<ntlvars &&
        vm_inittlvars(v)) {
        *out = v->tlvars.data[handle];
        success=true; 
    }
    return success;
}

/* **********************************************************************
 * Finalize functions
 * ********************************************************************** */

varray_value _finalizefns;

void morpho_addfinalizefn(morpho_finalizefn finalizefn) {
    varray_valuewrite(&_finalizefns, MORPHO_OBJECT(finalizefn));
}

/* **********************************************************************
* Initialization
* ********************************************************************** */

/** Initializes morpho */
void morpho_initialize(void) {
    varray_valueinit(&_finalizefns);
    
    random_initialize();
    error_initialize();
    
    value_initialize(); // } Must be first
    object_initialize(); // }
    
    builtin_initialize(); // Must come before initialization of any classes or similar
    resources_initialize(); // Must come before compiler and extensions
    
    lex_initialize();
    parse_initialize();
    compile_initialize();
    debugger_initialize();
    extensions_initialize();
    
#ifdef MORPHO_DEBUG_GCSIZETRACKING
    dictionary_init(&sizecheck);
#endif
    
    morpho_defineerror(ERROR_ALLOCATIONFAILED, ERROR_EXIT, ERROR_ALLOCATIONFAILED_MSG);
    morpho_defineerror(ERROR_INTERNALERROR, ERROR_EXIT, ERROR_INTERNALERROR_MSG);
    morpho_defineerror(ERROR_ERROR, ERROR_HALT, ERROR_ERROR_MSG);
    
    morpho_defineerror(VM_STCKOVFLW, ERROR_HALT, VM_STCKOVFLW_MSG);
    morpho_defineerror(VM_ERRSTCKOVFLW, ERROR_HALT, VM_ERRSTCKOVFLW_MSG);
    morpho_defineerror(VM_INVLDOP, ERROR_HALT, VM_INVLDOP_MSG);
    morpho_defineerror(VM_CNCTFLD, ERROR_HALT, VM_CNCTFLD_MSG);
    morpho_defineerror(VM_UNCALLABLE, ERROR_HALT, VM_UNCALLABLE_MSG);
    morpho_defineerror(VM_GLBLRTRN, ERROR_HALT, VM_GLBLRTRN_MSG);
    morpho_defineerror(VM_INSTANTIATEFAILED, ERROR_HALT, VM_INSTANTIATEFAILED_MSG);
    morpho_defineerror(VM_NOTANOBJECT, ERROR_HALT, VM_NOTANOBJECT_MSG);
    morpho_defineerror(VM_OBJECTLACKSPROPERTY, ERROR_HALT, VM_OBJECTLACKSPROPERTY_MSG);
    morpho_defineerror(VM_NOINITIALIZER, ERROR_HALT, VM_NOINITIALIZER_MSG);
    morpho_defineerror(VM_NOTANINSTANCE, ERROR_HALT, VM_NOTANINSTANCE_MSG);
    morpho_defineerror(VM_CLASSLACKSPROPERTY, ERROR_HALT, VM_CLASSLACKSPROPERTY_MSG);
    morpho_defineerror(VM_INVALIDARGS, ERROR_HALT, VM_INVALIDARGS_MSG);
    morpho_defineerror(VM_NOOPTARG, ERROR_HALT, VM_NOOPTARG_MSG);
    morpho_defineerror(VM_UNKNWNOPTARG, ERROR_HALT, VM_UNKNWNOPTARG_MSG);
    morpho_defineerror(VM_INVALIDARGSDETAIL, ERROR_HALT, VM_INVALIDARGSDETAIL_MSG);
    morpho_defineerror(VM_NOTINDEXABLE, ERROR_HALT, VM_NOTINDEXABLE_MSG);
    morpho_defineerror(VM_OUTOFBOUNDS, ERROR_HALT, VM_OUTOFBOUNDS_MSG);
    morpho_defineerror(VM_NONNUMINDX, ERROR_HALT, VM_NONNUMINDX_MSG);
    morpho_defineerror(VM_ARRAYWRONGDIM, ERROR_HALT, VM_ARRAYWRONGDIM_MSG);
    morpho_defineerror(VM_DVZR, ERROR_HALT, VM_DVZR_MSG);
	morpho_defineerror(VM_GETINDEXARGS, ERROR_HALT, VM_GETINDEXARGS_MSG);
    morpho_defineerror(VM_MLTPLDSPTCHFLD, ERROR_HALT, VM_MLTPLDSPTCHFLD_MSG);

    morpho_defineerror(VM_DBGQUIT, ERROR_HALT, VM_DBGQUIT_MSG);

    /* Selector for initializers */
    initselector=builtin_internsymbolascstring(MORPHO_INITIALIZER_METHOD);

    indexselector=builtin_internsymbolascstring(MORPHO_GETINDEX_METHOD);
    setindexselector=builtin_internsymbolascstring(MORPHO_SETINDEX_METHOD);

    addselector=builtin_internsymbolascstring(MORPHO_ADD_METHOD);
    addrselector=builtin_internsymbolascstring(MORPHO_ADDR_METHOD);
    subselector=builtin_internsymbolascstring(MORPHO_SUB_METHOD);
    subrselector=builtin_internsymbolascstring(MORPHO_SUBR_METHOD);
    mulselector=builtin_internsymbolascstring(MORPHO_MUL_METHOD);
    mulrselector=builtin_internsymbolascstring(MORPHO_MULR_METHOD);
    divselector=builtin_internsymbolascstring(MORPHO_DIV_METHOD);
    divrselector=builtin_internsymbolascstring(MORPHO_DIVR_METHOD);
    powselector=builtin_internsymbolascstring(MORPHO_POW_METHOD);
    powrselector=builtin_internsymbolascstring(MORPHO_POWR_METHOD);

    enumerateselector=builtin_internsymbolascstring(MORPHO_ENUMERATE_METHOD);
    countselector=builtin_internsymbolascstring(MORPHO_COUNT_METHOD);
    cloneselector=builtin_internsymbolascstring(MORPHO_CLONE_METHOD);

    printselector=builtin_internsymbolascstring(MORPHO_PRINT_METHOD);
}

/** Finalizes morpho, calling all finalize functions */
void morpho_finalize(void) {
    
    for (int i=_finalizefns.count-1; i>=0; i--) {
        morpho_finalizefn fn=(morpho_finalizefn) MORPHO_GETOBJECT(_finalizefns.data[i]);
        fn();
    }
  
    varray_valueclear(&_finalizefns);
}
