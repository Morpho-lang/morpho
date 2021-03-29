/** @file vm.c
 *  @author T J Atherton
 *
 *  @brief Morpho virtual machine
 */

#include <stdarg.h>
#include <time.h>
#include "vm.h"
#include "compile.h"
#include "veneer.h"
#include "morpho.h"
#include "debug.h"

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
value printselector = MORPHO_NIL;
value enumerateselector = MORPHO_NIL;
value countselector = MORPHO_NIL;
value cloneselector = MORPHO_NIL;

/* **********************************************************************
* Programs
* ********************************************************************** */

DEFINE_VARRAY(instruction, instruction);

/** @brief Initializes a program */
static void vm_programinit(program *p) {
    varray_instructioninit(&p->code);
    varray_debugannotationinit(&p->annotations);
    p->global=object_newfunction(MORPHO_PROGRAMSTART, MORPHO_NIL, NULL, 0);
    p->boundlist=NULL;
    dictionary_init(&p->symboltable);
    builtin_copysymboltable(&p->symboltable);
    p->nglobals=0;
}

/** @brief Clears a program, freeing associated data structures */
static void vm_programclear(program *p) {
    if (p->global) object_free((object *) p->global);
    varray_instructionclear(&p->code);
    debug_clear(&p->annotations);
    p->global=NULL;
    /* Free any objects bound to the program */
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("--Freeing objects bound to program.\n");
#endif
    while (p->boundlist!=NULL) {
        object *next = p->boundlist->next;
        object_free(p->boundlist);
        p->boundlist=next;
    }
    #ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
        printf("------\n");
    #endif
    dictionary_clear(&p->symboltable); /* Note we don't free the contents as they should be bound to the program */
}

/** @brief Creates and initializes a new program */
program *morpho_newprogram(void) {
    program *new = MORPHO_MALLOC(sizeof(program));
        
    if (new) vm_programinit(new);
        
    return new;
}

/** @brief Frees a program */
void morpho_freeprogram(program *p) {
    vm_programclear(p);
    MORPHO_FREE(p);
}

/** Sets the entry point of a program */
void program_setentry(program *p, instructionindx entry) {
    if (p->global) p->global->entry=entry;
}

/** Gets the entry point of a program  */
instructionindx program_getentry(program *p) {
    instructionindx out = MORPHO_PROGRAMSTART;
    if (p->global) out=p->global->entry;
    return out;
}

/** @brief Binds an object to a program
 *  @details Objects bound to the program are freed with the program; use for static data (e.g. held in constant tables) */
void program_bindobject(program *p, object *obj) {
    if (!obj->next && /* Object is not already bound to the program (or something else) */
        obj->status==OBJECT_ISUNMANAGED && /* Object is unmanaged */
        obj->type!=OBJECT_BUILTINFUNCTION && /* Object is not a built in function that is freed separately */
        (p->boundlist!=obj->next && p->boundlist!=NULL) /* To handle the case where the object is the only object */
        ) {
        
        obj->next=p->boundlist;
        p->boundlist=obj;
    }
}

/** @brief Interns a symbol into the programs symbol table.
 *  @details Note that the string is cloned if it does not exist already.
 *           Interning is used to accelerate dynamic lookups as the same string for a symbol will be used universally */
value program_internsymbol(program *p, value symbol) {
    value new = symbol, out;
#ifdef MORPHO_DEBUG_SYMBOLTABLE
    printf("Interning symbol '");
    morpho_printvalue(symbol);
#endif
    if (!dictionary_get(&p->symboltable, symbol, NULL)) {
       new = object_clonestring(symbol);
    }
    out = dictionary_intern(&p->symboltable, new);
#ifdef MORPHO_DEBUG_SYMBOLTABLE
    printf("' at %p\n", (void *) MORPHO_GETOBJECT(out));
#endif
    program_bindobject(p, MORPHO_GETOBJECT(out));
    return out;
}

/* **********************************************************************
 * The gray list
 * ********************************************************************** */

/* Initialize the gray list */
void vm_graylistinit(graylist *g) {
    g->graycapacity=0;
    g->graycount=0;
    g->list=NULL;
}

/* Clear the gray list */
void vm_graylistclear(graylist *g) {
    if (g->list) free(g->list);
    vm_graylistinit(g);
}

/* Add an object to the gray list */
void vm_graylistadd(graylist *g, object *obj) {
    if (g->graycount+1>=g->graycapacity) {
        g->graycapacity*=2;
        if (g->graycapacity<8) g->graycapacity=8;
        g->list=realloc(g->list, g->graycapacity*sizeof(object *));
    }
    
    if (g->list) {
        g->list[g->graycount]=obj;
        g->graycount++;
    }
}

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
    v->bound=0;
    v->nextgc=MORPHO_GCINITIAL;
    v->debug=false;
    vm_graylistinit(&v->gray);
    varray_valueinit(&v->stack);
    varray_valueinit(&v->globals);
    varray_valueresize(&v->stack, MORPHO_STACKINITIALSIZE);
    error_init(&v->err);
}

/** Clears a virtual machine */
static void vm_clear(vm *v) {
    varray_valueclear(&v->stack);
    varray_valueclear(&v->globals);
    vm_graylistclear(&v->gray);
    vm_freeobjects(v);
}

/** Frees all objects bound to a virtual machine */
void vm_freeobjects(vm *v) {
    long k=0;
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("--- Freeing objects bound to VM ---\n");
#endif
    object *next=NULL;
    for (object *e=v->objects; e!=NULL; e=next) {
        next = e->next;
        object_free(e);
        k++;
    }
    
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("--- Freed %li objects bound to VM ---\n", k);
#endif
}

#ifdef MORPHO_DEBUG_GCSIZETRACKING
dictionary sizecheck;
#endif

#include "object.h"
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

/* **********************************************************************
 * Garbage collector
 * ********************************************************************** */

/** Recalculates the size of bound objects to the VM */
size_t vm_gcrecalculatesize(vm *v) {
    size_t size = 0;
    for (object *ob=v->objects; ob!=NULL; ob=ob->next) {
        size+=object_size(ob);
    }
    return size;
}

/** Marks an object as reachable */
void vm_gcmarkobject(vm *v, object *obj) {
    if (!obj || obj->status!=OBJECT_ISUNMARKED) return;
    
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
        printf("Marking %p ", obj);
        object_print(MORPHO_OBJECT(obj));
        printf("\n");
#endif
    obj->status=OBJECT_ISMARKED;
    
    vm_graylistadd(&v->gray, obj);
}

/** Marks a value as reachable */
void vm_gcmarkvalue(vm *v, value val) {
    if (MORPHO_ISOBJECT(val)) {
        vm_gcmarkobject(v, MORPHO_GETOBJECT(val));
    }
}

/** Marks all entries in a dictionary */
void vm_gcmarkdictionary(vm *v, dictionary *dict) {
    for (unsigned int i=0; i<dict->capacity; i++) {
        if (!MORPHO_ISNIL(dict->contents[i].key)) {
            vm_gcmarkvalue(v, dict->contents[i].key);
            vm_gcmarkvalue(v, dict->contents[i].val);
        }
    }
}

/** Marks all entries in an array */
void vm_gcmarkarray(vm *v, varray_value *array) {
    if (array) for (unsigned int i=0; i<array->count; i++) {
        vm_gcmarkvalue(v, array->data[i]);
    }
}

/** Searches a vm for all reachable objects */
void vm_gcmarkroots(vm *v) {
    /** Mark anything on the stack */
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("> Stack.\n");
#endif
    for (value *s=v->stack.data+v->fp->roffset+v->fp->function->nregs-1; s>=v->stack.data; s--) {
        if (MORPHO_ISOBJECT(*s)) vm_gcmarkvalue(v, *s);
    }

#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("> Globals.\n");
#endif
    for (unsigned int i=0; i<v->globals.count; i++) {
        vm_gcmarkvalue(v, v->globals.data[i]);
    }
    
    /** Mark closure objects in use */
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("> Closures.\n");
#endif
    for (callframe *f=v->frame; f && v->fp && f<=v->fp; f++) {
        if (f->closure) vm_gcmarkobject(v, (object *) f->closure);
    }
    
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("> Open upvalues.\n");
#endif
    for (objectupvalue *u=v->openupvalues; u!=NULL; u=u->next) {
        vm_gcmarkobject(v, (object *) u);
    }
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("> End mark roots.\n");
#endif
}

void vm_gcmarkretainobject(vm *v, object *obj) {
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("Searching object %p ", (void *) obj);
    morpho_printvalue(MORPHO_OBJECT(obj));
    printf("\n");
#endif
    switch (obj->type) {
        case OBJECT_BUILTINFUNCTION:
        case OBJECT_MATRIX:
        case OBJECT_DOKKEY:
        case OBJECT_STRING:
        case OBJECT_RANGE:
            break;
        case OBJECT_UPVALUE:
            vm_gcmarkvalue(v, ((objectupvalue *) obj)->closed);
            break;
        case OBJECT_FUNCTION: {
            objectfunction *f = (objectfunction *) obj;
            vm_gcmarkvalue(v, f->name);
            vm_gcmarkarray(v, &f->konst);
        }
            break;
        case OBJECT_CLOSURE: {
            objectclosure *c = (objectclosure *) obj;
            vm_gcmarkobject(v, (object *) c->func);
            for (unsigned int i=0; i<c->nupvalues; i++) {
                vm_gcmarkobject(v, (object *) c->upvalues[i]);
            }
        }
            break;
        case OBJECT_CLASS: {
            objectclass *c = (objectclass *) obj;
            vm_gcmarkvalue(v, c->name);
            vm_gcmarkdictionary(v, &c->methods);
        }
            break;
        case OBJECT_INSTANCE: {
            objectinstance *c = (objectinstance *) obj;
            vm_gcmarkdictionary(v, &c->fields);
        }
            break;
        case OBJECT_INVOCATION: {
            objectinvocation *c = (objectinvocation *) obj;
            vm_gcmarkvalue(v, c->receiver);
            vm_gcmarkvalue(v, c->method);
        }
            break;
        case OBJECT_DICTIONARY: {
            objectdictionary *c = (objectdictionary *) obj;
            vm_gcmarkdictionary(v, &c->dict);
        }
            break;
        case OBJECT_LIST: {
            objectlist *c = (objectlist *) obj;
            vm_gcmarkarray(v, &c->val);
        }
            break;
        case OBJECT_ARRAY: {
            objectarray *c = (objectarray *) obj;
            for (unsigned int i=0; i<c->nelements+c->dimensions; i++) {
                vm_gcmarkvalue(v, c->data[i]);
            }
        }
            break;
        case OBJECT_SPARSE: {
            objectsparse *c = (objectsparse *) obj;
            vm_gcmarkdictionary(v, &c->dok.dict);
        }
            break;
        case OBJECT_MESH: {
            objectmesh *c = (objectmesh *) obj;
            if (c->vert) vm_gcmarkobject(v, (object *) c->vert);
            if (c->conn) vm_gcmarkretainobject(v, (object *) c->conn);
        }
            break;
        case OBJECT_SELECTION: {
            //objectselection *c = (objectselection *) obj;
        }
            break;
        case OBJECT_EXTERN: {
        }
            break;
    }
}

/** Trace all objects on the graylist */
void vm_gctrace(vm *v) {
    while (v->gray.graycount>0) {
        object *obj=v->gray.list[v->gray.graycount-1];
        v->gray.graycount--;
        vm_gcmarkretainobject(v, obj);
    }
}

/** Go through the VM's object list and free all unmarked objects */
void vm_gcsweep(vm *v) {
    object *prev=NULL;
    object *obj = v->objects;
    while (obj!=NULL) {
        if (obj->status==OBJECT_ISMARKED) {
            prev=obj;
            obj->status=OBJECT_ISUNMARKED; /* Clear for the next cycle */
            obj=obj->next;
        } else {
            object *unreached = obj;
            size_t size=object_size(obj);
#ifdef MORPHO_DEBUG_GCSIZETRACKING
            value xsize;
            if (dictionary_get(&sizecheck, MORPHO_OBJECT(unreached), &xsize)) {
                size_t isize = MORPHO_GETINTEGERVALUE(xsize);
                if (size!=isize) {
                    morpho_printvalue(MORPHO_OBJECT(unreached));
                    UNREACHABLE("Object doesn't match its declared size");
                }
            }
#endif
            
            v->bound-=size;
            
            /* Delink */
            obj=obj->next;
            if (prev!=NULL) {
                prev->next=obj;
            } else {
                v->objects=obj;
            }
            
#ifndef MORPHO_DEBUG_GCSIZETRACKING
            object_free(unreached);
#endif
        }
    }
}

/** Collects garbage */
void vm_collectgarbage(vm *v) {
#ifdef MORPHO_DEBUG_DISABLEGARBAGECOLLECTOR
    return;
#endif
    vm *vc = (v!=NULL ? v : globalvm);
    if (!vc) return;
    
    if (vc && vc->bound>0) {
        size_t init=vc->bound;
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
        printf("--- begin garbage collection ---\n");
#endif
        vm_gcmarkroots(vc);
        vm_gctrace(vc);
        vm_gcsweep(vc);
        
        if (vc->bound>init) {
#ifdef MORPHO_DEBUG_GCSIZETRACKING
            printf("GC collected %ld bytes (from %zu to %zu) next at %zu.\n", init-vc->bound, init, vc->bound, vc->bound*MORPHO_GCGROWTHFACTOR);
            UNREACHABLE("VM bound object size < 0");
#else
            // This catch has been put in to prevent the garbarge collector from completely seizing up.
            vc->bound=vm_gcrecalculatesize(v);
#endif
        }
        
        vc->nextgc=vc->bound*MORPHO_GCGROWTHFACTOR;
        
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
        printf("--- end garbage collection ---\n");
        if (vc) printf("    collected %ld bytes (from %zu to %zu) next at %zu.\n", init-vc->bound, init, vc->bound, vc->nextgc);
#endif
    }
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
    debug_infofromindx(v->current, iindx, &line, &posn, NULL, NULL);
    
    va_start(args, id);
    morpho_writeerrorwithidvalist(&v->err, id, line, posn, args);
    va_end(args);
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

/** Process variadic and optional arguments */
static inline bool vm_vargs(vm *v, ptrdiff_t iindx, objectfunction *func, unsigned int shift, unsigned int nargs, value *reg) {
    unsigned int nopt = func->opt.count, // No. of optional params
                 nfixed = func->nargs-nopt, // No. of fixed params
                 roffset = shift+nfixed+1, // Position of first optional parameter in output
                 n=0;
    value res[nopt];
    
    /* Copy across default values */
    for (unsigned int i=0; i<nopt; i++) {
        res[i]=func->konst.data[func->opt.data[i].def];
    }
    
    /* Identify the optional arguments by searching back from the end */
    for (n=0; 2*n<nargs; n+=1) {
        unsigned int k=0;
        for (; k<nopt; k++) if (MORPHO_ISSAME(func->opt.data[k].symbol, reg[nargs-1-2*n])) break;
        if (k>=nopt) break; // If we didn't find a match, we're done with optional arguments
        res[k]=reg[nargs-2*n];
    }
    
    if (nargs-2*n!=nfixed) { // Verify number of fixed args is correct 
        vm_runtimeerror(v, iindx, VM_INVALIDARGS, nfixed, nargs-2*n);
        return false;
    }
    
    /* Copy across the arguments */
    for (unsigned int i=0; i<nopt; i++) reg[roffset+i]=res[i];
    
    return true;
}


/** @brief Performs a function call
 *  @details A function call involves:
 *           1. Saving the program counter, register index and stacksize to the callframe stack;
 *           2. Advancing the frame pointer;
 *           3. Extracting the function from a closure if necessary;
 *           4. Expanding the stack if necessary
 *           5. Loading the constant table from the function definition
 *           6. Shifting the register base
 *           7. Moving the program counter to the function
 * @param[in]  v                         The virtual machine
 * @param[in]  fn                       Function to call
 * @param[in]  shift                 rshift becomes r0 in the new call frame
 * @param[in]  nargs                number of arguments
 * @param[out] pc                       program counter, updated
 * @param[out] reg                     register/stack pointer, updated */
static inline bool vm_call(vm *v, value fn, unsigned int shift, unsigned int nargs, instruction **pc, value **reg) {
    objectfunction *func = MORPHO_GETFUNCTION(fn);
    
    /* In the old frame... */
    v->fp->pc=*pc; /* Save the program counter */
    v->fp->stackcount=v->fp->function->nregs+(unsigned int) v->fp->roffset; /* Store the stacksize */
    unsigned int oldnregs = v->fp->function->nregs; /* Get the old number of registers */
    
    v->fp++; /* Advance frame pointer */
    v->fp->pc=*pc; /* We will also store the program counter in the new frame;
                      this will be used to detect whether the VM should return on OP_RETURN */
    
    if (MORPHO_ISCLOSURE(fn)) {
        objectclosure *closure=MORPHO_GETCLOSURE(fn); /* Closure object in use */
        func=closure->func;
        v->fp->closure=closure;
    } else {
        v->fp->closure=NULL;
    }
    
    v->fp->ret=false; /* Interpreter should not return from this frame */
    v->fp->function=func; /* Store the function */
    
    if (func->opt.count>0) {
        if (!vm_vargs(v, (*pc) - v->instructions, func, shift, nargs, *reg)) return false;
    } else if (func->nargs!=nargs) {
        vm_runtimeerror(v, (*pc) - v->instructions, VM_INVALIDARGS, func->nargs, nargs);
        return false;
    }
    
    /* Do we need to expand the stack? */
    if (shift+func->nregs > oldnregs) {
        /* We check this explicitly to avoid an unnecessary function call to vm_expandstack */
        unsigned int n=shift+func->nregs-oldnregs;
        if (v->stack.count+n>v->stack.capacity) {
            vm_expandstack(v, reg, n); /* Expand the stack */
        } else {
            v->stack.count+=n;
        }
    }

    v->konst = func->konst.data; /* Load the constant table */
    *reg += shift; /* Shift the register frame so
               that the register a becomes r0 */
    v->fp->roffset=*reg-v->stack.data; /* Store the register index */
    
    /* Zero out registers beyond args up to the top of the stack
       This has to be fast: memset was too slow. Zero seems to be faster than MORPHO_NIL */
    for (value *r = *reg + func->nregs-1; r > *reg + func->nargs; r--) *r = MORPHO_INTEGER(0);
    
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
        objectclass *klass = builtin_getveneerclass(MORPHO_GETOBJECTTYPE(obj));
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
#define MORPHO_DISASSEMBLE_INSRUCTION(bc,pc,k,r) { printf("  "); debug_disassembleinstruction(bc, pc-1, k, r); printf("\n"); }
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
    
/* Define the interpreter loop. Computed gotos or regular switch statements can be used here. */
#ifdef MORPHO_COMPUTED_GOTO
    /* The dispatch table, containing the entry points for each opcode */
    static void* dispatchtable[] = {
      #define OPCODE(name) &&code_##name,
      #include "opcodes.h"
      #undef OPCODE
    };
    
    /* The interpret loop begins by dispatching an instruction */
    #define INTERPRET_LOOP    DISPATCH();
    
    /* Create a label corresponding to each opcode */
    #define CASE_CODE(name)   code_##name
    
    /* Dispatch here means fetch the next instruction, decode and jump */
    #define DISPATCH()                                                       \
        do {                                                                 \
            bc=*pc++;                                                        \
            OPOPCODECNT(pp, bc)                                              \
            op=DECODE_OP(bc);                                                \
            OPCODECNT(op)                                                    \
            MORPHO_DISASSEMBLE_INSRUCTION(bc,pc-v->instructions,v->konst, reg)     \
            goto *dispatchtable[op];                                         \
        } while(false);
    
#else
    /* Every iteration of the interpret loop we fetch, decode and switch */
    #define INTERPRET_LOOP                                                   \
        loop:                                                                \
        bc=*pc++;                                                            \
        OPOPCODECNT(pp, bc)                                                  \
        op=DECODE_OP(bc);                                                    \
        OPCODECNT(op)                                                        \
        MORPHO_DISASSEMBLE_INSRUCTION(bc,pc-v->instructions,v->konst, reg)         \
        switch (op)
    
    /* Each opcode generates a case statement */
    #define CASE_CODE(name)  case OP_##name
    
    /* Dispatch means return to the beginning of the loop */
    #define DISPATCH() goto loop;
#endif
    
#define ERROR(id) { vm_runtimeerror(v, pc-v->instructions, id); goto vm_error; }
#define VERROR(id, ...) { vm_runtimeerror(v, pc-v->instructions, id, __VA_ARGS__); goto vm_error; }
#define ERRORCHK() if (v->err.cat!=ERROR_NONE) goto vm_error;
    
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
            left = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
            right = (DECODE_ISCCONSTANT(bc) ? v->konst[c] : reg[c]);
            
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
                }
            }
            
            if (MORPHO_ISOBJECT(left)) {
                if (vm_invoke(v, left, addselector, 1, &right, &reg[a])) {
                    ERRORCHK();
                    DISPATCH();
                }
            }
        
            if (MORPHO_ISOBJECT(right)) {
                if (vm_invoke(v, right, addrselector, 1, &left, &reg[a])) {
                    ERRORCHK();
                    DISPATCH();
                }
            }
        
            ERROR(VM_INVLDOP);
            DISPATCH();
        
        CASE_CODE(SUB):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
            right = (DECODE_ISCCONSTANT(bc) ? v->konst[c] : reg[c]);
            
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
        
            if (MORPHO_ISOBJECT(left)) {
                if (vm_invoke(v, left, subselector, 1, &right, &reg[a])) {
                    ERRORCHK();
                    DISPATCH();
                }
            }
        
            if (MORPHO_ISOBJECT(right)) {
                if (vm_invoke(v, right, subrselector, 1, &left, &reg[a])) {
                    ERRORCHK();
                    DISPATCH();
                }
            }
        
            ERROR(VM_INVLDOP);
            DISPATCH();
        
        CASE_CODE(MUL):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
            right = (DECODE_ISCCONSTANT(bc) ? v->konst[c] : reg[c]);
            
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
        
            if (MORPHO_ISOBJECT(left)) {
                if (vm_invoke(v, left, mulselector, 1, &right, &reg[a])) {
                    ERRORCHK();
                    DISPATCH();
                }
            }
        
            if (MORPHO_ISOBJECT(right)) {
                if (vm_invoke(v, right, mulrselector, 1, &left, &reg[a])) {
                    ERRORCHK();
                    DISPATCH();
                }
            }
        
            ERROR(VM_INVLDOP);
            DISPATCH();
        
        CASE_CODE(DIV):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
            right = (DECODE_ISCCONSTANT(bc) ? v->konst[c] : reg[c]);
            
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
        
            if (MORPHO_ISOBJECT(left)) {
                if (vm_invoke(v, left, divselector, 1, &right, &reg[a])) {
                    ERRORCHK();
                    DISPATCH();
                }
            }
        
            if (MORPHO_ISOBJECT(right)) {
                if (vm_invoke(v, right, divrselector, 1, &left, &reg[a])) {
                    ERRORCHK();
                    DISPATCH();
                }
            }
        
            ERROR(VM_INVLDOP);
            DISPATCH();
        
        CASE_CODE(POW):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
            right = (DECODE_ISCCONSTANT(bc) ? v->konst[c] : reg[c]);
            
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
        
            ERROR(VM_INVLDOP);
            DISPATCH();
        
        
        CASE_CODE(NOT):
            a=DECODE_A(bc); b=DECODE_B(bc);
            left = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
            if (MORPHO_ISBOOL(left)) {
                reg[a] = MORPHO_BOOL(!MORPHO_GETBOOLVALUE(left));
            } else {
                reg[a] = MORPHO_BOOL(MORPHO_ISNIL(left));
            }
            DISPATCH();
        
        CASE_CODE(EQ):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
            right = (DECODE_ISCCONSTANT(bc) ? v->konst[c] : reg[c]);

#define CHECKCMPTYPE(l, r) \
            if (!morpho_ofsametype(l, r)) { \
                if (MORPHO_ISINTEGER(l) && MORPHO_ISFLOAT(r)) { \
                    l = MORPHO_INTEGERTOFLOAT(l); \
                } else if (MORPHO_ISFLOAT(l) && MORPHO_ISINTEGER(r)) { \
                    r = MORPHO_INTEGERTOFLOAT(right); \
                } \
            }

            CHECKCMPTYPE(left,right);
        
            reg[a] = (morpho_comparevalue(left, right)==0 ? MORPHO_BOOL(true) : MORPHO_BOOL(false));
            DISPATCH();
        
        CASE_CODE(NEQ):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
            right = (DECODE_ISCCONSTANT(bc) ? v->konst[c] : reg[c]);
        
            CHECKCMPTYPE(left,right);
            reg[a] = (morpho_comparevalue(left, right)!=0 ? MORPHO_BOOL(true) : MORPHO_BOOL(false));
            DISPATCH();
        
        CASE_CODE(LT):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
            right = (DECODE_ISCCONSTANT(bc) ? v->konst[c] : reg[c]);
        
            if ( !( (MORPHO_ISFLOAT(left) || MORPHO_ISINTEGER(left)) &&
                   (MORPHO_ISFLOAT(right) || MORPHO_ISINTEGER(right)) ) ) {
                ERROR(VM_INVLDOP);
            }
        
            CHECKCMPTYPE(left,right);
            reg[a] = (morpho_comparevalue(left, right)>0 ? MORPHO_BOOL(true) : MORPHO_BOOL(false));
            DISPATCH();
        
        CASE_CODE(LE):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
            right = (DECODE_ISCCONSTANT(bc) ? v->konst[c] : reg[c]);

            if ( !( (MORPHO_ISFLOAT(left) || MORPHO_ISINTEGER(left)) &&
                   (MORPHO_ISFLOAT(right) || MORPHO_ISINTEGER(right)) ) ) {
                ERROR(VM_INVLDOP);
            }
        
            CHECKCMPTYPE(left,right);
            reg[a] = (morpho_comparevalue(left, right)>=0 ? MORPHO_BOOL(true) : MORPHO_BOOL(false));
            DISPATCH();
        
        CASE_CODE(B):
            b=DECODE_sBx(bc);
            pc+=b;
            DISPATCH();
        
        CASE_CODE(BIF):
            a=DECODE_A(bc);
            b=DECODE_sBx(bc);
            left=reg[a];
        
            if ((MORPHO_ISTRUE(left) ? true : false) == (DECODE_F(bc) ? true : false)) pc+=b;
        
            DISPATCH();
        
        CASE_CODE(CALL):
            a=DECODE_A(bc);
            left=reg[a];
            c=DECODE_B(bc); // We use c for consistency between call and invoke...
        
callfunction: // Jump here if an instruction becomes a call
            if (MORPHO_ISINVOCATION(left)) {
                /* An method invocation */
                objectinvocation *inv = MORPHO_GETINVOCATION(left);
                left=inv->method;
                reg[a]=inv->receiver;
            }
            
            if (MORPHO_ISFUNCTION(left) || MORPHO_ISCLOSURE(left)) {
                if (!vm_call(v, left, a, c, &pc, &reg)) goto vm_error;
                
            } else if (MORPHO_ISBUILTINFUNCTION(left)) {
                /* Save program counter in the old callframe */
                v->fp->pc=pc;
                
                objectbuiltinfunction *f = MORPHO_GETBUILTINFUNCTION(left);
                
                value ret = (f->function) (v, c, reg+a);
                reg=v->stack.data+v->fp->roffset; /* Ensure register pointer is correct */
                reg[a]=ret;
                
                ERRORCHK();
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
                        /* If so, call it */
                        if (MORPHO_ISFUNCTION(ifunc)) {
                            if (!vm_call(v, ifunc, a, c, &pc, &reg)) goto vm_error;
                        } else if (MORPHO_ISBUILTINFUNCTION(ifunc)) {
                            (MORPHO_GETBUILTINFUNCTION(ifunc)->function) (v, c, reg+a);
                            ERRORCHK();
                        }
                    } else {
                        if (c>0) {
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
        
        CASE_CODE(INVOKE):
            a=DECODE_A(bc);
            b=DECODE_B(bc);
            c=DECODE_C(bc);
            left=reg[a];
            right=(DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
        
            if (MORPHO_ISINSTANCE(left)) {
                objectinstance *instance = MORPHO_GETINSTANCE(left);
                value ifunc;
                
                /* Check if we have this method */
                if (dictionary_getintern(&instance->klass->methods, right, &ifunc)) {
                    /* If so, call it */
                    if (MORPHO_ISFUNCTION(ifunc)) {
                        if (!vm_call(v, ifunc, a, c, &pc, &reg)) goto vm_error;
                    } else if (MORPHO_ISBUILTINFUNCTION(ifunc)) {
                        reg[a] = (MORPHO_GETBUILTINFUNCTION(ifunc)->function) (v, c, reg+a);
                        ERRORCHK();
                    }
                } else if (dictionary_getintern(&instance->fields, right, &left)) {
                    /* Otherwise, if it's a property, try to call it */
                    if (MORPHO_ISFUNCTION(left) || MORPHO_ISCLOSURE(left) || MORPHO_ISBUILTINFUNCTION(left) || MORPHO_ISINVOCATION(left)) {
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
                    if (v->fp>v->frame) reg[a]=reg[0]; /* Copy self into r[a] and call */
                    
                    if (MORPHO_ISFUNCTION(ifunc)) {
                        if (!vm_call(v, ifunc, a, c, &pc, &reg)) goto vm_error;
                    } else if (MORPHO_ISBUILTINFUNCTION(ifunc)) {
                        reg[a] = (MORPHO_GETBUILTINFUNCTION(ifunc)->function) (v, c, reg+a);
                        ERRORCHK();
                    }
                } else {
                    /* Otherwise, raise an error */
                    char *p = (MORPHO_ISSTRING(right) ? MORPHO_GETCSTRING(right) : "");
                    VERROR(VM_CLASSLACKSPROPERTY, p);
                }
            } else if (MORPHO_ISOBJECT(left)) {
                /* If it's an object, it may have a veneer class */
                objectclass *klass = builtin_getveneerclass(MORPHO_GETOBJECTTYPE(left));
                if (klass) {
                    value ifunc;
                    if (dictionary_getintern(&klass->methods, right, &ifunc)) {
                        if (MORPHO_ISBUILTINFUNCTION(ifunc)) {
                            reg[a] = (MORPHO_GETBUILTINFUNCTION(ifunc)->function) (v, c, reg+a);
                            ERRORCHK();
                        }
                    } else {
                        char *p = (MORPHO_ISSTRING(right) ? MORPHO_GETCSTRING(right) : "");
                        VERROR(VM_CLASSLACKSPROPERTY, p);
                    }
                } else {
                    ERROR(VM_NOTANINSTANCE);
                }
            } else {
                ERROR(VM_NOTANINSTANCE);
            }
        
            DISPATCH();
        
        CASE_CODE(RETURN):
            a=DECODE_A(bc);
        
            if (v->openupvalues) { /* Close upvalues */
                vm_closeupvalues(v, reg);
            }
        
            if (a>0) {
                b=DECODE_B(bc);
                reg[0] = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
            } else {
                reg[0] = MORPHO_NIL; /* No return value; returns nil */
            }
            
            if (v->fp>v->frame) {
                bool shouldreturn = (v->fp->ret);
                v->fp--;
                v->konst=v->fp->function->konst.data; /* Restore the constant table */
                reg=v->fp->roffset+v->stack.data; /* Restore registers */
                v->stack.count=v->fp->stackcount; /* Restore the stack size */
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
            right = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
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
            left = (DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
            right = (DECODE_ISCCONSTANT(bc) ? v->konst[c] : reg[c]);
        
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
                } else {
                    /* Otherwise, raise an error */
                    char *p = (MORPHO_ISSTRING(right) ? MORPHO_GETCSTRING(right) : "");
                    VERROR(VM_OBJECTLACKSPROPERTY, p);
                }
            } else if (MORPHO_ISCLASS(left)) {
                /* If it's a class, we lookup a method and bind it to self, which is in r0 */
                objectclass *klass = MORPHO_GETCLASS(left);
                if (dictionary_get(&klass->methods, right, &reg[a])) {
                    objectinvocation *bound=object_newinvocation(reg[0], reg[a]);
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
            } else if (MORPHO_ISOBJECT(left)) {
                /* If it's an object, it may have a veneer class */
                objectclass *klass = builtin_getveneerclass(MORPHO_GETOBJECTTYPE(left));
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
            } else {
                ERROR(VM_NOTANOBJECT);
            }
            DISPATCH();
        
        CASE_CODE(SPR):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[a];
            right = (DECODE_ISCCONSTANT(bc) ? v->konst[c] : reg[c]);
            
            if (MORPHO_ISINSTANCE(left)) {
                objectinstance *instance = MORPHO_GETINSTANCE(left);
                dictionary_insertintern(&instance->fields, v->konst[b], right);
            } else {
                ERROR(VM_NOTANOBJECT);
            }
        
            DISPATCH();
        
        CASE_CODE(LIX):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[a];
        
            if (MORPHO_ISARRAY(left)) {
                objectarrayerror err=array_getelement(MORPHO_GETARRAY(left), c-b+1, &reg[b], &reg[b]);
                if (err!=ARRAY_OK) ERROR( array_error(err) );
            } else {
                if (!vm_invoke(v, left, indexselector, c-b+1, &reg[b], &reg[b])) {
                    ERROR(VM_NOTINDEXABLE);
                }
                ERRORCHK();
            }
        
            DISPATCH();
        
        CASE_CODE(SIX):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            left = reg[a];
        
            if (MORPHO_ISARRAY(left)) {
                objectarrayerror err=array_setelement(MORPHO_GETARRAY(left), c-b, &reg[b], reg[c]);
                if (err!=ARRAY_OK) ERROR( array_error(err) );
            } else {
                if (!vm_invoke(v, left, setindexselector, c-b+1, &reg[b], &right)) {
                    ERROR(VM_NOTINDEXABLE);
                }
                ERRORCHK();
            }
        
            DISPATCH();
        
        CASE_CODE(ARRAY):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            if (DECODE_ISBCONSTANT(bc)) {
                    
            } else {
                objectarray *new = object_arrayfromvalueindices((unsigned int) c-b+1, &reg[b]);
                if (new) {
                    reg[a]=MORPHO_OBJECT(new);
                    vm_bindobject(v, reg[a]);
                } else {
                    ERROR(ERROR_ALLOCATIONFAILED);
                }
            }
            
            DISPATCH();
        
        CASE_CODE(CAT):
            a=DECODE_A(bc); b=DECODE_B(bc); c=DECODE_C(bc);
            reg[a]=morpho_concatenatestringvalues(c-b+1, reg+b);
            vm_bindobject(v, reg[a]);
            DISPATCH();
        
        CASE_CODE(PRINT):
            b=DECODE_B(bc);
            left=(DECODE_ISBCONSTANT(bc) ? v->konst[b] : reg[b]);
#ifdef MORPHO_COLORTERMINAL
            printf("\033[1m");
#endif
            if (!vm_invoke(v, left, printselector, 0, NULL, &right)) {
                morpho_printvalue(left);
            }
#ifdef MORPHO_COLORTERMINAL
            printf("\033[0m");
#endif
            printf("\n");
            DISPATCH();
        
        CASE_CODE(RAISE):
            a=DECODE_A(bc);
            if (MORPHO_ISSTRING(reg[a])) {
                ERROR(MORPHO_GETCSTRING(reg[a]));
            }
            DISPATCH();
        
        CASE_CODE(BREAK):
            if (v->debug) {
                v->fp->pc=pc;
                v->fp->roffset=reg-v->stack.data;
                debugger(v);
                ERRORCHK();
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
                    printf("%s:\t\t%lu\n", opname[i], opcount[i]);
                }
                
                printf(",");
                for (unsigned int i=0; i<OP_END; i++) printf("%s, ", opname[i]);
                printf("\n");
                
                for (unsigned int i=0; i<OP_END; i++) {
                    printf("%s, ", opname[i]);
                    for (unsigned int j=0; j<OP_END; j++) {
                        printf("%lu ", opopcount[i][j]);
                        if (j<OP_END-1) printf(",");
                    }
                    printf("\n");
                }
            }
            #endif
            return true;
    }

#undef INTERPRET_LOOP
#undef CASE_CODE
#undef DISPATCH
    
vm_error:
    v->fp->pc=pc;
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

/** Returns a VM's error block */
error *morpho_geterror(vm *v) {
    return &v->err;
}

/** @brief Public interface to raise a runtime error
 * @param v        the virtual machine
 * @param id       error id
 * @param ...      additional data for sprintf. */
void morpho_runtimeerror(vm *v, errorid id, ...) {
    va_list args;
    
    va_start(args, id);
    morpho_writeerrorwithidvalist(&v->err, id, ERROR_POSNUNIDENTIFIABLE, ERROR_POSNUNIDENTIFIABLE, args);
    va_end(args);
}

/** @brief Binds a set of objects to a Virtual Machine; public interface.
 *  @details Any object created during execution should be bound to a VM; this object is then managed by the garbage collector.
 *  @param v      the virtual machine
 *  @param obj    objects to bind */
void morpho_bindobjects(vm *v, int nobj, value *obj) {
    /* Now bind the new objects in. */
    for (unsigned int i=0; i<nobj; i++) {
        object *ob = MORPHO_GETOBJECT(obj[i]);
        if (MORPHO_ISOBJECT(obj[i]) && ob->status==OBJECT_ISUNMANAGED) {
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
        /* Temporarily store these objects at the top of the globals array */
        int gcount=v->globals.count;
        varray_valueadd(&v->globals, obj, nobj);
        
        vm_collectgarbage(v);
        /* Restore globals count */
        v->globals.count=gcount;
    }
}

/** Runs a program
 * @param[in] v - the virtual machine to use
 * @param[in] p - program to run
 * @returns true on success, false if an error occurred */
bool morpho_run(vm *v, program *p) {
    /* Set the current program */
    v->current=p;
    
    /* Clear current error state */
    error_clear(&v->err);

    /* Set up the callframe stack */
    v->fp=v->frame; /* Set the frame pointer to the bottom of the stack */
    v->fp->function=p->global;
    v->fp->closure=NULL;
    v->fp->roffset=0;
    
    /* Initialize global variables */
    int oldsize = v->globals.count;
    varray_valueresize(&v->globals, p->nglobals);
    v->globals.count=p->nglobals;
    for (int i=oldsize; i<p->nglobals; i++) v->globals.data[i]=MORPHO_NIL; /* Zero out globals */
    
    /* Set instruction base */
    v->instructions = p->code.data;
    if (!v->instructions) return false;
    
    /* Set up the constant table */
    varray_value *konsttable=object_functiongetconstanttable(p->global);
    if (!konsttable) return false;
    v->konst = konsttable->data;
    
    /* and initially set the register pointer to the bottom of the stack */
    value *reg = v->stack.data;
    
    /* Expand and clear the stack if necessary */
    if (v->fp->function->nregs>v->stack.count) {
        unsigned int oldcount=v->stack.count;
        vm_expandstack(v, &reg, v->fp->function->nregs-v->stack.count);
        for (unsigned int i=oldcount; i<v->stack.count; i++) v->stack.data[i]=MORPHO_NIL;
    }
    
    instructionindx start = program_getentry(p);
    
    return morpho_interpret(v, reg, start);
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
    
    if (MORPHO_ISBUILTINFUNCTION(fn)) {
        objectbuiltinfunction *f = MORPHO_GETBUILTINFUNCTION(fn);
        
        /* Copy arguments across to comply with call standard */
        value xargs[nargs+1];
        xargs[0]=r0;
        for (unsigned int i=0; i<nargs; i++) xargs[i+1]=args[i];
        
        *ret=(f->function) (v, nargs, xargs);
        success=true;
    } else if (MORPHO_ISFUNCTION(fn) || MORPHO_ISCLOSURE(fn)) {
        ptrdiff_t aoffset=0;
        value *xargs=args;
        
        /* If the arguments are on the stack, we need to keep to track of this */
        bool argsonstack=(args>v->stack.data && args<v->stack.data+v->stack.capacity);
        if (argsonstack) aoffset=args-v->stack.data;
        
        value *reg=v->stack.data+v->fp->roffset;
        instruction *pc=v->fp->pc;
        
        /* Set up the function call, advancing the frame pointer and expanding the stack if necessary */
        if (vm_call(v, fn, v->fp->function->nregs, nargs, &pc, &reg)) {
            if (argsonstack) xargs=v->stack.data+aoffset;
            
            /* Now place the function (or self) and arguments on the stack */
            reg[0]=r0;
            for (unsigned int i=0; i<nargs; i++) reg[i+1]=xargs[i];
            
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
    }
    
    return success;
}

/** Finds a method */
bool morpho_lookupmethod(vm *v, value obj, value label, value *method) {
    if (MORPHO_ISINSTANCE(obj)) {
        objectinstance *instance=MORPHO_GETINSTANCE(obj);
        return dictionary_get(&instance->klass->methods, label, method);
    } else {
        objectclass *klass = builtin_getveneerclass(MORPHO_GETOBJECTTYPE(obj));
        if (klass) {
            return dictionary_get(&klass->methods, label, method);
        }
    }
    
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

/** Sets whether debugging is active or not for a virtual machine */
void morpho_setdebug(vm *v, bool active) {
    v->debug=active;
}

/* **********************************************************************
* Initialization
* ********************************************************************** */

/** Initializes morpho */
void morpho_initialize(void) {
    object_initialize(); // Must be first for zombie object tracking
    error_initialize();
    random_initialize();
    compile_initialize();
    builtin_initialize();
    
#ifdef MORPHO_DEBUG_GCSIZETRACKING
    dictionary_init(&sizecheck);
#endif
    
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
    morpho_defineerror(VM_NOTINDEXABLE, ERROR_HALT, VM_NOTINDEXABLE_MSG);
    morpho_defineerror(VM_OUTOFBOUNDS, ERROR_HALT, VM_OUTOFBOUNDS_MSG);
    morpho_defineerror(VM_NONNUMINDX, ERROR_HALT, VM_NONNUMINDX_MSG);
    morpho_defineerror(VM_ARRAYWRONGDIM, ERROR_HALT, VM_ARRAYWRONGDIM_MSG);
    morpho_defineerror(VM_DVZR, ERROR_HALT, VM_DVZR_MSG);
    
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
    
    enumerateselector=builtin_internsymbolascstring(MORPHO_ENUMERATE_METHOD);
    countselector=builtin_internsymbolascstring(MORPHO_COUNT_METHOD);
    cloneselector=builtin_internsymbolascstring(MORPHO_CLONE_METHOD);
    
    printselector=builtin_internsymbolascstring(MORPHO_PRINT_METHOD);
}

/** Finalizes morpho */
void morpho_finalize(void) {
    morpho_freeobject(initselector);
    
    error_finalize();
    compile_finalize();
    builtin_finalize();
    object_finalize(); // Must be last for zombie object tracking
}
