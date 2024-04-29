/** @file metafunction.c
 *  @author T J Atherton
 *
 *  @brief Implement objectmetafunctions and the Metafunction veneer class
 */

#include "morpho.h"
#include "classes.h"
#include "common.h"

/* **********************************************************************
 * objectmetafunction definitions
 * ********************************************************************** */

void objectmetafunction_freefn(object *obj) {
    objectmetafunction *f = (objectmetafunction *) obj;
    varray_valueclear(&f->fns);
}

void objectmetafunction_markfn(object *obj, void *v) {
    objectmetafunction *f = (objectmetafunction *) obj;
}

size_t objectmetafunction_sizefn(object *obj) {
    objectmetafunction *f = (objectmetafunction *) obj;
    return sizeof(objectmetafunction)+sizeof(value)*f->fns.count;
}

void objectmetafunction_printfn(object *obj, void *v) {
    objectmetafunction *f = (objectmetafunction *) obj;
    if (f) morpho_printf(v, "<fn %s>", (MORPHO_ISNIL(f->name) ? "" : MORPHO_GETCSTRING(f->name)));
}

objecttypedefn objectmetafunctiondefn = {
    .printfn=objectmetafunction_printfn,
    .markfn=objectmetafunction_markfn,
    .freefn=objectmetafunction_freefn,
    .sizefn=objectmetafunction_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/* **********************************************************************
 * objectmetafunction utility functions
 * ********************************************************************** */

/** Creates a new metafunction */
objectmetafunction *object_newmetafunction(value name) {
    objectmetafunction *new = (objectmetafunction *) object_new(sizeof(objectmetafunction), OBJECT_METAFUNCTION);

    if (new) {
        new->name=MORPHO_NIL;
        if (MORPHO_ISSTRING(name)) new->name=object_clonestring(name);
        varray_valueinit(&new->fns);
    }

    return new;
}

/** Wraps a function in a metafunction */
bool metafunction_wrap(value name, value fn, value *out) {
    if (!MORPHO_ISCALLABLE(fn)) return false;
    
    objectmetafunction *mf = object_newmetafunction(name);
    if (!mf) return false;
    
    metafunction_add(mf, fn);
    *out = MORPHO_OBJECT(mf);
    
    return true;
}

/** Adds a function to a metafunction */
bool metafunction_add(objectmetafunction *f, value fn) {
    return varray_valuewrite(&f->fns, fn);
}

/** Extracts a type from a value */
bool metafunction_typefromvalue(value v, value *out) {
    objectclass *clss = NULL;
    value type = MORPHO_NIL;
    
    if (MORPHO_ISINSTANCE(v)) {
        clss=MORPHO_GETINSTANCE(v)->klass;
    } else if (MORPHO_ISOBJECT(v)) {
        clss = object_getveneerclass(MORPHO_GETOBJECT(v)->type);
    } else clss = value_getveneerclass(v);
    
    if (clss) *out = MORPHO_OBJECT(clss);
    return clss;
}

/** Checks if val matches a given type */
bool metafunction_matchtype(value type, value val) {
    value match;
    if (!metafunction_typefromvalue(val, &match)) return false;
    
    if (MORPHO_ISNIL(type) || // If type is unset, we always match
        MORPHO_ISEQUAL(type, match)) return true; // Or if the types are the same
    
    return false;
}

signature *_getsignature(value fn) {
    if (MORPHO_ISFUNCTION(fn)) {
        return &MORPHO_GETFUNCTION(fn)->sig;
    } else if (MORPHO_ISBUILTINFUNCTION(fn)) {
        return &MORPHO_GETBUILTINFUNCTION(fn)->sig;
    }
    return NULL;
}

/** Resolves a metafunction given calling arguments */
bool metafunction_slowresolve(objectmetafunction *f, int nargs, value *args, value *out) {
    for (int i=0; i<f->fns.count; i++) {
        signature *s = _getsignature(f->fns.data[i]);
        if (!s) continue;
        
        int nparams; value *ptypes;
        signature_paramlist(s, &nparams, &ptypes);
        if (nargs!=nparams) continue;
        
        int j;
        for (j=0; j<nparams; j++) {
            if (!metafunction_matchtype(ptypes[j], args[j])) break;
        }
        if (j==nparams) { *out=f->fns.data[i]; return true; }
    }
    
    return false;
}

/* **********************************************************************
 * Fast metafunction resolver
 * ********************************************************************** */

enum {
    MF_CHECKNARGS,
    MF_BRANCHNARGS,
    MF_RESOLVE,
    MF_FAIL
};

DEFINE_VARRAY(mfinstruction, mfinstruction);

#define MFINSTRUCTION_EMPTY -1

#define MFINSTRUCTION_FAIL { .opcode=MF_FAIL, .branch=MFINSTRUCTION_EMPTY }
#define MFINSTRUCTION_RESOLVE(fn) { .opcode=MF_RESOLVE, .data.resolvefn=fn, .branch=MFINSTRUCTION_EMPTY }
#define MFINSTRUCTION_CHECKNARG(n, brnch) { .opcode=MF_CHECKNARGS, .data.nargs=n, .branch=brnch }
#define MFINSTRUCTION_BRANCHNARG(table, brnch) { .opcode=MF_BRANCHNARGS, .data.btable=table, .branch=brnch }

typedef struct {
    signature *sig; /** Signature of the target */
    value fn; /** The target */
} mfresult;

typedef struct {
    int count;
    mfresult *rlist;
} mfset;

/** Static intiializer for the mfset */
#define MFSET(c, l) { .count=c, .rlist=l }

DECLARE_VARRAY(mfset, mfset)
DEFINE_VARRAY(mfset, mfset)

typedef struct {
    objectmetafunction *fn;
    dictionary pcount;
    varray_mfset set; /** A stack of possible sets */
} mfcompiler;

/** Initialize the metafunction compiler */
void mfcompiler_init(mfcompiler *c, objectmetafunction *fn) {
    c->fn=fn;
    dictionary_init(&c->pcount);
    varray_mfsetinit(&c->set);
}

/** Clear the metafunction compiler */
void mfcompiler_clear(mfcompiler *c, objectmetafunction *fn) {
    dictionary_clear(&c->pcount);
    varray_mfsetclear(&c->set);
}

/** Pushes a set onto the stack */
void mfcompiler_pushset(mfcompiler *c, mfset *set) {
    varray_mfsetadd(&c->set, set, 1);
}

/** Pops a set off the stack, optionally returning it */
bool mfcompiler_popset(mfcompiler *c, mfset *set) {
    if (c->set.count<=0) return false;
    if (set) *set = c->set.data[c->set.count-1];
    c->set.count--;
}

/** Disassemble */
void mfcompiler_disassemble(mfcompiler *c) {
    int ninstr = c->fn->resolver.count;
    morpho_printvalue(NULL, MORPHO_OBJECT(c->fn));
    printf(":\n");
    for (int i=0; i<ninstr; i++) {
        mfinstruction *instr = &c->fn->resolver.data[i];
        printf("%5i : ", i) ;
        switch(instr->opcode) {
            case MF_CHECKNARGS: {
                printf("checkargs (%i) -> (%i)", instr->data.nargs, i+instr->branch+1);
                break;
            }
            case MF_BRANCHNARGS: {
                printf("branchargs (%i) -> (%i)\n", instr->data.nargs, i+instr->branch+1);
                for (int k=0; k<instr->data.btable.count; k++) {
                    printf("        %i -> %i\n", k, i+instr->data.btable.data[k]+1);
                }
                
                break;
            }
            case MF_RESOLVE: {
                printf("resolve ");
                signature *sig = _getsignature(instr->data.resolvefn);
                printf(" ");
                if (sig) signature_print(sig);
                break;
            }
            case MF_FAIL: printf("fail"); break;
        }
        printf("\n");
    }
}

/** Counts the range of parameters for the function call */
void mfcompile_countparams(mfcompiler *c, mfset *set, int *min, int *max) {
    int imin=INT_MAX, imax=INT_MIN;
    for (int i=0; i<set->count; i++) {
        int nparams;
        signature_paramlist(set->rlist[i].sig, &nparams, NULL);
        if (nparams<imin) imin=nparams;
        if (nparams>imax) imax=nparams;
    }
    if (min) *min = imin;
    if (max) *max = imax;
}

/** Places the various outcomes for a parameter into a dictionary */
bool mfcompile_outcomes(mfcompiler *c, mfset *set, int i, dictionary *out) {
    out->count=0;
    for (int k=0; i<set->count; i++) { // Loop over outcomes
        int nparams; value *ptypes;
        signature_paramlist(set->rlist[k].sig, &nparams, &ptypes);
        if (i>=nparams) continue;
        value val = MORPHO_INTEGER(1);
        if (dictionary_get(out, ptypes[i], &val)) val=MORPHO_INTEGER(MORPHO_GETINTEGERVALUE(val)+1);
        if (!dictionary_insert(out, ptypes[i], val)) return false;
    }
    return true;
}

/** Count the divergent outcomes of each parameter */
bool mfcompile_countoutcomes(mfcompiler *c, mfset *set, int *best) {
    varray_int count;
    varray_intinit(&count);
    
    dictionary dict;
    dictionary_init(&dict);
    
    int k=0; // Loop over parameters
    do {
        mfcompile_outcomes(c, set, k, &dict);
        varray_intwrite(&count, dict.count);
        k++;
    } while (dict.count);
    
    dictionary_clear(&dict);
    varray_intclear(&count);
    
    return false;
}

mfindx mfcompile_insertinstruction(mfcompiler *c, mfinstruction instr) {
    return varray_mfinstructionwrite(&c->fn->resolver, instr);
}

mfindx mfcompile_currentinstruction(mfcompiler *c) {
    return c->fn->resolver.count-1;
}

void mfcompile_setbranch(mfcompiler *c, mfindx i, mfindx branch) {
    if (i>=c->fn->resolver.count) return;
    c->fn->resolver.data[i].branch=branch;
}

void mfcompile_fail(mfcompiler *c);
void mfcompile_resolve(mfcompiler *c, mfset *set);
void mfcompile_dispatchonparam(mfcompiler *c, mfset *set, int i);
void mfcompile_dispatchonnarg(mfcompiler *c, mfset *set, int min, int max);
void mfcompile_set(mfcompiler *c, mfset *set);

/** Inserts a fail instruction */
void mfcompile_fail(mfcompiler *c) {
    mfinstruction fail = MFINSTRUCTION_FAIL;
    mfcompile_insertinstruction(c, fail);
}

/** Compiles a single result */
void mfcompile_resolve(mfcompiler *c, mfset *set) {
    // Should check all arguments have been resolved

    mfinstruction instr = MFINSTRUCTION_RESOLVE(set->rlist->fn);
    mfcompile_insertinstruction(c, instr);
}

/** Attempts to dispatch based on a parameter i */
void mfcompile_dispatchonparam(mfcompiler *c, mfset *set, int i) {
    
}

/** Attempts to dispatch based on the number of arguments */
void mfcompile_dispatchonnarg(mfcompiler *c, mfset *set, int min, int max) {
    if (set->count==2) {
        for (int i=0; i<2; i++) {
            signature *sig = set->rlist[i].sig;
            int nparams;
            signature_paramlist(sig, &nparams, NULL); // Get the number of parameters
            mfinstruction instr = MFINSTRUCTION_CHECKNARG(nparams, 0);
            mfindx cindx = mfcompile_insertinstruction(c, instr); // Write the check nargs instruction
            
            mfset res = MFSET(1, &set->rlist[i]); // If it works, resolve on this implementation
            mfcompile_resolve(c, &res);
            
            // Fix the branch instruction
            mfindx eindx = mfcompile_currentinstruction(c);
            mfcompile_setbranch(c, cindx, eindx-cindx);
        }
        mfcompile_fail(c);
    } else {
        varray_int btable;
        varray_intinit(&btable);
        for (int i=0; i<=max; i++) varray_intwrite(&btable, 0);
        
        // Insert the branch table instruction
        mfinstruction table = MFINSTRUCTION_BRANCHNARG(btable, 0);
        mfindx tindx=mfcompile_insertinstruction(c, table);
        // Immediately follow by a fail instruction if this falls through
        mfcompile_fail(c);
        
        // Count the number of implementations for each parameter count
        for (int k=0; k<set->count; k++) {
            btable.data[signature_countparams(set->rlist[k].sig)]++;
        }

        // Compile the outcomes for each parameter count
        for (int n=0; n<=max; n++) {
            int nimp = btable.data[n];
            if (!nimp) continue;
            
            // Select all implementations that match this parameter value
            mfresult rlist[nimp];
            mfset iset = MFSET(nimp, rlist);
            for (int k=0, j=0; k<set->count; k++) {
                int m=signature_countparams(set->rlist[k].sig);
                if (m==n) { rlist[j]=set->rlist[k]; j++; }
            }
            
            btable.data[n]=mfcompile_currentinstruction(c)-tindx;
            mfcompile_set(c, &iset);
        }
    }
}

/** Attempts to discriminate between a list of possible signatures */
void mfcompile_set(mfcompiler *c, mfset *set) {
    if (set->count==1) mfcompile_resolve(c, set);
    
    int min, max; // Count the range of possible parameters
    mfcompile_countparams(c, set, &min, &max);
    
    // Dispatch on the number of parameters if it's in doubt
    if (min!=max) return mfcompile_dispatchonnarg(c, set, min, max);
    
    // If just one parameter, dispatch on it
    if (min==1) return mfcompile_dispatchonparam(c, set, 0);
    
    int best;
    if (mfcompile_countoutcomes(c, set, &best)) return mfcompile_dispatchonparam(c, set, best);
}

/** Compiles the metafunction resolver */
void metafunction_compile(objectmetafunction *fn) {
    mfset set;
    set.count = fn->fns.count;
    if (!set.count) return;
    
    mfresult rlist[set.count];
    set.rlist=rlist;
    for (int i=0; i<set.count; i++) {
        rlist[i].sig=_getsignature(fn->fns.data[i]);
        rlist[i].fn=fn->fns.data[i];
    }
    
    mfcompiler compiler;
    mfcompiler_init(&compiler, fn);
    
    mfcompile_set(&compiler, &set);
    
    //mfcompiler_disassemble(&compiler);
    
    mfcompiler_clear(&compiler, fn);
}

/** Execute the metafunction's resolver */
bool metafunction_resolve(objectmetafunction *fn, int nargs, value *args, value *out) {
    mfinstruction *pc = fn->resolver.data;
    if (!pc) return metafunction_slowresolve(fn, nargs, args, out);
    
    do {
        switch(pc->opcode) {
            case MF_CHECKNARGS:
                if (pc->data.nargs!=nargs) pc+=pc->branch;
                break;
            case MF_BRANCHNARGS:
                if (nargs<=pc->data.btable.count) {
                    pc+=pc->data.btable.data[nargs];
                } else pc+=pc->branch;
                break;
            case MF_RESOLVE:
                *out = pc->data.resolvefn;
                return true;
            case MF_FAIL:
                return false;
        }
        pc++;
    } while(true);
}

/* **********************************************************************
 * Metafunction veneer class
 * ********************************************************************** */


/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

objecttype objectmetafunctiontype;

void metafunction_initialize(void) {
    // Create function object type
    objectmetafunctiontype=object_addtype(&objectmetafunctiondefn);
    
    // Locate the Object class to use as the parent class of Metafunction
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // Create function veneer class
    //value metafunctionclass=builtin_addclass(METAFUNCTION_CLASSNAME, MORPHO_GETCLASSDEFINITION(Metafunction), objclass);
    //object_setveneerclass(OBJECT_FUNCTION, functionclass);
    
    // No constructor as objectmetafunctions are generated by the compiler
    
    // Metafunction error messages
}
