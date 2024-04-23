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
    MF_RESOLVE,
    MF_FAIL
};

DEFINE_VARRAY(mfinstruction, mfinstruction);

typedef int mfindx;

typedef struct {
    objectmetafunction *fn;
    varray_int checkedargs; /** List of arguments already checked */
} mfcompiler;

typedef struct {
    signature *sig; /** Signature of the target */
    value fn; /** The target */
} mfresult;

/** Initialize the metafunction compiler */
void mfcompiler_init(mfcompiler *c, objectmetafunction *fn) {
    c->fn=fn;
    varray_intinit(&c->checkedargs);
}

/** Clear the metafunction compiler */
void mfcompiler_clear(mfcompiler *c, objectmetafunction *fn) {
    varray_intclear(&c->checkedargs);
}

mfindx mfcompile_insertinstruction(mfcompiler *c, mfinstruction instr) {
    return varray_mfinstructionwrite(&c->fn->resolver, instr);
}

/** Compiles a single result */
void mfcompile_singleresult(mfcompiler *c, mfresult *result) {
    /** Should check remaining args */
    
    mfinstruction instr = { .opcode=MF_RESOLVE, .data.resolvefn=result->fn };
    mfcompile_insertinstruction(c, instr);
}

/** Attempts to discriminate between a list of possible signatures */
void mfcompile_set(mfcompiler *fn, int nres, mfresult *rlist) {
    if (nres==1) mfcompile_singleresult(fn, rlist);
}

/** Compiles the metafunction resolver */
void metafunction_compile(objectmetafunction *fn) {
    int nfn = fn->fns.count;
    if (!nfn) return;
    
    mfresult rlist[nfn];
    for (int i=0; i<nfn; i++) {
        rlist[i].sig=_getsignature(fn->fns.data[i]);
        rlist[i].fn=fn->fns.data[i];
    }
    
    mfcompiler compiler;
    mfcompiler_init(&compiler, fn);
    
    mfcompile_set(&compiler, nfn, rlist);
    
    mfcompiler_clear(&compiler, fn);
}

/** Execute the metafunction's resolver */
bool metafunction_resolve(objectmetafunction *fn, int nargs, value *args, value *out) {
    mfinstruction *pc = fn->resolver.data;
    if (!pc) return metafunction_slowresolve(fn, nargs, args, out);
    
    do {
        switch(pc->opcode) {
            case MF_RESOLVE:
                *out = pc->data.resolvefn;
                return true;
            case MF_FAIL:
                return false;
        }
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
