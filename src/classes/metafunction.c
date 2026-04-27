/** @file metafunction.c
 *  @author T J Atherton
 *
 *  @brief Implement objectmetafunctions and the Metafunction veneer class
 */

#include <limits.h>
#include <stdlib.h>

#include "morpho.h"
#include "classes.h"
#include "common.h"

#define ERR_CHECK(expr, label) do { if (!(expr)) goto label; } while (0)
#define ERR_CHECK_RETURN(expr) do { if (!(expr)) return false; } while (0)

/* **********************************************************************
 * objectmetafunction definitions
 * ********************************************************************** */

void objectmetafunction_freefn(object *obj) {
    objectmetafunction *f = (objectmetafunction *) obj;
    morpho_freeobject(f->name);
    varray_valueclear(&f->fns);
    metafunction_clearinstructions(f);
}

void objectmetafunction_markfn(object *obj, void *v) {
    objectmetafunction *f = (objectmetafunction *) obj;
    morpho_markvalue(v, f->name); // Mark the name
    morpho_markvarrayvalue(v, &f->fns); // Preserve implementations while building/frozen
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
        new->klass=NULL; 
        new->state=METAFUNCTION_BUILDING;
        varray_valueinit(&new->fns);
        varray_mfinstructioninit(&new->resolver);
        new->entry=0;
    }

    return new;
}

/** Clone a metafunction */
objectmetafunction *metafunction_clone(objectmetafunction *f) {
    objectmetafunction *new = object_newmetafunction(f->name);
    
    if (new) {
        varray_valueadd(&new->fns, f->fns.data, f->fns.count);
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
    if (f->state==METAFUNCTION_FROZEN) {
        metafunction_clearinstructions(f);
        f->state=METAFUNCTION_BUILDING;
    }
    return varray_valuewrite(&f->fns, fn);
}

/** Checks if val matches a given type */
bool metafunction_matchtype(value type, value val) {
    return value_istype(val, type);
}

/** Sets the parent class of a metafunction */
void metafunction_setclass(objectmetafunction *f, objectclass *klass) {
    f->klass=klass;
}

/** Returns a metafunction's class if any */
objectclass *metafunction_class(objectmetafunction *f) {
    return f->klass;
}

/** Finds whether an implementation f occurs in a metafunction */
bool metafunction_matchfn(objectmetafunction *fn, value f) {
    for (int i=0; i<fn->fns.count; i++) if (MORPHO_ISEQUAL(fn->fns.data[i], f)) return true;
    return false;
}

/** Checks if a metafunction matches a given list of implementations */
bool metafunction_matchset(objectmetafunction *fn, int n, value *fns) {
    for (int i=0; i<n; i++) {
        if (!metafunction_matchfn(fn, fns[i])) return false;
    }
    return true;
}

/** Infer the return type from the contents of a metafunction, if known */
void metafunction_inferreturntype(objectmetafunction *fn, value *type) {
    value rtype = MORPHO_NIL;
    
    for (int i=0; i<fn->fns.count; i++) {
        signature *sig = metafunction_getsignature(fn->fns.data[i]);
        if (i==0) {
            rtype=sig->ret;
        } else {
            if (!MORPHO_ISEQUAL(sig->ret,rtype)) rtype=MORPHO_NIL; 
        }
    }
    
    *type=rtype;
}

/** Clears the compiled code from a given metafunction. */
void metafunction_clearinstructions(objectmetafunction *fn) {
    varray_mfinstructionclear(&fn->resolver);
    fn->entry=0;
}

/** Finalizes a metafunction. */
bool metafunction_finalize(objectmetafunction *fn, error *err) {
    if (fn->state==METAFUNCTION_FROZEN) return true;
    if (!metafunction_compile(fn, err)) return false;
    //metafunction_disassemble(fn);
    fn->state=METAFUNCTION_FROZEN;
    return true;
}

/** Finalizes any metafunctions stored in a linked object list. */
bool metafunction_finalizelist(object *list, error *err) {
    for (object *obj=list; obj!=NULL; obj=obj->next) {
        if (obj->type==OBJECT_METAFUNCTION &&
            !metafunction_finalize((objectmetafunction *) obj, err)) {
            return false;
        }
    }

    return true;
}

signature *metafunction_getsignature(value fn) {
    if (MORPHO_ISFUNCTION(fn)) {
        return &MORPHO_GETFUNCTION(fn)->sig;
    } else if (MORPHO_ISBUILTINFUNCTION(fn)) {
        return &MORPHO_GETBUILTINFUNCTION(fn)->sig;
    } else if (MORPHO_ISCLOSURE(fn)) {
        return &MORPHO_GETCLOSURE(fn)->func->sig;
    }
    return NULL;
}

value metafunction_getname(value fn) {
    if (MORPHO_ISFUNCTION(fn)) {
        return MORPHO_GETFUNCTION(fn)->name;
    } else if (MORPHO_ISBUILTINFUNCTION(fn)) {
        return MORPHO_GETBUILTINFUNCTION(fn)->name;
    } else if (MORPHO_ISCLOSURE(fn)) {
        return MORPHO_GETCLOSURE(fn)->func->name;
    }
    return MORPHO_NIL;
}

/* **********************************************************************
 * Metafunction base-case resolver
 * ********************************************************************** */

/** Define a possible resolution */
typedef struct {
    value fn;
    signature *sig;
} mfresolution;

/** Checks if a resolution matches a given number of arguments */
bool mfresolution_checkarity(mfresolution *res, int nargs) {
    if (!res->sig) return false;
    int nparams = signature_countparams(res->sig);
    return (nparams == nargs) || (nargs >= nparams - 1 && signature_isvarg(res->sig));
}

/** Checks if a resolution matches a given set of arguments based on types */
bool mfresolution_checktypes(mfresolution *res, int nargs, value *args) {
    if (!res->sig) return false;
    int nparams = signature_countparams(res->sig);
    
    for (int i=0; i<nargs && i<nparams; i++) { // Note this works as vargs are currently untyped
        if (!value_istype(args[i], res->sig->types.data[i])) return false;
    }
    
    return true;
}

/** Rank a parameter type by its distance from an actual argument type. */
static bool mfresolution_rankparamtype(value actual, value type, int *out) {
    if (MORPHO_ISEQUAL(actual, type)) { *out=0; return true; }
    if (MORPHO_ISNIL(type)) { *out=INT_MAX; return true; } // Ensures a wildcare loses to an actual type

    if (MORPHO_ISCLASS(actual) && MORPHO_ISCLASS(type)) {
        if (class_comparedistance(MORPHO_GETCLASS(actual), MORPHO_GETCLASS(type), out)) {
            *out = abs(*out); return true;
        }
    }

    return false;
}

/** Compare the specificity of two parameter types relative to an actual argument type.
 *  @returns true if the types are comparable, setting out to the signed comparison. */
static bool mfresolution_compareparamtypes(value actual, value a, value b, int *out) {
    int arank, brank;
    if (mfresolution_rankparamtype(actual, a, &arank) &&
        mfresolution_rankparamtype(actual, b, &brank)) {
        *out = arank-brank;
        return true;
    }
    return false;
}

/** Compare resolutions by whether they are variadic. Non-variadic resolutions are more specific. */
static int mfresolution_comparevarg(mfresolution *a, mfresolution *b) {
    bool avarg = signature_isvarg(a->sig);
    bool bvarg = signature_isvarg(b->sig);
    if (avarg==bvarg) return 0;
    return (avarg ? 1 : -1);
}

/** Determine how many parameters should be checked when comparing resolutions. */
static int _min(int a, int b, int c) {
    return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

/** Returns the sign of an integer. */
static int _sign(int x) {
    return (x>0) - (x<0);
}

/** Compare the specificity of two resolutions.
 *  Returns <0 if a is more specific, >0 if b is more specific, 0 if they are equal
 *  or incomparable. */
static int mfresolution_comparespecificity(mfresolution *a, mfresolution *b, int nargs, value *args) {
    if (!a->sig || !b->sig) return 0;

    int ncheck=_min(signature_countparams(a->sig), signature_countparams(b->sig), nargs);
    int cmp[ncheck];
    for (int i=0; i<ncheck; i++) { // Compare each arg
        cmp[i]=0;
        value actual;
        if (value_type(args[i], &actual)) {
            mfresolution_compareparamtypes(actual, a->sig->types.data[i], b->sig->types.data[i], &cmp[i]);
        }
    }

    int firstsign = 0;
    for (int i=0; i<ncheck; i++) { // Check all entries have the same sign
        if (cmp[i]!=0) {
            if (firstsign==0) firstsign = _sign(cmp[i]);
            else if (firstsign!=_sign(cmp[i])) return 0;
        }
    }

    if (firstsign!=0) return firstsign; // If they did have the same sign, we have a winner
    return mfresolution_comparevarg(a, b); // Tiebreak on varg
}

/** A set of possible resolutions */
typedef struct {
    int count;
    mfresolution *data;
} mfresolutionset;

/** @brief Collapses a set, removing any resolutions that lack a signature */
void mfresolutionset_collapse(mfresolutionset *set) {
    int n=0;
    
    for (int i=0; i<set->count; i++) {
        if (set->data[i].sig) {
            set->data[n] = set->data[i];
            n++;
        }
    }
    
    set->count=n;
}

/** @brief Initialize the set of resolutions from the metafunction. */
void mfresolutionset_init(mfresolutionset *set, mfresolution *res, objectmetafunction *fn) {
    set->count = fn->fns.count;
    set->data=res;
    for (int i=0; i<set->count; i++) {
        res[i].fn = fn->fns.data[i];
        res[i].sig = metafunction_getsignature(res[i].fn);
    }
    mfresolutionset_collapse(set);
}

/** @brief Filter the resolution set by arity */
void mfresolutionset_filterbyarity(mfresolutionset *set, int nargs) {
    for (int i=0; i<set->count; i++) {
        if (!mfresolution_checkarity(&set->data[i], nargs)) set->data[i].sig = NULL;
    }
    mfresolutionset_collapse(set);
}

/** @brief Filter the resolution set by types of the arguments */
void mfresolutionset_filterbytypes(mfresolutionset *set, int nargs, value *args) {
    for (int i=0; i<set->count; i++) {
        if (!mfresolution_checktypes(&set->data[i], nargs, args)) set->data[i].sig = NULL;
    }
    mfresolutionset_collapse(set);
}

/** @brief Filter the resolution set to maximal resolutions by specificity. */
void mfresolutionset_filterbyspecificity(mfresolutionset *set, int nargs, value *args) {
    for (int i=0; i<set->count; i++) { // Compare all possible pairs and remove resolutions dominated by at least one other candidate
        if (!set->data[i].sig) continue;
        for (int j=i+1; j<set->count; j++) {
            if (!set->data[j].sig) continue;
            int cmp = mfresolution_comparespecificity(&set->data[i], &set->data[j], nargs, args);
            if (cmp < 0) set->data[j].sig = NULL;
            else if (cmp > 0) { set->data[i].sig = NULL; break; }
        }
    }
    mfresolutionset_collapse(set);
}

/** @brief Find a resolution, if any exists, for given arguments by direct comparison.
 @details Slow direct comparison acts as a source of truth for metafunction compiler.
 @param[in] fn - the metafunction to resolve
 @param[in] nargs - number of positional arguments
 @param[in] args - positional arguments @warning: the first user-visible argument should be in the zero position
 @param[out] err - error block to be filled out
 @param[out] out - resolved function
 @returns true if the metafunction was successfully resolved */
static bool metafunction_resolveslow(objectmetafunction *fn, int nargs, value *args, error *err, value *out) {
    if (fn->state!=METAFUNCTION_FROZEN) {
        if (err) error_writewithid(err, METAFUNCTION_UNFROZEN); return false;
    }

    int nres = fn->fns.count;
    
    mfresolutionset set; // Initial set of resolutions
    mfresolution res[nres];
    
    mfresolutionset_init(&set, res, fn);
    
    mfresolutionset_filterbyarity(&set, nargs);
    mfresolutionset_filterbytypes(&set, nargs, args);
    mfresolutionset_filterbyspecificity(&set, nargs, args);
    
    switch (set.count) {
        case 0: if (err) error_writewithid(err, VM_MLTPLDSPTCHFLD); return false;
        case 1: if (out) *out = set.data[0].fn; return true;
        default: if (err) error_writewithid(err, METAFUNCTION_CMPLAMBGS); return false;
    }
}

/* **********************************************************************
 * Fast metafunction resolver
 * ********************************************************************** */

DEFINE_VARRAY(mfinstruction, mfinstruction);

/** Bytecodes */
enum {
    MFOP_SLOW,
    MFOP_RESOLVE,
    MFOP_FAIL,
    MFOP_GETUID,
    MFOP_SPARSE
};

/* --------------------------
 * Fast resolver Compiler
 * -------------------------- */

/** Candidate metadata used by the metafunction compiler. */
typedef struct {
    int fnindex;
    signature *sig;
    int nparams;
    int minarity;
    int maxarity;
    bool varg;
    bool typed;
} mfcompileresolution;

/** Compiler state for building a metafunction resolver. */
typedef struct {
    objectmetafunction *fn;
    error *err;
    mfcompileresolution *resolutions;
    int nresolutions;
    bool hasvarg;
    bool hastyped;
} mfcompiler;

/** Initialize compiler state. */
static void mfcompiler_init(mfcompiler *compiler, objectmetafunction *fn, error *err) {
    compiler->fn = fn;
    compiler->err = err;
    compiler->nresolutions = fn->fns.count;
    compiler->hasvarg = false;
    compiler->hastyped = false;
    compiler->resolutions = malloc(sizeof(mfcompileresolution)*compiler->nresolutions);
}

/** Clear compiler state. */
static void mfcompiler_clear(mfcompiler *compiler) {
    if (compiler->resolutions) free(compiler->resolutions);
    compiler->resolutions = NULL;
    compiler->nresolutions = 0;
}

/** Write a compiler error. */
static void mfcompiler_error(mfcompiler *compiler, errorid id) {
    if (compiler->err) error_writewithid(compiler->err, id);
}

/** Analyze one candidate resolution. */
static bool mfcompiler_analyzecandidate(mfcompiler *compiler, int i) {
    ERR_CHECK_RETURN(compiler->resolutions);
    
    value fn = compiler->fn->fns.data[i];
    signature *sig = metafunction_getsignature(fn);
    ERR_CHECK_RETURN(sig);
    
    mfcompileresolution *resolution = &compiler->resolutions[i];
    resolution->fnindex = i;
    resolution->sig = sig;
    resolution->nparams = signature_countparams(sig);
    resolution->varg = signature_isvarg(sig);
    resolution->typed = signature_istyped(sig);
    resolution->minarity = (resolution->varg ? resolution->nparams-1 : resolution->nparams);
    resolution->maxarity = (resolution->varg ? -1 : resolution->nparams);
    
    compiler->hasvarg = compiler->hasvarg || resolution->varg;
    compiler->hastyped = compiler->hastyped || resolution->typed;
    return true;
}

/** Analyze the candidate set for a metafunction. */
static bool mfcompiler_analyze(mfcompiler *compiler) {
    for (int i=0; i<compiler->nresolutions; i++) {
        ERR_CHECK_RETURN(mfcompiler_analyzecandidate(compiler, i));
    }
    return true;
}

/** Check for duplicate implementations that are unavoidably ambiguous. */
static bool mfcompiler_checkduplicates(mfcompiler *compiler) {
    for (int i=0; i<compiler->nresolutions; i++) {
        for (int j=i+1; j<compiler->nresolutions; j++) {
            if (compiler->resolutions[i].varg==compiler->resolutions[j].varg &&
                signature_isequal(compiler->resolutions[i].sig, compiler->resolutions[j].sig)) {
                mfcompiler_error(compiler, METAFUNCTION_CMPLAMBGS);
                return false;
            }
        }
    }
    return true;
}

/** Emit one instruction into the resolver. */
static bool mfcompiler_emit(mfcompiler *compiler, mfinstruction instruction, mfindx *entry) {
    if (entry) *entry = compiler->fn->resolver.count;
    return varray_mfinstructionadd(&compiler->fn->resolver, &instruction, 1);
}

/** Emit multiple instructions into the resolver. */
static bool mfcompiler_emitmulti(mfcompiler *compiler, int n, mfinstruction *instructions, mfindx *entry) {
    if (entry) *entry = compiler->fn->resolver.count;
    return varray_mfinstructionadd(&compiler->fn->resolver, instructions, n);
}

/** Emit a resolver block that falls back to the slow path. */
static bool mfcompiler_emitslow(mfcompiler *compiler, mfindx *entry) {
    return mfcompiler_emit(compiler, MFOP_SLOW, entry);
}

/** Emit a resolver block that fails dispatch. */
static bool mfcompiler_emitfail(mfcompiler *compiler, mfindx *entry) {
    return mfcompiler_emit(compiler, MFOP_FAIL, entry);
}

/** Emit a resolver block that loads the uid of a given argument. */
static bool mfcompiler_emitgetuid(mfcompiler *compiler, int arg, mfindx *entry) {
    mfinstruction instructions[2] = { MFOP_GETUID, arg };
    return mfcompiler_emitmulti(compiler, 2, instructions, entry);
}

/** Emit a resolver block that resolves to a specific implementation. */
static bool mfcompiler_emitresolve(mfcompiler *compiler, int fnindex, mfindx *entry) {
    mfinstruction instructions[2] = { MFOP_RESOLVE, fnindex };
    return mfcompiler_emitmulti(compiler, 2, instructions, entry);
}

/** One entry in a sparse branch table. */
typedef struct {
    int value;
    mfindx target;
} mfcompilersparseentry;

/** Emit a sparse branch over a table of value/target matches. */
static bool mfcompiler_emitsparse(mfcompiler *compiler, int ncases, mfcompilersparseentry *table, mfindx deflt, mfindx *entry) {
    mfinstruction header[3] = { MFOP_SPARSE, ncases, deflt };
    ERR_CHECK_RETURN(mfcompiler_emitmulti(compiler, 3, header, entry));

    for (int i=0; i<ncases; i++) {
        mfinstruction match[2] = { table[i].value, table[i].target };
        ERR_CHECK_RETURN(mfcompiler_emitmulti(compiler, 2, match, NULL));
    }

    return true;
}

/** Check if any resolutions in a candidate set use typed parameters. */
static bool mfcompiler_hastypedresolutions(int nresolutions, mfcompileresolution *resolutions) {
    for (int i=0; i<nresolutions; i++) if (resolutions[i].typed) return true;
    return false;
}

/** Count number of possible types per parameter. */
static int mfcompiler_counttypesforparam(int nresolutions, mfcompileresolution *resolutions, int param) {
    bool wildcard=false;
    dictionary dict;
    dictionary_init(&dict);
    for (int j=0; j<nresolutions; j++) {
        value type=MORPHO_NIL;
        if (signature_getparamtype(resolutions[j].sig, param, &type) &&
            MORPHO_ISCLASS(type)) dictionary_insert(&dict, type, MORPHO_NIL);
        else wildcard=true;
    }
    int count = dict.count + ((dict.count>0 && wildcard) ? 1 : 0);
    dictionary_clear(&dict);
    return count;
}

/** One typed branch in a compiled dispatch plan. */
typedef struct {
    int uid;
    int fnindex;
} mfcompilertypecase;

DECLARE_VARRAY(mfcompilertypecase, mfcompilertypecase)
DEFINE_VARRAY(mfcompilertypecase, mfcompilertypecase)

/** A compiled plan for typed dispatch on one argument. */
typedef struct {
    int arg;
    int defaultfnindex;
    varray_mfcompilertypecase cases;
} mfcompilertypeplan;

/** Clear a typed dispatch plan. */
static void mfcompiler_cleartypeplan(mfcompilertypeplan *plan) {
    varray_mfcompilertypecaseclear(&plan->cases);
    plan->defaultfnindex = -1;
}

/** Choose a typed argument to branch on, if one exists. */
static bool mfcompiler_choosetypeparam(int nresolutions, mfcompileresolution *resolutions, int *param) {
    int bestparam = -1, bestcount = 0;

    for (int i=0; i<resolutions[0].nparams; i++) {
        int count = mfcompiler_counttypesforparam(nresolutions, resolutions, i);
        if (count>bestcount) { bestcount = count; bestparam = i; }
    }

    *param = bestparam;
    return (bestparam>=0);
}

/** Insert a class and all of its descendants into a dictionary. */
static bool mfcompiler_insertchildren(dictionary *dict, value type) {
    if (!MORPHO_ISCLASS(type) || dictionary_get(dict, type, NULL)) return true;
    
    ERR_CHECK_RETURN(dictionary_insert(dict, type, MORPHO_NIL));
    objectclass *klass = MORPHO_GETCLASS(type);
    for (int i=0; i<klass->children.count; i++) {
        ERR_CHECK_RETURN(mfcompiler_insertchildren(dict, klass->children.data[i]));
    }

    return true;
}

/** Choose the best resolution for a given runtime class at one parameter. */
static bool mfcompiler_resolvetypecase(int nresolutions, mfcompileresolution *resolutions, int param, value actual, int *fnindex) {
    int best = -1, bestrank = INT_MAX, bestcount = 0, rank;

    for (int i=0; i<nresolutions; i++) {
        value type = MORPHO_NIL;
        ERR_CHECK_RETURN(signature_getparamtype(resolutions[i].sig, param, &type));
        if (!mfresolution_rankparamtype(actual, type, &rank)) continue;

        if (rank<bestrank) {
            best = i;
            bestrank = rank;
            bestcount = 1;
        } else if (rank==bestrank) bestcount++;
    }

    if (best<0) return false;
    if (bestcount!=1) return false;
    *fnindex = resolutions[best].fnindex;
    return true;
}

/** Decide whether a typed branch still needs slow resolution. */
static bool mfcompiler_typeneedsslowcase(int nresolutions, mfcompileresolution *resolutions, int param, value actual) {
    int best = -1, bestrank = INT_MAX, bestcount = 0, rank;
    for (int i=0; i<nresolutions; i++) {
        value type = MORPHO_NIL;
        if (!signature_getparamtype(resolutions[i].sig, param, &type)) return true;
        if (!mfresolution_rankparamtype(actual, type, &rank)) continue;

        if (rank<bestrank) {
            best = i;
            bestrank = rank;
            bestcount = 1;
        } else if (rank==bestrank) bestcount++;
    }

    if (best<0 || bestcount!=1) return true;

    for (int i=0; i<resolutions[best].nparams; i++) {
        value type = MORPHO_NIL;
        if (i==param) continue;
        if (!signature_getparamtype(resolutions[best].sig, i, &type)) return true;
        if (!MORPHO_ISNIL(type)) return true;
    }

    return false;
}

/** Build a typed dispatch plan for one argument. */
static bool mfcompiler_buildtypeplan(int nresolutions, mfcompileresolution *resolutions, int param, mfcompilertypeplan *plan) {
    varray_mfcompilertypecaseinit(&plan->cases);
    plan->defaultfnindex = -1;
    
    dictionary children;
    dictionary_init(&children); // Maintain dictionary of resolution classes and their children

    for (int i=0; i<nresolutions; i++) { // Build list of all relevant types from each resolution
        value type = MORPHO_NIL;
        ERR_CHECK(signature_getparamtype(resolutions[i].sig, param, &type), mfcompiler_buildtypeplan_cleanup);

        if (MORPHO_ISNIL(type)) {
            ERR_CHECK(plan->defaultfnindex<0, mfcompiler_buildtypeplan_cleanup);
            plan->defaultfnindex = resolutions[i].fnindex;
            continue;
        } else ERR_CHECK(MORPHO_ISCLASS(type), mfcompiler_buildtypeplan_cleanup);
        ERR_CHECK(mfcompiler_insertchildren(&children, type), mfcompiler_buildtypeplan_cleanup);
    }

    for (unsigned int i=0; i<children.capacity; i++) { // Now build a resolution for each type
        value type = children.contents[i].key;
        if (MORPHO_ISNIL(type)) continue;
        
        int fnindex;
        ERR_CHECK(mfcompiler_resolvetypecase(nresolutions, resolutions, param, type, &fnindex), mfcompiler_buildtypeplan_cleanup);
        if (mfcompiler_typeneedsslowcase(nresolutions, resolutions, param, type)) fnindex = -1;

        if (fnindex==plan->defaultfnindex) continue;

        mfcompilertypecase newcase = { .uid = MORPHO_GETCLASS(type)->uid, .fnindex = fnindex };
        ERR_CHECK(varray_mfcompilertypecaseadd(&plan->cases, &newcase, 1), mfcompiler_buildtypeplan_cleanup);
    }

    dictionary_clear(&children);
    return (plan->cases.count>0);

mfcompiler_buildtypeplan_cleanup:
    dictionary_clear(&children);
    mfcompiler_cleartypeplan(plan);
    return false;
}

/** Emit bytecode for a typed dispatch plan. */
static bool mfcompiler_emitbranchtarget(mfcompiler *compiler, int fnindex, mfindx *entry) {
    if (fnindex>=0) return mfcompiler_emitresolve(compiler, fnindex, entry);
    return mfcompiler_emitslow(compiler, entry);
}

/** Emit bytecode for a typed dispatch plan. */
static bool mfcompiler_emittypeplan(mfcompiler *compiler, mfcompilertypeplan *plan, mfindx *entry) {
    mfindx deflt;
    // If a default resolution exists, generate a resolution
    ERR_CHECK_RETURN(mfcompiler_emitbranchtarget(compiler, plan->defaultfnindex, &deflt));

    mfcompilersparseentry table[plan->cases.count]; // Generate resolutions compile table
    for (int i=0; i<plan->cases.count; i++) {
        table[i].value = plan->cases.data[i].uid;
        ERR_CHECK_RETURN(mfcompiler_emitbranchtarget(compiler, plan->cases.data[i].fnindex, &table[i].target));
    }

    // Output bytecode for the branch
    ERR_CHECK_RETURN(mfcompiler_emitgetuid(compiler, plan->arg, entry));
    return mfcompiler_emitsparse(compiler, plan->cases.count, table, deflt, NULL);
}

/** Try to emit typed dispatch for one exact-arity candidate set. */
static bool mfcompiler_emittypedcase(mfcompiler *compiler, int nresolutions, mfcompileresolution *resolutions, mfindx *entry) {
    int param;
    mfcompilertypeplan plan = { .arg=-1, .defaultfnindex=-1 };
    varray_mfcompilertypecaseinit(&plan.cases);

    ERR_CHECK(mfcompiler_choosetypeparam(nresolutions, resolutions, &param), mfcompiler_emittypedcase_cleanup);
    ERR_CHECK(mfcompiler_buildtypeplan(nresolutions, resolutions, param, &plan), mfcompiler_emittypedcase_cleanup);

    plan.arg = param;
    ERR_CHECK(mfcompiler_emittypeplan(compiler, &plan, entry), mfcompiler_emittypedcase_cleanup);
    mfcompiler_cleartypeplan(&plan);
    return true;

mfcompiler_emittypedcase_cleanup:
    mfcompiler_cleartypeplan(&plan);
    return false;
}

/** Compare resolutions by arity for sorting. */
static int mfcompiler_compareresolutionarity(const void *a, const void *b) {
    int xi=((mfcompileresolution *) a)->nparams, yi=((mfcompileresolution *) b)->nparams;
    return (xi > yi) - (xi < yi); // Ascending order
}

/** Emit an untyped exact-arity case if a unique fixed-arity winner exists. */
static bool mfcompiler_emitfixedaritywinner(mfcompiler *compiler, int nresolutions, mfcompileresolution *resolutions, mfindx *entry) {
    int winner = -1;

    for (int i=0; i<nresolutions; i++) {
        if (resolutions[i].varg) continue;
        if (winner>=0) return false;
        winner = i;
    }

    if (winner>=0) return mfcompiler_emitresolve(compiler, resolutions[winner].fnindex, entry);
    if (nresolutions==1) return mfcompiler_emitresolve(compiler, resolutions[0].fnindex, entry);
    return false;
}

/** Emit a resolver block for one exact-arity candidate set. */
static bool mfcompiler_emitaritycase(mfcompiler *compiler, int nresolutions, mfcompileresolution *resolutions, mfindx *entry) {
    bool hastyped = mfcompiler_hastypedresolutions(nresolutions, resolutions);

    if (nresolutions==1 && !resolutions[0].typed) {
        return mfcompiler_emitresolve(compiler, resolutions[0].fnindex, entry);
    }

    if (hastyped) {
        if (mfcompiler_emittypedcase(compiler, nresolutions, resolutions, entry)) return true;
    } else {
        if (mfcompiler_emitfixedaritywinner(compiler, nresolutions, resolutions, entry)) return true;
    }

    return mfcompiler_emitslow(compiler, entry);
}

/** Emit a conservative exact-arity resolver directly into bytecode. */
static bool mfcompiler_emitarityresolver(mfcompiler *compiler, mfindx *entry) {
    // Sort resolutions by arity
    mfcompileresolution resolutions[compiler->nresolutions];
    memcpy(resolutions, compiler->resolutions, sizeof(mfcompileresolution)*compiler->nresolutions);
    qsort(resolutions, compiler->nresolutions, sizeof(mfcompileresolution), mfcompiler_compareresolutionarity);

    mfindx deflt; // Emit default resolution
    if (compiler->hasvarg) ERR_CHECK_RETURN(mfcompiler_emitslow(compiler, &deflt));
    else ERR_CHECK_RETURN(mfcompiler_emitfail(compiler, &deflt));

    mfcompilersparseentry table[compiler->nresolutions];
    int ncases = 0;
    for (int i=0; i<compiler->nresolutions; ) { // Loop over resolutions
        if (resolutions[i].varg) {
            i++;
            continue;
        }

        int arity = resolutions[i].nparams;
        while (i<compiler->nresolutions &&
               !resolutions[i].varg &&
               resolutions[i].nparams==arity) i++;

        mfcompileresolution bucket[compiler->nresolutions];
        int count = 0;
        for (int j=0; j<compiler->nresolutions; j++) {
            if (resolutions[j].minarity<=arity &&
                (resolutions[j].maxarity<0 || arity<=resolutions[j].maxarity)) {
                bucket[count++] = resolutions[j];
            }
        }

        table[ncases].value = arity; // Generate code for this arity
        ERR_CHECK_RETURN(mfcompiler_emitaritycase(compiler, count, bucket, &table[ncases].target));
        ncases++;
    }

    return mfcompiler_emitsparse(compiler, ncases, table, deflt, entry); // Emit the sparse table
}

/** Compiles the resolver for a metafunction that is still being assembled. */
bool metafunction_compile(objectmetafunction *fn, error *err) {
    if (fn->fns.count<=0) return false;

    mfcompiler compiler;
    mfcompiler_init(&compiler, fn, err);
    metafunction_clearinstructions(fn);

    ERR_CHECK(mfcompiler_analyze(&compiler), metafunction_compile_cleanup);
    ERR_CHECK(mfcompiler_checkduplicates(&compiler), metafunction_compile_cleanup);
    ERR_CHECK(mfcompiler_emitarityresolver(&compiler, &fn->entry), metafunction_compile_cleanup);
    mfcompiler_clear(&compiler);
    return true;

metafunction_compile_cleanup:
    mfcompiler_clear(&compiler);
    metafunction_clearinstructions(fn);
    return false;
}

/* --------------------------
 * Disassembler
 * -------------------------- */

/** Disassemble a RESOLVE instruction. */
static mfindx metafunction_disassembleresolve(objectmetafunction *fn, mfindx pc) {
    if (pc+1>=fn->resolver.count) { printf("resolve <missing operand>"); return pc+1; }
    
    int index=fn->resolver.data[pc+1];
    printf("resolve %i ", index);
    if (index<0 || index>=fn->fns.count) printf("<invalid index>");
    else {
        signature *sig = metafunction_getsignature(fn->fns.data[index]);
        if (sig) signature_print(sig);
    }
    return pc+2;
}

/** Disassemble a SPARSE instruction. */
static mfindx metafunction_disassemblesparse(objectmetafunction *fn, mfindx pc) {
    if (pc+2>=fn->resolver.count) { printf("sparse <missing operands>"); return fn->resolver.count; }
    
    int ncases=fn->resolver.data[pc+1];
    mfindx defaultpc=fn->resolver.data[pc+2];
    printf("sparse default -> %i", defaultpc);
    
    mfindx next=pc+3;
    for (int k=0; k<ncases; k++) {
        if (next+1>=fn->resolver.count) { printf(" <missing case>"); return fn->resolver.count; }
        printf(" %i -> %i", fn->resolver.data[next], fn->resolver.data[next+1]);
        next+=2;
    }
    return next;
}

/** Disassemble a GETUID instruction. */
static mfindx metafunction_disassemblegetuid(objectmetafunction *fn, mfindx pc) {
    if (pc+1>=fn->resolver.count) { printf("getuid <missing operand>"); return pc+1; }
    printf("getuid %i", fn->resolver.data[pc+1]);
    return pc+2;
}

/** Print a disassembly of the metafunction resolver bytecode. */
void metafunction_disassemble(objectmetafunction *fn) {
    printf("Resolver for ");
    morpho_printvalue(NULL, MORPHO_OBJECT(fn));
    printf(":\n");
    
    for (mfindx pc=0; pc<fn->resolver.count; ) {
        printf("%s%3i : ", (pc==fn->entry ? "->" : "  "), pc);
        switch (fn->resolver.data[pc]) {
            case MFOP_SLOW: printf("slow"); pc++; break;
            case MFOP_FAIL: printf("fail"); pc++; break;
            case MFOP_RESOLVE: pc=metafunction_disassembleresolve(fn, pc); break;
            case MFOP_GETUID: pc=metafunction_disassemblegetuid(fn, pc); break;
            case MFOP_SPARSE: pc=metafunction_disassemblesparse(fn, pc); break;
            default:
                printf("unknown %i", fn->resolver.data[pc]);
                pc++;
                break;
        }
        printf("\n");
    }
}


/* --------------------------
 * Fast resolver VM
 * -------------------------- */

/** Execute the new resolver VM. */
static bool metafunction_runresolver(objectmetafunction *fn, int nargs, value *args, error *err, value *out) {
    mfinstruction *instructions = fn->resolver.data;
    if (!instructions) return metafunction_resolveslow(fn, nargs, args, err, out);
    
    mfindx pc = fn->entry;
    int reg = nargs; // Single register initialized with nargs
    
    while (true) {
        switch (instructions[pc]) {
            case MFOP_SLOW:
                return metafunction_resolveslow(fn, nargs, args, err, out);
            case MFOP_RESOLVE: {
                pc++; *out=fn->fns.data[instructions[pc]];
                return true;
            }
            case MFOP_FAIL:
                error_writewithid(err, VM_MLTPLDSPTCHFLD); return false;
            case MFOP_GETUID: {
                int arg = instructions[++pc];
                value type;
                if (value_type(args[arg], &type) && MORPHO_ISCLASS(type)) {
                    reg = MORPHO_GETCLASS(type)->uid;
                    break;
                }
                error_writewithid(err, VM_MLTPLDSPTCHFLD); return false;
            }
            case MFOP_SPARSE: {
                int ncases=instructions[pc+1];
                mfindx cases=pc+3;
                pc=instructions[pc+2]; // Default branch
                for (int i=0; i<2*ncases; i+=2) {
                    if (instructions[cases+i]==reg) { // match
                        pc=instructions[cases+i+1];
                        break;
                    }
                }
                continue;
            }
        }
        pc++;
    }
}

/** Resolve a metafunction using the compiled resolver VM. */
bool metafunction_resolve(objectmetafunction *fn, int nargs, value *args, error *err, value *out) {
    if (fn->state!=METAFUNCTION_FROZEN) {
        error_writewithid(err, METAFUNCTION_UNFROZEN); return false;
    }
    
    return metafunction_runresolver(fn, nargs, args, err, out);
}

/* **********************************************************************
 * Metafunction veneer class
 * ********************************************************************** */

/** Constructor function for Metafunctions */
value metafunction_constructor(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    
    if (nargs==0) return MORPHO_NIL;
    
    value name = metafunction_getname(MORPHO_GETARG(args, 0));
    if (!MORPHO_ISSTRING(name)) return MORPHO_NIL;
    
    objectmetafunction *new = object_newmetafunction(name);
    
    if (new) {
        for (int i=0; i<nargs; i++) {
            metafunction_add(new, MORPHO_GETARG(args, i));
        }
        
        error err;
        error_init(&err);
        if (!metafunction_finalize(new, &err)) morpho_runtimeerror(v, err.id);
        error_clear(&err);
        
        out=morpho_wrapandbind(v, (object *) new);
    }
    
    return out;
}

/** Count the number of implementations in a metafunction */
value Metafunction_count(vm *v, int nargs, value *args) {
    objectmetafunction *fn = MORPHO_GETMETAFUNCTION(MORPHO_SELF(args));
    
    return MORPHO_INTEGER(fn->fns.count);
}

value Metafunction_tostring(vm *v, int nargs, value *args) {
    objectmetafunction *func=MORPHO_GETMETAFUNCTION(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    varray_char buffer;
    varray_charinit(&buffer);

    varray_charadd(&buffer, "<fn ", 4);
    morpho_printtobuffer(v, func->name, &buffer);
    varray_charwrite(&buffer, '>');

    out = object_stringfromvarraychar(&buffer);
    if (MORPHO_ISSTRING(out)) {
        morpho_bindobjects(v, 1, &out);
    }
    varray_charclear(&buffer);

    return out;
}

MORPHO_BEGINCLASS(Metafunction)
MORPHO_METHOD(MORPHO_TOSTRING_METHOD, Metafunction_tostring, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Metafunction_count, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

objecttype objectmetafunctiontype;

void metafunction_initialize(void) {
    // Create function object type
    objectmetafunctiontype=object_addtype(&objectmetafunctiondefn);
    
    // Locate the Object class to use as the parent class of Metafunction
    value objclass = builtin_findclassfromcstring(CALLABLE_CLASSNAME);
    
    // Metafunction constructor function
    morpho_addfunction(METAFUNCTION_CLASSNAME, METAFUNCTION_CLASSNAME " (...)", metafunction_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
    
    // Create function veneer class
    value metafunctionclass=builtin_addclass(METAFUNCTION_CLASSNAME, MORPHO_GETCLASSDEFINITION(Metafunction), objclass);
    object_setveneerclass(OBJECT_METAFUNCTION, metafunctionclass);
    
    // Metafunction error messages
    morpho_defineerror(METAFUNCTION_CMPLAMBGS, ERROR_PARSE, METAFUNCTION_CMPLAMBGS_MSG);
    morpho_defineerror(METAFUNCTION_UNFROZEN, ERROR_HALT, METAFUNCTION_UNFROZEN_MSG);
}
