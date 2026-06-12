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

/** Merges an incoming callable into an existing entry.
 * @param[in] name      name of the callable
 * @param[in] existing  the existing dictionary entry
 * @param[in] incoming  the callable to merge into the entry
 * @param[in] klass     owner class; if non-NULL, an existing metafunction must either be unowned or already belong to this class
 * @param[out] out      the merged entry
 * @returns true on success */
bool metafunction_merge(value name, value existing, value incoming, objectclass *klass, value *out) {
    value entry=existing;

    if (MORPHO_ISNIL(entry)) {
        *out=incoming;
        return true;
    }

    if (!MORPHO_ISMETAFUNCTION(entry)) {
        if (!metafunction_wrap(name, entry, &entry)) return false;
    }

    objectmetafunction *dest = MORPHO_GETMETAFUNCTION(entry);
    if (klass) {
        if (dest->klass && dest->klass!=klass) return false;
        dest->klass=klass;
    }

    if (MORPHO_ISMETAFUNCTION(incoming)) {
        objectmetafunction *src = MORPHO_GETMETAFUNCTION(incoming);
        for (int i=0; i<src->fns.count; i++) {
            if (!metafunction_add(dest, src->fns.data[i])) return false;
        }
    } else {
        if (!metafunction_add(dest, incoming)) return false;
    }

    *out=entry;
    return true;
}

/** Adds a function to a metafunction */
bool metafunction_add(objectmetafunction *f, value fn) {
    if (f->state==METAFUNCTION_FROZEN) {
        metafunction_clearinstructions(f);
        f->state=METAFUNCTION_BUILDING;
    }
    return varray_valueadd(&f->fns, &fn, 1);
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

/** Check whether known argument types fully determine a signature. */
static bool mfresolution_isterminal(signature *sig, int nargs, value *args);

/** Compare the specificity of two resolutions.
 *  Returns <0 if a is more specific, >0 if b is more specific, 0 if they are equal
 *  or incomparable. */
static int mfresolution_comparespecificity(mfresolution *a, mfresolution *b, int nargs, value *args) {
    if (!a->sig || !b->sig) return 0;
    if (!mfresolution_isterminal(a->sig, nargs, args) ||
        !mfresolution_isterminal(b->sig, nargs, args)) return 0;

    int ncheck=_min(signature_countparams(a->sig), signature_countparams(b->sig), nargs);
    int cmp[ncheck];
    for (int i=0; i<ncheck; i++) { // Compare each arg
        cmp[i]=0;
        value actual = args[i];
        if (MORPHO_ISCLASS(actual) || value_type(args[i], &actual)) {
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

/** @brief Filter the resolution set by known argument types. */
void mfresolutionset_filterbyknowntypes(mfresolutionset *set, int nargs, value *args) {
    for (int i=0; i<set->count; i++) {
        if (!set->data[i].sig) continue;
        for (int j=0; j<nargs; j++) {
            value actual = args[j], type = MORPHO_NIL;
            if (MORPHO_ISNIL(actual)) continue;
            if (!signature_getparamtype(set->data[i].sig, j, &type) ||
                !mfresolution_rankparamtype(actual, type, &(int) { 0 })) {
                set->data[i].sig = NULL;
                break;
            }
        }
    }
    mfresolutionset_collapse(set);
}

/** @brief Check whether known argument types fully determine a signature. */
static bool mfresolution_isterminal(signature *sig, int nargs, value *args) {
    int nparams = signature_countparams(sig);

    for (int i=0; i<nargs && i<nparams; i++) {
        value type = MORPHO_NIL;
        if (!signature_getparamtype(sig, i, &type)) return false;
        if (!MORPHO_ISNIL(type) && MORPHO_ISNIL(args[i])) return false;
    }

    return true;
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
} mfcompiler;

/** Path-local facts accumulated while compiling one resolver branch. */
typedef struct {
    int knownarity;
    bool aritychecked;
    int nparams;
    value *known;
} mfcompilerpath;

/** Initialize compiler state. */
static void mfcompiler_init(mfcompiler *compiler, objectmetafunction *fn, error *err) {
    compiler->fn = fn;
    compiler->err = err;
    compiler->nresolutions = fn->fns.count;
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

/** One emitted child subtree that can be reused by later sparse cases. */
typedef struct {
    int nsubset;
    value *known;
    mfcompileresolution *subset;
    mfindx target;
} mfcompileremittedcase;

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

/** Find the maximum declared parameter count in a candidate set. */
static int mfcompiler_maxparams(int nresolutions, mfcompileresolution *resolutions) {
    int max = 0;
    for (int i=0; i<nresolutions; i++) {
        if (resolutions[i].nparams>max) max = resolutions[i].nparams;
    }
    return max;
}

/** Check if a resolution matches a known exact arity. */
static bool mfcompiler_matcharity(mfcompileresolution *resolution, int arity) {
    return (resolution->minarity<=arity &&
            (resolution->maxarity<0 || arity<=resolution->maxarity));
}

/** Check if a resolution matches the known runtime classes on this path. */
static bool mfcompiler_matchknown(mfcompileresolution *resolution, int nparams, value *known) {
    for (int i=0; i<nparams && i<resolution->nparams; i++) {
        value type = MORPHO_NIL;
        if (MORPHO_ISNIL(known[i])) continue;
        if (!signature_getparamtype(resolution->sig, i, &type)) return false;
        if (!mfresolution_rankparamtype(known[i], type, &(int) { 0 })) return false;
    }
    return true;
}

/** Count the number of distinct arities in a candidate set. */
static int mfcompiler_countarities(int nresolutions, mfcompileresolution *resolutions) {
    dictionary seen;
    dictionary_init(&seen);

    for (int i=0; i<nresolutions; i++) {
        if (!resolutions[i].varg) dictionary_insert(&seen, MORPHO_INTEGER(resolutions[i].nparams), MORPHO_NIL);
    }

    int narities = seen.count;
    dictionary_clear(&seen);
    return narities;
}

/** Check if any resolutions in a candidate set are variadic. */
static bool mfcompiler_hasvargresolutions(int nresolutions, mfcompileresolution *resolutions) {
    for (int i=0; i<nresolutions; i++) if (resolutions[i].varg) return true;
    return false;
}

/** Check if a parameter's runtime class is already known. */
static inline bool mfcompiler_paramisknown(value *known, int param) {
    return (known && !MORPHO_ISNIL(known[param]));
}

/** Count possible runtime types for a parameter. */
static int mfcompiler_paramtypecount(int nresolutions, mfcompileresolution *resolutions, int param, value *known) {
    if (mfcompiler_paramisknown(known, param)) return 0;

    bool wildcard = false;
    dictionary dict;
    dictionary_init(&dict);
    for (int j=0; j<nresolutions; j++) {
        value type = MORPHO_NIL;
        if (signature_getparamtype(resolutions[j].sig, param, &type) &&
            MORPHO_ISCLASS(type)) dictionary_insert(&dict, type, MORPHO_NIL);
        else wildcard = true;
    }

    int count = dict.count + ((dict.count>0 && wildcard) ? 1 : 0);
    dictionary_clear(&dict);
    return count;
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

/** Collect all runtime classes reachable from a parameter's declared types. */
static bool mfcompiler_paramchildren(int nresolutions, mfcompileresolution *resolutions,
                                     int param, dictionary *children) {
    for (int i=0; i<nresolutions; i++) {
        value type = MORPHO_NIL;
        if (!signature_getparamtype(resolutions[i].sig, param, &type)) return false;
        if (MORPHO_ISCLASS(type) && !mfcompiler_insertchildren(children, type)) return false;
    }

    return true;
}

/** Filter resolutions by known arity and runtime classes. */
static int mfcompiler_collectsubset(int nresolutions, mfcompileresolution *resolutions, int knownarity,
                                    int nparams, value *known, mfcompileresolution *out) {
    int count = 0;

    for (int i=0; i<nresolutions; i++) {
        if (knownarity>=0 && !mfcompiler_matcharity(&resolutions[i], knownarity)) continue;
        if (!mfcompiler_matchknown(&resolutions[i], nparams, known)) continue;
        out[count++] = resolutions[i];
    }

    return count;
}

/** Check if a resolution is fully determined on this path. */
static bool mfcompiler_resolutionisterminal(mfcompileresolution *resolution, int nparams, value *known) {
    for (int i=0; i<nparams && i<resolution->nparams; i++) {
        value type = MORPHO_NIL;
        if (!signature_getparamtype(resolution->sig, i, &type)) return false;
        if (!MORPHO_ISNIL(type) && MORPHO_ISNIL(known[i])) return false;
    }

    return true;
}

/** Check for any remaining unchecked typed parameter. */
static bool mfcompiler_haspendingtypes(int nresolutions, mfcompileresolution *resolutions,
                                       int nparams, value *known) {
    for (int i=0; i<nparams; i++) {
        if (mfcompiler_paramisknown(known, i)) continue;
        if (mfcompiler_paramtypecount(nresolutions, resolutions, i, known)>0) return true;
    }

    return false;
}

/** Count distinct declared non-wildcard types for one parameter in a subset. */
static int mfcompiler_countparamtypes(int nresolutions, mfcompileresolution *resolutions, int param) {
    value seen[nresolutions];
    int count = 0;

    for (int i=0; i<nresolutions; i++) {
        value type = MORPHO_NIL;
        if (!signature_getparamtype(resolutions[i].sig, param, &type)) return 0;
        if (MORPHO_ISNIL(type)) continue;

        int j = 0;
        while (j<count && !MORPHO_ISEQUAL(seen[j], type)) j++;
        if (j==count) seen[count++] = type;
    }

    return count;
}

/** Remove known runtime classes that no longer affect dispatch within a subset. */
static void mfcompiler_pruneknown(int nresolutions, mfcompileresolution *resolutions, int nparams, value *known) {
    for (int i=0; i<nparams; i++) {
        if (!mfcompiler_paramisknown(known, i)) continue;
        if (mfcompiler_countparamtypes(nresolutions, resolutions, i)<=1) known[i] = MORPHO_NIL;
    }
}

/** Compare two candidate subsets for structural equality. */
static bool mfcompiler_samesubset(int nresolutions, mfcompileresolution *a, mfcompileresolution *b) {
    for (int i=0; i<nresolutions; i++) {
        if (a[i].fnindex!=b[i].fnindex ||
            a[i].sig!=b[i].sig ||
            a[i].typed!=b[i].typed ||
            a[i].varg!=b[i].varg ||
            a[i].nparams!=b[i].nparams ||
            a[i].minarity!=b[i].minarity ||
            a[i].maxarity!=b[i].maxarity) return false;
    }

    return true;
}

/** Compare two known-type path states. */
static bool mfcompiler_sameknown(int nparams, value *a, value *b) {
    for (int i=0; i<nparams; i++) {
        if (!MORPHO_ISEQUAL(a[i], b[i])) return false;
    }

    return true;
}

/**
 * Build the surviving subset for one typed child branch.
 *
 * `childknown` carries the actual runtime class learned on this branch and
 * must be preserved for recursive compilation. `canonknown`, when requested,
 * prunes any facts that no longer distinguish the surviving subset so
 * structurally identical residual subproblems can reuse one emitted subtree.
 */
static int mfcompiler_collectchildsubset(int nresolutions, mfcompileresolution *resolutions,
                                         int knownarity, int nparams, value *sourceknown, value *childknown,
                                         int param, value actual, mfcompileresolution *subset, value *canonknown) {
    memcpy(childknown, sourceknown, sizeof(value)*nparams);
    childknown[param] = actual;

    int nsubset = mfcompiler_collectsubset(nresolutions, resolutions, knownarity, nparams, childknown, subset);
    if (nsubset<=0) return nsubset;

    if (canonknown) {
        memcpy(canonknown, childknown, sizeof(value)*nparams);
        mfcompiler_pruneknown(nsubset, subset, nparams, canonknown);
    }

    return nsubset;
}

/** Find a previously emitted child subtree that matches this residual state. */
static int mfcompiler_findemittedcase(int nemitted, mfcompileremittedcase *emitted, int nsubset, mfcompileresolution *subset,
                                      int nparams, value *known) {
    for (int i=0; i<nemitted; i++) {
        if (emitted[i].nsubset==nsubset &&
            mfcompiler_samesubset(nsubset, subset, emitted[i].subset) &&
            mfcompiler_sameknown(nparams, known, emitted[i].known)) return i;
    }

    return -1;
}

/** Compare two resolutions using the runtime classes already known on this path. */
static int mfcompiler_compareknownspecificity(mfcompileresolution *a, mfcompileresolution *b, int nparams, value *known) {
    int ncheck = _min(a->nparams, b->nparams, nparams);
    int firstsign = 0;

    for (int i=0; i<ncheck; i++) {
        int cmp = 0;
        value atype = MORPHO_NIL, btype = MORPHO_NIL;
        if (MORPHO_ISNIL(known[i])) continue;
        if (!signature_getparamtype(a->sig, i, &atype) ||
            !signature_getparamtype(b->sig, i, &btype)) return 0;
        if (!mfresolution_compareparamtypes(known[i], atype, btype, &cmp)) return 0;
        if (cmp!=0) {
            if (firstsign==0) firstsign = _sign(cmp);
            else if (firstsign!=_sign(cmp)) return 0;
        }
    }

    if (firstsign!=0) return firstsign;
    if (a->varg==b->varg) return 0;
    return (a->varg ? 1 : -1);
}

/** Resolve a subset directly if known path facts determine all typed parameters. */
static bool mfcompiler_resolveknownsubset(int nresolutions, mfcompileresolution *resolutions,
                                          int nparams, value *known, int *fnindex) {
    for (int i=0; i<nresolutions; i++) {
        if (!mfcompiler_resolutionisterminal(&resolutions[i], nparams, known)) return false;
    }

    bool alive[nresolutions];
    for (int i=0; i<nresolutions; i++) alive[i] = true;

    for (int i=0; i<nresolutions; i++) {
        if (alive[i]) for (int j=i+1; j<nresolutions; j++) {
            if (alive[j]) {
                int cmp = mfcompiler_compareknownspecificity(&resolutions[i], &resolutions[j], nparams, known);
                if (cmp<0) alive[j] = false;
                else if (cmp>0) {
                    alive[i] = false;
                    break;
                }
            }
        }
    }

    int winner = -1;
    for (int i=0; i<nresolutions; i++) {
        if (alive[i]) {
            if (winner>=0) return false;
            winner = i;
        }
    }

    if (winner<0) return false;
    *fnindex = resolutions[winner].fnindex;
    return true;
}

/** Check if a surviving subset is worth recursing into. */
static bool mfcompiler_subsetisuseful(int nresolutions, mfcompileresolution *resolutions,
                                      int nparams, value *known) {
    int winner;

    if (nresolutions<=0) return false;
    if (nresolutions==1 && mfcompiler_resolutionisterminal(&resolutions[0], nparams, known)) return true;
    if (mfcompiler_resolveknownsubset(nresolutions, resolutions, nparams, known, &winner)) return true;
    return mfcompiler_haspendingtypes(nresolutions, resolutions, nparams, known);
}

/** Count useful runtime-class branches for a parameter. */
static int mfcompiler_parambranchcount(int nresolutions, mfcompileresolution *resolutions,
                                       int knownarity, int nparams, value *known, int param) {
    int useful = 0;
    if (mfcompiler_paramisknown(known, param)) return 0;

    value childknown[nparams];
    mfcompileresolution subset[nresolutions];
    dictionary children;
    dictionary_init(&children);
    ERR_CHECK(mfcompiler_paramchildren(nresolutions, resolutions, param, &children), mfcompiler_parambranchcount_cleanup);

    for (unsigned int i=0; i<children.capacity; i++) {
        value actual = children.contents[i].key;
        if (MORPHO_ISNIL(actual)) continue;

        int count = mfcompiler_collectchildsubset(nresolutions, resolutions, knownarity, nparams,
                                                  known, childknown, param, actual, subset, NULL);
        if (count>0 && (count<nresolutions || mfcompiler_subsetisuseful(count, subset, nparams, childknown))) useful++;
    }

mfcompiler_parambranchcount_cleanup:
    dictionary_clear(&children);
    return useful;
}

/** Choose a typed parameter to branch on. */
static bool mfcompiler_choosetypeparam(int nresolutions, mfcompileresolution *resolutions,
                                       int knownarity, int nparams, value *known, int *param) {
    int bestparam = -1, bestuseful = 0, bestcount = 0;

    for (int i=0; i<resolutions[0].nparams; i++) {
        int count = mfcompiler_paramtypecount(nresolutions, resolutions, i, known);
        int useful = mfcompiler_parambranchcount(nresolutions, resolutions, knownarity, nparams, known, i);
        if (useful>bestuseful || (useful==bestuseful && count>bestcount)) {
            bestuseful = useful;
            bestcount = count;
            bestparam = i;
        }
    }

    *param = bestparam;
    return (bestparam>=0 && bestuseful>0);
}

static bool mfcompiler_emitresolver(mfcompiler *compiler, int nresolutions, mfcompileresolution *resolutions, mfcompilerpath *path, mfindx *entry);

/** Compile the resolver entry point. */
static bool mfcompiler_compileentry(mfcompiler *compiler, mfindx *entry) {
    int nparams = mfcompiler_maxparams(compiler->nresolutions, compiler->resolutions);
    if (nparams<1) nparams = 1;

    value known[nparams];
    for (int i=0; i<nparams; i++) known[i] = MORPHO_NIL;

    mfcompilerpath path = { .knownarity = -1, .aritychecked = false, .nparams = nparams, .known = known };
    return mfcompiler_emitresolver(compiler, compiler->nresolutions, compiler->resolutions, &path, entry);
}

/** Compare resolutions by arity for sorting. */
static int mfcompiler_compareresolutionarity(const void *a, const void *b) {
    int xi=((mfcompileresolution *) a)->nparams, yi=((mfcompileresolution *) b)->nparams;
    return (xi > yi) - (xi < yi); // Ascending order
}

/** Emit an exact-arity case if a unique winner is already fully determined. */
static bool mfcompiler_emitfixedaritywinner(mfcompiler *compiler, int nresolutions, mfcompileresolution *resolutions, int nparams, value *known, mfindx *entry) {
    int winner = -1;

    for (int i=0; i<nresolutions; i++) {
        if (resolutions[i].varg) continue;
        if (winner>=0) return false;
        winner = i;
    }

    if (winner<0) {
        if (nresolutions!=1) return false;
        winner = 0;
    }
    if (!mfcompiler_resolutionisterminal(&resolutions[winner], nparams, known)) return false;
    return mfcompiler_emitresolve(compiler, resolutions[winner].fnindex, entry);
}

/** Collect wildcard resolutions for a typed split. */
static bool mfcompiler_defaultsubset(int nresolutions, mfcompileresolution *resolutions, int param, mfcompileresolution *subset, int *count) {
    int ndefault = 0;

    for (int i=0; i<nresolutions; i++) {
        value type = MORPHO_NIL;
        ERR_CHECK_RETURN(signature_getparamtype(resolutions[i].sig, param, &type));
        if (MORPHO_ISNIL(type)) subset[ndefault++] = resolutions[i];
    }

    *count = ndefault;
    return true;
}

/** Emit typed child branches for one split parameter. */
static bool mfcompiler_emitchildcases(mfcompiler *compiler, int nresolutions, mfcompileresolution *resolutions, mfcompilerpath *path, int param, dictionary *children, mfcompilersparseentry *table, int *ncases) {
    value childknown[children->count][path->nparams];
    value canonknown[children->count][path->nparams];
    mfcompileresolution subset[nresolutions];
    mfcompileremittedcase emitted[children->count];
    mfcompileresolution emittedsubset[children->count][nresolutions];
    int count = 0, nemitted = 0;

    for (unsigned int i=0; i<children->capacity; i++) {
        value actual = children->contents[i].key;
        if (MORPHO_ISNIL(actual)) continue;

        value *known = childknown[count];
        value *canon = canonknown[count];
        mfcompilerpath childpath = *path;
        childpath.known = known;

        int nsubset = mfcompiler_collectchildsubset(nresolutions, resolutions, path->knownarity, path->nparams,
                                                    path->known, known, param, actual, subset, canon);
        if (nsubset<=0) continue;

        table[count].value = MORPHO_GETCLASS(actual)->uid;

        int match = mfcompiler_findemittedcase(nemitted, emitted, nsubset, subset, path->nparams, canon);
        if (match>=0) {
            table[count].target = emitted[match].target;
        } else {
            ERR_CHECK_RETURN(mfcompiler_emitresolver(compiler, nsubset, subset, &childpath, &table[count].target));
            memcpy(emittedsubset[nemitted], subset, sizeof(mfcompileresolution)*nsubset);
            emitted[nemitted].nsubset = nsubset;
            emitted[nemitted].known = canon;
            emitted[nemitted].subset = emittedsubset[nemitted];
            emitted[nemitted].target = table[count].target;
            nemitted++;
        }

        count++;
    }

    *ncases = count;
    return true;
}

/** Emit a typed resolver for one exact-arity candidate set. */
static bool mfcompiler_emittypedresolver(mfcompiler *compiler, int nresolutions, mfcompileresolution *resolutions, mfcompilerpath *path, mfindx *entry) {
    int param, defaultcount, ncases;
    dictionary children;
    mfcompileresolution defaultsubset[nresolutions];
    mfindx deflt;

    dictionary_init(&children);

    ERR_CHECK(mfcompiler_choosetypeparam(nresolutions, resolutions, path->knownarity, path->nparams, path->known, &param), mfcompiler_emittypedresolver_cleanup);
    ERR_CHECK(mfcompiler_paramchildren(nresolutions, resolutions, param, &children), mfcompiler_emittypedresolver_cleanup);

    {
        /* Branch count is bounded by reachable runtime classes, not declared resolutions. */
        mfcompilersparseentry table[children.count];

        ERR_CHECK(mfcompiler_defaultsubset(nresolutions, resolutions, param, defaultsubset, &defaultcount), mfcompiler_emittypedresolver_cleanup);
        if (defaultcount>0) {
            ERR_CHECK(mfcompiler_emitresolver(compiler, defaultcount, defaultsubset, path, &deflt), mfcompiler_emittypedresolver_cleanup);
        } else ERR_CHECK(mfcompiler_emitslow(compiler, &deflt), mfcompiler_emittypedresolver_cleanup);

        ERR_CHECK(mfcompiler_emitchildcases(compiler, nresolutions, resolutions, path,
                                            param, &children, table, &ncases), mfcompiler_emittypedresolver_cleanup);
        if (ncases<=0) goto mfcompiler_emittypedresolver_cleanup;
        ERR_CHECK(mfcompiler_emitgetuid(compiler, param, entry), mfcompiler_emittypedresolver_cleanup);
        ERR_CHECK(mfcompiler_emitsparse(compiler, ncases, table, deflt, NULL), mfcompiler_emittypedresolver_cleanup);
    }

    dictionary_clear(&children);
    return true;

mfcompiler_emittypedresolver_cleanup:
    dictionary_clear(&children);
    return false;
}

/** Emit an arity-first resolver for a candidate set. */
static bool mfcompiler_emitarityresolver(mfcompiler *compiler, int nresolutions, mfcompileresolution *resolutions, mfcompilerpath *path, mfindx *entry) {
    mfcompileresolution sorted[nresolutions];
    mfcompileresolution vargsubset[nresolutions];
    memcpy(sorted, resolutions, sizeof(mfcompileresolution)*nresolutions);
    qsort(sorted, nresolutions, sizeof(mfcompileresolution), mfcompiler_compareresolutionarity);

    mfindx deflt;
    int nvarg = 0;
    for (int i=0; i<nresolutions; i++) {
        if (sorted[i].varg &&
            mfcompiler_matchknown(&sorted[i], path->nparams, path->known)) {
            vargsubset[nvarg++] = sorted[i];
        }
    }
    if (nvarg<=0) {
        ERR_CHECK_RETURN(mfcompiler_emitfail(compiler, &deflt));
    } else {
        mfcompilerpath defltpath = *path;
        defltpath.aritychecked = true;
        ERR_CHECK_RETURN(mfcompiler_emitresolver(compiler, nvarg, vargsubset, &defltpath, &deflt));
    }

    mfcompilersparseentry table[nresolutions];
    int ncases = 0;
    for (int i=0; i<nresolutions; ) {
        if (sorted[i].varg) { i++; continue; }

        int arity = sorted[i].nparams;
        while (i<nresolutions && !sorted[i].varg && sorted[i].nparams==arity) i++;

        mfcompileresolution bucket[nresolutions];
        int count = 0;
        for (int j=0; j<nresolutions; j++) {
            if (!sorted[j].varg &&
                sorted[j].nparams==arity &&
                mfcompiler_matchknown(&sorted[j], path->nparams, path->known)) {
                bucket[count++] = sorted[j];
            }
        }

        table[ncases].value = arity;
        mfcompilerpath bucketpath = *path;
        bucketpath.knownarity = arity;
        bucketpath.aritychecked = true;
        ERR_CHECK_RETURN(mfcompiler_emitresolver(compiler, count, bucket, &bucketpath, &table[ncases].target));
        ncases++;
    }

    return mfcompiler_emitsparse(compiler, ncases, table, deflt, entry);
}

/** Emit a resolver block for a filtered candidate set. */
static bool mfcompiler_emitresolver(mfcompiler *compiler, int nresolutions, mfcompileresolution *resolutions, mfcompilerpath *path, mfindx *entry) {
    mfcompilerpath exactpath;

    /* No candidates remain on this path. */
    if (nresolutions<=0) return mfcompiler_emitfail(compiler, entry);

    /* A single fully-determined resolution can be emitted directly. */
    if (nresolutions==1 && mfcompiler_resolutionisterminal(&resolutions[0], path->nparams, path->known)) {
        return mfcompiler_emitresolve(compiler, resolutions[0].fnindex, entry);
    }

    bool hastyped = mfcompiler_hastypedresolutions(nresolutions, resolutions);
    if (!path->aritychecked &&
        !mfcompiler_hasvargresolutions(nresolutions, resolutions) &&
        mfcompiler_countarities(nresolutions, resolutions)==1) {
        exactpath = *path;
        exactpath.knownarity = resolutions[0].nparams;
        exactpath.aritychecked = true;
        path = &exactpath;
    }

    /* For exact arity without typed params, emit the lone fixed-arity winner. */
    if (!hastyped && path->knownarity>=0) {
        if (mfcompiler_emitfixedaritywinner(compiler, nresolutions, resolutions, path->nparams, path->known, entry)) return true;
    }

    /* Known path facts may already determine the winner. */
    if (path->aritychecked) {
        int fnindex;
        if (mfcompiler_resolveknownsubset(nresolutions, resolutions, path->nparams, path->known, &fnindex)) {
            return mfcompiler_emitresolve(compiler, fnindex, entry);
        }
    }

    /* Split by arity before considering typed dispatch. */
    if (!path->aritychecked &&
        (mfcompiler_countarities(nresolutions, resolutions)>1 || mfcompiler_hasvargresolutions(nresolutions, resolutions))) {
        return mfcompiler_emitarityresolver(compiler, nresolutions, resolutions, path, entry);
    }

    /* Otherwise try a typed split on the current arity-filtered subset. */
    if (hastyped) {
        if (mfcompiler_emittypedresolver(compiler, nresolutions, resolutions, path, entry)) return true;
    }

    /* Fall back to the runtime resolver when no fast split is worthwhile. */
    return mfcompiler_emitslow(compiler, entry);
}

/** Compile a resolver for a metafunction being assembled. */
bool metafunction_compile(objectmetafunction *fn, error *err) {
    if (fn->fns.count<=0) return false;

    mfcompiler compiler;
    mfcompiler_init(&compiler, fn, err);
    metafunction_clearinstructions(fn);

    ERR_CHECK(mfcompiler_analyze(&compiler), metafunction_compile_cleanup);
    ERR_CHECK(mfcompiler_checkduplicates(&compiler), metafunction_compile_cleanup);
    ERR_CHECK(mfcompiler_compileentry(&compiler, &fn->entry), metafunction_compile_cleanup);
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
 * Specialize a metafunction given type information
 * ********************************************************************** */

/** Reduce a metafunction using known argument types. */
bool metafunction_reduce(objectmetafunction *fn, int nargs, value *args, error *err, value *out) {
    int nres = fn->fns.count;
    mfresolutionset set;
    mfresolution res[nres];
    objectmetafunction *reduced = NULL;
    
    mfresolutionset_init(&set, res, fn);
    mfresolutionset_filterbyarity(&set, nargs);
    mfresolutionset_filterbyknowntypes(&set, nargs, args);
    mfresolutionset_filterbyspecificity(&set, nargs, args);

    if (set.count<=0) { // No resolutions left
        error_writewithid(err, VM_MLTPLDSPTCHFLD); return false;
    } else if (set.count==1 &&
               mfresolution_isterminal(set.data[0].sig, nargs, args)) { // Resolution is completely specified
        *out = set.data[0].fn; return true;
    } else if (set.count==fn->fns.count) { // Resolution set is unchanged
        *out = MORPHO_OBJECT(fn); return true;
    }

    ERR_CHECK(reduced = object_newmetafunction(fn->name), metafunction_reduce_cleanup);

    metafunction_setclass(reduced, metafunction_class(fn));
    for (int i=0; i<set.count; i++) {
        ERR_CHECK(metafunction_add(reduced, set.data[i].fn), metafunction_reduce_cleanup);
    }
    ERR_CHECK(metafunction_finalize(reduced, err), metafunction_reduce_cleanup);

    *out = MORPHO_OBJECT(reduced);
    return true;

metafunction_reduce_cleanup:
    if (reduced) object_free((object *) reduced);
    return false;
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
MORPHO_METHOD(MORPHO_TOSTRING_METHOD, Metafunction_tostring, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Metafunction_count, MORPHO_FN_PUREFN)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

objecttype objectmetafunctiontype;

void metafunction_initialize(void) {
    // Locate the Callable class to use as the parent class of Metafunction
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
