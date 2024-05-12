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
    metafunction_clearinstructions(f);
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
        varray_mfinstructioninit(&new->resolver);
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

signature *_getsignature(value fn) {
    if (MORPHO_ISFUNCTION(fn)) {
        return &MORPHO_GETFUNCTION(fn)->sig;
    } else if (MORPHO_ISBUILTINFUNCTION(fn)) {
        return &MORPHO_GETBUILTINFUNCTION(fn)->sig;
    } else if (MORPHO_ISCLOSURE(fn)) {
        return &MORPHO_GETCLOSURE(fn)->func->sig;
    }
    return NULL;
}

value _getname(value fn) {
    if (MORPHO_ISFUNCTION(fn)) {
        return MORPHO_GETFUNCTION(fn)->name;
    } else if (MORPHO_ISBUILTINFUNCTION(fn)) {
        return MORPHO_GETBUILTINFUNCTION(fn)->name;
    } else if (MORPHO_ISCLOSURE(fn)) {
        return MORPHO_GETCLOSURE(fn)->func->name;
    }
    return MORPHO_NIL;
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
    MF_FAIL,
    MF_CHECKNARGSNEQ,
    MF_CHECKNARGSLT,
    MF_CHECKVALUE,
    MF_CHECKOBJECT,
    MF_CHECKINSTANCE,
    MF_BRANCH,
    MF_BRANCHNARGS,
    MF_BRANCHVALUETYPE,
    MF_BRANCHOBJECTTYPE,
    MF_BRANCHINSTANCE
};

DEFINE_VARRAY(mfinstruction, mfinstruction);

#define MFINSTRUCTION_EMPTY -1

#define MFINSTRUCTION_FAIL { .opcode=MF_FAIL, .branch=MFINSTRUCTION_EMPTY }
#define MFINSTRUCTION_RESOLVE(fn) { .opcode=MF_RESOLVE, .data.resolvefn=fn, .branch=MFINSTRUCTION_EMPTY }
#define MFINSTRUCTION_OPTARGS { .opcode=MF_OPTARGS, .branch=MFINSTRUCTION_EMPTY }
#define MFINSTRUCTION_CHECKNARGS(op, n, brnch) { .opcode=op, .narg=n, .branch=brnch }
#define MFINSTRUCTION_CHECKTYPE(op, n, t, brnch) { .opcode=op, .data.tindx=t, .narg=n, .branch=brnch }
#define MFINSTRUCTION_BRANCH(brnch) { .opcode=MF_BRANCH, .branch=brnch }
#define MFINSTRUCTION_BRANCHNARG(table, brnch) { .opcode=MF_BRANCHNARGS, .data.btable=table, .branch=brnch }
#define MFINSTRUCTION_BRANCHOBJECTTYPE(n, table, brnch) { .opcode=MF_BRANCHOBJECTTYPE, .narg=n, .data.btable=table, .branch=brnch }
#define MFINSTRUCTION_BRANCHVALUETYPE(n, table, brnch) { .opcode=MF_BRANCHVALUETYPE, .narg=n, .data.btable=table, .branch=brnch }
#define MFINSTRUCTION_BRANCHINSTANCE(n, table, brnch) { .opcode=MF_BRANCHINSTANCE, .narg=n, .data.btable=table, .branch=brnch }

typedef struct {
    signature *sig; /** Signature of the target */
    value fn; /** The target */
    int indx; /** Used to sort */
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
    varray_int checked; // Stack of checked parameters
} mfcompiler;

/** Initialize the metafunction compiler */
void mfcompiler_init(mfcompiler *c, objectmetafunction *fn) {
    c->fn=fn;
    varray_intinit(&c->checked);
}

/** Clear the metafunction compiler */
void mfcompiler_clear(mfcompiler *c, objectmetafunction *fn) {
    varray_intclear(&c->checked);
}

/** Pushes a parameter check onto the stack*/
void mfcompiler_pushcheck(mfcompiler *c, int i) {
    varray_intwrite(&c->checked, i);
}

/** Pops a parameter check from the stack*/
int mfcompiler_popcheck(mfcompiler *c) {
    c->checked.count--;
    return c->checked.data[c->checked.count];
}

/** Tests if a parameter has been checked according to the check stack */
bool mfcompiler_ischecked(mfcompiler *c, int i) {
    for (int j=0; j<c->checked.count; j++) if (i==c->checked.data[j]) return true;
    return false;
}

void _mfcompiler_disassemblebranchtable(mfinstruction *instr, mfindx i) {
    for (int k=0; k<instr->data.btable.count; k++) {
        printf("        %i -> %i\n", k, i+instr->data.btable.data[k]+1);
    }
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
            case MF_CHECKNARGSNEQ: {
                printf("checknargsneq %i -> (%i)", instr->narg, i+instr->branch+1);
                break;
            }
            case MF_CHECKNARGSLT: {
                printf("checknargslt %i -> (%i)", instr->narg, i+instr->branch+1);
                break;
            }
            case MF_CHECKVALUE: {
                objectclass *klass=value_veneerclassfromtype(instr->data.tindx);
                printf("checkvalue (%i) [%s] -> (%i)", instr->narg,  MORPHO_GETCSTRING(klass->name), i+instr->branch+1);
                break;
            }
            case MF_CHECKOBJECT: {
                objectclass *klass=object_getveneerclass(instr->data.tindx);
                printf("checkobject (%i) [%s] -> (%i)", instr->narg, MORPHO_GETCSTRING(klass->name), i+instr->branch+1);
                break;
            }
            case MF_CHECKINSTANCE: {
                printf("checkinstance (%i) -> (%i)", instr->narg, i+instr->branch+1);
                break;
            }
            case MF_BRANCH: {
                printf("branch -> (%i)", i+instr->branch+1);
                break;
            }
            case MF_BRANCHNARGS: {
                printf("branchargs (%i) -> (%i)\n", instr->narg, i+instr->branch+1);
                _mfcompiler_disassemblebranchtable(instr, i);
                break;
            }
            case MF_BRANCHVALUETYPE: {
                printf("branchvalue (%i) -> (%i)\n", instr->narg, i+instr->branch+1);
                for (int k=0; k<instr->data.btable.count; k++) {
                    if (instr->data.btable.data[k]==0) continue;
                    objectclass *klass=value_veneerclassfromtype(k);
                    printf("        %i [%s] -> %i\n", k, (klass ? MORPHO_GETCSTRING(klass->name) : ""), i+instr->data.btable.data[k]+1);
                }
                break;
            }
            case MF_BRANCHOBJECTTYPE: {
                printf("branchobjtype (%i) -> (%i)\n", instr->narg, i+instr->branch+1);
                for (int k=0; k<instr->data.btable.count; k++) {
                    if (instr->data.btable.data[k]==0) continue;
                    objectclass *klass=object_getveneerclass(k);
                    printf("        %i [%s] -> %i\n", k, (klass ? MORPHO_GETCSTRING(klass->name) : ""), i+instr->data.btable.data[k]+1);
                }
                break;
            }
            case MF_BRANCHINSTANCE: {
                printf("branchinstance (%i) -> (%i)\n", instr->narg, i+instr->branch+1);
                _mfcompiler_disassemblebranchtable(instr, i);
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
    for (int k=0; k<set->count; k++) { // Loop over outcomes
        value type;
        if (!signature_getparamtype(set->rlist[k].sig, i, &type)) continue;
        if (!dictionary_insert(out, (MORPHO_ISNIL(type) ? MORPHO_FALSE : type), MORPHO_NIL)) return false;
    }
    return true;
}

/** Find the parameter number that has most variety in types */
bool mfcompile_countoutcomes(mfcompiler *c, mfset *set, int *best) {
    varray_int count;
    varray_intinit(&count);
    
    dictionary dict;
    dictionary_init(&dict);
    
    // Loop over parameters, counting the number of outcomes.
    while (true) {
        mfcompile_outcomes(c, set, count.count, &dict);
        if (!dict.count) break;
        varray_intwrite(&count, dict.count);
        dictionary_clear(&dict); // Not needed if dict.count was zero
    };
    
    // Find the parameter that has most variability
    int max=count.data[0], maxindx=0;
    for (int i=1; i<count.count; i++) {
        if (count.data[i]>max) { max=count.data[i]; maxindx=i; }
    }
    if (best) *best = maxindx;
    
    varray_intclear(&count);
    
    return true;
}

mfindx mfcompile_insertinstruction(mfcompiler *c, mfinstruction instr) {
    return varray_mfinstructionwrite(&c->fn->resolver, instr);
}

mfindx mfcompile_currentinstruction(mfcompiler *c) {
    return c->fn->resolver.count-1;
}

mfindx mfcompile_nextinstruction(mfcompiler *c) {
    return c->fn->resolver.count;
}

void mfcompile_setbranch(mfcompiler *c, mfindx i, mfindx branch) {
    if (i>=c->fn->resolver.count) return;
    c->fn->resolver.data[i].branch=branch;
}

void mfcompile_replaceinstruction(mfcompiler *c, mfindx i, mfinstruction instr) {
    if (i>=c->fn->resolver.count) return;
    c->fn->resolver.data[i] = instr;
}

enum {
    MF_VENEERVALUE,
    MF_VENEEROBJECT,
    MF_INSTANCE,
    MF_ANY
};

/** Detects the kind of type */
int _detecttype(value type, int *tindx) {
    if (MORPHO_ISCLASS(type)) {
        objectclass *klass = MORPHO_GETCLASS(type);
        if (object_veneerclasstotype(klass, tindx)) {
            return MF_VENEEROBJECT;
        } else if (value_veneerclasstotype(klass, tindx)) {
            return MF_VENEERVALUE;
        } else {
            if (tindx) *tindx=klass->uid;
            return MF_INSTANCE;
        }
    }
    return MF_ANY;
}

mfindx mfcompile_fail(mfcompiler *c);
mfindx mfcompile_resolve(mfcompiler *c, mfresult *res);
mfindx mfcompile_dispatchonparam(mfcompiler *c, mfset *set, int i);
mfindx mfcompile_dispatchonnarg(mfcompiler *c, mfset *set, int min, int max);
mfindx mfcompile_set(mfcompiler *c, mfset *set);

/** Inserts a fail instruction */
mfindx mfcompile_fail(mfcompiler *c) {
    mfinstruction fail = MFINSTRUCTION_FAIL;
    return mfcompile_insertinstruction(c, fail);
}

/** Checks a parameter i for type */
mfindx mfcompile_check(mfcompiler *c, int i, value type) {
    int tindx;
    int opcode[MF_ANY] = { MF_CHECKVALUE, MF_CHECKOBJECT, MF_CHECKINSTANCE };
    int k=_detecttype(type, &tindx);
    
    if (k==MF_ANY) return MFINSTRUCTION_EMPTY;
        
    mfinstruction check = MFINSTRUCTION_CHECKTYPE(opcode[k], i, tindx, 0);
    return mfcompile_insertinstruction(c, check);
}

/** Compiles a single result */
mfindx mfcompile_resolve(mfcompiler *c, mfresult *res) {
    mfindx start = mfcompile_nextinstruction(c);
    
    // Check all arguments have been resolved
    signature *sig = res->sig;
    for (int i=0; i<sig->types.count; i++) {
        if (MORPHO_ISNIL(sig->types.data[i]) ||
            mfcompiler_ischecked(c, i)) continue;
        
        mfcompile_check(c, i, sig->types.data[i]);
    }
    
    mfindx end = mfcompile_nextinstruction(c);
    
    mfinstruction instr = MFINSTRUCTION_RESOLVE(res->fn);
    mfcompile_insertinstruction(c, instr);
    
    if (start!=end) {
        mfindx fail = mfcompile_fail(c);
        for (mfindx i=start; i<end; i++) mfcompile_setbranch(c, i, fail-start-1);
    }
        
    return start;
}

/** Compile a branch table from a sorted set */
void mfcompile_branchtable(mfcompiler *c, mfset *set, mfindx bindx, varray_int *btable) {
    int k=0;
    // Values with negative indices shouldn't be included in the branch table
    while (set->rlist[k].indx<0 && k<set->count) k++;
    
    // Deal with each outcome
    while (k<set->count) {
        int indx=set->rlist[k].indx, n=0;
        while (k+n<set->count && set->rlist[k+n].indx==indx) n++;
        
        mfset out = MFSET(n, &set->rlist[k]);
        
        // Set the branch point
        btable->data[indx]=mfcompile_currentinstruction(c)-bindx;
        mfcompile_set(c, &out);
        
        k+=n;
    }
}

int _mfresultsortfn (const void *a, const void *b) {
    mfresult *aa = (mfresult *) a, *bb = (mfresult *) b;
    return aa->indx-bb->indx;
}

typedef mfindx (mfcompile_dispatchfn) (mfcompiler *c, mfset *set, int i);

/** Branch table on object type */
mfindx mfcompile_dispatchveneerobj(mfcompiler *c, mfset *set, int i) {
    value type;
    
    // Extract the type index for each member of the set
    for (int k=0; k<set->count; k++) {
        if (!signature_getparamtype(set->rlist[k].sig, i, &type)) return MFINSTRUCTION_EMPTY;
        if (_detecttype(type, &set->rlist[k].indx)!=MF_VENEEROBJECT) set->rlist[k].indx=-1;
    }
    
    // Sort the set on the type index
    qsort(set->rlist, set->count, sizeof(mfresult), _mfresultsortfn);
    
    // Create the branch table
    int maxindx=set->rlist[set->count-1].indx;
    varray_int btable;
    varray_intinit(&btable);
    for (int i=0; i<=maxindx; i++) varray_intwrite(&btable, 0);
    
    // Insert the branch instruction
    mfinstruction instr = MFINSTRUCTION_BRANCHOBJECTTYPE(i, btable, 0);
    mfindx bindx = mfcompile_insertinstruction(c, instr);
    
    // Fail if an object type isn't in the table
    mfcompile_fail(c);
    
    // Compile the branch table
    mfcompile_branchtable(c, set, bindx, &btable);
    
    return bindx;
}

/** Branch table on value type */
mfindx mfcompile_dispatchveneervalue(mfcompiler *c, mfset *set, int i) {
    value type;
    
    // Extract the type index for each member of the set
    for (int k=0; k<set->count; k++) {
        if (!signature_getparamtype(set->rlist[k].sig, i, &type)) return MFINSTRUCTION_EMPTY;
        if (_detecttype(type, &set->rlist[k].indx)!=MF_VENEERVALUE) set->rlist[k].indx=-1;
    }
    
    // Sort the set on the type index
    qsort(set->rlist, set->count, sizeof(mfresult), _mfresultsortfn);
    
    // Create the branch table
    int maxindx=set->rlist[set->count-1].indx;
    varray_int btable;
    varray_intinit(&btable);
    for (int i=0; i<=maxindx; i++) varray_intwrite(&btable, 0);
    
    // Insert the branch instruction
    mfinstruction instr = MFINSTRUCTION_BRANCHVALUETYPE(i, btable, 0);
    mfindx bindx = mfcompile_insertinstruction(c, instr);
    
    // Fail if an object type isn't in the table
    mfcompile_fail(c);
    
    // Compile the branch table
    mfcompile_branchtable(c, set, bindx, &btable);
    
    return bindx;
}

/** Branch table on instance type */
mfindx mfcompile_dispatchinstance(mfcompiler *c, mfset *set, int i) {
    value type;
    
    // Extract the type index for each member of the set
    for (int k=0; k<set->count; k++) {
        if (!signature_getparamtype(set->rlist[k].sig, i, &type)) return MFINSTRUCTION_EMPTY;
        if (_detecttype(type, &set->rlist[k].indx)!=MF_INSTANCE) set->rlist[k].indx=-1;
    }
    
    // Sort the set on the type index
    qsort(set->rlist, set->count, sizeof(mfresult), _mfresultsortfn);
    
    // Create the branch table
    int maxindx=set->rlist[set->count-1].indx;
    varray_int btable;
    varray_intinit(&btable);
    for (int i=0; i<=maxindx; i++) varray_intwrite(&btable, 0);
    
    // Insert the branch instruction
    mfinstruction instr = MFINSTRUCTION_BRANCHINSTANCE(i, btable, 0);
    mfindx bindx = mfcompile_insertinstruction(c, instr);
    
    // Fail if an object type isn't in the table
    mfcompile_fail(c);
    
    // Compile the branch table
    mfcompile_branchtable(c, set, bindx, &btable);
    
    return bindx;
}

/** Handle implementations that accept any type */
mfindx mfcompile_dispatchany(mfcompiler *c, mfset *set, int i) {
    mfresult rlist[set->count];
    int n=0;
    
    // Find implementations that accept any type
    for (int k=0; k<set->count; k++) {
        value type;
        if (!signature_getparamtype(set->rlist[k].sig, i, &type)) return MFINSTRUCTION_EMPTY;
        if (_detecttype(type, &set->rlist[k].indx)==MF_ANY) {
            rlist[n] = set->rlist[k]; n++;
        }
    }
    
    mfindx bindx = mfcompile_nextinstruction(c);
    
    mfset anyset = MFSET(n, rlist);
    mfcompile_set(c, &anyset);
    
    return bindx;
}

/** Fixes a fallthrough fail */
void mfcompile_fixfallthrough(mfcompiler *c, mfindx i, mfindx branchto) {
    mfinstruction instr = MFINSTRUCTION_BRANCH(branchto-i-1);
    mfcompile_replaceinstruction(c, i, instr);
}

/** Attempts to dispatch based on a parameter i */
mfindx mfcompile_dispatchonparam(mfcompiler *c, mfset *set, int i) {
    mfcompiler_pushcheck(c, i);
    int typecount[MF_ANY+1] = { 0, 0, 0, 0};
    
    // Determine what types are present
    for (int k=0; k<set->count; k++) {
        value type;
        if (!signature_getparamtype(set->rlist[k].sig, i, &type)) continue;
        typecount[_detecttype(type, NULL)]++;
    }
    
    mfindx bindx[MF_ANY+1];
    mfcompile_dispatchfn *dfn[MF_ANY+1] = { mfcompile_dispatchveneervalue,
                                            mfcompile_dispatchveneerobj,
                                            mfcompile_dispatchinstance,
                                            mfcompile_dispatchany};
    
    // Cycle through all value types, building a chain of branchtables
    int n=0;
    for (int j=0; j<=MF_ANY; j++) {
        if (typecount[j] && dfn[j]) {
            bindx[n]=(dfn[j]) (c, set, i);
            if (n>0) mfcompile_setbranch(c, bindx[n-1], bindx[n]-bindx[n-1]-1);
            n++;
        }
    }
    
    if (typecount[MF_ANY]) { // Fix branch table fallthroughs to point to any
        for (int j=0; j<n-1; j++) {
            if (bindx[j]!=MFINSTRUCTION_EMPTY) mfcompile_fixfallthrough(c, bindx[j]+1, bindx[n-1]);
        }
    }
    
    mfcompiler_popcheck(c);
    return bindx[0];
}

/** Compiles an argument number check for a single result */
mfindx mfcompile_checknarg(mfcompiler *c, mfresult *res) {
    mfinstruction instr = MFINSTRUCTION_CHECKNARGS((signature_isvarg(res->sig) ? MF_CHECKNARGSLT : MF_CHECKNARGSNEQ ), signature_countparams(res->sig), 0);
    mfindx bindx = mfcompile_insertinstruction(c, instr); // Write the check nargs instruction
    
    mfcompile_resolve(c, res);
    
    return bindx;
}

/** Attempts to dispatch based on the number of arguments */
mfindx mfcompile_dispatchonnarg(mfcompiler *c, mfset *set, int min, int max) {
    mfindx bindx = MFINSTRUCTION_EMPTY;
    
    // Sort the set into order given by number of parameters
    for (int k=0; k<set->count; k++) {
        set->rlist[k].indx=signature_countparams(set->rlist[k].sig);
    }
    qsort(set->rlist, set->count, sizeof(mfresult), _mfresultsortfn);
    
    if (set->count==2) {
        for (int i=0; i<2; i++) {
            bindx = mfcompile_checknarg(c, &set->rlist[i]);
            
            mfindx eindx = mfcompile_currentinstruction(c);
            mfcompile_setbranch(c, bindx, eindx-bindx);
        }
        mfcompile_fail(c);
    } else { // If more than two options, generate a branch table
        varray_int btable;
        varray_intinit(&btable);
        for (int i=0; i<=max; i++) varray_intwrite(&btable, 0);
        
        // Insert the branch table instruction
        mfinstruction table = MFINSTRUCTION_BRANCHNARG(btable, 0);
        bindx = mfcompile_insertinstruction(c, table);
        
        // Immediately follow by a fail instruction if this falls through
        mfcompile_fail(c);

        // Compile the branch table
        mfcompile_branchtable(c, set, bindx, &btable);
        
        // Correct branchargs branch destination to point to the varg resolution
        if (signature_isvarg(set->rlist[set->count-1].sig)) {
            mfcompile_setbranch(c, bindx, btable.data[btable.count-1]);
        }
    }
    return bindx;
}

/** Attempts to discriminate between a list of possible signatures */
mfindx mfcompile_set(mfcompiler *c, mfset *set) {
    if (set->count==1) return mfcompile_resolve(c, set->rlist);
    
    int min, max; // Count the range of possible parameters
    mfcompile_countparams(c, set, &min, &max);
    
    // Dispatch on the number of parameters if it's in doubt
    if (min!=max) return mfcompile_dispatchonnarg(c, set, min, max);
    
    // If just one parameter, dispatch on it
    if (min==1) return mfcompile_dispatchonparam(c, set, 0);
    
    int best;
    if (mfcompile_countoutcomes(c, set, &best)) return mfcompile_dispatchonparam(c, set, best);
    
    return MFINSTRUCTION_EMPTY;
}

/** Clears the compiled code from a given metafunction */
void metafunction_clearinstructions(objectmetafunction *fn) {
    for (int i=0; i<fn->resolver.count; i++) {
        mfinstruction *mf = &fn->resolver.data[i];
        if (mf->opcode>=MF_BRANCHNARGS && mf->opcode<=MF_BRANCHINSTANCE) varray_intclear(&mf->data.btable);
    }
    varray_mfinstructionclear(&fn->resolver);
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

unsigned int vm_countpositionalargs(unsigned int nargs, value *args);

/** Execute the metafunction's resolver */
bool metafunction_resolve(objectmetafunction *fn, int nargs, value *args, value *out) {
    int n=vm_countpositionalargs(nargs, args);;
    mfinstruction *pc = fn->resolver.data;
    if (!pc) return metafunction_slowresolve(fn, n, args, out);
    
    do {
        switch(pc->opcode) {
            case MF_CHECKNARGSNEQ:
                if (n!=pc->narg) pc+=pc->branch;
                break;
            case MF_CHECKNARGSLT:
                if (n<pc->narg) pc+=pc->branch;
                break;
            case MF_CHECKVALUE: {
                if (!MORPHO_ISOBJECT(args[pc->narg])) {
                    int tindx = (int) MORPHO_GETORDEREDTYPE(args[pc->narg]);
                    if (pc->data.tindx!=tindx) pc+=pc->branch;
                } else pc+=pc->branch;
            }
                break;
            case MF_CHECKOBJECT: {
                if (MORPHO_ISOBJECT(args[pc->narg])) {
                    int tindx = (int) MORPHO_GETOBJECTTYPE(args[pc->narg]);
                    if (pc->data.tindx!=tindx) pc+=pc->branch;
                } else pc+=pc->branch;
            }
                break;
            case MF_CHECKINSTANCE: {
                if (MORPHO_ISINSTANCE(args[pc->narg])) {
                    int tindx = MORPHO_GETINSTANCE(args[pc->narg])->klass->uid;
                    if (pc->data.tindx!=tindx) pc+=pc->branch;
                } else pc+=pc->branch;
            }
                break;
            case MF_BRANCH:
                pc+=pc->branch;
                break;
            case MF_BRANCHNARGS:
                if (n<pc->data.btable.count) {
                    pc+=pc->data.btable.data[n];
                } else pc+=pc->branch;
                break;
            case MF_BRANCHVALUETYPE: {
                if (!MORPHO_ISOBJECT(args[pc->narg])) {
                    // TODO: Check for btable bound
                    int type = (int) MORPHO_GETORDEREDTYPE(args[pc->narg]);
                    pc+=pc->data.btable.data[type];
                } else pc+=pc->branch;
            }
                break;
            case MF_BRANCHOBJECTTYPE: {
                if (MORPHO_ISOBJECT(args[pc->narg])) {
                    // TODO: Check for btable bound
                    int type = MORPHO_GETOBJECTTYPE(args[pc->narg]);
                    pc+=pc->data.btable.data[type];
                } else pc+=pc->branch;
            }
                break;
            case MF_BRANCHINSTANCE: {
                if (MORPHO_ISINSTANCE(args[pc->narg])) {
                    // TODO: Check for btable bound
                    objectclass *klass = MORPHO_GETINSTANCE(args[pc->narg])->klass;
                    pc+=pc->data.btable.data[klass->uid];
                } else pc+=pc->branch;
            }
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

/** Constructor function for Metafunctions */
value metafunction_constructor(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    
    if (nargs==0) return MORPHO_NIL;
    
    value name = _getname(MORPHO_GETARG(args, 0));
    if (!MORPHO_ISSTRING(name)) return MORPHO_NIL;
    
    objectmetafunction *new = object_newmetafunction(name);
    
    if (new) {
        for (int i=0; i<nargs; i++) {
            metafunction_add(new, MORPHO_GETARG(args, i));
        }
        
        metafunction_compile(new);
        out=MORPHO_OBJECT(new);
    }
    
    return out;
}

/** Count the number of implementations in a metafunction */
value Metafunction_count(vm *v, int nargs, value *args) {
    objectmetafunction *fn = MORPHO_GETMETAFUNCTION(MORPHO_SELF(args));
    
    return MORPHO_INTEGER(fn->fns.count);
}

MORPHO_BEGINCLASS(Metafunction)
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
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // List constructor function
    builtin_addfunction(METAFUNCTION_CLASSNAME, metafunction_constructor, MORPHO_FN_CONSTRUCTOR);
    
    // Create function veneer class
    value metafunctionclass=builtin_addclass(METAFUNCTION_CLASSNAME, MORPHO_GETCLASSDEFINITION(Metafunction), objclass);
    object_setveneerclass(OBJECT_METAFUNCTION, metafunctionclass);
    
    // Metafunction error messages
}
