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
    MF_BRANCH,
    MF_BRANCHNARGS,
    MF_BRANCHVALUETYPE,
    MF_BRANCHOBJECTTYPE,
    MF_BRANCHINSTANCE,
    MF_RESOLVE,
    MF_FAIL
};

DEFINE_VARRAY(mfinstruction, mfinstruction);

#define MFINSTRUCTION_EMPTY -1

#define MFINSTRUCTION_FAIL { .opcode=MF_FAIL, .branch=MFINSTRUCTION_EMPTY }
#define MFINSTRUCTION_RESOLVE(fn) { .opcode=MF_RESOLVE, .data.resolvefn=fn, .branch=MFINSTRUCTION_EMPTY }
#define MFINSTRUCTION_CHECKNARG(n, brnch) { .opcode=MF_CHECKNARGS, .narg=n, .branch=brnch }
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
            case MF_CHECKNARGS: {
                printf("checkargs (%i) -> (%i)", instr->narg, i+instr->branch+1);
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
                    printf("        %i [%s] -> %i\n", k, MORPHO_GETCSTRING(klass->name), i+instr->data.btable.data[k]+1);
                }
                break;
            }
            case MF_BRANCHOBJECTTYPE: {
                printf("branchobjtype (%i) -> (%i)\n", instr->narg, i+instr->branch+1);
                for (int k=0; k<instr->data.btable.count; k++) {
                    if (instr->data.btable.data[k]==0) continue;
                    objectclass *klass=object_getveneerclass(k);
                    printf("        %i [%s] -> %i\n", k, MORPHO_GETCSTRING(klass->name), i+instr->data.btable.data[k]+1);
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

mfindx mfcompile_fail(mfcompiler *c);
mfindx mfcompile_resolve(mfcompiler *c, mfset *set);
mfindx mfcompile_dispatchonparam(mfcompiler *c, mfset *set, int i);
mfindx mfcompile_dispatchonnarg(mfcompiler *c, mfset *set, int min, int max);
mfindx mfcompile_set(mfcompiler *c, mfset *set);

/** Inserts a fail instruction */
mfindx mfcompile_fail(mfcompiler *c) {
    mfinstruction fail = MFINSTRUCTION_FAIL;
    return mfcompile_insertinstruction(c, fail);
}

/** Compiles a single result */
mfindx mfcompile_resolve(mfcompiler *c, mfset *set) {
    // Should check all arguments have been resolved

    mfinstruction instr = MFINSTRUCTION_RESOLVE(set->rlist->fn);
    return mfcompile_insertinstruction(c, instr);
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
    int typecount[MF_ANY+1] = { 0, 0, 0, 0};
    
    // Determine what types are present
    for (int k=0; k<set->count; k++) {
        value type;
        if (!signature_getparamtype(set->rlist[k].sig, i, &type)) return;
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
    
    return bindx[0];
}

/** Attempts to dispatch based on the number of arguments */
mfindx mfcompile_dispatchonnarg(mfcompiler *c, mfset *set, int min, int max) {
    mfindx bindx = MFINSTRUCTION_EMPTY;
    
    if (set->count==2) {
        for (int i=0; i<2; i++) {
            signature *sig = set->rlist[i].sig;
            int nparams;
            signature_paramlist(sig, &nparams, NULL); // Get the number of parameters
            mfinstruction instr = MFINSTRUCTION_CHECKNARG(nparams, 0);
            bindx = mfcompile_insertinstruction(c, instr); // Write the check nargs instruction
            
            mfset res = MFSET(1, &set->rlist[i]); // If it works, resolve on this implementation
            mfcompile_resolve(c, &res);
            
            // Fix the branch instruction
            mfindx eindx = mfcompile_currentinstruction(c);
            mfcompile_setbranch(c, bindx, eindx-bindx);
        }
        mfcompile_fail(c);
    } else { // If more than two options, generate a branch table
        varray_int btable;
        varray_intinit(&btable);
        for (int i=0; i<=max; i++) varray_intwrite(&btable, 0);
        
        // Count the number of implementations for each parameter count
        for (int k=0; k<set->count; k++) {
            set->rlist[k].indx=signature_countparams(set->rlist[k].sig);
        }
        
        // Sort the set on the type index
        qsort(set->rlist, set->count, sizeof(mfresult), _mfresultsortfn);
        
        // Insert the branch table instruction
        mfinstruction table = MFINSTRUCTION_BRANCHNARG(btable, 0);
        bindx = mfcompile_insertinstruction(c, table);
        
        // Immediately follow by a fail instruction if this falls through
        mfcompile_fail(c);

        // Compile the branch table
        mfcompile_branchtable(c, set, bindx, &btable);
    }
    return bindx;
}

/** Attempts to discriminate between a list of possible signatures */
mfindx mfcompile_set(mfcompiler *c, mfset *set) {
    if (set->count==1) return mfcompile_resolve(c, set);
    
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
    
    mfcompiler_disassemble(&compiler);
    
    mfcompiler_clear(&compiler, fn);
}

/** Execute the metafunction's resolver */
bool metafunction_resolve(objectmetafunction *fn, int nargs, value *args, value *out) {
    mfinstruction *pc = fn->resolver.data;
    if (!pc) return metafunction_slowresolve(fn, nargs, args, out);
    
    do {
        switch(pc->opcode) {
            case MF_CHECKNARGS:
                if (pc->narg!=nargs) pc+=pc->branch;
                break;
            case MF_BRANCH:
                pc+=pc->branch;
                break;
            case MF_BRANCHNARGS:
                if (nargs<pc->data.btable.count) {
                    pc+=pc->data.btable.data[nargs];
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
