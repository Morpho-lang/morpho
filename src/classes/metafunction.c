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
}

/** Finalizes a metafunction. */
bool metafunction_finalize(objectmetafunction *fn, error *err) {
    if (fn->state==METAFUNCTION_FROZEN) return true;
    if (!metafunction_compile(fn, err)) return false;
    metafunction_disassemble(fn);
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
 * Fast metafunction resolver
 * ********************************************************************** */

// DEFINE_VARRAY(mfinstruction, mfinstruction);
//
// #define MFINSTRUCTION_EMPTY -1
//
// #define MFINSTRUCTION_FAIL { .opcode=MF_FAIL, .branch=MFINSTRUCTION_EMPTY }
// #define MFINSTRUCTION_RESOLVE(fn) { .opcode=MF_RESOLVE, .data.resolvefn=fn, .branch=MFINSTRUCTION_EMPTY }
// #define MFINSTRUCTION_OPTARGS { .opcode=MF_OPTARGS, .branch=MFINSTRUCTION_EMPTY }
// #define MFINSTRUCTION_CHECKNARGS(op, n, brnch) { .opcode=op, .narg=n, .branch=brnch }
// #define MFINSTRUCTION_CHECKTYPE(op, n, t, brnch) { .opcode=op, .data.tindx=t, .narg=n, .branch=brnch }
// #define MFINSTRUCTION_BRANCH(brnch) { .opcode=MF_BRANCH, .branch=brnch }
// #define MFINSTRUCTION_BRANCHNARG(table, brnch) { .opcode=MF_BRANCHNARGS, .data.btable=table, .branch=brnch }
// #define MFINSTRUCTION_BRANCHTABLE(op, n, table, brnch) { .opcode=op, .narg=n, .data.btable=table, .branch=brnch }
//     
// typedef struct {
//     signature *sig; /** Signature of the target */
//     value fn; /** The target */
//     int indx; /** Used to sort */
// } mfresult;
//
// typedef struct {
//     int count;
//     mfresult *rlist;
// } mfset;
//
// /** Static intiializer for the mfset */
// #define MFSET(c, l) { .count=c, .rlist=l }
//
// DECLARE_VARRAY(mfset, mfset)
// DEFINE_VARRAY(mfset, mfset)
//
// typedef struct {
//     objectmetafunction *fn;
//     varray_int checked; // Stack of checked parameters
//     bool varchecked; // Have we checked for a variadic function?
//     error err;
// } mfcompiler;
//
// /** Initialize the metafunction compiler */
// void mfcompiler_init(mfcompiler *c, objectmetafunction *fn) {
//     c->fn=fn;
//     varray_intinit(&c->checked);
//     error_init(&c->err);
//     c->varchecked=false;
// }
//
// /** Clear the metafunction compiler */
// void mfcompiler_clear(mfcompiler *c, objectmetafunction *fn) {
//     varray_intclear(&c->checked);
//     error_clear(&c->err);
// }
//
// /** Report an error during metafunction compilation */
// void mfcompiler_error(mfcompiler *c, errorid id) {
//     morpho_writeerrorwithid(&c->err, id, NULL, ERROR_POSNUNIDENTIFIABLE, ERROR_POSNUNIDENTIFIABLE);
// }
//
// /** Pushes a parameter check onto the stack*/
// void mfcompiler_pushcheck(mfcompiler *c, int i) {
//     varray_intwrite(&c->checked, i);
// }
//
// /** Pops a parameter check from the stack*/
// int mfcompiler_popcheck(mfcompiler *c) {
//     c->checked.count--;
//     return c->checked.data[c->checked.count];
// }
//
// /** Tests if a parameter has been checked according to the check stack */
// bool mfcompiler_ischecked(mfcompiler *c, int i) {
//     for (int j=0; j<c->checked.count; j++) if (i==c->checked.data[j]) return true;
//     return false;
// }
//
// void _mfcompiler_disassemblebranchtable(mfinstruction *instr, mfindx i) {
//     for (int k=0; k<instr->data.btable.count; k++) {
//         printf("        %i -> %i\n", k, i+instr->data.btable.data[k]+1);
//     }
// }
//
// /** Disassemble */
// void mfcompiler_disassemble(mfcompiler *c) {
//     int ninstr = c->fn->resolver.count;
//     morpho_printvalue(NULL, MORPHO_OBJECT(c->fn));
//     printf(":\n");
//     for (int i=0; i<ninstr; i++) {
//         mfinstruction *instr = &c->fn->resolver.data[i];
//         printf("%5i : ", i) ;
//         switch(instr->opcode) {
//             case MF_CHECKNARGSNEQ: {
//                 printf("checknargsneq %i -> (%i)", instr->narg, i+instr->branch+1);
//                 break;
//             }
//             case MF_CHECKNARGSLT: {
//                 printf("checknargslt %i -> (%i)", instr->narg, i+instr->branch+1);
//                 break;
//             }
//             case MF_CHECKVALUE: {
//                 objectclass *klass=value_veneerclassfromtype(instr->data.tindx);
//                 printf("checkvalue (%i) [%s] -> (%i)", instr->narg,  MORPHO_GETCSTRING(klass->name), i+instr->branch+1);
//                 break;
//             }
//             case MF_CHECKOBJECT: {
//                 objectclass *klass=object_getveneerclass(instr->data.tindx);
//                 printf("checkobject (%i) [%s] -> (%i)", instr->narg, MORPHO_GETCSTRING(klass->name), i+instr->branch+1);
//                 break;
//             }
//             case MF_CHECKINSTANCE: {
//                 printf("checkinstance (%i) [%i] -> (%i)", instr->narg, instr->data.tindx, i+instr->branch+1);
//                 break;
//             }
//             case MF_BRANCH: {
//                 printf("branch -> (%i)", i+instr->branch+1);
//                 break;
//             }
//             case MF_BRANCHNARGS: {
//                 printf("branchnargs (%i) -> (%i)\n", instr->narg, i+instr->branch+1);
//                 _mfcompiler_disassemblebranchtable(instr, i);
//                 break;
//             }
//             case MF_BRANCHVALUETYPE: {
//                 printf("branchvalue (%i) -> (%i)\n", instr->narg, i+instr->branch+1);
//                 for (int k=0; k<instr->data.btable.count; k++) {
//                     if (instr->data.btable.data[k]==0) continue;
//                     objectclass *klass=value_veneerclassfromtype(k);
//                     printf("        %i [%s] -> %i\n", k, (klass ? MORPHO_GETCSTRING(klass->name) : ""), i+instr->data.btable.data[k]+1);
//                 }
//                 break;
//             }
//             case MF_BRANCHOBJECTTYPE: {
//                 printf("branchobjtype (%i) -> (%i)\n", instr->narg, i+instr->branch+1);
//                 for (int k=0; k<instr->data.btable.count; k++) {
//                     if (instr->data.btable.data[k]==0) continue;
//                     objectclass *klass=object_getveneerclass(k);
//                     printf("        %i [%s] -> %i\n", k, (klass ? MORPHO_GETCSTRING(klass->name) : ""), i+instr->data.btable.data[k]+1);
//                 }
//                 break;
//             }
//             case MF_BRANCHINSTANCE: {
//                 printf("branchinstance (%i) -> (%i)\n", instr->narg, i+instr->branch+1);
//                 _mfcompiler_disassemblebranchtable(instr, i);
//                 break;
//             }
//             case MF_RESOLVE: {
//                 printf("resolve ");
//                 signature *sig = metafunction_getsignature(instr->data.resolvefn);
//                 printf(" ");
//                 if (sig) signature_print(sig);
//                 break;
//             }
//             case MF_FAIL: printf("fail"); break;
//         }
//         printf("\n");
//     }
// }
//
// /** Counts the range of parameters for the function call */
// void mfcompile_countparams(mfcompiler *c, mfset *set, int *min, int *max) {
//     int imin=INT_MAX, imax=INT_MIN;
//     for (int i=0; i<set->count; i++) {
//         int nparams;
//         signature_paramlist(set->rlist[i].sig, &nparams, NULL);
//         if (nparams<imin) imin=nparams;
//         if (nparams>imax) imax=nparams;
//     }
//     if (min) *min = imin;
//     if (max) *max = imax;
// }
//
// /** Check if a set contains a variadic */
// bool mfcompile_containsvariadic(mfcompiler *c, mfset *set) {
//     for (int i=0; i<set->count; i++) {
//         if (signature_isvarg(set->rlist[i].sig)) return true;
//     }
//     return false;
// }
//
// /** Places the various outcomes for a parameter into a dictionary */
// bool mfcompile_outcomes(mfcompiler *c, mfset *set, int i, dictionary *out) {
//     for (int k=0; k<set->count; k++) { // Loop over outcomes
//         value type;
//         if (!signature_getparamtype(set->rlist[k].sig, i, &type)) continue;
//         if (!dictionary_insert(out, (MORPHO_ISNIL(type) ? MORPHO_FALSE : type), MORPHO_NIL)) return false;
//     }
//     return true;
// }
//
// /** Find the parameter number that has most variety in types */
// bool mfcompile_countoutcomes(mfcompiler *c, mfset *set, int *best) {
//     varray_int count;
//     varray_intinit(&count);
//     
//     dictionary dict;
//     dictionary_init(&dict);
//     
//     // Loop over parameters, counting the number of outcomes.
//     while (true) {
//         mfcompile_outcomes(c, set, count.count, &dict);
//         if (!dict.count) break;
//         varray_intwrite(&count, dict.count);
//         dictionary_clear(&dict); // Not needed if dict.count was zero
//     };
//     
//     // Find the parameter that has most variability that has not already been checked
//     int max=0, maxindx=-1;
//     for (int i=0; i<count.count; i++) {
//         if (mfcompiler_ischecked(c, i)) continue;
//         if (count.data[i]>max) { max=count.data[i]; maxindx=i; }
//     }
//     
//     varray_intclear(&count);
//     
//     if (maxindx<0) return false;
//     if (best) *best = maxindx;
//     
//     return true;
// }
//
// mfindx mfcompile_insertinstruction(mfcompiler *c, mfinstruction instr) {
//     return varray_mfinstructionwrite(&c->fn->resolver, instr);
// }
//
// mfindx mfcompile_currentinstruction(mfcompiler *c) {
//     return c->fn->resolver.count-1;
// }
//
// mfindx mfcompile_nextinstruction(mfcompiler *c) {
//     return c->fn->resolver.count;
// }
//
// void mfcompile_setbranch(mfcompiler *c, mfindx i, mfindx branch) {
//     if (i>=c->fn->resolver.count) return;
//     c->fn->resolver.data[i].branch=branch;
// }
//
// void mfcompile_replaceinstruction(mfcompiler *c, mfindx i, mfinstruction instr) {
//     if (i>=c->fn->resolver.count) return;
//     c->fn->resolver.data[i] = instr;
// }
//
// enum {
//     MF_VENEERVALUE,
//     MF_INSTANCE,
//     MF_VENEEROBJECT,
//     MF_ANY
// };
//
// /** Detects the kind of type */
// int _detecttype(value type, int *tindx) {
//     if (MORPHO_ISCLASS(type)) {
//         objectclass *klass = MORPHO_GETCLASS(type);
//         if (object_veneerclasstotype(klass, tindx)) {
//             return MF_VENEEROBJECT;
//         } else if (value_veneerclasstotype(klass, tindx)) {
//             return MF_VENEERVALUE;
//         } else {
//             if (tindx) *tindx=klass->uid;
//             return MF_INSTANCE;
//         }
//     }
//     return MF_ANY;
// }
//
// mfindx mfcompile_fail(mfcompiler *c);
// mfindx mfcompile_resolve(mfcompiler *c, mfresult *res);
// mfindx mfcompile_dispatchonparam(mfcompiler *c, mfset *set, int i);
// mfindx mfcompile_dispatchonnarg(mfcompiler *c, mfset *set, int min, int max);
// mfindx mfcompile_set(mfcompiler *c, mfset *set);
//
// /** Inserts a fail instruction */
// mfindx mfcompile_fail(mfcompiler *c) {
//     mfinstruction fail = MFINSTRUCTION_FAIL;
//     return mfcompile_insertinstruction(c, fail);
// }
//
// /** Checks a parameter i for type */
// mfindx mfcompile_check(mfcompiler *c, int i, value type) {
//     int tindx;
//     int opcode[MF_ANY] = { MF_CHECKVALUE, MF_CHECKINSTANCE, MF_CHECKOBJECT };
//     int k=_detecttype(type, &tindx);
//     
//     if (k==MF_ANY) return MFINSTRUCTION_EMPTY;
//         
//     mfinstruction check = MFINSTRUCTION_CHECKTYPE(opcode[k], i, tindx, 0);
//     return mfcompile_insertinstruction(c, check);
// }
//
// /** Compiles a single result */
// mfindx mfcompile_resolve(mfcompiler *c, mfresult *res) {
//     mfindx start = mfcompile_nextinstruction(c);
//     
//     // Check all arguments have been resolved
//     signature *sig = res->sig;
//     for (int i=0; i<sig->types.count; i++) {
//         if (MORPHO_ISNIL(sig->types.data[i]) ||
//             mfcompiler_ischecked(c, i)) continue;
//         
//         mfcompile_check(c, i, sig->types.data[i]);
//     }
//     
//     mfindx end = mfcompile_nextinstruction(c);
//     
//     mfinstruction instr = MFINSTRUCTION_RESOLVE(res->fn);
//     mfcompile_insertinstruction(c, instr);
//     
//     if (start!=end) {
//         mfindx fail = mfcompile_fail(c);
//         for (mfindx i=start; i<end; i++) mfcompile_setbranch(c, i, fail-start-1);
//     }
//         
//     return start;
// }
//
// /** Compile a branch table from a sorted set */
// void mfcompile_branchtable(mfcompiler *c, mfset *set, mfindx bindx, varray_int *btable) {
//     int k=0;
//     // Values with negative indices shouldn't be included in the branch table
//     while (k<set->count && set->rlist[k].indx<0) k++;
//     
//     // Deal with each outcome
//     while (k<set->count) {
//         int indx=set->rlist[k].indx, n=0;
//         while (k+n<set->count && set->rlist[k+n].indx==indx) n++;
//         
//         mfset out = MFSET(n, &set->rlist[k]);
//         
//         // Set the branch point
//         btable->data[indx]=mfcompile_currentinstruction(c)-bindx;
//         mfcompile_set(c, &out);
//         
//         k+=n;
//     }
// }
//
// void _insertchildren(dictionary *dict, value v) {
//     if (!MORPHO_ISCLASS(v) ||
//         dictionary_get(dict, v, NULL)) return;
//     
//     dictionary_insert(dict, v, MORPHO_NIL); // Insert the class
//     objectclass *klass = MORPHO_GETCLASS(v); // and its children
//     for (int i=0; i<klass->children.count; i++) _insertchildren(dict, klass->children.data[i]);
// }
//
// bool _resolve(objectclass *klass, dictionary *types, value *out) {
//     for (int k=0; k<klass->linearization.count; k++) {
//         if (dictionary_get(types, klass->linearization.data[k], NULL)) {
//             *out = klass->linearization.data[k];
//             return true;
//         }
//     }
//     return false;
// }
//
// int _maxindx(dictionary *dict) {
//     int indx=0, maxindx=0;
//     for (int i=0; i<dict->capacity; i++) {
//         _detecttype(dict->contents[i].key, &indx);
//         if (indx>maxindx) maxindx=indx;
//     }
//     return maxindx;
// }
//
// int _mfresultsortfn (const void *a, const void *b) {
//     mfresult *aa = (mfresult *) a, *bb = (mfresult *) b;
//     
//     int ai = aa->indx, bi = bb->indx;
//     if (aa->sig->varg) ai=-1; // Ensure vargs end up first
//     if (bb->sig->varg) bi=-1;
//     
//     return ai-bi;
// }
//
// /** Constructs a dispatch table from the set of implementations */
// mfindx mfcompile_dispatchtable(mfcompiler *c, mfset *set, int i, int otype, int opcode) {
//     dictionary types, children;
//     dictionary_init(&types); // Keep track of the available types provided by the implementation
//     dictionary_init(&children); // and all of their children
//     
//     // Extract the type index for each member of the set
//     for (int k=0; k<set->count; k++) {
//         value type;
//         if (!signature_getparamtype(set->rlist[k].sig, i, &type)) UNREACHABLE("Incorrect parameter type");
//         if (_detecttype(type, &set->rlist[k].indx)==otype) {
//             dictionary_insert(&types, type, MORPHO_NIL);
//             _insertchildren(&children, type);
//         } else set->rlist[k].indx=-1; // Exclude from the branch table
//     }
//     
//     // Sort the set on the type index
//     qsort(set->rlist, set->count, sizeof(mfresult), _mfresultsortfn);
//     
//     // Create the branch table
//     int maxindx=_maxindx(&children);
//     varray_int btable;
//     varray_intinit(&btable);
//     for (int i=0; i<=maxindx; i++) varray_intwrite(&btable, 0);
//     
//     // Insert the branch instruction
//     mfinstruction instr = MFINSTRUCTION_BRANCHTABLE(opcode, i, btable, 0);
//     mfindx bindx = mfcompile_insertinstruction(c, instr);
//     
//     // Fail if an object type isn't in the table
//     mfcompile_fail(c);
//     
//     // Compile the branch table
//     mfcompile_branchtable(c, set, bindx, &btable);
//     
//     // Fix branch table to include child classes
//     for (int i=0; i<children.capacity; i++) {
//         int ifrom, ito;
//         value mapto;
//         if (_detecttype(children.contents[i].key, &ifrom)==otype &&
//             _resolve(MORPHO_GETCLASS(children.contents[i].key), &types, &mapto) &&
//             _detecttype(mapto, &ito)==otype) {
//             btable.data[ifrom]=btable.data[ito];
//         }
//     }
//     
//     // Clear temporary data structures
//     dictionary_clear(&types);
//     dictionary_clear(&children);
//     
//     return bindx;
// }
//
// /** Branch table on object type */
// mfindx mfcompile_dispatchveneerobj(mfcompiler *c, mfset *set, int i) {
//     return mfcompile_dispatchtable(c, set, i, MF_VENEEROBJECT, MF_BRANCHOBJECTTYPE);
// }
//
// /** Branch table on value type */
// mfindx mfcompile_dispatchveneervalue(mfcompiler *c, mfset *set, int i) {
//     return mfcompile_dispatchtable(c, set, i, MF_VENEERVALUE, MF_BRANCHVALUETYPE);
// }
//
// /** Branch table on instance type */
// mfindx mfcompile_dispatchinstance(mfcompiler *c, mfset *set, int i) {
//     return mfcompile_dispatchtable(c, set, i, MF_INSTANCE, MF_BRANCHINSTANCE);
// }
//
// /** Handle implementations that accept any type */
// mfindx mfcompile_dispatchany(mfcompiler *c, mfset *set, int i) {
//     mfresult rlist[set->count];
//     int n=0;
//     
//     // Find implementations that accept any type
//     for (int k=0; k<set->count; k++) {
//         value type;
//         if (!signature_getparamtype(set->rlist[k].sig, i, &type)) return MFINSTRUCTION_EMPTY;
//         if (_detecttype(type, &set->rlist[k].indx)==MF_ANY) {
//             rlist[n] = set->rlist[k]; n++;
//         }
//     }
//     
//     mfindx bindx = mfcompile_nextinstruction(c);
//     
//     mfset anyset = MFSET(n, rlist);
//     mfcompile_set(c, &anyset);
//     
//     return bindx;
// }
//
// /** Fixes a fallthrough fail */
// void mfcompile_fixfallthrough(mfcompiler *c, mfindx i, mfindx branchto) {
//     mfinstruction instr = MFINSTRUCTION_BRANCH(branchto-i-1);
//     mfcompile_replaceinstruction(c, i, instr);
// }
//
// typedef mfindx (mfcompile_dispatchfn) (mfcompiler *c, mfset *set, int i);
//
// /** Attempts to dispatch based on a parameter i */
// mfindx mfcompile_dispatchonparam(mfcompiler *c, mfset *set, int i) {
//     mfcompiler_pushcheck(c, i);
//     int typecount[MF_ANY+1] = { 0, 0, 0, 0};
//     
//     // Determine what types are present
//     for (int k=0; k<set->count; k++) {
//         value type;
//         if (!signature_getparamtype(set->rlist[k].sig, i, &type)) continue;
//         typecount[_detecttype(type, NULL)]++;
//     }
//     
//     mfindx bindx[MF_ANY+1];
//     mfcompile_dispatchfn *dfn[MF_ANY+1] = { mfcompile_dispatchveneervalue,
//                                             mfcompile_dispatchinstance,
//                                             mfcompile_dispatchveneerobj,
//                                             mfcompile_dispatchany};
//     
//     // Cycle through all value types, building a chain of branchtables
//     int n=0;
//     for (int j=0; j<=MF_ANY; j++) {
//         if (typecount[j] && dfn[j]) {
//             bindx[n]=(dfn[j]) (c, set, i);
//             if (n>0) mfcompile_setbranch(c, bindx[n-1], bindx[n]-bindx[n-1]-1);
//             n++;
//         }
//     }
//     
//     if (typecount[MF_ANY]) { // Fix branch table fallthroughs to point to any
//         for (int j=0; j<n-1; j++) {
//             if (bindx[j]!=MFINSTRUCTION_EMPTY) mfcompile_fixfallthrough(c, bindx[j]+1, bindx[n-1]);
//         }
//     }
//     
//     mfcompiler_popcheck(c);
//     return bindx[0];
// }
//
// /** Compiles an argument number check for a single result */
// mfindx mfcompile_checknarg(mfcompiler *c, mfresult *res) {
//     mfinstruction instr = MFINSTRUCTION_CHECKNARGS((signature_isvarg(res->sig) ? MF_CHECKNARGSLT : MF_CHECKNARGSNEQ ), signature_countparams(res->sig), 0);
//     mfindx bindx = mfcompile_insertinstruction(c, instr); // Write the check nargs instruction
//     
//     mfcompile_resolve(c, res);
//     
//     return bindx;
// }
//
// /** Attempts to dispatch based on the number of arguments */
// mfindx mfcompile_dispatchonnarg(mfcompiler *c, mfset *set, int min, int max) {
//     mfindx bindx = MFINSTRUCTION_EMPTY;
//     
//     // Sort the set into order given by number of parameters; resolution with varg is always first
//     for (int k=0; k<set->count; k++) {
//         set->rlist[k].indx=signature_countparams(set->rlist[k].sig);
//     }
//     qsort(set->rlist, set->count, sizeof(mfresult), _mfresultsortfn);
//     
//     if (set->count==2) {
//         for (int i=1; i>=0; i--) {
//             bindx = mfcompile_checknarg(c, &set->rlist[i]);
//             
//             mfindx eindx = mfcompile_currentinstruction(c);
//             mfcompile_setbranch(c, bindx, eindx-bindx);
//         }
//         mfcompile_fail(c);
//     } else { // If more than two options, generate a branch table
//         varray_int btable;
//         varray_intinit(&btable);
//         for (int i=0; i<=max; i++) varray_intwrite(&btable, 0);
//         
//         // Insert the branch table instruction
//         mfinstruction table = MFINSTRUCTION_BRANCHNARG(btable, 0);
//         bindx = mfcompile_insertinstruction(c, table);
//         
//         // Immediately follow by a fail instruction if this falls through
//         mfindx fail = mfcompile_fail(c);
//         
//         // Compile the branch table
//         mfcompile_branchtable(c, set, bindx, &btable);
//         
//         // Insert variadic resolution into branch table if necessary
//         if (set->rlist[0].sig->varg) {
//             mfinstruction vargres = MFINSTRUCTION_RESOLVE(set->rlist[0].fn);
//             mfindx varg = mfcompile_insertinstruction(c, vargres)-1;
//             
//             int nmin = set->rlist[0].sig->types.count-1; // varg can match this many args or more
//             for (int i=nmin; i<btable.count; i++) {
//                 if (btable.data[i]==fail-1) btable.data[i]=varg;
//             }
//             
//             mfcompile_setbranch(c, bindx, varg); // Correct branchargs branch destination to point to the varg resolution
//         }
//     }
//     
//     return bindx;
// }
//
// /** Attempts to discriminate between a list of possible signatures */
// mfindx mfcompile_set(mfcompiler *c, mfset *set) {
//     if (set->count==1) return mfcompile_resolve(c, set->rlist);
//     
//     int min, max; // Count the range of possible parameters
//     mfcompile_countparams(c, set, &min, &max);
//     
//     bool isvariadic = mfcompile_containsvariadic(c, set);
//     
//     // Dispatch on the number of parameters if it's in doubt
//     if (min!=max || (isvariadic && !c->varchecked)) {
//         c->varchecked=true;
//         return mfcompile_dispatchonnarg(c, set, min, max);
//     }
//     
//     // If just one parameter, dispatch on it
//     if (min==1 && !mfcompiler_ischecked(c, 0)) {
//         return mfcompile_dispatchonparam(c, set, 0);
//     }
//     
//     int best;
//     if (mfcompile_countoutcomes(c, set, &best)) return mfcompile_dispatchonparam(c, set, best);
//     
//     mfcompiler_error(c, METAFUNCTION_CMPLAMBGS);
//     return MFINSTRUCTION_EMPTY;
// }
//
// /** Clears the compiled code from a given metafunction */
// void metafunction_clearinstructions(objectmetafunction *fn) {
//     for (int i=0; i<fn->resolver.count; i++) {
//         mfinstruction *mf = &fn->resolver.data[i];
//         if (mf->opcode>=MF_BRANCHNARGS && mf->opcode<=MF_BRANCHINSTANCE) varray_intclear(&mf->data.btable);
//     }
//     varray_mfinstructionclear(&fn->resolver);
// }
//
// /** Compiles the resolver for a metafunction that is still being assembled */
// bool metafunction_compile(objectmetafunction *fn, error *err) {
//     mfset set;
//     set.count = fn->fns.count;
//     if (!set.count) return false;
//     
//     mfresult rlist[set.count];
//     set.rlist=rlist;
//     for (int i=0; i<set.count; i++) {
//         rlist[i].sig=metafunction_getsignature(fn->fns.data[i]);
//         rlist[i].fn=fn->fns.data[i];
//     }
//     
//     mfcompiler compiler;
//     mfcompiler_init(&compiler, fn);
//     
//     mfcompile_set(&compiler, &set);
//     //mfcompiler_disassemble(&compiler);
//     
//     bool success=!morpho_checkerror(&compiler.err);
//     if (!success && err) *err=compiler.err;
//     
//     mfcompiler_clear(&compiler, fn);
//     
//     return success;
// }
//
// /** Finalizes a metafunction, compiling its resolver once the implementation set is complete */
// bool metafunction_finalize(objectmetafunction *fn, error *err) {
//     if (fn->state==METAFUNCTION_FROZEN) return true;
//
//     if (!metafunction_compile(fn, err)) return false;
//     fn->state=METAFUNCTION_FROZEN;
//
//     return true;
// }
//
// /** Finalizes any metafunctions stored in a linked object list */
// bool metafunction_finalizelist(object *list, error *err) {
//     for (object *obj=list; obj!=NULL; obj=obj->next) {
//         if (obj->type==OBJECT_METAFUNCTION &&
//             !metafunction_finalize((objectmetafunction *) obj, err)) {
//             return false;
//         }
//     }
//
//     return true;
// }
//
// /** Attempt to find the desired class uid in the linearization of a given class */
// bool _finduidinlinearization(objectclass *klass, int uid) {
//     for (int k=0; k<klass->linearization.count; k++) {
//         if (MORPHO_GETCLASS(klass->linearization.data[k])->uid == uid) return true;
//     }
//     return false;
// }
//
// /** Execute the resolver for a finalized metafunction. 
//  @param[in] fn - the metafunction to resolve
//  @param[in] nargs - number of positional arguments
//  @param[in] args - positional arguments @warning: the first user-visible argument should be in the zero position
//  @param[out] err - error block to be filled out
//  @param[out] out - resolved function
//  @returns true if the metafunction was successfully resolved */
// bool metafunction_xresolve(objectmetafunction *fn, int nargs, value *args, error *err, value *out) {
//     if (fn->state!=METAFUNCTION_FROZEN) {
//         if (err) morpho_writeerrorwithid(err, METAFUNCTION_UNFROZEN, NULL, ERROR_POSNUNIDENTIFIABLE, ERROR_POSNUNIDENTIFIABLE);
//         return false;
//     }
//     mfinstruction *pc = fn->resolver.data;
//     if (!pc) return false;
//     
//     do {
//         switch(pc->opcode) {
//             case MF_CHECKNARGSNEQ:
//                 if (nargs!=pc->narg) pc+=pc->branch;
//                 break;
//             case MF_CHECKNARGSLT:
//                 if (nargs<pc->narg) pc+=pc->branch;
//                 break;
//             case MF_CHECKVALUE: {
//                 if (!MORPHO_ISOBJECT(args[pc->narg])) {
//                     int tindx = (int) MORPHO_GETORDEREDTYPE(args[pc->narg]);
//                     if (pc->data.tindx!=tindx) pc+=pc->branch;
//                 } else pc+=pc->branch;
//             }
//                 break;
//             case MF_CHECKOBJECT: {
//                 if (MORPHO_ISOBJECT(args[pc->narg])) {
//                     int tindx = (int) MORPHO_GETOBJECTTYPE(args[pc->narg]);
//                     if (pc->data.tindx!=tindx) pc+=pc->branch;
//                 } else pc+=pc->branch;
//             }
//                 break;
//             case MF_CHECKINSTANCE: {
//                 if (MORPHO_ISINSTANCE(args[pc->narg])) {
//                     objectclass *klass = MORPHO_GETINSTANCE(args[pc->narg])->klass;
//                     
//                     if (!(klass->uid==pc->data.tindx ||
//                         _finduidinlinearization(klass, pc->data.tindx))) {
//                         pc+=pc->branch;
//                     }
//                 } else pc+=pc->branch;
//             }
//                 break;
//             case MF_BRANCH:
//                 pc+=pc->branch;
//                 break;
//             case MF_BRANCHNARGS:
//                 if (nargs<pc->data.btable.count) {
//                     pc+=pc->data.btable.data[nargs];
//                 } else pc+=pc->branch;
//                 break;
//             case MF_BRANCHVALUETYPE: {
//                 if (!MORPHO_ISOBJECT(args[pc->narg])) {
//                     int type = (int) MORPHO_GETORDEREDTYPE(args[pc->narg]);
//                     if (type<pc->data.btable.count) pc+=pc->data.btable.data[type];
//                 } else pc+=pc->branch;
//             }
//                 break;
//             case MF_BRANCHOBJECTTYPE: {
//                 if (MORPHO_ISOBJECT(args[pc->narg])) {
//                     int type = MORPHO_GETOBJECTTYPE(args[pc->narg]);
//                     if (type<pc->data.btable.count) pc+=pc->data.btable.data[type];
//                 } else pc+=pc->branch;
//             }
//                 break;
//             case MF_BRANCHINSTANCE: {
//                 if (MORPHO_ISINSTANCE(args[pc->narg])) {
//                     // TODO: Check for btable bound
//                     objectclass *klass = MORPHO_GETINSTANCE(args[pc->narg])->klass;
//                     if (klass->uid<pc->data.btable.count) pc+=pc->data.btable.data[klass->uid];
//                 } else pc+=pc->branch;
//             }
//                 break;
//             case MF_RESOLVE:
//                 *out = pc->data.resolvefn;
//                 return true;
//             case MF_FAIL:
//                 return false;
//         }
//         pc++;
//     } while(true);
// }
//

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
    mfcompileresolution *resolutions;
    int nresolutions;
    bool hasvarg;
    bool hastyped;
} mfcompiler;

/** Initialize compiler state. */
static void mfcompiler_init(mfcompiler *compiler, objectmetafunction *fn) {
    compiler->fn = fn;
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

/** Analyze one candidate resolution. */
static bool mfcompiler_analyzecandidate(mfcompiler *compiler, int i) {
    if (!compiler->resolutions) return false;
    
    value fn = compiler->fn->fns.data[i];
    signature *sig = metafunction_getsignature(fn);
    if (!sig) return false;
    
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
        if (!mfcompiler_analyzecandidate(compiler, i)) return false;
    }
    return true;
}

/** Compare resolutions by arity for sorting. */
static int mfcompiler_compareresolutionarity(const void *a, const void *b) {
    int xi=((mfcompileresolution *) a)->nparams, yi=((mfcompileresolution *) b)->nparams;
    return (xi > yi) - (xi < yi); // Ascending order
}

/** Emit one instruction into the resolver. */
static int mfcompiler_emit(mfcompiler *compiler, mfinstruction instruction) {
    return varray_mfinstructionwrite(&compiler->fn->resolver, instruction);
}

/** Emit a conservative exact-arity resolver directly into bytecode. */
static bool mfcompiler_emitarityresolver(mfcompiler *compiler) {
    if (compiler->hasvarg) return (mfcompiler_emit(compiler, MFOP_SLOW)>=0);

    mfcompileresolution resolutions[compiler->nresolutions];
    memcpy(resolutions, compiler->resolutions, sizeof(mfcompileresolution)*compiler->nresolutions);
    qsort(resolutions, compiler->nresolutions, sizeof(mfcompileresolution), mfcompiler_compareresolutionarity);

    int nmatches = 0;
    for (int i=0; i<compiler->nresolutions; ) {
        int count = 1;
        while (i+count<compiler->nresolutions &&
               resolutions[i+count].nparams==resolutions[i].nparams) count++;
        nmatches++;
        i += count;
    }

    if (mfcompiler_emit(compiler, MFOP_SPARSE)<0) return false;
    if (mfcompiler_emit(compiler, nmatches)<0) return false;

    int defaultpc = mfcompiler_emit(compiler, 0);
    if (defaultpc<0) return false;

    int targetpc[nmatches];
    int match = 0;
    for (int i=0; i<compiler->nresolutions; ) {
        int arity = resolutions[i].nparams;
        int count = 1;
        while (i+count<compiler->nresolutions &&
               resolutions[i+count].nparams==arity) count++;
        if (mfcompiler_emit(compiler, arity)<0) return false;
        targetpc[match] = mfcompiler_emit(compiler, 0);
        if (targetpc[match]<0) return false;
        match++;
        i += count;
    }

    compiler->fn->resolver.data[defaultpc] = compiler->fn->resolver.count;
    if (mfcompiler_emit(compiler, MFOP_FAIL)<0) return false;

    match = 0;
    for (int i=0; i<compiler->nresolutions; ) {
        int count = 1;
        while (i+count<compiler->nresolutions &&
               resolutions[i+count].nparams==resolutions[i].nparams) count++;

        compiler->fn->resolver.data[targetpc[match]] = compiler->fn->resolver.count;
        if (count==1 && !resolutions[i].typed) {
            if (mfcompiler_emit(compiler, MFOP_RESOLVE)<0) return false;
            if (mfcompiler_emit(compiler, resolutions[i].fnindex)<0) return false;
        } else {
            if (mfcompiler_emit(compiler, MFOP_SLOW)<0) return false;
        }

        match++;
        i += count;
    }

    return true;
}

/** Compiles the resolver for a metafunction that is still being assembled. */
bool metafunction_compile(objectmetafunction *fn, error *err) {
    if (fn->fns.count<=0) return false;

    mfcompiler compiler;
    mfcompiler_init(&compiler, fn);
    metafunction_clearinstructions(fn);

    ERR_CHECK(mfcompiler_analyze(&compiler), metafunction_compile_cleanup);
    ERR_CHECK(mfcompiler_emitarityresolver(&compiler), metafunction_compile_cleanup);
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

/** Print a disassembly of the metafunction resolver bytecode. */
void metafunction_disassemble(objectmetafunction *fn) {
    printf("Resolver for ");
    morpho_printvalue(NULL, MORPHO_OBJECT(fn));
    printf(":\n");
    
    for (mfindx pc=0; pc<fn->resolver.count; ) {
        printf("%5i : ", pc);
        switch (fn->resolver.data[pc]) {
            case MFOP_SLOW: printf("slow"); pc++; break;
            case MFOP_FAIL: printf("fail"); pc++; break;
            case MFOP_RESOLVE: pc=metafunction_disassembleresolve(fn, pc); break;
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
    
    mfindx pc = 0;
    int reg = nargs; // Single register initialized with nargs
    
    while (true) {
        switch (instructions[pc]) {
            case MFOP_SLOW: return metafunction_resolveslow(fn, nargs, args, err, out);
            case MFOP_RESOLVE: {
                pc++; *out=fn->fns.data[instructions[pc]];
                return true;
            }
            case MFOP_FAIL: if (err) error_writewithid(err, VM_MLTPLDSPTCHFLD); return false;
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
        if (err) error_writewithid(err, METAFUNCTION_UNFROZEN); return false;
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
