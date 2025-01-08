/** @file tuple.c
 *  @author T J Atherton
 *
 *  @brief Defines tuple object type and Tuple class
 */

#include "morpho.h"
#include "classes.h"

/* **********************************************************************
 * Tuple objects
 * ********************************************************************** */

/** Tuple object definitions */
void objecttuple_printfn(object *obj, void *v) {
    objecttuple *t = (objecttuple *) obj;
    morpho_printf(v, "(");
    for (unsigned int i=0; i<t->length; i++) {
        morpho_printvalue(v, t->tuple[i]);
        if (i<t->length-1) morpho_printf(v, ", ");
    }
    morpho_printf(v, ")");
}

void objecttuple_markfn(object *obj, void *v) {
    objecttuple *t = (objecttuple *) obj;
    for (unsigned int i=0; i<t->length; i++) morpho_markvalue(v, t->tuple[i]);
}

size_t objecttuple_sizefn(object *obj) {
    return sizeof(objecttuple)+(((objecttuple *) obj)->length)*sizeof(value);
}

hash objecttuple_hashfn(object *obj) {
    objecttuple *tuple = (objecttuple *) obj;
    return dictionary_hashvaluelist(tuple->length, tuple->tuple);
}

int objecttuple_cmpfn(object *a, object *b) {
    objecttuple *atuple = (objecttuple *) a;
    objecttuple *btuple = (objecttuple *) b;
    
    if (atuple->length!=btuple->length) return MORPHO_NOTEQUAL;

    int cmp=0;
    for (unsigned int i=0; i<atuple->length && cmp==0; i++) {
        cmp=morpho_comparevalue(atuple->tuple[i], btuple->tuple[i]);
    }
    
    return cmp;
}

objecttypedefn objecttupledefn = {
    .printfn = objecttuple_printfn,
    .markfn = objecttuple_markfn,
    .freefn = NULL,
    .sizefn = objecttuple_sizefn,
    .hashfn = objecttuple_hashfn,
    .cmpfn = objecttuple_cmpfn
};

/** @brief Creates a tuple from an existing C array of values
 *  @param length length of list
 *  @param in list of values
 *  @returns the object or NULL on failure */
objecttuple *object_newtuple(unsigned int length, value *in) {
    objecttuple *new = (objecttuple *) object_new(sizeof(objecttuple) + sizeof(value)*length, OBJECT_TUPLE);

    if (new) {
        new->tuple=new->tupledata;
        new->length=length;
        if (in) memcpy(new->tuple, in, sizeof(value)*length);
        else for (unsigned int i=0; i<length; i++) new->tuple[i]=MORPHO_NIL;
    }
    return new;
}

/* **********************************************************************
 * Tuple interface
 * ********************************************************************** */

/** Returns the length of a tuple */
unsigned int tuple_length(objecttuple *tuple) {
    return tuple->length;
}

/** Gets an element from the tuple */
bool tuple_getelement(objecttuple *tuple, int i, value *out) {
    if (!(i>=-(int) tuple->length && i<(int) tuple->length)) return false;
    if (i>=0) *out=tuple->tuple[i];
    else *out=tuple->tuple[tuple->length+i];
    return true;
}

/** Tests if a value is a member of a list */
bool tuple_ismember(objecttuple *tuple, value v) {
    for (unsigned int i=0; i<tuple->length; i++) {
        if (MORPHO_ISEQUAL(tuple->tuple[i], v)) return true;
    }
    return false;
}

/** Concatenates two tuples */
objecttuple *tuple_concatenate(objecttuple *a, objecttuple *b) {
    unsigned int newlength = a->length+b->length;
    objecttuple *new=object_newtuple(newlength, NULL);

    if (new) {
        memcpy(new->tuple, a->tuple, sizeof(value)*a->length);
        memcpy(new->tuple + a->length, b->tuple, sizeof(value)*b->length);
        new->length=newlength;
    }

    return new;
}

/* -------------------------------------------------------
 * Slicing
 * ------------------------------------------------------- */

/* Constructs a new list of a given size with a generic interface */
void tuple_sliceconstructor(unsigned int *slicesize, unsigned int ndim, value *out){
    objecttuple *tuple = object_newtuple(slicesize[0], NULL);
    *out = MORPHO_OBJECT(tuple);
}

/* Return number of dimensions to slice-there's only one */
bool tuple_slicedim(value *a, unsigned int ndim){
    if (ndim>1||ndim<0) return false;
    return true;
}

/* Copies data from tuple a at position indx to tuple out at position newindx with a generic interface */
objectarrayerror tuple_slicecopy(value *a,value *out, unsigned int ndim, unsigned int *indx, unsigned int *newindx){
    value data;
    objecttuple *outtuple = MORPHO_GETTUPLE(*out);

    if (tuple_getelement(MORPHO_GETTUPLE(*a),indx[0],&data)){
        outtuple->tuple[newindx[0]] = data;
    } else return ARRAY_OUTOFBOUNDS;
    return ARRAY_OK;
}

/** Converts an array error into an list error code for use in slices*/
errorid array_to_tuple_error(objectarrayerror err) {
    switch (err) {
        case ARRAY_OUTOFBOUNDS: return VM_OUTOFBOUNDS;
        case ARRAY_WRONGDIM: return TUPLE_NUMARGS;
        case ARRAY_NONINTINDX: return TUPLE_ARGS;
        case ARRAY_ALLOC_FAILED: return ERROR_ALLOCATIONFAILED;
        case ARRAY_OK: UNREACHABLE("array_to_tuple_error called incorrectly.");
    }
    UNREACHABLE("Unhandled array error.");
    return VM_OUTOFBOUNDS;
}

/* **********************************************************************
 * Tuple class
 * ********************************************************************** */

/** Constructor function for tuples */
value tuple_constructor(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    objecttuple *new=object_newtuple(nargs, & MORPHO_GETARG(args, 0));
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    return out;
}

/** Find a tuple's length */
value Tuple_count(vm *v, int nargs, value *args) {
    objecttuple *slf = MORPHO_GETTUPLE(MORPHO_SELF(args));

    return MORPHO_INTEGER(slf->length);
}

/** Clones a tuple */
value Tuple_clone(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    objecttuple *slf = MORPHO_GETTUPLE(MORPHO_SELF(args));
    objecttuple *new = object_newtuple(slf->length, slf->tuple);
    
    if (new) {
        out = MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

/** Get an element */
value Tuple_getindex(vm *v, int nargs, value *args) {
    objecttuple *slf = MORPHO_GETTUPLE(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (nargs==1) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

            if (!tuple_getelement(slf, i, &out)) morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        } else {
            objectarrayerror err = getslice(&MORPHO_SELF(args), tuple_slicedim, tuple_sliceconstructor, tuple_slicecopy, nargs, &MORPHO_GETARG(args, 0), &out);
            if (err!=ARRAY_OK) MORPHO_RAISE(v, array_to_tuple_error(err) );
            if (MORPHO_ISOBJECT(out)){
                morpho_bindobjects(v,1,&out);
            } else MORPHO_RAISE(v, VM_NONNUMINDX);
        }
    } else MORPHO_RAISE(v, LIST_NUMARGS)

    return out;
}

/** Setindex just raises an error */
value Tuple_setindex(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, OBJECT_IMMUTABLE);
    return MORPHO_NIL;
}

/** Enumerate members of a tuple */
value Tuple_enumerate(vm *v, int nargs, value *args) {
    objecttuple *slf = MORPHO_GETTUPLE(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

        if (n<0) {
            out=MORPHO_INTEGER(slf->length);
        } else {
            if (n<slf->length) {
                out=slf->tuple[n];
            } else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        }
    } else MORPHO_RAISE(v, ENUMERATE_ARGS);

    return out;
}

/** Joins two tuples together  */
value Tuple_join(vm *v, int nargs, value *args) {
    objecttuple *slf = MORPHO_GETTUPLE(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    if (nargs==1 && MORPHO_ISTUPLE(MORPHO_GETARG(args, 0))) {
        objecttuple *operand = MORPHO_GETTUPLE(MORPHO_GETARG(args, 0));
        objecttuple *new = tuple_concatenate(slf, operand);

        if (new) {
            out = MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        }

    } else morpho_runtimeerror(v, LIST_ADDARGS);

    return out;
}

/** Tests if a tuple has a value as a member */
value Tuple_ismember(vm *v, int nargs, value *args) {
    objecttuple *slf = MORPHO_GETTUPLE(MORPHO_SELF(args));

    if (nargs==1) {
        return MORPHO_BOOL(tuple_ismember(slf, MORPHO_GETARG(args, 0)));
    } else morpho_runtimeerror(v, ISMEMBER_ARG, 1, nargs);

    return MORPHO_NIL;
}

MORPHO_BEGINCLASS(Tuple)
MORPHO_METHOD(MORPHO_COUNT_METHOD, Tuple_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Tuple_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, Tuple_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, Tuple_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, Tuple_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_JOIN_METHOD, Tuple_join, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_ISMEMBER_METHOD, Tuple_ismember, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CONTAINS_METHOD, Tuple_ismember, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

objecttype objecttupletype;

void tuple_initialize(void) {
    // Create tuple object type
    objecttupletype=object_addtype(&objecttupledefn);
    
    // Locate the Object class to use as the parent class of Range
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // Create tuple veneer class
    value tupleclass=builtin_addclass(TUPLE_CLASSNAME, MORPHO_GETCLASSDEFINITION(Tuple), objclass);
    object_setveneerclass(OBJECT_TUPLE, tupleclass);
    
    // Tuple constructor function
    morpho_addfunction(TUPLE_CLASSNAME, TUPLE_CLASSNAME " (...)", tuple_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
    
    // Tuple error messages
    morpho_defineerror(TUPLE_ARGS, ERROR_HALT, TUPLE_ARGS_MSG);
    morpho_defineerror(TUPLE_NUMARGS, ERROR_HALT, TUPLE_NUMARGS_MSG);
}
