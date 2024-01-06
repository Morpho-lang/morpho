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

MORPHO_BEGINCLASS(Tuple)
MORPHO_METHOD(MORPHO_COUNT_METHOD, Tuple_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Tuple_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, Tuple_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, Tuple_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_JOIN_METHOD, Tuple_join, BUILTIN_FLAGSEMPTY)
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
    builtin_addfunction(TUPLE_CLASSNAME, tuple_constructor, BUILTIN_FLAGSEMPTY);
    value tupleclass=builtin_addclass(TUPLE_CLASSNAME, MORPHO_GETCLASSDEFINITION(Tuple), objclass);
    object_setveneerclass(OBJECT_TUPLE, tupleclass);
}
