/** @file range.c
 *  @author T J Atherton
 *
 *  @brief Implements the Range class
 */

#include "morpho.h"
#include "object.h"
#include "builtin.h"
#include "classes.h"

/* **********************************************************************
 * objectrange definitions
 * ********************************************************************** */

void objectrange_printfn(object *obj) {
    objectrange *r = (objectrange *) obj;
    morpho_printvalue(r->start);
    printf("..");
    morpho_printvalue(r->end);
    if (!MORPHO_ISNIL(r->step)) {
        printf(":");
        morpho_printvalue(r->step);
    }
}

size_t objectrange_sizefn(object *obj) {
    return sizeof(objectrange);
}

objecttypedefn objectrangedefn = {
    .printfn=objectrange_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectrange_sizefn
};

/** Create a new range. Step may be set to MORPHO_NIL to use the default value of 1 */
objectrange *object_newrange(value start, value end, value step) {
    value v[3]={start, end, step};

    /* Ensure all three values are either integer or floating point */
    if (!value_promotenumberlist((MORPHO_ISNIL(step) ? 2 : 3), v)) return NULL;

    objectrange *new = (objectrange *) object_new(sizeof(objectrange), OBJECT_RANGE);

    if (new) {
        new->start=v[0];
        new->end=v[1];
        new->step=v[2];
        new->nsteps=range_count(new);
    }

    return new;
}

/* **********************************************************************
 * objectrange utility functions
 * ********************************************************************** */

/** Calculate the number of steps in a range */
int range_count(objectrange *range) {
    int out=0;
    if (MORPHO_ISFLOAT(range->start)) {
        double diff=MORPHO_GETFLOATVALUE(range->end)-MORPHO_GETFLOATVALUE(range->start);
        double stp=(MORPHO_ISNIL(range->step) ? 1 : MORPHO_GETFLOATVALUE(range->step));
        double cnt = ceil(diff / stp);
        if (isfinite(cnt)) out = cnt + (fabs(cnt * stp - diff) <= DBL_EPSILON);
    } else {
        int diff=MORPHO_GETINTEGERVALUE(range->end)-MORPHO_GETINTEGERVALUE(range->start);
        int stp=(MORPHO_ISNIL(range->step) ? 1 : MORPHO_GETINTEGERVALUE(range->step));
        if (stp != 0) out = diff / stp + 1;
    }
    if (out < 0) out=0;
    return out;
}

/** Find the ith value of a range object */
value range_iterate(objectrange *range, unsigned int i) {
    if (MORPHO_ISFLOAT(range->start)) {
        return MORPHO_FLOAT( MORPHO_GETFLOATVALUE(range->start) +
                            i*(MORPHO_ISNIL(range->step) ? 1.0 : MORPHO_GETFLOATVALUE(range->step)));
    } else {
        return MORPHO_INTEGER( MORPHO_GETINTEGERVALUE(range->start) +
                            i*(MORPHO_ISNIL(range->step) ? 1 : MORPHO_GETINTEGERVALUE(range->step)));
    }
}

/* **********************************************************************
 * Range veneer class
 * ********************************************************************** */

/** Constructor function for ranges */
value range_constructor(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectrange *new=NULL;

    /* Check args are numerical */
    for (unsigned int i=0; i<nargs; i++) {
        if (!(MORPHO_ISINTEGER(MORPHO_GETARG(args, i)) || MORPHO_ISFLOAT(MORPHO_GETARG(args, i)))) {
            MORPHO_RAISE(v, RANGE_ARGS);
        }
    }

    if (nargs==2) {
        new=object_newrange(MORPHO_GETARG(args, 0), MORPHO_GETARG(args, 1), MORPHO_NIL);
    } else if (nargs==3) {
        new=object_newrange(MORPHO_GETARG(args, 0), MORPHO_GETARG(args, 1), MORPHO_GETARG(args, 2));
    } else MORPHO_RAISE(v, RANGE_ARGS);

    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }

    return out;
}

/** Gets a specified element from a range */
value Range_getindex(vm *v, int nargs, value *args) {
    objectrange *slf = MORPHO_GETRANGE(MORPHO_SELF(args));

    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

        if (n<slf->nsteps) return range_iterate(slf, n);
        else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
    }

    return MORPHO_SELF(args);
}

/** Enumerate members of a range */
value Range_enumerate(vm *v, int nargs, value *args) {
    objectrange *slf = MORPHO_GETRANGE(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

        if (n<0) return MORPHO_INTEGER(slf->nsteps);
        else return range_iterate(slf, n);
    } else MORPHO_RAISE(v, ENUMERATE_ARGS);

    return out;
}

/** Count number of items in a range */
value Range_count(vm *v, int nargs, value *args) {
    objectrange *slf = MORPHO_GETRANGE(MORPHO_SELF(args));

    return MORPHO_INTEGER(slf->nsteps);
}

/** Clones a range */
value Range_clone(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    objectrange *slf = MORPHO_GETRANGE(MORPHO_SELF(args));
    objectrange *new = object_newrange(slf->start, slf->end, slf->step);

    if (new) {
        out = MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

MORPHO_BEGINCLASS(Range)
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, Range_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, Range_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Range_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Range_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

objecttype objectrangetype;

void range_initialize(void) {
    // Create range object type
    objectrangetype=object_addtype(&objectrangedefn);
    
    // Locate the Object class to use as the parent class of Range
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // Create range veneer class
    builtin_addfunction(RANGE_CLASSNAME, range_constructor, BUILTIN_FLAGSEMPTY);
    value rangeclass=builtin_addclass(RANGE_CLASSNAME, MORPHO_GETCLASSDEFINITION(Range), objclass);
    object_setveneerclass(OBJECT_RANGE, rangeclass);
    
    // Range error messages
    morpho_defineerror(RANGE_ARGS, ERROR_HALT, RANGE_ARGS_MSG);
}

void range_finalize(void) {
}
