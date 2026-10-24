/** @file range.c
 *  @author T J Atherton
 *
 *  @brief Implements the Range class
 */

#include <float.h>
#include <limits.h>

#include "morpho.h"
#include "classes.h"

/* **********************************************************************
 * objectrange definitions
 * ********************************************************************** */

void objectrange_printfn(object *obj, void *v) {
    objectrange *r = (objectrange *) obj;
    morpho_printvalue(v, r->start);
    morpho_printf(v, (r->inclusive ? ".." : "..."));
    morpho_printvalue(v, r->end);
    if (!MORPHO_ISNIL(r->step)) {
        morpho_printf(v, ":");
        morpho_printvalue(v, r->step);
    }
}

size_t objectrange_sizefn(object *obj) {
    return sizeof(objectrange);
}

objecttypedefn objectrangedefn = {
    .printfn=objectrange_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectrange_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/** Determine the number of steps in a range */
bool _range_count(objectrange *range) {
    double start, end, step=1.0;

    morpho_valuetofloat(range->start, &start);
    morpho_valuetofloat(range->end, &end);
    if (!MORPHO_ISNIL(range->step)) morpho_valuetofloat(range->step, &step);

    if (step==0.0) return false;

    double q=(end-start)/step;
    if (!isfinite(q)) return false;

    if (q<0.0) {
        range->nsteps=0;
        return true;
    }

    /* Snap a quotient that is numerically an integer onto that integer. */
    double nearest=round(q);
    if (MORPHO_ISEQUAL(MORPHO_FLOAT(q), MORPHO_FLOAT(nearest))) q=nearest;

    double nsteps=(range->inclusive ? floor(q)+1.0 : ceil(q));
    if (nsteps>(double) INT_MAX) return false;

    range->nsteps=(unsigned int) nsteps;
    return true;
}

/** Create a new range. Step may be set to MORPHO_NIL to use the default value of 1.
    @param[out] errid - errid is filled in if the range can't be initialized */
objectrange *object_newrange(value start, value end, value step, bool inclusive, errorid *errid) {
    value v[3]={start, end, step};

    /* Ensure all three values are either integer or floating point */
    if (!value_promotenumberlist((MORPHO_ISNIL(step) ? 2 : 3), v)) {
        *errid = RANGE_ARGS;
        return NULL;
    }

    objectrange *new = (objectrange *) object_new(sizeof(objectrange), OBJECT_RANGE);

    if (new) {
        new->start=v[0];
        new->end=v[1];
        new->step=v[2];
        new->inclusive=inclusive;
        if (!_range_count(new)) {
            *errid = RANGE_STPSZ;
            object_free((object *) new);
            new=NULL;
        }
    } else *errid = ERROR_ALLOCATIONFAILED;

    return new;
}

/* **********************************************************************
 * objectrange utility functions
 * ********************************************************************** */

/** Return the number of steps in a range */
int range_count(objectrange *range) {
    return range->nsteps;
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

static value _rangeconstructor(vm *v, int nargs, value *args, bool inclusive) {
    value out=MORPHO_NIL;
    objectrange *new=NULL;

    value in[3] = { MORPHO_NIL, MORPHO_NIL, MORPHO_NIL};
    for (int i=0; i<nargs; i++) in[i]=MORPHO_GETARG(args, i);
    
    errorid errid=RANGE_ARGS;
    new=object_newrange(in[0], in[1], in[2], inclusive, &errid);
    
    if (new) {
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, errid);

    return out;
}

/** Constructor functions for ranges */
value range_constructor(vm *v, int nargs, value *args) {
    return _rangeconstructor(v, nargs, args, false);
}

value range_inclusiveconstructor(vm *v, int nargs, value *args) {
    return _rangeconstructor(v, nargs, args, true);
}

/** Default if incorrect args passed */
value range_invldconstructor(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, RANGE_ARGS);
    return MORPHO_NIL;
}

/** Gets a specified element from a range */
value Range_getindex(vm *v, int nargs, value *args) {
    objectrange *slf = MORPHO_GETRANGE(MORPHO_SELF(args));
    int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    
    if (n<0 || n>=slf->nsteps) {
        morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        return MORPHO_NIL;
    }
    
    return range_iterate(slf, n);
}

/** Enumerate members of a range */
value Range_enumerate(vm *v, int nargs, value *args) {
    objectrange *slf = MORPHO_GETRANGE(MORPHO_SELF(args));
    int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    
    if (n<0) return MORPHO_INTEGER(slf->nsteps);
    if (n>=slf->nsteps) {
        morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        return MORPHO_NIL;
    }
    
    return range_iterate(slf, n);
}

/** Iterate protocol */
value Range_iterate(vm *v, int nargs, value *args) {
    objectrange *slf = MORPHO_GETRANGE(MORPHO_SELF(args));
    int i=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    if (i<0) { morpho_runtimeerror(v, VM_INVALIDARGS); return MORPHO_NIL; }
    
    if (i < slf->nsteps) {
        value ret[3] = { MORPHO_INTEGER(i+1), range_iterate(slf, i), MORPHO_INTEGER(i) };
        return morpho_return(v, 3, ret);
    }
    value ret[3] = { MORPHO_NIL, MORPHO_NIL, MORPHO_NIL };
    return morpho_return(v, 3, ret);
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
    errorid errid=RANGE_ARGS;
    objectrange *new = object_newrange(slf->start, slf->end, slf->step, slf->inclusive, &errid);

    if (new) {
        out = MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, errid);
    
    return out;
}

MORPHO_BEGINCLASS(Range)
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, " (Int)", Range_getindex, MORPHO_FN_THROWS),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, MORPHO_FN_IO),
MORPHO_METHOD_SIGNATURE(MORPHO_ENUMERATE_METHOD, " (Int)", Range_enumerate, MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ITERATE_METHOD, " (Int)", Range_iterate, MORPHO_FN_PUREFN|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int ()", Range_count, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, "Range ()", Range_clone, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

objecttype objectrangetype;

void range_initialize(void) {
    // Create range object type
    objectrangetype=object_addtype(&objectrangedefn);
    
    // Locate the Object class to use as the parent class of Range
    value objclass = builtin_findclassfromcstring(OBJECT_CLASSNAME);
    
    // Create range veneer class
    value rangeclass=builtin_addclass(RANGE_CLASSNAME, MORPHO_GETCLASSDEFINITION(Range), objclass);
    object_setveneerclass(OBJECT_RANGE, rangeclass);
    
    // Range constructor function
    morpho_addfunction(RANGE_CLASSNAME, RANGE_CLASSNAME " (_,_)", range_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(RANGE_CLASSNAME, RANGE_CLASSNAME " (_,_,_)", range_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(RANGE_CLASSNAME, RANGE_CLASSNAME " (...)", range_invldconstructor, MORPHO_FN_CONSTRUCTOR, NULL);
    
    // Inclusive range constructor
    morpho_addfunction(RANGE_INCLUSIVE_CONSTRUCTOR, RANGE_CLASSNAME " (_,_)", range_inclusiveconstructor, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(RANGE_INCLUSIVE_CONSTRUCTOR, RANGE_CLASSNAME " (_,_,_)", range_inclusiveconstructor, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(RANGE_INCLUSIVE_CONSTRUCTOR, RANGE_CLASSNAME " (...)", range_invldconstructor, MORPHO_FN_CONSTRUCTOR, NULL);
    
    // Range error messages
    morpho_defineerror(RANGE_ARGS, ERROR_HALT, RANGE_ARGS_MSG);
    morpho_defineerror(RANGE_STPSZ, ERROR_HALT, RANGE_STPSZ_MSG);
}
