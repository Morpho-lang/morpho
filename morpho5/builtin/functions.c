/** @file functions.c
 *  @author T J Atherton
 *
 *  @brief Built in functions
 */

#include <time.h>
#include <stdlib.h>
#include "functions.h"
#include "random.h"
#include "builtin.h"
#include "common.h"

/* **********************************************************************
 * Built in functions
 * ********************************************************************** */

/* ************************************
 * Math
 * *************************************/

#define BUILTIN_MATH(function) \
value builtin_##function(vm *v, int nargs, value *args) { \
        if (nargs==1) { \
            value arg = MORPHO_GETARG(args, 0); \
            if (MORPHO_ISFLOAT(arg)) { \
                return MORPHO_FLOAT(function(MORPHO_GETFLOATVALUE(arg))); \
            } else if (MORPHO_ISINTEGER(arg)) { \
                return MORPHO_FLOAT(function((double) MORPHO_GETINTEGERVALUE(arg))); \
            } else { \
                morpho_runtimeerror(v, MATH_ARGS, #function);\
            } \
        } \
        morpho_runtimeerror(v, MATH_NUMARGS, #function);\
        return MORPHO_NIL; \
        }

/** Math functions */
BUILTIN_MATH(fabs)
BUILTIN_MATH(exp)
BUILTIN_MATH(log)
BUILTIN_MATH(log10)

BUILTIN_MATH(sin)
BUILTIN_MATH(cos)
BUILTIN_MATH(tan)
BUILTIN_MATH(asin)
BUILTIN_MATH(acos)

BUILTIN_MATH(sinh)
BUILTIN_MATH(cosh)
BUILTIN_MATH(tanh)
BUILTIN_MATH(sqrt)

BUILTIN_MATH(floor)
BUILTIN_MATH(ceil)

BUILTIN_MATH(isfinite)
BUILTIN_MATH(isinf)
BUILTIN_MATH(isnan)

#undef BUILTIN_MATH

/** The arctan function is special; it can either take one or two arguments */
value builtin_arctan(vm *v, int nargs, value *args) {
    double x[2];
    for (unsigned int i=0; i<nargs; i++) {
        if (!morpho_valuetofloat(MORPHO_GETARG(args, i), x+i)) {
            morpho_runtimeerror(v, MATH_ARGS, "arctan");
            return MORPHO_NIL;
        }
    }
    
    if (nargs==1) {
        return MORPHO_FLOAT(atan(x[0]));
    } else if (nargs==2) {
        return MORPHO_FLOAT(atan2(x[1], x[0])); // Note Morpho uses the opposite order to C!
    }
        
    morpho_runtimeerror(v, MATH_NUMARGS, "#function");
    return MORPHO_NIL;
}

/* ************************************
 * Random numbers
 * *************************************/

/** Generate a random float between 0 and 1 */
value builtin_random(vm *v, int nargs, value *args) {
    return MORPHO_FLOAT(random_double());
}

/** Generate a random normally distributed number */
value builtin_randomnormal(vm *v, int nargs, value *args) {
    double x,y,r;

    do {
        x=2.0*random_double()-1.0;
        y=2.0*random_double()-1.0;
      
        r=x*x+y*y;
    } while (r>=1.0);
    
    return MORPHO_FLOAT(x*sqrt((-2.0*log(r))/r));
}

/** Generate a random integer with a bound.
 Efficient and unbiased algorithm from: https://www.pcg-random.org/posts/bounded-rands.html */
value builtin_randomint(vm *v, int nargs, value *args) {
    uint32_t x = random_int();
    /* Leave quickly if no range was asked for */
    if (nargs==0) return MORPHO_INTEGER((int) x);
    
    /* Otherwise, generate a number in range. */
    int r=0;
    if (!morpho_valuetoint(MORPHO_GETARG(args, 0), &r)||r<0) {

        morpho_runtimeerror(v, VM_INVALIDARGSDETAIL,FUNCTION_RANDOMINT, 1, "positive integer");
    }
    
    uint32_t range=(uint32_t) r;
    uint64_t m = (uint64_t) x  * (uint64_t) range;
    uint32_t l = (uint32_t) m;
    
    if (l < range) {
        uint32_t t = -range;
        if (t >= range) {
            t -= range;
            if (t >= range)
                t %= range;
        }
        while (l < t) {
            x = random_int();
            m = (uint64_t) x * (uint64_t) range;
            l = (uint32_t) m;
        }
    }
    return MORPHO_INTEGER(m >> 32);
}

/* ************************************
 * Type checking and conversion
 * *************************************/

/*value builtin_isint(vm *v, int nargs, value *args) {
    if (nargs==1) {
        return MORPHO_BOOL(MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)));
    } else morpho_runtimeerror(v, TYPE_NUMARGS, FUNCTION_INT);
    
    return MORPHO_NIL;
}*/

/** Typecheck functions to test for the type of a quantity */
#define BUILTIN_TYPECHECK(type, test) \
    value builtin_##type(vm *v, int nargs, value *args) { \
        if (nargs==1) { \
            return MORPHO_BOOL(test(MORPHO_GETARG(args, 0))); \
            } else morpho_runtimeerror(v, TYPE_NUMARGS, #type); \
        \
        return MORPHO_NIL; \
    }
    

BUILTIN_TYPECHECK(isnil, MORPHO_ISNIL)
BUILTIN_TYPECHECK(isint, MORPHO_ISINTEGER)
BUILTIN_TYPECHECK(isfloat, MORPHO_ISFLOAT)
BUILTIN_TYPECHECK(isnumber, MORPHO_ISNUMBER)
BUILTIN_TYPECHECK(isbool, MORPHO_ISBOOL)
BUILTIN_TYPECHECK(isobject, MORPHO_ISOBJECT)
BUILTIN_TYPECHECK(isstring, MORPHO_ISSTRING)
BUILTIN_TYPECHECK(isclass, MORPHO_ISCLASS)
BUILTIN_TYPECHECK(isrange, MORPHO_ISRANGE)
BUILTIN_TYPECHECK(isdictionary, MORPHO_ISDICTIONARY)
BUILTIN_TYPECHECK(islist, MORPHO_ISLIST)
BUILTIN_TYPECHECK(isarray, MORPHO_ISARRAY)
BUILTIN_TYPECHECK(ismatrix, MORPHO_ISMATRIX)
BUILTIN_TYPECHECK(issparse, MORPHO_ISSPARSE)
BUILTIN_TYPECHECK(ismesh, MORPHO_ISMESH)
BUILTIN_TYPECHECK(isselection, MORPHO_ISSELECTION)
BUILTIN_TYPECHECK(isfield, MORPHO_ISFIELD)

#undef BUILTIN_TYPECHECK

/** Check if something is callable */
value builtin_iscallablefunction(vm *v, int nargs, value *args) {
    if (nargs==1) {
        if (builtin_iscallable(MORPHO_GETARG(args, 0))) return MORPHO_TRUE;
    } else morpho_runtimeerror(v, TYPE_NUMARGS, FUNCTION_ISCALLABLE);
    return MORPHO_FALSE;
}

/** Convert something to an integer */
value builtin_int(vm *v, int nargs, value *args) {
    if (nargs==1) {
        if (MORPHO_ISFLOAT(MORPHO_GETARG(args, 0))) {
            return MORPHO_FLOATTOINTEGER(MORPHO_GETARG(args, 0));
        } else if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            return MORPHO_GETARG(args, 0);
        }
    }
    morpho_runtimeerror(v, MATH_NUMARGS, FUNCTION_INT);
    return MORPHO_NIL;
}

/** Convert to a floating point number */
value builtin_float(vm *v, int nargs, value *args) {
    if (nargs==1) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            return MORPHO_INTEGERTOFLOAT(MORPHO_GETARG(args, 0));
        } else {
            return MORPHO_GETARG(args, 0);
        }
    }
    morpho_runtimeerror(v, MATH_NUMARGS, FUNCTION_FLOAT);
    return MORPHO_NIL;
}

/** Convert to a boolean */
value builtin_bool(vm *v, int nargs, value *args) {
    if (nargs==1) {
        return MORPHO_BOOL(MORPHO_ISTRUE(MORPHO_GETARG(args, 0)));
    }
    morpho_runtimeerror(v, MATH_NUMARGS, FUNCTION_BOOL);
    return MORPHO_NIL;
}

/** Remainder */
value builtin_mod(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    if (nargs==2) {
        value a = MORPHO_GETARG(args, 0);
        value b = MORPHO_GETARG(args, 1);
        
        if (MORPHO_ISINTEGER(a) && MORPHO_ISINTEGER(b)) {
            out=MORPHO_INTEGER(MORPHO_GETINTEGERVALUE(a) % MORPHO_GETINTEGERVALUE(b));
        } else {
            if (MORPHO_ISINTEGER(a)) a=MORPHO_INTEGERTOFLOAT(a);
            if (MORPHO_ISINTEGER(b)) b=MORPHO_INTEGERTOFLOAT(b);
            
            if (MORPHO_ISFLOAT(a) && MORPHO_ISFLOAT(b)) {
                out=MORPHO_FLOAT(fmod(MORPHO_GETFLOATVALUE(a), MORPHO_GETFLOATVALUE(b)));
            } else morpho_runtimeerror(v, MATH_NUMARGS, FUNCTION_INT);
        }
    } else morpho_runtimeerror(v, VM_INVALIDARGS, 2, nargs);
    return out;
}


/** Find the minimum and maximum values in an enumerable object */
typedef struct {
    value min;
    value max;
} minmaxstruct;

static bool minmaxfn(vm *v, indx i, value val, void *ref) {
    minmaxstruct *m=(minmaxstruct *) ref;
    
    if (i==0 || morpho_comparevalue(m->min, val)<0) m->min=val;
    if (i==0 || morpho_comparevalue(m->max, val)>0) m->max=val;
    
    return true;
}

static bool builtin_minmax(vm *v, value obj, value *min, value *max) {
    minmaxstruct m;
    // intialize the minmaxstuct
    m.max = MORPHO_NIL;
    m.min = MORPHO_NIL;

    if (!builtin_enumerateloop(v, obj, minmaxfn, &m)) return false;
        
    if (min) *min = m.min;
    if (max) *max = m.max;
    
    return true;
}


/** Find the minimum and maximum values in an enumerable object */
static value builtin_bounds(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    
    if (nargs==1) {
        value bounds[2];
        
        if (builtin_minmax(v, MORPHO_GETARG(args, 0), &bounds[0], &bounds[1])) {
            objectlist *list = object_newlist(2, bounds);
            if (list) {
                out = MORPHO_OBJECT(list);
                morpho_bindobjects(v, 1, &out);
            } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        }
    } else morpho_runtimeerror(v, VM_INVALIDARGS, 1, nargs);
    
    return out;
}

/** Find the minimum value in an enumerable object */
static value builtin_min(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    // ensure we have 1 argument and that it is a list or a matrix
    if (nargs==1 && (MORPHO_ISLIST(MORPHO_GETARG(args, 0)) || \
                    MORPHO_ISMATRIX(MORPHO_GETARG(args, 0)))) {
            builtin_minmax(v, MORPHO_GETARG(args, 0), &out, NULL);
    } else morpho_runtimeerror(v, VM_INVALIDARGSDETAIL,FUNCTION_MIN, 1, "list or matrix");
    
    return out;
}

/** Find the maximum value in an enumerable object */
static value builtin_max(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    // ensure we have 1 argument and that it is a list or a matrix
    if (nargs==1 && (MORPHO_ISLIST(MORPHO_GETARG(args, 0)) || \
                    MORPHO_ISMATRIX(MORPHO_GETARG(args, 0)))) {
        builtin_minmax(v, MORPHO_GETARG(args, 0), NULL, &out);
    } else morpho_runtimeerror(v, VM_INVALIDARGSDETAIL,FUNCTION_MAX, 1, "list or matrix");
    
    return out;
}

/* ************************************
 * Apply
 * *************************************/

/** Apply a function to a list of arguments */
value builtin_apply(vm *v, int nargs, value *args) {
    value ret = MORPHO_NIL;
    morpho_call(v, MORPHO_GETARG(args, 0), nargs-1, &MORPHO_GETARG(args, 1), &ret);
    
    return ret;
}

/* ************************************
 * System
 * *************************************/

/** Call the operating system */
value builtin_system(vm *v, int nargs, value *args) {
    if (nargs==1) {
        value arg=MORPHO_GETARG(args, 0);
        if (MORPHO_ISSTRING(arg)) {
            return MORPHO_INTEGER(system(MORPHO_GETCSTRING(arg)));
        }
    }
    return MORPHO_NIL;
}

/** Clock */
value builtin_clock(vm *v, int nargs, value *args) {
    clock_t time;
    time = clock();
    return MORPHO_FLOAT( ((double) time)/((double) CLOCKS_PER_SEC) );
}

#define BUILTIN_MATH(function) \
    builtin_addfunction(#function, builtin_##function, BUILTIN_FLAGSEMPTY);

#define BUILTIN_TYPECHECK(function) \
    builtin_addfunction(#function, builtin_##function, BUILTIN_FLAGSEMPTY);

void functions_initialize(void) {
    builtin_addfunction(FUNCTION_CLOCK, builtin_clock, BUILTIN_FLAGSEMPTY);
    builtin_addfunction(FUNCTION_RANDOM, builtin_random, BUILTIN_FLAGSEMPTY);
    builtin_addfunction(FUNCTION_RANDOMINT, builtin_randomint, BUILTIN_FLAGSEMPTY);
    builtin_addfunction(FUNCTION_RANDOMNORMAL, builtin_randomnormal, BUILTIN_FLAGSEMPTY);
    
    builtin_addfunction(FUNCTION_SYSTEM, builtin_system, BUILTIN_FLAGSEMPTY);
    builtin_addfunction(FUNCTION_ARCTAN, builtin_arctan, BUILTIN_FLAGSEMPTY);
    
    builtin_addfunction(FUNCTION_ABS, builtin_fabs, BUILTIN_FLAGSEMPTY);
    
    BUILTIN_MATH(exp)
    BUILTIN_MATH(log)
    BUILTIN_MATH(log10)

    BUILTIN_MATH(sin)
    BUILTIN_MATH(cos)
    BUILTIN_MATH(tan)
    BUILTIN_MATH(asin)
    BUILTIN_MATH(acos)

    BUILTIN_MATH(sinh)
    BUILTIN_MATH(cosh)
    BUILTIN_MATH(tanh)
    BUILTIN_MATH(sqrt)

    BUILTIN_MATH(floor)
    BUILTIN_MATH(ceil)

    BUILTIN_MATH(isfinite)
    BUILTIN_MATH(isinf)
    BUILTIN_MATH(isnan)

    
    BUILTIN_TYPECHECK(isnil)
    BUILTIN_TYPECHECK(isint)
    BUILTIN_TYPECHECK(isfloat)
    BUILTIN_TYPECHECK(isnumber)
    BUILTIN_TYPECHECK(isbool)
    BUILTIN_TYPECHECK(isobject)
    BUILTIN_TYPECHECK(isstring)
    BUILTIN_TYPECHECK(isclass)
    BUILTIN_TYPECHECK(isrange)
    BUILTIN_TYPECHECK(isdictionary)
    BUILTIN_TYPECHECK(islist)
    BUILTIN_TYPECHECK(isarray)
    BUILTIN_TYPECHECK(ismatrix)
    BUILTIN_TYPECHECK(issparse)
    BUILTIN_TYPECHECK(ismesh)
    BUILTIN_TYPECHECK(isselection)
    BUILTIN_TYPECHECK(isfield)
    
    builtin_addfunction(FUNCTION_ISCALLABLE, builtin_iscallablefunction, BUILTIN_FLAGSEMPTY);
    
    builtin_addfunction(FUNCTION_INT, builtin_int, BUILTIN_FLAGSEMPTY);
    builtin_addfunction(FUNCTION_FLOAT, builtin_float, BUILTIN_FLAGSEMPTY);
    builtin_addfunction(FUNCTION_BOOL, builtin_bool, BUILTIN_FLAGSEMPTY);
    
    builtin_addfunction(FUNCTION_MOD, builtin_mod, BUILTIN_FLAGSEMPTY);
    
    builtin_addfunction(FUNCTION_BOUNDS, builtin_bounds, BUILTIN_FLAGSEMPTY);
    builtin_addfunction(FUNCTION_MIN, builtin_min, BUILTIN_FLAGSEMPTY);
    builtin_addfunction(FUNCTION_MAX, builtin_max, BUILTIN_FLAGSEMPTY);
    
    builtin_addfunction(FUNCTION_APPLY, builtin_apply, BUILTIN_FLAGSEMPTY);
    
    morpho_defineerror(MATH_ARGS, ERROR_HALT, MATH_ARGS_MSG);
    morpho_defineerror(MATH_NUMARGS, ERROR_HALT, MATH_NUMARGS_MSG);
    morpho_defineerror(MATH_ATANARGS, ERROR_HALT, MATH_ATANARGS_MSG);
    morpho_defineerror(TYPE_NUMARGS, ERROR_HALT, TYPE_NUMARGS_MSG);   
}

#undef BUILTIN_MATH
