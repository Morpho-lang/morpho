/** @file functiondefs.c
 *  @author T J Atherton
 *
 *  @brief Built in function definitions
 */

#include <time.h>
#include <stdlib.h>
#include <complex.h>

#include "functiondefs.h"
#include "random.h"
#include "builtin.h"
#include "common.h"
#include "cmplx.h"

#include "linalg.h"
#include "sparse.h"

#include "mesh.h"
#include "field.h"
#include "selection.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

/* **********************************************************************
 * Built in functions
 * ********************************************************************** */

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

/* ************************************
 * Apply
 * *************************************/

static value builtin_apply__tuple(vm *v, int nargs, value *args) {
    value ret = MORPHO_NIL;
    objecttuple *t = MORPHO_GETTUPLE(MORPHO_GETARG(args, 1));
    morpho_call(v, MORPHO_GETARG(args, 0), tuple_length(t), t->tuple, &ret);
    return ret;
}

static value builtin_apply__list(vm *v, int nargs, value *args) {
    value ret = MORPHO_NIL;
    objectlist *lst = MORPHO_GETLIST(MORPHO_GETARG(args, 1));
    morpho_call(v, MORPHO_GETARG(args, 0), list_length(lst), lst->val.data, &ret);
    return ret;
}

/** Apply a function to a list of arguments */
value builtin_apply(vm *v, int nargs, value *args) {
    value ret = MORPHO_NIL;
    morpho_call(v, MORPHO_GETARG(args, 0), nargs-1, &MORPHO_GETARG(args, 1), &ret);
    return ret;
}

static value builtin_apply__err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, APPLY_ARGS);
    return MORPHO_NIL;
}

/* ************************************
 * Random numbers
 * *************************************/

/** Generate a random float between 0 and 1 */
value builtin_random(vm *v, int nargs, value *args) {
    return MORPHO_FLOAT(random_double());
}

/** Generate a random integer with a bound.
 Efficient and unbiased algorithm from: https://www.pcg-random.org/posts/bounded-rands.html */
value builtin_randomint_norange(vm *v, int nargs, value *args) {
    uint32_t x = random_int();
    return MORPHO_INTEGER((int) x);
}

value builtin_randomint(vm *v, int nargs, value *args) {
    uint32_t x = random_int();
    
    /* Generate a number in range. */
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

/* ************************************
 * Value constructors
 * *************************************/

/** Convert something to an integer */
value builtin_int__int(vm *v, int nargs, value *args) {
    return MORPHO_GETARG(args, 0);
}

value builtin_int__float(vm *v, int nargs, value *args) {
    return MORPHO_FLOATTOINTEGER(MORPHO_GETARG(args, 0));
}

value builtin_int__string(vm *v, int nargs, value *args) {
    value arg = MORPHO_GETARG(args, 0);
    string_tonumber(MORPHO_GETSTRING(arg), &arg);
    if (MORPHO_ISFLOAT(arg)) return MORPHO_FLOATTOINTEGER(arg);
    else if (MORPHO_ISINTEGER(arg)) return arg;
    
    morpho_runtimeerror(v, MATH_NUMARGS, FUNCTION_INT);
    return MORPHO_INTEGER(0);
}

value builtin_int__err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, MATH_NUMARGS, FUNCTION_INT);
    return MORPHO_INTEGER(0);
}

/** Convert to a floating point number */
value builtin_float__int(vm *v, int nargs, value *args) {
    return MORPHO_INTEGERTOFLOAT(MORPHO_GETARG(args, 0));
}

value builtin_float__float(vm *v, int nargs, value *args) {
    return MORPHO_GETARG(args, 0);
}

value builtin_float__string(vm *v, int nargs, value *args) {
    value arg = MORPHO_GETARG(args, 0);
    string_tonumber(MORPHO_GETSTRING(arg), &arg);
    if (MORPHO_ISFLOAT(arg)) return arg;
    else if (MORPHO_ISINTEGER(arg)) return MORPHO_INTEGERTOFLOAT(arg);
    
    morpho_runtimeerror(v, MATH_NUMARGS, FUNCTION_FLOAT);
    return MORPHO_INTEGER(0);
}

value builtin_float__err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, MATH_NUMARGS, FUNCTION_FLOAT);
    return MORPHO_FLOAT(0.0);
}

/** Convert to a boolean */
value builtin_bool(vm *v, int nargs, value *args) {
    return MORPHO_BOOL(MORPHO_ISTRUE(MORPHO_GETARG(args, 0)));
}

value builtin_bool_err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, MATH_NUMARGS, FUNCTION_BOOL);
    return MORPHO_FALSE;
}

/* ************************************
 * Math
 * *************************************/

#define BUILTIN_VARMATH(function, type) \
value builtin_float_##function(vm *v, int nargs, value *args) {                \
    value arg = MORPHO_GETARG(args, 0);                                        \
    return type(function(MORPHO_GETFLOATVALUE(arg)));                          \
}                                                                              \
                                                                               \
value builtin_int_##function(vm *v, int nargs, value *args) {                  \
    value arg = MORPHO_GETARG(args, 0);                                        \
    return type(function((double) MORPHO_GETINTEGERVALUE(arg)));               \
}                                                                              \
                                                                               \
value builtin_cmplx_##function(vm *v, int nargs, value *args) {                \
    value arg = MORPHO_GETARG(args, 0);                                        \
    return complex_builtin##function(v, MORPHO_GETCOMPLEX(arg));               \
}                                                                              \
                                                                               \
value builtin_numargserr_##function(vm *v, int nargs, value *args) {           \
    morpho_runtimeerror(v, MATH_NUMARGS, #function);                           \
    return MORPHO_NIL;                                                         \
}                                                                              \

#define BUILTIN_MATH(function) BUILTIN_VARMATH(function, MORPHO_FLOAT)

#define BUILTIN_MATH_BOOL(function) BUILTIN_VARMATH(function, MORPHO_BOOL)

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

BUILTIN_MATH(floor)
BUILTIN_MATH(ceil)

BUILTIN_MATH_BOOL(isfinite)
BUILTIN_MATH_BOOL(isinf)
BUILTIN_MATH_BOOL(isnan)

#undef BUILTIN_VARMATH
#undef BUILTIN_MATH_BOOL
#undef BUILTIN_MATH

/* ************************************
 * Math functions with special cases
 * *************************************/

/** The sqrt function is needs to be able to return a complex number for negative arguments */
value builtin_sqrt(vm *v, int nargs, value *args) { 
    if (nargs==1) { 
        value arg = MORPHO_GETARG(args, 0); 
        if (MORPHO_ISCOMPLEX(arg)){
            return complex_builtinsqrt(v,MORPHO_GETCOMPLEX(arg));
        }
        else if (MORPHO_ISNUMBER(arg)) {
            double val;
            if (morpho_valuetofloat(arg,&val)) {
                if (val<0) {// need to use complex sqrt
                    objectcomplex C = MORPHO_STATICCOMPLEX(val, 0);
                    return complex_builtinsqrt(v, &C);
                } else { 
                    return MORPHO_FLOAT(sqrt(val));
                }
            } else morpho_runtimeerror(v, MATH_ARGS, "sqrt"); 
        } else morpho_runtimeerror(v, MATH_ARGS, "sqrt"); 
    }
    morpho_runtimeerror(v, MATH_NUMARGS, "sqrt");
    return MORPHO_NIL; 
}

/** The arctan function is special; it can either take one or two arguments */
static value builtin_arctan__number(vm *v, int nargs, value *args) {
    double x;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &x);
    return MORPHO_FLOAT(atan(x));
}

static value builtin_arctan__number_number(vm *v, int nargs, value *args) {
    double x, y;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &x);
    morpho_valuetofloat(MORPHO_GETARG(args, 1), &y);
    return MORPHO_FLOAT(atan2(y, x)); // Note Morpho uses the opposite order to C!
}

static value builtin_arctan__complex(vm *v, int nargs, value *args) {
    return complex_builtinatan(v, MORPHO_GETARG(args, 0));
}

static value builtin_arctan__complex2(vm *v, int nargs, value *args) {
    return complex_builtinatan2(v, MORPHO_GETARG(args, 1), MORPHO_GETARG(args, 0));
}

static value builtin_arctan__err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, MATH_NUMARGS, "arctan");
    return MORPHO_NIL;
}

/** Remainder */
value builtin_mod__int_int(vm *v, int nargs, value *args) {
    value a = MORPHO_GETARG(args, 0);
    value b = MORPHO_GETARG(args, 1);
    return MORPHO_INTEGER(MORPHO_GETINTEGERVALUE(a) % MORPHO_GETINTEGERVALUE(b));
}

value builtin_mod__float_float(vm *v, int nargs, value *args) {
    value a = MORPHO_GETARG(args, 0);
    value b = MORPHO_GETARG(args, 1);
    return MORPHO_FLOAT(fmod(MORPHO_GETFLOATVALUE(a), MORPHO_GETFLOATVALUE(b)));
}

value builtin_mod__int_float(vm *v, int nargs, value *args) {
    return MORPHO_FLOAT(fmod((double) MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                             MORPHO_GETFLOATVALUE(MORPHO_GETARG(args, 1))));
}

value builtin_mod__float_int(vm *v, int nargs, value *args) {
    return MORPHO_FLOAT(fmod(MORPHO_GETFLOATVALUE(MORPHO_GETARG(args, 0)),
                             (double) MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1))));
}

value builtin_mod__err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, VM_INVALIDARGS, 2, nargs);
    return MORPHO_NIL;
}

/** find the sign of a number */
static value builtin_sign__value(vm *v, int nargs, value *args){
    double val;
    if (!morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
        morpho_runtimeerror(v, MATH_ARGS, FUNCTION_SIGN);
        return MORPHO_NIL;
    }

    if (val>0) return MORPHO_FLOAT(1);
    if (val<0) return MORPHO_FLOAT(-1);
    return MORPHO_FLOAT(0);
}

static value builtin_sign__float(vm *v, int nargs, value *args){
    return builtin_sign__value(v, nargs, args);
}

static value builtin_sign__int(vm *v, int nargs, value *args){
    return builtin_sign__value(v, nargs, args);
}

static value builtin_sign__err(vm *v, int nargs, value *args){
    morpho_runtimeerror(v, MATH_NUMARGS, FUNCTION_SIGN);
    return MORPHO_NIL;
}

/* ************************************
 * Elementary complex functions
 * *************************************/

value builtin_real__number(vm *v, int nargs, value *args) {
    return MORPHO_GETARG(args, 0);
}

value builtin_real__complex(vm *v, int nargs, value *args) {
    objectcomplex *c=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
    double val;
    complex_getreal(c,&val);
    return MORPHO_FLOAT(val);
}

value builtin_imag__number(vm *v, int nargs, value *args) {
    return MORPHO_FLOAT(0);
}

value builtin_imag__complex(vm *v, int nargs, value *args) {
    objectcomplex *c=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
    double val;
    complex_getimag(c,&val);
    return MORPHO_FLOAT(val);
}

value builtin_angle__number(vm *v, int nargs, value *args) {
    double val;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &val);
    if (val>=0) return MORPHO_FLOAT(0);
    return MORPHO_FLOAT(M_PI);
}

value builtin_angle__complex(vm *v, int nargs, value *args) {
    objectcomplex *c=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
    double val;
    complex_angle(c,&val);
    return MORPHO_FLOAT(val);
}

value builtin_conj__number(vm *v, int nargs, value *args) {
    return MORPHO_GETARG(args, 0);
}

value builtin_conj__complex(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
    value out=MORPHO_NIL;
    objectcomplex *new = object_newcomplex(0,0);
    if (new) {
        complex_conj(a, new);
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    return out;
}

/* ************************************
 * Min/max
 * *************************************/

/** Find the minimum and maximum values in an enumerable object */
typedef struct {
    value min;
    value max;
} minmaxstruct;

static bool minmaxfn(vm *v, indx i, value val, void *ref) {
    minmaxstruct *m=(minmaxstruct *) ref;
    value l=m->min, r=val;
    if (i==0 || morpho_extendedcomparevalue(l, r)<0) m->min=val;
    
    l=m->max; r=val;
    if (i==0 || morpho_extendedcomparevalue(l, r)>0) m->max=val;
    
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

bool builtin_minmaxargs(vm *v, int nargs, value *args, value *min, value *max, char *fname) {
    for (unsigned int i=0; i<nargs; i++) {
        value arg = MORPHO_GETARG(args, i);
        if (MORPHO_ISOBJECT(arg)) {
            if (!builtin_minmax(v, arg, (min ? &min[i] : NULL), (max ? &max[i]: NULL))) return false;
        } else if (morpho_isnumber(arg)) {
            if (min) min[i]=arg;
            if (max) max[i]=arg;
        } else {
            morpho_runtimeerror(v, MAX_ARGS, fname);
            return false;
        }
    }
    return true;
}

/** Find the minimum and maximum values in an enumerable object */
static value builtin_bounds(vm *v, int nargs, value *args) {
    value minlist[nargs+1],maxlist[nargs+1];
    value out = MORPHO_NIL;
    
    if (builtin_minmaxargs(v, nargs, args, minlist, maxlist, FUNCTION_BOUNDS)) {
        if (nargs>0) {
            value bounds[2];
            value_minmax(nargs, minlist, &bounds[0], NULL);
            value_minmax(nargs, maxlist, NULL, &bounds[1]);
            
            objectlist *list = object_newlist(2, bounds);
            if (list) {
                out = MORPHO_OBJECT(list);
                morpho_bindobjects(v, 1, &out);
            } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
            
        } else morpho_runtimeerror(v, MAX_ARGS, FUNCTION_BOUNDS);
    }
    
    return out;
}

static value builtin_bounds__err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, MAX_ARGS, FUNCTION_BOUNDS);
    return MORPHO_NIL;
}

/** Find the minimum value in an enumerable object */
static value builtin_min(vm *v, int nargs, value *args) {
    value m[nargs+1];
    value out = MORPHO_NIL;
    
    if (builtin_minmaxargs(v, nargs, args, m, NULL, FUNCTION_MIN)) {
        value_minmax(nargs, m, &out, NULL);
    }
    
    return out;
}

static value builtin_min__err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, MAX_ARGS, FUNCTION_MIN);
    return MORPHO_NIL;
}

/** Find the maximum value in an enumerable object */
static value builtin_max(vm *v, int nargs, value *args) {
    value m[nargs+1];
    value out = MORPHO_NIL;
    
    if (builtin_minmaxargs(v, nargs, args, NULL, m, FUNCTION_MAX)) {
        value_minmax(nargs, m, NULL, &out);
    }
    
    return out;
}

static value builtin_max__err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, MAX_ARGS, FUNCTION_MAX);
    return MORPHO_NIL;
}

/* ************************************
 * Type checking and conversion
 * *************************************/

/** Typecheck functions to test for the type of a quantity */

#define BUILTIN_TYPECHECK(type, test) \
value builtin_##type(vm *v, int nargs, value *args) {                          \
    return MORPHO_BOOL(test(MORPHO_GETARG(args, 0)));                          \
}                                                                              \
                                                                               \
value builtin_numargserr_##type(vm *v, int nargs, value *args) {               \
    morpho_runtimeerror(v, TYPE_NUMARGS, #type);                               \
    return MORPHO_FALSE;                                                       \
}                                                                              \
    
BUILTIN_TYPECHECK(isnil, MORPHO_ISNIL)
BUILTIN_TYPECHECK(isint, MORPHO_ISINTEGER)
BUILTIN_TYPECHECK(isfloat, MORPHO_ISFLOAT)
BUILTIN_TYPECHECK(isnumber, MORPHO_ISNUMBER)
BUILTIN_TYPECHECK(isarray, MORPHO_ISARRAY)
BUILTIN_TYPECHECK(isbool, MORPHO_ISBOOL)
BUILTIN_TYPECHECK(isclass, MORPHO_ISCLASS)
BUILTIN_TYPECHECK(isclosure, MORPHO_ISCLOSURE)
BUILTIN_TYPECHECK(iscomplex, MORPHO_ISCOMPLEX)
BUILTIN_TYPECHECK(isdictionary, MORPHO_ISDICTIONARY)
BUILTIN_TYPECHECK(isobject, MORPHO_ISOBJECT)
BUILTIN_TYPECHECK(isstring, MORPHO_ISSTRING)
BUILTIN_TYPECHECK(isrange, MORPHO_ISRANGE)
BUILTIN_TYPECHECK(islist, MORPHO_ISLIST)
BUILTIN_TYPECHECK(istuple, MORPHO_ISTUPLE)

#ifdef MORPHO_INCLUDE_LINALG
BUILTIN_TYPECHECK(ismatrix, MORPHO_ISMATRIX)
#endif

#ifdef MORPHO_INCLUDE_SPARSE
BUILTIN_TYPECHECK(issparse, MORPHO_ISSPARSE)
#endif

#ifdef MORPHO_INCLUDE_GEOMETRY
BUILTIN_TYPECHECK(ismesh, MORPHO_ISMESH)
BUILTIN_TYPECHECK(isselection, MORPHO_ISSELECTION)
BUILTIN_TYPECHECK(isfield, MORPHO_ISFIELD)
#endif

#undef BUILTIN_TYPECHECK

/** Check if something is callable */
value builtin_iscallablefunction(vm *v, int nargs, value *args) {
    if (MORPHO_ISCALLABLE(MORPHO_GETARG(args, 0))) return MORPHO_TRUE;
    return MORPHO_FALSE;
}

value builtin_iscallablefunction_err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, TYPE_NUMARGS, FUNCTION_ISCALLABLE);
    return MORPHO_FALSE;
}

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

#define BUILTIN_MATH_OLD2(function) \
    builtin_addfunction(#function, builtin_##function, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS);

#define BUILTIN_VARMATH_RET(label, function, realret, realflags, cmplxret, cmplxflags) \
    morpho_addfunction(label, realret " (Int)", builtin_int_##function, realflags, NULL); \
    morpho_addfunction(label, realret " (Float)", builtin_float_##function, realflags, NULL); \
    morpho_addfunction(label, cmplxret " (Complex)", builtin_cmplx_##function, cmplxflags, NULL); \
    morpho_addfunction(label, "(...)", builtin_numargserr_##function, MORPHO_FN_THROWS, NULL);

#define BUILTIN_VARMATH(label, function) \
    BUILTIN_VARMATH_RET(label, function, "Float", MORPHO_FN_PUREFN, "Complex", MORPHO_FN_ALLOCATES)

#define BUILTIN_MATH_BOOL(function) \
    BUILTIN_VARMATH_RET(#function, function, "Bool", MORPHO_FN_PUREFN, "Bool", MORPHO_FN_PUREFN)

#define BUILTIN_MATH(function) BUILTIN_VARMATH(#function, function)

#define BUILTIN_TYPECHECK(function) \
    morpho_addfunction(#function, "Bool (_)", builtin_##function, MORPHO_FN_PUREFN, NULL); \
    morpho_addfunction(#function, "Bool ()", builtin_numargserr_##function, MORPHO_FN_THROWS, NULL); \
    morpho_addfunction(#function, "Bool (_,_,...)", builtin_numargserr_##function, MORPHO_FN_THROWS, NULL);

void functiondefs_initialize(void) {
    // System
    builtin_addfunction(FUNCTION_SYSTEM, builtin_system, MORPHO_FN_IO);
    
    // Clock
    morpho_addfunction(FUNCTION_CLOCK, "Float ()", builtin_clock, MORPHO_FN_IO|MORPHO_FN_NONDETERMINISTIC, NULL);

    // Apply
    morpho_addfunction(FUNCTION_APPLY, "(Callable,Tuple)", builtin_apply__tuple, MORPHO_FN_REENTRANT|MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_APPLY, "(Callable,List)", builtin_apply__list, MORPHO_FN_REENTRANT|MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_APPLY, "(Callable,_,...)", builtin_apply, MORPHO_FN_REENTRANT|MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_APPLY, "()", builtin_apply__err, MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_APPLY, "(_)", builtin_apply__err, MORPHO_FN_THROWS, NULL);
    
    // Random numbers
    morpho_addfunction(FUNCTION_RANDOM, "Float ()", builtin_random, MORPHO_FN_NONDETERMINISTIC, NULL);
    morpho_addfunction(FUNCTION_RANDOMINT, "Int ()", builtin_randomint_norange, MORPHO_FN_NONDETERMINISTIC, NULL);
    morpho_addfunction(FUNCTION_RANDOMINT, "Int (_)", builtin_randomint, MORPHO_FN_NONDETERMINISTIC|MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_RANDOMNORMAL, "Float ()", builtin_randomnormal, MORPHO_FN_NONDETERMINISTIC, NULL);
    
    // Value constructors
    morpho_addfunction(FUNCTION_INT, "Int (Int)", builtin_int__int, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_INT, "Int (Float)", builtin_int__float, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_INT, "Int (String)", builtin_int__string, MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_INT, "Int (...)", builtin_int__err, MORPHO_FN_THROWS, NULL);
    
    morpho_addfunction(FUNCTION_FLOAT, "Float (Int)", builtin_float__int, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_FLOAT, "Float (Float)", builtin_float__float, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_FLOAT, "Float (String)", builtin_float__string, MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_FLOAT, "Float (...)", builtin_float__err, MORPHO_FN_THROWS, NULL);
    
    morpho_addfunction(FUNCTION_BOOL, "Bool (_)", builtin_bool, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_BOOL, "Bool ()", builtin_bool_err, MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_BOOL, "Bool (_,_,...)", builtin_bool_err, MORPHO_FN_THROWS, NULL);
    
    // Math functions
    BUILTIN_VARMATH_RET(FUNCTION_ABS, fabs, "Float", MORPHO_FN_PUREFN, "Float", MORPHO_FN_PUREFN)
    
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

    BUILTIN_MATH(floor)
    BUILTIN_MATH(ceil)

    // Math properties
    BUILTIN_MATH_BOOL(isfinite)
    BUILTIN_MATH_BOOL(isinf)
    BUILTIN_MATH_BOOL(isnan)
    
    // Math functions with special cases
    morpho_addfunction(FUNCTION_ARCTAN, "Float (Int)", builtin_arctan__number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_ARCTAN, "Float (Float)", builtin_arctan__number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_ARCTAN, "Float (Int,Int)", builtin_arctan__number_number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_ARCTAN, "Float (Int,Float)", builtin_arctan__number_number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_ARCTAN, "Float (Float,Int)", builtin_arctan__number_number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_ARCTAN, "Float (Float,Float)", builtin_arctan__number_number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_ARCTAN, "Complex (Complex)", builtin_arctan__complex, MORPHO_FN_ALLOCATES, NULL);
    morpho_addfunction(FUNCTION_ARCTAN, "Complex (Complex,Complex)", builtin_arctan__complex2, MORPHO_FN_ALLOCATES, NULL);
    morpho_addfunction(FUNCTION_ARCTAN, "Complex (Complex,_)", builtin_arctan__complex2, MORPHO_FN_ALLOCATES, NULL);
    morpho_addfunction(FUNCTION_ARCTAN, "Complex (_,Complex)", builtin_arctan__complex2, MORPHO_FN_ALLOCATES, NULL);
    morpho_addfunction(FUNCTION_ARCTAN, "Float (...)", builtin_arctan__err, MORPHO_FN_THROWS, NULL);
    BUILTIN_MATH_OLD2(sqrt);
    morpho_addfunction(FUNCTION_MOD, "Int (Int,Int)", builtin_mod__int_int, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_MOD, "Float (Float,Float)", builtin_mod__float_float, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_MOD, "Float (Int,Float)", builtin_mod__int_float, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_MOD, "Float (Float,Int)", builtin_mod__float_int, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_MOD, "Float (...)", builtin_mod__err, MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_SIGN, "Float (Int)", builtin_sign__int, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_SIGN, "Float (Float)", builtin_sign__float, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_SIGN, "Float (...)", builtin_sign__err, MORPHO_FN_THROWS, NULL);
    
    // Complex
    morpho_addfunction(FUNCTION_REAL, "Float (Complex)", builtin_real__complex, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_REAL, "Int (Int)", builtin_real__number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_REAL, "Float (Float)", builtin_real__number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_IMAG, "Float (Complex)", builtin_imag__complex, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_IMAG, "Float (Int)", builtin_imag__number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_IMAG, "Float (Float)", builtin_imag__number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_ANGLE, "Float (Complex)", builtin_angle__complex, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_ANGLE, "Float (Int)", builtin_angle__number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_ANGLE, "Float (Float)", builtin_angle__number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_CONJ, "Complex (Complex)", builtin_conj__complex, MORPHO_FN_ALLOCATES, NULL);
    morpho_addfunction(FUNCTION_CONJ, "Int (Int)", builtin_conj__number, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_CONJ, "Float (Float)", builtin_conj__number, MORPHO_FN_PUREFN, NULL);
    
    // Min/max
    morpho_addfunction(FUNCTION_BOUNDS, "List (_)", builtin_bounds, MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_BOUNDS, "List (_,...)", builtin_bounds, MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_BOUNDS, "List ()", builtin_bounds__err, MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_MIN, "(_,...)", builtin_min, MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_MIN, "()", builtin_min__err, MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_MAX, "(_,...)", builtin_max, MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_MAX, "()", builtin_max__err, MORPHO_FN_THROWS, NULL);
    
    // Type checking
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
    BUILTIN_TYPECHECK(istuple)
    BUILTIN_TYPECHECK(isarray)
    
#ifdef MORPHO_INCLUDE_LINALG
    BUILTIN_TYPECHECK(ismatrix)
#endif
    
#ifdef MORPHO_INCLUDE_SPARSE
    BUILTIN_TYPECHECK(issparse)
#endif
    
#ifdef MORPHO_INCLUDE_GEOMETRY
    BUILTIN_TYPECHECK(ismesh)
    BUILTIN_TYPECHECK(isselection)
    BUILTIN_TYPECHECK(isfield)
#endif
    
    morpho_addfunction(FUNCTION_ISCALLABLE, "Bool (_)", builtin_iscallablefunction, MORPHO_FN_PUREFN, NULL);
    morpho_addfunction(FUNCTION_ISCALLABLE, "Bool ()", builtin_iscallablefunction_err, MORPHO_FN_THROWS, NULL);
    morpho_addfunction(FUNCTION_ISCALLABLE, "Bool (_,_,...)", builtin_iscallablefunction_err, MORPHO_FN_THROWS, NULL);
    
    /* Define errors */
    morpho_defineerror(MATH_ARGS, ERROR_HALT, MATH_ARGS_MSG);
    morpho_defineerror(MATH_NUMARGS, ERROR_HALT, MATH_NUMARGS_MSG);
    morpho_defineerror(TYPE_NUMARGS, ERROR_HALT, TYPE_NUMARGS_MSG);
    morpho_defineerror(MAX_ARGS, ERROR_HALT, MAX_ARGS_MSG);
    morpho_defineerror(APPLY_ARGS, ERROR_HALT, APPLY_ARGS_MSG);
    morpho_defineerror(APPLY_NOTCALLABLE, ERROR_HALT, APPLY_NOTCALLABLE_MSG);
}

#undef BUILTIN_MATH
#undef BUILTIN_MATH_BOOL
