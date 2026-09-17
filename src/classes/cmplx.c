/** @file complex.c
 *  @author D Hellstein and T J Atherton
 *
 *  @brief Complex number type
 */

#include <string.h>
#include <complex.h>

#include "morpho.h"
#include "classes.h"
#include "common.h"

/** Live MorphoComplex (owned or view). Never access Z[] on a view. */
#define COMPLEX_VAL(c) (*(c)->val)

/* **********************************************************************
 * Complex objects
 * ********************************************************************** */

objecttype objectcomplextype;

/** Complex object definitions */
size_t objectcomplex_sizefn(object *obj) {
    objectcomplex *c=(objectcomplex *) obj;
    return sizeof(objectcomplex) + (c->val==c->Z ?  sizeof(MorphoComplex) : 0 );
} 

void objectcomplex_printfn(object *obj, void *v) {
    complex_print(v, (objectcomplex *) obj);
}

int objectcomplex_cmpfn(object *a, object *b) {
    objectcomplex *acomp = (objectcomplex *) a;
    objectcomplex *bcomp = (objectcomplex *) b;
    return (complex_isequal(acomp, bcomp)? MORPHO_EQUAL: MORPHO_NOTEQUAL);
}

objecttypedefn objectcomplexdefn = {
    .printfn=objectcomplex_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectcomplex_sizefn,
    .hashfn=NULL,
    .cmpfn=objectcomplex_cmpfn
};

/** Creates a complex object */
objectcomplex *object_newcomplex(double real,double imag) {
    objectcomplex *new = (objectcomplex *) object_new(sizeof(objectcomplex)+sizeof(MorphoComplex), OBJECT_COMPLEX);
    
    if (new) {
        new->val=new->Z;
        COMPLEX_VAL(new)=MCBuild(real,imag);
    }
    
    return new;
}

/* **********************************************************************
 * Other constructors
 * ********************************************************************** */

/** Create complex number from a float */
objectcomplex *object_complexfromfloat(double val) {
    objectcomplex *ret=object_newcomplex(val,0.0);
    return ret;
}

/** Create complex number from a complex */
objectcomplex *object_complexfromcomplex(MorphoComplex val) {
    objectcomplex *ret=object_newcomplex(creal(val),cimag(val));
    return ret;
}

/** Clone a complex */
objectcomplex *object_clonecomplex(objectcomplex *in) {
    objectcomplex *new = object_newcomplex(creal(COMPLEX_VAL(in)),cimag(COMPLEX_VAL(in)));
    return new;
}

/** Clone a Complex Stored in a value*/
value object_clonecomplexvalue(value val) {
    value out = MORPHO_NIL;
    if (MORPHO_ISCOMPLEX(val)) {
        objectcomplex *c = MORPHO_GETCOMPLEX(val);
        out=MORPHO_OBJECT(object_clonecomplex(c));
    }
    return out;
}

/* **********************************************************************
 * Complex operations
 * ********************************************************************* */

/** @brief Gets a complex numbers real part */
void complex_getreal(objectcomplex *c, double *value) {
    *value = creal(COMPLEX_VAL(c));
}

/** @brief Gets a complex numbers imaginary part */
void complex_getimag(objectcomplex *c, double *value) {
    *value = cimag(COMPLEX_VAL(c));
}

/** @brief checks equality on two complex numbers */
bool complex_isequal(objectcomplex *a, objectcomplex *b) {
    return MCEq(COMPLEX_VAL(a),COMPLEX_VAL(b));
}

/** @brief checks equality between a complex number and a value */
bool complex_isequaltonumber(objectcomplex *a, value b) {
    if (MORPHO_ISNUMBER(b)){
        double val;
        morpho_valuetofloat(b,&val);
        return MCEq(COMPLEX_VAL(a), MCBuild(val,0.0));
    }
    return false;
}

/** Prints a complex number */
void complex_print(vm *v, objectcomplex *a) {
    char sign = '+';
    if (cimag(COMPLEX_VAL(a))<0) {
        sign = '-';
    }

    double Zr = creal(COMPLEX_VAL(a)), Zi = cimag(COMPLEX_VAL(a)), R = cabs(COMPLEX_VAL(a));

    double showZr = ( fabs(Zr) < MORPHO_RELATIVE_EPS*R ? 0 : Zr);
    double showZi = ( fabs(Zi) < MORPHO_RELATIVE_EPS*R ? 0 : fabs(Zi));

    morpho_printf(v, "%g %c %gim", showZr, sign, showZi);
}

/* **********************************************************************
 * Complex arithmetic
 * ********************************************************************* */

/** performs out = a + b */
void complex_add(objectcomplex *a, objectcomplex *b, objectcomplex *out){
    COMPLEX_VAL(out) = MCAdd(COMPLEX_VAL(a),COMPLEX_VAL(b));
}

/** performs out = a + b where a is not complex */
void complex_add_real(objectcomplex *a, double b, objectcomplex *out){
    COMPLEX_VAL(out) = MCAdd(COMPLEX_VAL(a), MCBuild(b,0));
}

/** performs out = a - b  */
void complex_sub(objectcomplex *a, objectcomplex *b, objectcomplex *out) {
    COMPLEX_VAL(out) = MCSub(COMPLEX_VAL(a), COMPLEX_VAL(b));
}

/** performs out = a * b */
void complex_mul(objectcomplex *a, objectcomplex *b, objectcomplex *out){
    COMPLEX_VAL(out) = MCMul(COMPLEX_VAL(a), COMPLEX_VAL(b));
}

/** performs out = a * b where b is real */
void complex_mul_real(objectcomplex *a, double b, objectcomplex *out){
    COMPLEX_VAL(out) = MCScale(COMPLEX_VAL(a), b);
}

/** performs out = a */
void complex_copy(objectcomplex *a, objectcomplex *out) {
    COMPLEX_VAL(out) = COMPLEX_VAL(a);
}

/** performs out = a ^ b  where b is real*/
void complex_power(objectcomplex *a, double exponent, objectcomplex *out){
    COMPLEX_VAL(out) = cpow(COMPLEX_VAL(a),MCBuild(exponent, 0));
}

/** performs out = a ^ b  for complex numbers*/
void complex_cpower(objectcomplex *a, objectcomplex *b, objectcomplex *out){
    COMPLEX_VAL(out) = cpow(COMPLEX_VAL(a),COMPLEX_VAL(b));
}

/** performs out = a / b */
void complex_div(objectcomplex *a, objectcomplex *b, objectcomplex *out){
    COMPLEX_VAL(out) = MCDiv(COMPLEX_VAL(a),COMPLEX_VAL(b));
}

/** performs out = 1/a */
void complex_invert(objectcomplex *a, objectcomplex *out){
    COMPLEX_VAL(out) = MCDiv(MCBuild(1,0), COMPLEX_VAL(a));
}

/** performs out = conj(a) by negating the imaginary part */
void complex_conj(objectcomplex *a, objectcomplex *out) {
    COMPLEX_VAL(out) = conj(COMPLEX_VAL(a));
}

/** calculates theta in the complex representation a = r e^{i theta}  */
void complex_angle(objectcomplex *a, double *out){
    *out = carg(COMPLEX_VAL(a));
}

void complex_abs(objectcomplex *a, double *out) {
    *out = cabs(COMPLEX_VAL(a));
}

/* **********************************************************************
 * Builtin Mathematical Funtions For Complex Numbers
 * ********************************************************************* */

// Macro for creating a value from a new complex that copies val 
#define RET_COMPLEX(z) \
    return morpho_wrapandbind(v, (object *) object_complexfromcomplex(z));

#define COMPLEX_BUILTIN(fcn,type,MAKEVAL)\
value complex_builtin##fcn(vm * v, objectcomplex *c) {\
    type val = c##fcn(COMPLEX_VAL(c));\
    MAKEVAL(val)\
}

value complex_builtinfabs(vm * v, objectcomplex *c) {
    double val = cabs(COMPLEX_VAL(c));
    return MORPHO_FLOAT(val);
}

COMPLEX_BUILTIN(exp,MorphoComplex,RET_COMPLEX)
COMPLEX_BUILTIN(log,MorphoComplex,RET_COMPLEX)

value complex_builtinlog10(vm * v, objectcomplex *c) {
    MorphoComplex val = MCScale(clog(COMPLEX_VAL(c)), 1.0/log(10));
    RET_COMPLEX(val)
}

COMPLEX_BUILTIN(sin,MorphoComplex,RET_COMPLEX)
COMPLEX_BUILTIN(cos,MorphoComplex,RET_COMPLEX)
COMPLEX_BUILTIN(tan,MorphoComplex,RET_COMPLEX)
COMPLEX_BUILTIN(asin,MorphoComplex,RET_COMPLEX)
COMPLEX_BUILTIN(acos,MorphoComplex,RET_COMPLEX)

COMPLEX_BUILTIN(sinh,MorphoComplex,RET_COMPLEX)
COMPLEX_BUILTIN(cosh,MorphoComplex,RET_COMPLEX)
COMPLEX_BUILTIN(tanh,MorphoComplex,RET_COMPLEX)
COMPLEX_BUILTIN(sqrt,MorphoComplex,RET_COMPLEX)

value complex_builtinfloor(vm * v, objectcomplex *c) {
    MorphoComplex val = MCBuild(floor(creal(COMPLEX_VAL(c))),floor(cimag(COMPLEX_VAL(c))));
    RET_COMPLEX(val)
}

value complex_builtinceil(vm * v, objectcomplex *c) {
    MorphoComplex val = MCBuild(ceil(creal(COMPLEX_VAL(c))), ceil(cimag(COMPLEX_VAL(c))));
    RET_COMPLEX(val)
}

#undef COMPLEX_BUILTIN
#undef RET_COMPLEX

#define COMPLEX_BUILTIN_BOOL(fcn,logicalop)\
value complex_builtin##fcn(vm *v, objectcomplex *c) {\
    bool val = fcn(creal(COMPLEX_VAL(c))) logicalop fcn(cimag(COMPLEX_VAL(c)));\
    return MORPHO_BOOL(val);\
}

COMPLEX_BUILTIN_BOOL(isfinite,&&)
COMPLEX_BUILTIN_BOOL(isinf,||)
COMPLEX_BUILTIN_BOOL(isnan,||)

#undef COMPLEX_BUILTIN_BOOL

value complex_builtinatan(vm *v, value c){
    return morpho_wrapandbind(v, (object *) object_complexfromcomplex(catan(MORPHO_GETDOUBLECOMPLEX(c))));
}

value complex_builtinatan2(vm *v, value c1, value c2){
    MorphoComplex val=MCBuild(0,0);
    
    if (MORPHO_ISCOMPLEX(c1) && MORPHO_ISCOMPLEX(c2)) {
        val = catan(MCDiv(MORPHO_GETDOUBLECOMPLEX(c1), MORPHO_GETDOUBLECOMPLEX(c2)));
    } else if (MORPHO_ISCOMPLEX(c1) && MORPHO_ISNUMBER(c2)) {
        double num;
        morpho_valuetofloat(c2,&num);
        val = catan(MCScale(MORPHO_GETDOUBLECOMPLEX(c1), 1.0/num));
    } else if (MORPHO_ISNUMBER(c1) && MORPHO_ISCOMPLEX(c2)) {
        double num;
        morpho_valuetofloat(c1,&num);
        val = catan(MCDiv(MCBuild(num,0), MORPHO_GETDOUBLECOMPLEX(c2)));
    } else {
        morpho_runtimeerror(v, COMPLEX_INVLDNARG);
        return MORPHO_NIL;
    }
     
    return morpho_wrapandbind(v, (object *) object_complexfromcomplex(val));
}


/* **********************************************************************
 * Complex veneer class
 * ********************************************************************* */

/** Constructs a Complex object from two numbers */
value complex_constructor(vm *v, int nargs, value *args) {
    double real=0, imag=0;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &real);
    morpho_valuetofloat(MORPHO_GETARG(args, 1), &imag);
    return morpho_wrapandbind(v, (object *) object_newcomplex(real, imag));
}

/** Gets the real part of a complex number */
value Complex_getreal(vm *v, int nargs, value *args) {
    objectcomplex *c=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double real;
    complex_getreal(c, &real);
    return MORPHO_FLOAT(real);
}

/** Gets the imaginary part of a complex number */
value Complex_getimag(vm *v, int nargs, value *args) {
    objectcomplex *c=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double imag;
    complex_getimag(c, &imag);
    return MORPHO_FLOAT(imag);
}

/** Prints a complex */
value Complex_print(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    if (!MORPHO_ISCOMPLEX(self)) return Object_print(v, nargs, args);
    
    objectcomplex *c=MORPHO_GETCOMPLEX(self);
    complex_print(v, c);
    return MORPHO_NIL;
}

/** Complex add */
value Complex_add__complex(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    objectcomplex *b=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
    
    objectcomplex *new = object_newcomplex(0, 0);
    if (new) complex_add(a, b, new);
    
    return morpho_wrapandbind(v, (object *) new);
}

value Complex_add__number(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &val);

    objectcomplex *new = object_newcomplex(0, 0);
    if (new) complex_add_real(a, val, new);
    return morpho_wrapandbind(v, (object *) new);
}

/** Complex subtract */
value Complex_sub__complex(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    objectcomplex *b=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
    
    objectcomplex *new = object_newcomplex(0, 0);
    if (new) complex_sub(a, b, new);
    
    return morpho_wrapandbind(v, (object *) new);
}

value Complex_sub__number(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &val);

    objectcomplex *new = object_newcomplex(0, 0);
    if (new) complex_add_real(a, -val, new);
    return morpho_wrapandbind(v, (object *) new);
}

/** Right subtract: number - self */
value Complex_subr__number(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &val);

    objectcomplex *new = object_clonecomplex(a);
    if (new) {
        complex_mul_real(new, -1, new);
        complex_add_real(new, val, new);
    }
    return morpho_wrapandbind(v, (object *) new);
}

/** Complex multiply */
value Complex_mul__complex(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    objectcomplex *b=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
    
    objectcomplex *new = object_newcomplex(0, 0);
    if (new) complex_mul(a, b, new);
    
    return morpho_wrapandbind(v, (object *) new);
}

value Complex_mul__number(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &val);

    objectcomplex *new = object_newcomplex(0, 0);
    if (new) complex_mul_real(a, val, new);
    return morpho_wrapandbind(v, (object *) new);
}

/** Complex divide */
value Complex_div__complex(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    objectcomplex *b=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
    
    objectcomplex *new = object_newcomplex(0, 0);
    if (new) complex_div(a, b, new);
    
    return morpho_wrapandbind(v, (object *) new);
}

value Complex_div__number(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &val);

    objectcomplex *new = object_newcomplex(0, 0);
    if (new) complex_mul_real(a, 1.0/val, new);
    return morpho_wrapandbind(v, (object *) new);
}

/** Complex right divide: number / self */
value Complex_divr__number(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &val);

    objectcomplex *new = object_newcomplex(0, 0);
    if (new) {
        complex_invert(a, new);
        complex_mul_real(new, val, new);
    }
    return morpho_wrapandbind(v, (object *) new);
}

/** Complex exponentiation */
value Complex_power__complex(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    objectcomplex *b=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
    
    objectcomplex *new = object_newcomplex(0, 0);
    if (new) complex_cpower(a, b, new);
    
    return morpho_wrapandbind(v, (object *) new);
}

value Complex_power__number(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &val);

    objectcomplex *new = object_newcomplex(0, 0);
    if (new) complex_power(a, val, new);
    return morpho_wrapandbind(v, (object *) new);
}

/** Complex right exponentiation: arg ^ self */
value Complex_powerr__complex(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    objectcomplex *b=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
    
    objectcomplex *new = object_newcomplex(0, 0);
    if (new) complex_cpower(b, a, new);
    
    return morpho_wrapandbind(v, (object *) new);
}

value Complex_powerr__number(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &val);

    objectcomplex *new = object_newcomplex(val, 0);
    if (new) complex_cpower(new, a, new);
    return morpho_wrapandbind(v, (object *) new);
}

/** Angle of a complex number  */
value Complex_angle(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    complex_angle(a, &val);
    return MORPHO_FLOAT(val);
}

/** Absolute value of a complex number  */
value Complex_abs(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    complex_abs(a, &val);
    return MORPHO_FLOAT(val);
}

/** Conjugate of a complex */
value Complex_conjugate(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    objectcomplex *new = object_newcomplex(0,0);
    if (new) complex_conj(a, new);
    return morpho_wrapandbind(v, (object *) new);
}

/** Clones a complex */
value Complex_clone(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    return morpho_wrapandbind(v, (object *) object_clonecomplex(a));
}

/** Left-arithmetic catch-all: decline so OPREDIRECT can try the right-hand method. */
value Complex_arith__x(vm *v, int nargs, value *args) {
    return MORPHO_NIL;
}

#define COMPLEX_ARITH_FLGS (MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS)

MORPHO_BEGINCLASS(ComplexNum)
MORPHO_METHOD(MORPHO_PRINT_METHOD, Complex_print, MORPHO_FN_IO),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Complex (Complex)", Complex_add__complex, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Complex (Int)", Complex_add__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Complex (Float)", Complex_add__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Complex (_)", Complex_arith__x, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "Complex (Complex)", Complex_sub__complex, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "Complex (Int)", Complex_sub__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "Complex (Float)", Complex_sub__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "Complex (_)", Complex_arith__x, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "Complex (Complex)", Complex_mul__complex, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "Complex (Int)", Complex_mul__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "Complex (Float)", Complex_mul__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "Complex (_)", Complex_arith__x, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "Complex (Complex)", Complex_div__complex, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "Complex (Int)", Complex_div__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "Complex (Float)", Complex_div__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "Complex (_)", Complex_arith__x, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "Complex (Complex)", Complex_add__complex, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "Complex (Int)", Complex_add__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "Complex (Float)", Complex_add__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "Complex (Int)", Complex_subr__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "Complex (Float)", Complex_subr__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "Complex (Complex)", Complex_mul__complex, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "Complex (Int)", Complex_mul__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "Complex (Float)", Complex_mul__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_DIVR_METHOD, "Complex (Int)", Complex_divr__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_DIVR_METHOD, "Complex (Float)", Complex_divr__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_POW_METHOD, "Complex (Complex)", Complex_power__complex, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_POW_METHOD, "Complex (Int)", Complex_power__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_POW_METHOD, "Complex (Float)", Complex_power__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_POW_METHOD, "Complex (_)", Complex_arith__x, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_POWR_METHOD, "Complex (Complex)", Complex_powerr__complex, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_POWR_METHOD, "Complex (Int)", Complex_powerr__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(MORPHO_POWR_METHOD, "Complex (Float)", Complex_powerr__number, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(COMPLEX_ANGLE_METHOD, "Float ()", Complex_angle, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(COMPLEX_CONJUGATE_METHOD, "Complex ()", Complex_conjugate, COMPLEX_ARITH_FLGS),
MORPHO_METHOD_SIGNATURE(COMPLEX_REAL_METHOD, "Float ()", Complex_getreal, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(COMPLEX_IMAG_METHOD, "Float ()", Complex_getimag, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(COMPLEX_ABS_METHOD, "Float ()", Complex_abs, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, "Complex ()", Complex_clone, COMPLEX_ARITH_FLGS)
MORPHO_ENDCLASS
#undef COMPLEX_ARITH_FLGS

/* **********************************************************************
 * Initialization
 * ********************************************************************* */

void complex_initialize(void) {
    // Define complex object type
    objectcomplextype=object_addtype(&objectcomplexdefn);
    
    // Complex constructor function
#define COMPLEX_CONS_FLGS (MORPHO_FN_CONSTRUCTOR|MORPHO_FN_ALLOCATES)
    morpho_addfunction(COMPLEX_CLASSNAME, "Complex (Int, Int)", complex_constructor, COMPLEX_CONS_FLGS, NULL);
    morpho_addfunction(COMPLEX_CLASSNAME, "Complex (Int, Float)", complex_constructor, COMPLEX_CONS_FLGS, NULL);
    morpho_addfunction(COMPLEX_CLASSNAME, "Complex (Float, Int)", complex_constructor, COMPLEX_CONS_FLGS, NULL);
    morpho_addfunction(COMPLEX_CLASSNAME, "Complex (Float, Float)", complex_constructor, COMPLEX_CONS_FLGS, NULL);
#undef COMPLEX_CONS_FLGS
    
    value objclass = builtin_findclassfromcstring(OBJECT_CLASSNAME);
    
    // Define Complex class
    value complexclass=builtin_addclass(COMPLEX_CLASSNAME, MORPHO_GETCLASSDEFINITION(ComplexNum), objclass);
    object_setveneerclass(OBJECT_COMPLEX, complexclass);

    // Complex error messages
    morpho_defineerror(COMPLEX_INVLDNARG, ERROR_HALT, COMPLEX_INVLDNARG_MSG);
}
