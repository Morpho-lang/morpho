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

/* **********************************************************************
 * Complex objects
 * ********************************************************************** */

objecttype objectcomplextype;

/** Complex object definitions */
size_t objectcomplex_sizefn(object *obj) {
    return sizeof(objectcomplex);
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
    objectcomplex *new = (objectcomplex *) object_new(sizeof(objectcomplex), OBJECT_COMPLEX);
    
    if (new) {
        new->Z=MCBuild(real,imag);
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
    objectcomplex *new = object_newcomplex(creal(in->Z),cimag(in->Z));
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
    *value = creal(c->Z);
}

/** @brief Gets a complex numbers imaginary part */
void complex_getimag(objectcomplex *c, double *value) {
    *value = cimag(c->Z);
}

/** @brief checks equality on two complex numbers */
bool complex_isequal(objectcomplex *a, objectcomplex *b) {
    return MCEq(a->Z,b->Z);
}

/** @brief checks equality between a complex number and a value */
bool complex_isequaltonumber(objectcomplex *a, value b) {
    if (MORPHO_ISNUMBER(b)){
        double val;
        morpho_valuetofloat(b,&val);
        return MCEq(a->Z, MCBuild(val,0.0));
    }
    return false;
}

/** Prints a complex number */
void complex_print(vm *v, objectcomplex *a) {
    char sign = '+';
    if (cimag(a->Z)<0) {
        sign = '-';
    }

    double Zr = creal(a->Z), Zi = cimag(a->Z), R = cabs(a->Z);

    double showZr = ( fabs(Zr) < MORPHO_RELATIVE_EPS*R ? 0 : Zr);
    double showZi = ( fabs(Zi) < MORPHO_RELATIVE_EPS*R ? 0 : fabs(Zi));

    morpho_printf(v, "%g %c %gim", showZr, sign, showZi);
}

/* **********************************************************************
 * Complex arithmetic
 * ********************************************************************* */

/** performs out = a + b */
void complex_add(objectcomplex *a, objectcomplex *b, objectcomplex *out){
    out->Z = MCAdd(a->Z,b->Z);
}

/** performs out = a + b where a is not complex */
void complex_add_real(objectcomplex *a, double b, objectcomplex *out){
    out->Z = MCAdd(a->Z, MCBuild(b,0));
}

/** performs out = a - b  */
void complex_sub(objectcomplex *a, objectcomplex *b, objectcomplex *out) {
    out->Z = MCSub(a->Z, b->Z);
}

/** performs out = a * b */
void complex_mul(objectcomplex *a, objectcomplex *b, objectcomplex *out){
    out->Z = MCMul(a->Z, b->Z);
}

/** performs out = a * b where b is real */
void complex_mul_real(objectcomplex *a, double b, objectcomplex *out){
    out->Z = MCScale(a->Z, b);
}

/** performs out = a */
void complex_copy(objectcomplex *a, objectcomplex *out) {
    out->Z = a->Z;
}

/** performs out = a ^ b  where b is real*/
void complex_power(objectcomplex *a, double exponent, objectcomplex *out){
    out->Z = cpow(a->Z,MCBuild(exponent, 0));
}

/** performs out = a ^ b  for complex numbers*/
void complex_cpower(objectcomplex *a, objectcomplex *b, objectcomplex *out){
    out->Z = cpow(a->Z,b->Z);
}

/** performs out = a / b */
void complex_div(objectcomplex *a, objectcomplex *b, objectcomplex *out){
    out->Z = MCDiv(a->Z,b->Z);
}

/** performs out = 1/a */
void complex_invert(objectcomplex *a, objectcomplex *out){
    out->Z = MCDiv(MCBuild(1,0), a->Z);
}

/** performs out = conj(a) by negating the imaginary part */
void complex_conj(objectcomplex *a, objectcomplex *out) {
    out->Z = conj(a->Z);
}

/** calculates theta in the complex representation a = r e^{i theta}  */
void complex_angle(objectcomplex *a, double *out){
    *out = carg(a->Z);
}

void complex_abs(objectcomplex *a, double *out) {
    *out = cabs(a->Z);
}

/* **********************************************************************
 * Builtin Mathematical Funtions For Complex Numbers
 * ********************************************************************* */

// Macro for creating a value from a new complex that copies val 
#define RET_COMPLEX(val,out) \
    objectcomplex *new=NULL;\
    new = object_complexfromcomplex(val);\
    if (new) {\
        out=MORPHO_OBJECT(new);\
        morpho_bindobjects(v, 1, &out);\
    }

// Macro for creating a value bool 
#define RET_DOUBLE(val,out) \
    out = MORPHO_FLOAT(val);


#define COMPLEX_BUILTIN(fcn,type,MAKEVAL)\
value complex_builtin##fcn(vm * v, objectcomplex *c) {\
    value out = MORPHO_NIL;\
    type val = c##fcn(c->Z);\
    MAKEVAL(val,out)\
    return out;\
}

value complex_builtinfabs(vm * v, objectcomplex *c) {
    double val = cabs(c->Z);
    return MORPHO_FLOAT(val);
}

COMPLEX_BUILTIN(exp,MorphoComplex,RET_COMPLEX)
COMPLEX_BUILTIN(log,MorphoComplex,RET_COMPLEX)

value complex_builtinlog10(vm * v, objectcomplex *c) {
    value out = MORPHO_NIL;
    MorphoComplex val = MCScale(clog(c->Z), 1.0/log(10));
    RET_COMPLEX(val,out)
    return out;
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
    value out = MORPHO_NIL;
    MorphoComplex val = MCBuild(floor(creal(c->Z)),floor(cimag(c->Z)));
    RET_COMPLEX(val,out)
    return out;
}

value complex_builtinceil(vm * v, objectcomplex *c) {
    value out = MORPHO_NIL;
    MorphoComplex val = MCBuild(ceil(creal(c->Z)), ceil(cimag(c->Z)));
    RET_COMPLEX(val,out)
    return out;
}

#undef COMPLEX_BUILTIN
#undef RET_COMPLEX
#undef RET_DOUBLE

#define COMPLEX_BUILTIN_BOOL(fcn,logicalop)\
value complex_builtin##fcn(objectcomplex *c) {\
    bool val = fcn(creal(c->Z)) logicalop fcn(cimag(c->Z));\
    return MORPHO_BOOL(val);\
}

COMPLEX_BUILTIN_BOOL(isfinite,&&)
COMPLEX_BUILTIN_BOOL(isinf,||)
COMPLEX_BUILTIN_BOOL(isnan,||)

#undef COMPLEX_BUILTIN_BOOL

value complex_builtinatan(vm *v, value c){
    value out = MORPHO_NIL;
    MorphoComplex val = catan(MORPHO_GETCOMPLEX(c)->Z);
    objectcomplex *new = NULL;
    new = object_complexfromcomplex(val);
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    return out;
}

value complex_builtinatan2(vm *v, value c1, value c2){
    value out = MORPHO_NIL;
    MorphoComplex val=MCBuild(0,0);
    
    if (MORPHO_ISCOMPLEX(c1) && MORPHO_ISCOMPLEX(c2)) {
        val = catan(MCDiv(MORPHO_GETCOMPLEX(c1)->Z, MORPHO_GETCOMPLEX(c2)->Z));
    } else if (MORPHO_ISCOMPLEX(c1) && MORPHO_ISNUMBER(c2)) {
        double num;
        morpho_valuetofloat(c2,&num);
        val = catan(MCScale(MORPHO_GETCOMPLEX(c1)->Z, 1.0/num));
    } else if (MORPHO_ISNUMBER(c1) && MORPHO_ISCOMPLEX(c2)) {
        double num;
        morpho_valuetofloat(c1,&num);
        val = catan(MCDiv(MCBuild(num,0), MORPHO_GETCOMPLEX(c2)->Z));
    } else {
        morpho_runtimeerror(v, COMPLEX_INVLDNARG);
        return MORPHO_NIL;
    }
     
    objectcomplex *new = NULL;
    new = object_complexfromcomplex(val);
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}


/* **********************************************************************
 * Complex veneer class
 * ********************************************************************* */

/** Constructs a Complex object */
value complex_constructor(vm *v, int nargs, value *args) {
    double real=0, imag=0;
    objectcomplex *new=NULL;
    value out=MORPHO_NIL;
    // expect 2 aruments

    if (nargs==2){
        // make sure both are numbers and cast them to floats
        if (MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
            morpho_valuetofloat(MORPHO_GETARG(args, 0), &real);
        } else goto complex_constructor_error;
        
        if (MORPHO_ISNUMBER(MORPHO_GETARG(args, 1))) {
            morpho_valuetofloat(MORPHO_GETARG(args, 1), &imag);
        } else goto complex_constructor_error;

    } else goto complex_constructor_error;

    new = object_newcomplex(real, imag);
    
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
    
complex_constructor_error:
    morpho_runtimeerror(v, COMPLEX_CONSTRUCTOR);
    
    return MORPHO_NIL;
}

/** Gets the real part of a complex number */
value Complex_getreal(vm *v, int nargs, value *args) {
    objectcomplex *c=MORPHO_GETCOMPLEX(MORPHO_SELF(args));

    value out = MORPHO_NIL;
	if (nargs>0){
		morpho_runtimeerror(v, COMPLEX_INVLDNARG);
		return out;
	}
    
    double real;
    complex_getreal(c, &real);
    out = MORPHO_FLOAT(real);
    return out;
}

/** Gets the imaginary part of a complex number */
value Complex_getimag(vm *v, int nargs, value *args) {
    objectcomplex *c=MORPHO_GETCOMPLEX(MORPHO_SELF(args));

    value out = MORPHO_NIL;
	if (nargs>0){
		morpho_runtimeerror(v, COMPLEX_INVLDNARG);
		return out;
	}
    
    double imag;
    complex_getimag(c, &imag);
    out = MORPHO_FLOAT(imag);
    return out;
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
value Complex_add(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISCOMPLEX(MORPHO_GETARG(args, 0))) {
        objectcomplex *b=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
        
        objectcomplex *new = object_newcomplex(0, 0);
        if (new) {
            out=MORPHO_OBJECT(new);
            complex_add(a, b, new);
        }
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double val;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
            objectcomplex *new = object_newcomplex(0,0);
            if (new) {
                out=MORPHO_OBJECT(new);
                complex_add_real(a, val, new);
            }
        }
    } else morpho_runtimeerror(v, COMPLEX_ARITHARGS);
    
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    
    return out;
}

/** Complex subtract */
value Complex_sub(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISCOMPLEX(MORPHO_GETARG(args, 0))) {
        objectcomplex *b=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
        
        objectcomplex *new = object_newcomplex(0, 0);
        if (new) {
            out=MORPHO_OBJECT(new);
            complex_sub(a, b, new);
        }
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double val;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
            objectcomplex *new = object_newcomplex(0,0);
            if (new) {
                out=MORPHO_OBJECT(new);
                complex_add_real(a, -val, new);
            }
        }
    } else morpho_runtimeerror(v, COMPLEX_ARITHARGS);
    
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    
    return out;
}

/** Right subtract */
value Complex_subr(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double val;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
            objectcomplex *new = object_clonecomplex(a);
            complex_mul_real(new,-1,new);

            if (new) {
                out=MORPHO_OBJECT(new);
                complex_add_real(new, val, new);
            }
        }
    } else morpho_runtimeerror(v, COMPLEX_ARITHARGS);
    
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    
    return out;
}

/** Complex multiply */
value Complex_mul(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISCOMPLEX(MORPHO_GETARG(args, 0))) {
        objectcomplex *b=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
        
        objectcomplex *new = object_newcomplex(0, 0);
        if (new) {
            out=MORPHO_OBJECT(new);
            complex_mul(a, b, new);
        }
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double val;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
            objectcomplex *new = object_newcomplex(0,0);
            if (new) {
                out=MORPHO_OBJECT(new);
                complex_mul_real(a, val, new);
            }
        }
    } else morpho_runtimeerror(v, COMPLEX_ARITHARGS);
    
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    
    return out;
}

/** Complex divide */
value Complex_div(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISCOMPLEX(MORPHO_GETARG(args, 0))) {
        objectcomplex *b=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
        
        objectcomplex *new = object_newcomplex(0, 0);
        if (new) {
            out=MORPHO_OBJECT(new);
            complex_div(a, b, new);
        }
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double val;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
            objectcomplex *new = object_newcomplex(0,0);
            if (new) {
                out=MORPHO_OBJECT(new);
                complex_mul_real(a, 1.0/val, new);
            }
        }
    } else morpho_runtimeerror(v, COMPLEX_ARITHARGS);
    
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    
    return out;
}

/** Complex right divide */
value Complex_divr(vm *v, int nargs, value *args) {
    // this gets called when we divide a nonobject (number) by a complex number
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double val;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {

            objectcomplex *new = object_newcomplex(0,0);
            complex_invert(a,new);
            complex_mul_real(new,val,new);

            if (new) {
                out=MORPHO_OBJECT(new);
                morpho_bindobjects(v, 1, &out);
            }
        } else UNREACHABLE("Number did not return float value");
    } else morpho_runtimeerror(v, COMPLEX_ARITHARGS);
    
    return out;
}

/** Complex exponentiation */
value Complex_power(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (nargs==1 && MORPHO_ISCOMPLEX(MORPHO_GETARG(args, 0))) {
        // raise a complex number to a complex power
        objectcomplex *b=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
        
        objectcomplex *new = object_newcomplex(0, 0);
        if (new) {
            out=MORPHO_OBJECT(new);
            complex_cpower(a, b, new);
        }
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        // raise complex power to a number
        double val;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
            objectcomplex *new = object_newcomplex(0,0);
            if (new) {
                out=MORPHO_OBJECT(new);
                complex_power(a, val, new);
            }
        }
    } else morpho_runtimeerror(v, COMPLEX_ARITHARGS);
    
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    
    return out;
}

/** Complex right exponentiation */
value Complex_powerr(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (nargs==1 && MORPHO_ISCOMPLEX(MORPHO_GETARG(args, 0))) {
        // raise a complex number to a complex power
        objectcomplex *b=MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0));
        
        objectcomplex *new = object_newcomplex(0, 0);
        if (new) {
            out=MORPHO_OBJECT(new);
            complex_cpower(a, b, new);
        }
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        // raise a number to a complex power
        double val;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
            objectcomplex *new = object_newcomplex(val,0);
            if (new) {
                out=MORPHO_OBJECT(new);
                complex_cpower(new, a, new);
            }
        }
    } else morpho_runtimeerror(v, COMPLEX_ARITHARGS);
    
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    
    return out;

}

/** Angle of a complex number  */
value Complex_angle(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    complex_angle(a, &val);
    return MORPHO_FLOAT(val);
}
value Complex_abs(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    complex_abs(a, &val);
    return MORPHO_FLOAT(val);
}

/** Conjugate of a complex */
value Complex_conjugate(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    objectcomplex *new = object_newcomplex(0,0);
    if (new) {
        complex_conj(a, new);
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

/** Clones a complex */
value Complex_clone(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    objectcomplex *new=object_clonecomplex(a);
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    return out;
}

MORPHO_BEGINCLASS(ComplexNum)
MORPHO_METHOD(MORPHO_PRINT_METHOD, Complex_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADD_METHOD, Complex_add, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUB_METHOD, Complex_sub, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_MUL_METHOD, Complex_mul, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_DIV_METHOD, Complex_div, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADDR_METHOD, Complex_add, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUBR_METHOD, Complex_subr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_MULR_METHOD, Complex_mul, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_DIVR_METHOD, Complex_divr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_POW_METHOD, Complex_power, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_POWR_METHOD, Complex_powerr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(COMPLEX_ANGLE_METHOD, Complex_angle, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(COMPLEX_CONJUGATE_METHOD, Complex_conjugate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(COMPLEX_REAL_METHOD, Complex_getreal, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(COMPLEX_IMAG_METHOD, Complex_getimag, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(COMPLEX_ABS_METHOD, Complex_abs, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Complex_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************* */

void complex_initialize(void) {
    // Define complex object type
    objectcomplextype=object_addtype(&objectcomplexdefn);
    
    // Complex constructor function
    morpho_addfunction(COMPLEX_CLASSNAME, COMPLEX_CLASSNAME " (...)", complex_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // Define Complex class
    value complexclass=builtin_addclass(COMPLEX_CLASSNAME, MORPHO_GETCLASSDEFINITION(ComplexNum), objclass);
    object_setveneerclass(OBJECT_COMPLEX, complexclass);

    // Complex error messages
    morpho_defineerror(COMPLEX_CONSTRUCTOR, ERROR_HALT, COMPLEX_CONSTRUCTOR_MSG);
    morpho_defineerror(COMPLEX_ARITHARGS, ERROR_HALT, COMPLEX_ARITHARGS_MSG);
    morpho_defineerror(COMPLEX_INVLDNARG, ERROR_HALT, COMPLEX_INVLDNARG_MSG);
}
