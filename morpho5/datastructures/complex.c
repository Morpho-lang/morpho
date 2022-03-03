/** @file complex.c
 *  @author Danny Goldstein
 *
 *  @brief Complex number type
 */

#include <string.h>
#include "object.h"
#include "complex.h"
#include "morpho.h"
#include "builtin.h"
#include "veneer.h"
#include "common.h"

/* **********************************************************************
 * Complex objects
 * ********************************************************************** */

objecttype objectcomplextype;

/** Function object definitions */
size_t objectcomplex_sizefn(object *obj) {
    return sizeof(objectcomplextype)+sizeof(double) * 2;
}

void objectcomplex_printfn(object *obj) {
    printf("<Complex>");
}

objecttypedefn objectcomplexdefn = {
    .printfn=objectcomplex_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectcomplex_sizefn
};

/** Creates a complex object */
objectcomplex *object_newcomplex(double real,double imag) {
    objectcomplex *new = (objectcomplex *) object_new(sizeof(objectcomplex), OBJECT_COMPLEX);
    
    if (new) {
        new->real=real;
        new->imag=imag;
    }
    
    return new;
}

/* **********************************************************************
 * Other constructors
 * ********************************************************************** */

/*
 * Create complex number from a float
 */

/** Creates a new array from a float */
objectcomplex *object_complexfromfloat(double val) {
    objectcomplex *ret=object_newcomplex(val,0.0);
    return ret;
}

/*
 * Clone complex
 */

/** Clone a complex */
objectcomplex *object_clonecomplex(objectcomplex *in) {
    objectcomplex *new = object_newcomplex(in->real,in->imag);
    return new;
}

/* **********************************************************************
 * Complex operations
 * ********************************************************************* */

/** @brief Sets the real part of a complex number.*/
void complex_setreal(objectcomplex *c, double value) {
    c->real = value;
}
/** @brief Sets the imaginary part of a complex number.*/
void complex_setimag(objectcomplex *c, double value) {
    c->imag = value;
}

/** @brief Gets a complex numbers real part */
void complex_getreal(objectcomplex *c, double *value) {
    *value = c->real;
}

/** @brief Gets a complex numbers imaginary part */
void complex_getimag(objectcomplex *c, double *value) {
    *value = c->imag;
}


/* **********************************************************************
 * Complex arithmetic
 * ********************************************************************* */

/** performs out = a + b */
void complex_add(objectcomplex *a, objectcomplex *b, objectcomplex *out){
    out->real = a->real + b->real;
    out->imag = a->imag + b->imag;

}

/** performs out = a + b where a is not complex */
void complex_add_real(objectcomplex *a, double b, objectcomplex *out){
    out->real = a->real + b;
    out->imag = a->imag;

}

/** performs out = a - b  */
void complex_sub(objectcomplex *a, objectcomplex *b, objectcomplex *out) {
    out->real = a->real - b->real;
    out->imag = a->imag - b->imag;
}

/** performs out = a * b */
void complex_mul(objectcomplex *a, objectcomplex *b, objectcomplex *out){
    out->real = a->real * b->real - a->imag * b->imag;
    out->imag = a->real * b->imag + a->imag * b->real;
}

/** performs out = a * b where b is real */
void complex_mul_real(objectcomplex *a, double b, objectcomplex *out){
    out->real = a->real * b;
    out->imag = a->imag * b;

}

/** performs out = a */
void complex_copy(objectcomplex *a, objectcomplex *out) {
    out->real = a->real;
    out->imag = a->real;
}

/** performs out = a ^ b  where be is real*/
void complex_power(objectcomplex *a, double exponent, objectcomplex *out){
    double r;
    double theta;
    complex_abs(a,&r);
    complex_angle(a,&theta);

    out->real = pow(r,exponent) * cos(theta * exponent);
    out->imag = pow(r,exponent) * sin(theta * exponent);

}

/** performs out = a ^ b */
void complex_cpower(objectcomplex *a, objectcomplex *exponent, objectcomplex *out){
    double r;
    double theta;
    complex_abs(a,&r);
    complex_angle(a,&theta);
    double lnr = log(r);
    double newr = exp(exponent->real*lnr - exponent->imag * theta);
    double newtheta = exponent->imag*lnr+exponent->real*theta;
    out->real = newr*cos(newtheta);
    out->imag = newr*sin(newtheta);

}

/** performs out = a / b */
void complex_div(objectcomplex *a, objectcomplex *b, objectcomplex *out){
    // (a_r + i * a_i)/(b_r + i * b_i) = a*b^c/|b|^2
    // (a_r + i * a_i) * (b_r - i * b_i)/ (b_r^2 + b_i^2)
    double denom = b->real*b->real + b->imag * b->imag;

    out->real = (a->real * b->real + a->imag * b->imag)/denom;
    out->imag = (a->imag * b->real - a->real * b->imag)/denom;
}

/** performs out = 1/a */
void complex_invert(objectcomplex *a, objectcomplex *out){
    double rsq = a->real*a->real + a->imag*a->imag;
    out->real = a->real/rsq;
    out->imag = -a->imag/rsq;
}

/** performs out = conj(a) by negating the imaginary part */
void complex_conj(objectcomplex *a, objectcomplex *out) {
    out->real = a->real;
    out->imag = -a->imag;
}

/** performs out = |a| */
void complex_abs(objectcomplex *a, double *out){
    *out = sqrt(a->real*a->real+a->imag*a->imag);
}

/** calculates theta in the complex representation a = r e^{i theta}  */
void complex_angle(objectcomplex *a, double *out){
    *out = atan2(a->imag,a->real);
}

/** Prints a complex */
void complex_print(objectcomplex *a) {
    char sign = '+';
    if (a->imag<0) {
        sign = '-';
    }

    printf("%g %c %gim",(fabs(a->real)<2*MORPHO_EPS ? 0 : a->real),sign,(fabs(a->imag)<2*MORPHO_EPS ? 0 : fabs(a->imag)));
}

/* **********************************************************************
 * Complex veneer class
 * ********************************************************************* */

/** Constructs a Complex object */
value complex_constructor(vm *v, int nargs, value *args) {
    double real, imag;
    objectcomplex *new=NULL;
    value out=MORPHO_NIL;
    // expect 2 aruments

    if ( nargs==2){

        // make sure both are numbers and cast them to floats
        if (MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
            morpho_valuetofloat(MORPHO_GETARG(args, 0), &real);
        } else morpho_runtimeerror(v, COMPLEX_CONSTRUCTOR);
        

        if (MORPHO_ISNUMBER(MORPHO_GETARG(args, 1))) {
            morpho_valuetofloat(MORPHO_GETARG(args, 1), &imag);
        } else morpho_runtimeerror(v, COMPLEX_CONSTRUCTOR);

    } else morpho_runtimeerror(v, COMPLEX_CONSTRUCTOR);

    new = object_newcomplex(real, imag);
    
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
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
    objectcomplex *c=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    complex_print(c);
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
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
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


/** Complex norm */
value Complex_abs(vm *v, int nargs, value *args) {
    objectcomplex *a=MORPHO_GETCOMPLEX(MORPHO_SELF(args));
    double val;
    complex_abs(a,&val);
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
MORPHO_METHOD(COMPLEX_ABS_METHOD, Complex_abs, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(COMPLEX_CONJUGATE_METHOD, Complex_conjugate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(COMPLEX_REAL_METHOD, Complex_getreal, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(COMPLEX_IMAG_METHOD, Complex_getimag, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Complex_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************* */

void complex_initialize(void) {
    objectcomplextype=object_addtype(&objectcomplexdefn);
    
    builtin_addfunction(COMPLEX_CLASSNAME, complex_constructor, BUILTIN_FLAGSEMPTY);
    
    value complexclass=builtin_addclass(COMPLEX_CLASSNAME, MORPHO_GETCLASSDEFINITION(ComplexNum), MORPHO_NIL);
    object_setveneerclass(OBJECT_COMPLEX, complexclass);

    morpho_defineerror(COMPLEX_CONSTRUCTOR, ERROR_HALT, COMPLEX_CONSTRUCTOR_MSG);
    morpho_defineerror(COMPLEX_ARITHARGS, ERROR_HALT, COMPLEX_ARITHARGS_MSG);
    morpho_defineerror(COMPLEX_INVLDNARG, ERROR_HALT, COMPLEX_INVLDNARG_MSG);
}
