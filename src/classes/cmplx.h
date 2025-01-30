/** @file cmplx.h
 *  @author D Hellstein and T J Atherton
 *
 *  @brief Veneer class over the objectcomplex type
 */

#ifndef cmplx_h
#define cmplx_h

#include <stdio.h>
#include <complex.h>
#include "classes.h"
#include "platform.h"

/* -------------------------------------------------------
 * Complex objects
 * ------------------------------------------------------- */

extern objecttype objectcomplextype;
#define OBJECT_COMPLEX objectcomplextype

typedef struct {
    object obj;
    MorphoComplex Z;
} objectcomplex;

/** Creates a static complex number */
#define MORPHO_STATICCOMPLEX(real,imag)      { .obj.type=OBJECT_COMPLEX, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .Z=MCBuild(real,imag)}

/** Tests whether an object is a complex */
#define MORPHO_ISCOMPLEX(val) object_istype(val, OBJECT_COMPLEX)

/** Gets the object as a complex */
#define MORPHO_GETCOMPLEX(val)   ((objectcomplex *) MORPHO_GETOBJECT(val))

/** Gets the object as a C-style MorphoComplex */
#define MORPHO_GETDOUBLECOMPLEX(val)   ((MorphoComplex) ((objectcomplex *) MORPHO_GETOBJECT(val))->Z)

/** Creates a complex object */
objectcomplex *object_newcomplex(double real, double imag);

/** Creates a new complex from an existing complex */
objectcomplex *object_clonecomplex(objectcomplex *array);

/** Clones a value that holds a complex */
value object_clonecomplexvalue(value val);
/** creates a complex object from a real value */
objectcomplex *object_complexfromfloat(double val);

/** tests the equality of two complex numbers */
bool complex_isequal(objectcomplex *a, objectcomplex *b);

/** tests equality between a complex number and a value */
bool complex_isequaltonumber(objectcomplex *a, value b);

/* -------------------------------------------------------
 * Complex class
 * ------------------------------------------------------- */

#define COMPLEX_CLASSNAME                   "Complex"

#define COMPLEX_CONJUGATE_METHOD            "conj"
#define COMPLEX_ABS_METHOD                  "abs"
#define COMPLEX_REAL_METHOD                 "real"
#define COMPLEX_IMAG_METHOD                 "imag"
#define COMPLEX_ANGLE_METHOD                "angle"

/* -------------------------------------------------------
 * Complex error messages
 * ------------------------------------------------------- */

#define COMPLEX_CONSTRUCTOR                "CmplxCns"
#define COMPLEX_CONSTRUCTOR_MSG            "Complex() constructor should be called with two floats"

#define COMPLEX_ARITHARGS                  "CmplxInvldArg"
#define COMPLEX_ARITHARGS_MSG              "Complex arithmetic methods expect a complex or number as their argument."

#define COMPLEX_INVLDNARG                  "CmpxArg"
#define COMPLEX_INVLDNARG_MSG              "Complex Operation did not exect those arguments."

/* -------------------------------------------------------
 * Complex interface
 * ------------------------------------------------------- */

void complex_copy(objectcomplex *a, objectcomplex *out);
void complex_add(objectcomplex *a, objectcomplex *b, objectcomplex *out);
void complex_sub(objectcomplex *a, objectcomplex *b, objectcomplex *out);
void complex_mul(objectcomplex *a, objectcomplex *b, objectcomplex *out);
void complex_div(objectcomplex *a, objectcomplex *b, objectcomplex *out);
void complex_conj(objectcomplex *a, objectcomplex *out);
void complex_abs(objectcomplex *a, double *out);
void complex_angle(objectcomplex *a, double *out);
void complex_getreal(objectcomplex *c, double *value);
void complex_getimag(objectcomplex *c, double *value);

void complex_print(vm *v, objectcomplex *m);

/* Built-in fucntions */

value complex_builtinexp(vm *v, objectcomplex *c);
value complex_builtinfabs(vm *v, objectcomplex *c);
value complex_builtinexp(vm *v, objectcomplex *c);
value complex_builtinlog(vm *v, objectcomplex *c);
value complex_builtinlog10(vm *v, objectcomplex *c);

value complex_builtinsin(vm *v, objectcomplex *c);
value complex_builtincos(vm *v, objectcomplex *c);
value complex_builtintan(vm *v, objectcomplex *c);
value complex_builtinasin(vm *v, objectcomplex *c);
value complex_builtinacos(vm *v, objectcomplex *c);

value complex_builtinsinh(vm *v, objectcomplex *c);
value complex_builtincosh(vm *v, objectcomplex *c);
value complex_builtintanh(vm *v, objectcomplex *c);
value complex_builtinsqrt(vm *v, objectcomplex *c);

value complex_builtinfloor(vm *v, objectcomplex *c);
value complex_builtinceil(vm *v, objectcomplex *c);

value complex_builtinisfinite(objectcomplex *c);
value complex_builtinisinf(objectcomplex *c);
value complex_builtinisnan(objectcomplex *c);

value complex_builtinatan(vm *v, value c);
value complex_builtinatan2(vm *v, value c1, value c2);

/* Complex methods */

value Complex_getreal(vm *v, int nargs, value *args);
value Complex_getimag(vm *v, int nargs, value *args);
value Complex_angle(vm *v, int nargs, value *args);
value Complex_conj(vm *v, int nargs, value *args);

void complex_initialize(void);

#endif /* complex_h */
