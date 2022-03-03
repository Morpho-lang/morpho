/** @file complex.h
 *  @author Danny Goldstein
 *
 *  @brief Veneer class over the objectcomplex type
 */

#ifndef complex_h
#define complex_h

#include <stdio.h>
#include "veneer.h"

/* -------------------------------------------------------
 * Complex objects
 * ------------------------------------------------------- */

extern objecttype objectcomplextype;
#define OBJECT_COMPLEX objectcomplextype

typedef struct {
    object obj;
    double real;
    double imag;
} objectcomplex;

/** Tests whether an object is a complex */
#define MORPHO_ISCOMPLEX(val) object_istype(val, OBJECT_COMPLEX)

/** Gets the object as an complex */
#define MORPHO_GETCOMPLEX(val)   ((objectcomplex *) MORPHO_GETOBJECT(val))

/** Creates a complex object */
objectcomplex *object_newcomplex(double real, double imag);

/** Creates a new complex from an existing complex */
objectcomplex *object_clonecomplex(objectcomplex *array);

/* -------------------------------------------------------
 * Complex class
 * ------------------------------------------------------- */

#define COMPLEX_CLASSNAME "Complex"

#define COMPLEX_CONJUGATE_METHOD "conj"
#define COMPLEX_ABS_METHOD "abs"

#define COMPLEX_CONSTRUCTOR                "CmplxCns"
#define COMPLEX_CONSTRUCTOR_MSG            "Complex() constructor should be called either two floats"

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

void complex_print(objectcomplex *m);

void complex_initialize(void);

#endif /* complex_h */
