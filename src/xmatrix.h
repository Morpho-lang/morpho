/** @file xmatrix.h
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#ifndef xmatrix_h
#define xmatrix_h

/* -------------------------------------------------------
 * Matrix object type
 * ------------------------------------------------------- */

extern objecttype objectxmatrixtype;
#define OBJECT_XMATRIX objectxmatrixtype

extern objecttypedefn objectxmatrixdefn;

/** Matrices are a purely numerical collection type oriented toward linear algebra.
    Elements are stored in column-major format, i.e.
        [ 1 2 ]
        [ 3 4 ]
    is stored ( 1, 3, 2, 4 ) in memory. This is for compatibility with standard linear algebra packages */

typedef struct {
    object obj;
    int nrows;
    int ncols;
    int nels;
    double *elements;
    double matrixdata[];
} objectxmatrix;

/** Tests whether an object is a matrix */
#define MORPHO_ISXMATRIX(val) object_istype(val, OBJECT_XMATRIX)

/** Gets the object as an matrix */
#define MORPHO_GETXMATRIX(val)   ((objectxmatrix *) MORPHO_GETOBJECT(val))

/* -------------------------------------------------------
 * Matrix veneer class
 * ------------------------------------------------------- */

#define XMATRIX_CLASSNAME                   "XMatrix"

void xmatrix_initialize(void);

#endif
