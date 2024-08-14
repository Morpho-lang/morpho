/** @file complexmatrix.h
 *  @author T J Atherton
 *
 *  @brief Linear algebra interface to BLAS and LAPACK
 */

#ifndef complexmatrix_h
#define complexmatrix_h

#include "newlinalg.h"
#include "complex.h"

/* -------------------------------------------------------
 * ComplexMatrix object type
 * ------------------------------------------------------- */

extern objecttype objectcomplexmatrixtype;
#define OBJECT_COMPLEXMATRIX objectcomplexmatrixtype

typedef struct {
    object obj;
    int nrows;
    int ncols;
    long nels;
    complex double *elements;
    complex double matrixdata[];
} objectcomplexmatrix;

/* -------------------------------------------------------
 * ComplexMatrix veneer class
 * ------------------------------------------------------- */

#define COMPLEXMATRIX_CLASSNAME                   "ComplexMatrix"

void complexmatrix_initialize(void);

#endif /* newlinalg_h */
