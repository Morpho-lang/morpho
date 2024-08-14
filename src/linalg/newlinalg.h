/** @file newlinalg.h
 *  @author T J Atherton
 *
 *  @brief Linear algebra interface to BLAS and LAPACK
 */

#ifndef newlinalg_h
#define newlinalg_h

#include "complex.h"

/* -------------------------------------------------------
 * Matrix object type
 * ------------------------------------------------------- */

extern objecttype objectxmatrixtype;
#define OBJECT_XMATRIX objectxmatrixtype

typedef struct {
    object obj;
    int nrows;
    int ncols;
    long nels;
    double *elements;
    double matrixdata[];
} objectxmatrix;

/* -------------------------------------------------------
 * XMatrix veneer class
 * ------------------------------------------------------- */

#define XMATRIX_CLASSNAME                   "XMatrix"

void xmatrix_initialize(void);

#endif /* newlinalg_h */
