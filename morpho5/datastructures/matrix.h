/** @file matrix.h
 *  @author T J Atherton
 *
 *  @brief Veneer class over the objectmatrix type that interfaces with blas and lapack
 */

#ifndef matrix_h
#define matrix_h

#include <stdio.h>

/** Use Apple's Accelerate library for LAPACK and BLAS */
#ifdef __APPLE__
#ifdef MORPHO_LINALG_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#define MATRIX_LAPACK_PRESENT
#endif
#endif

/** Otherwise, use LAPACKE */
#ifndef MATRIX_LAPACK_PRESENT
#include <cblas.h>
#include <lapacke.h>
#define MORPHO_LINALG_USE_LAPACKE
#define MATRIX_LAPACK_PRESENT
#endif

#define MATRIX_CLASSNAME "Matrix"

#define MATRIX_TRANSPOSE_METHOD "transpose"
#define MATRIX_TRACE_METHOD "trace"
#define MATRIX_INNER_METHOD "inner"
#define MATRIX_DET_METHOD "det"
#define MATRIX_EIGENVALUES_METHOD "eigenvalues"
#define MATRIX_EIGENSYSTEM_METHOD "eigensystem"
#define MATRIX_NORM_METHOD "norm"
#define MATRIX_GETCOLUMN_METHOD "column"
#define MATRIX_SETCOLUMN_METHOD "setcolumn"

#define MATRIX_DIMENSIONS_METHOD "dimensions"

#define MATRIX_INDICESOUTSIDEBOUNDS       "MtrxBnds"
#define MATRIX_INDICESOUTSIDEBOUNDS_MSG   "Matrix index out of bounds."

#define MATRIX_INVLDINDICES               "MtrxInvldIndx"
#define MATRIX_INVLDINDICES_MSG           "Matrix indices must be integers."

#define MATRIX_CONSTRUCTOR                "MtrxCns"
#define MATRIX_CONSTRUCTOR_MSG            "Matrix() constructor should be called either with dimensions or an array, list or matrix initializer."

#define MATRIX_INVLDARRAYINIT             "MtrxInvldInit"
#define MATRIX_INVLDARRAYINIT_MSG         "Invalid initializer passed to Matrix()."

#define MATRIX_ARITHARGS                  "MtrxInvldArg"
#define MATRIX_ARITHARGS_MSG              "Matrix arithmetic methods expect a matrix or number as their argument."

#define MATRIX_INCOMPATIBLEMATRICES       "MtrxIncmptbl"
#define MATRIX_INCOMPATIBLEMATRICES_MSG   "Matrices have incompatible shape."

#define MATRIX_SINGULAR                   "MtrxSnglr"
#define MATRIX_SINGULAR_MSG               "Matrix is singular."

#define MATRIX_NOTSQ                      "MtrxNtSq"
#define MATRIX_NOTSQ_MSG                  "Matrix is not square."

#define MATRIX_SETCOLARGS                 "MtrxStClArgs"
#define MATRIX_SETCOLARGS_MSG             "Method setcolumn expects an integer column index and a column matrix as arguments."

/** Macro to decide if a matrix is 'small' or 'large' and hence static or dynamic allocation should be used. */
#define MATRIX_ISSMALL(m) (m->nrows*m->ncols<MORPHO_MAXIMUMSTACKALLOC)

typedef enum { MATRIX_OK, MATRIX_INCMPTBLDIM, MATRIX_SING, MATRIX_INVLD, MATRIX_BNDS, MATRIX_NSQ, MATRIX_ALLOC } objectmatrixerror;

bool matrix_getarraydimensions(objectarray *array, unsigned int dim[], unsigned int maxdim, unsigned int *ndim);
value matrix_getarrayelement(objectarray *array, unsigned int ndim, unsigned int *indx);

bool matrix_getlistdimensions(objectlist *list, unsigned int dim[], unsigned int maxdim, unsigned int *ndim);
bool matrix_getlistelement(objectlist *list, unsigned int ndim, unsigned int *indx, value *val);

bool matrix_setelement(objectmatrix *matrix, unsigned int row, unsigned int col, double value);
bool matrix_getelement(objectmatrix *matrix, unsigned int row, unsigned int col, double *value);

bool matrix_getcolumn(objectmatrix *matrix, unsigned int col, double **v);
bool matrix_setcolumn(objectmatrix *matrix, unsigned int col, double *v);
bool matrix_addtocolumn(objectmatrix *m, unsigned int col, double alpha, double *v);

objectmatrixerror matrix_copy(objectmatrix *a, objectmatrix *out);
objectmatrixerror matrix_add(objectmatrix *a, objectmatrix *b, objectmatrix *out);
objectmatrixerror matrix_accumulate(objectmatrix *a, double lambda, objectmatrix *b);
objectmatrixerror matrix_sub(objectmatrix *a, objectmatrix *b, objectmatrix *out);
objectmatrixerror matrix_mul(objectmatrix *a, objectmatrix *b, objectmatrix *out);
objectmatrixerror matrix_inner(objectmatrix *a, objectmatrix *b, double *out);
objectmatrixerror matrix_divs(objectmatrix *a, objectmatrix *b, objectmatrix *out);
objectmatrixerror matrix_divl(objectmatrix *a, objectmatrix *b, objectmatrix *out);
objectmatrixerror matrix_inverse(objectmatrix *a, objectmatrix *out);
objectmatrixerror matrix_transpose(objectmatrix *a, objectmatrix *out);
objectmatrixerror matrix_trace(objectmatrix *a, double *out);
objectmatrixerror matrix_scale(objectmatrix *a, double scale);
objectmatrixerror matrix_identity(objectmatrix *a);
double matrix_sum(objectmatrix *a);
//objectmatrixerror matrix_det(objectmatrix *a, double *out);
//objectmatrixerror matrix_eigensystem(objectmatrix *a, double *val, objectmatrix *vec);

void matrix_print(objectmatrix *m);

void matrix_initialize(void);

#endif /* matrix_h */
