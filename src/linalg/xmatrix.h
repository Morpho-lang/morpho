/** @file matrix.h
 *  @author T J Atherton
 *
 *  @brief Veneer class over the objectmatrix type that interfaces with blas and lapack
 */

#ifndef matrix_h
#define matrix_h

#include "build.h"
#ifdef MORPHO_INCLUDE_LINALG

#include <stdio.h>
#include "classes.h"
/** Use Apple's Accelerate library for LAPACK and BLAS */
#ifdef __APPLE__
#ifdef MORPHO_LINALG_USE_ACCELERATE
#define ACCELERATE_NEW_LAPACK
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

#include "cmplx.h"
#include "list.h"

/* -------------------------------------------------------
 * Matrix objects
 * ------------------------------------------------------- */

extern objecttype objectmatrixtype;
#define OBJECT_MATRIX objectmatrixtype

/** Matrices are a purely numerical collection type oriented toward linear algebra.
    Elements are stored in column-major format, i.e.
        [ 1 2 ]
        [ 3 4 ]
    is stored ( 1, 3, 2, 4 ) in memory. This is for compatibility with standard linear algebra packages */

typedef struct {
    object obj;
    unsigned int nrows;
    unsigned int ncols;
    double *elements;
    double matrixdata[];
} objectmatrix;

/** Tests whether an object is a matrix */
#define MORPHO_ISMATRIX(val) object_istype(val, OBJECT_MATRIX)

/** Gets the object as an matrix */
#define MORPHO_GETMATRIX(val)   ((objectmatrix *) MORPHO_GETOBJECT(val))

/** Creates a matrix object */
objectmatrix *object_newmatrix(unsigned int nrows, unsigned int ncols, bool zero);

/** Creates a new matrix from an array */
objectmatrix *object_matrixfromarray(objectarray *array);

/** Creates a new matrix from an existing matrix */
objectmatrix *object_clonematrix(objectmatrix *array);

/** @brief Use to create static matrices on the C stack
    @details Intended for small matrices; Caller needs to supply a double array of size nr*nc. */
#define MORPHO_STATICMATRIX(darray, nr, nc)      { .obj.type=OBJECT_MATRIX, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .elements=darray, .nrows=nr, .ncols=nc }

/** Macro to decide if a matrix is 'small' or 'large' and hence static or dynamic allocation should be used. */
#define MATRIX_ISSMALL(m) (m->nrows*m->ncols<MORPHO_MAXIMUMSTACKALLOC)

/* -------------------------------------------------------
 * Matrix veneer class
 * ------------------------------------------------------- */

#define MATRIX_CLASSNAME "Matrix"

#define MATRIX_IDENTITYCONSTRUCTOR "IdentityMatrix"

#define MATRIX_INVERSE_METHOD "inverse"
#define MATRIX_TRANSPOSE_METHOD "transpose"
#define MATRIX_TRACE_METHOD "trace"
#define MATRIX_INNER_METHOD "inner"
#define MATRIX_OUTER_METHOD "outer"
#define MATRIX_DET_METHOD "det"
#define MATRIX_EIGENVALUES_METHOD "eigenvalues"
#define MATRIX_EIGENSYSTEM_METHOD "eigensystem"
#define MATRIX_NORM_METHOD "norm"
#define MATRIX_GETCOLUMN_METHOD "column"
#define MATRIX_SETCOLUMN_METHOD "setcolumn"
#define MATRIX_RESHAPE_METHOD "reshape"
#define MATRIX_EIGENVALUES_METHOD "eigenvalues"
#define MATRIX_EIGENSYSTEM_METHOD "eigensystem"

#define MATRIX_DIMENSIONS_METHOD "dimensions"

/* -------------------------------------------------------
 * Matrix error messages
 * ------------------------------------------------------- */

#define MATRIX_INDICESOUTSIDEBOUNDS       "MtrxBnds"
#define MATRIX_INDICESOUTSIDEBOUNDS_MSG   "Matrix index out of bounds."

#define MATRIX_INVLDINDICES               "MtrxInvldIndx"
#define MATRIX_INVLDINDICES_MSG           "Matrix indices must be integers."

#define MATRIX_INVLDNUMINDICES            "MtrxInvldNumIndx"
#define MATRIX_INVLDNUMINDICES_MSG        "Matrix expects two arguments for indexing."

#define MATRIX_CONSTRUCTOR                "MtrxCns"
#define MATRIX_CONSTRUCTOR_MSG            "Matrix() constructor should be called either with dimensions or an array, list or matrix initializer."

#define MATRIX_IDENTCONSTRUCTOR           "MtrxIdnttyCns"
#define MATRIX_IDENTCONSTRUCTOR_MSG       "IdentityMatrix expects the dimension as its argument."

#define MATRIX_INVLDARRAYINIT             "MtrxInvldInit"
#define MATRIX_INVLDARRAYINIT_MSG         "Invalid initializer passed to Matrix()."

#define MATRIX_ARITHARGS                  "MtrxInvldArg"
#define MATRIX_ARITHARGS_MSG              "Matrix arithmetic methods expect a matrix or number as their argument."

#define MATRIX_RESHAPEARGS                "MtrxRShpArg"
#define MATRIX_RESHAPEARGS_MSG            "Reshape requires two integer arguments."

#define MATRIX_INCOMPATIBLEMATRICES       "MtrxIncmptbl"
#define MATRIX_INCOMPATIBLEMATRICES_MSG   "Matrices have incompatible shape."

#define MATRIX_SINGULAR                   "MtrxSnglr"
#define MATRIX_SINGULAR_MSG               "Matrix is singular."

#define MATRIX_NOTSQ                      "MtrxNtSq"
#define MATRIX_NOTSQ_MSG                  "Matrix is not square."

#define MATRIX_OPFAILED                   "MtrxOpFld"
#define MATRIX_OPFAILED_MSG               "Matrix operation failed."

#define MATRIX_SETCOLARGS                 "MtrxStClArgs"
#define MATRIX_SETCOLARGS_MSG             "Method setcolumn expects an integer column index and a column matrix as arguments."

#define MATRIX_NORMARGS                   "MtrxNrmArgs"
#define MATRIX_NORMARGS_MSG               "Method norm expects an (optional) numerical argument."

/* -------------------------------------------------------
 * objectmatrixerror type
 * ------------------------------------------------------- */

typedef enum { MATRIX_OK, MATRIX_INCMPTBLDIM, MATRIX_SING, MATRIX_INVLD, MATRIX_BNDS, MATRIX_NSQ, MATRIX_FAILED, MATRIX_ALLOC } objectmatrixerror;

/* -------------------------------------------------------
 * Matrix interface
 * ------------------------------------------------------- */

bool matrix_getarraydimensions(objectarray *array, unsigned int dim[], unsigned int maxdim, unsigned int *ndim);
value matrix_getarrayelement(objectarray *array, unsigned int ndim, unsigned int *indx);

bool matrix_getlistdimensions(objectlist *list, unsigned int dim[], unsigned int maxdim, unsigned int *ndim);
bool matrix_getlistelement(objectlist *list, unsigned int ndim, unsigned int *indx, value *val);

bool matrix_setelement(objectmatrix *matrix, unsigned int row, unsigned int col, double value);
bool matrix_getelement(objectmatrix *matrix, unsigned int row, unsigned int col, double *value);

bool matrix_getcolumn(objectmatrix *matrix, unsigned int col, double **v);
bool matrix_setcolumn(objectmatrix *matrix, unsigned int col, double *v);
bool matrix_addtocolumn(objectmatrix *m, unsigned int col, double alpha, double *v);

unsigned int matrix_countdof(objectmatrix *a);

objectmatrixerror matrix_copy(objectmatrix *a, objectmatrix *out);
objectmatrixerror matrix_copyat(objectmatrix *a, objectmatrix *out, int row0, int col0);
objectmatrixerror matrix_add(objectmatrix *a, objectmatrix *b, objectmatrix *out);
objectmatrixerror matrix_accumulate(objectmatrix *a, double lambda, objectmatrix *b);
objectmatrixerror matrix_sub(objectmatrix *a, objectmatrix *b, objectmatrix *out);
objectmatrixerror matrix_mul(objectmatrix *a, objectmatrix *b, objectmatrix *out);
objectmatrixerror matrix_inner(objectmatrix *a, objectmatrix *b, double *out);
objectmatrixerror matrix_outer(objectmatrix *a, objectmatrix *b, objectmatrix *out);
objectmatrixerror matrix_divs(objectmatrix *a, objectmatrix *b, objectmatrix *out);
objectmatrixerror matrix_divl(objectmatrix *a, objectmatrix *b, objectmatrix *out);
objectmatrixerror matrix_inverse(objectmatrix *a, objectmatrix *out);
objectmatrixerror matrix_transpose(objectmatrix *a, objectmatrix *out);
objectmatrixerror matrix_trace(objectmatrix *a, double *out);
objectmatrixerror matrix_scale(objectmatrix *a, double scale);
objectmatrixerror matrix_identity(objectmatrix *a);
objectmatrixerror matrix_zero(objectmatrix *a);
double matrix_sum(objectmatrix *a);
objectmatrixerror matrix_eigensystem(objectmatrix *a, double *wr, double *wi, objectmatrix *vec);
bool matrix_eigen(vm *v, objectmatrix *a, value *evals, value *evecs);

double matrix_norm(objectmatrix *a);
double matrix_L1norm(objectmatrix *a);
double matrix_Lnnorm(objectmatrix *a, double n);
double matrix_Linfnorm(objectmatrix *a);

void matrix_print(vm *v, objectmatrix *m);

void matrix_initialize(void);

#endif

#endif /* matrix_h */
