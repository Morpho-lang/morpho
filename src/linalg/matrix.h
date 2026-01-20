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

#include "linalg.h"

#define LINALG_MAXMATRIXDEFNS 4

/* -------------------------------------------------------
 * Matrix object type
 * ------------------------------------------------------- */

extern objecttype objectmatrixtype;
#define OBJECT_MATRIX objectmatrixtype

extern objecttypedefn objectmatrixdefn;

typedef int MatrixIdx_t; // Type used for matrix indices
typedef size_t MatrixCount_t; // Type used to count total number of elements

/** Matrices are a purely numerical collection type oriented toward linear algebra.
    Elements are stored in column-major format, i.e.
        [ 1 2 ]
        [ 3 4 ]
    is stored ( 1, 3, 2, 4 ) in memory. This is for compatibility with standard linear algebra packages */

typedef struct {
    object obj;
    MatrixIdx_t nrows;    // Number of rows
    MatrixIdx_t ncols;    // Number of columns
    MatrixIdx_t nvals;    // Number of doubles per entry
    MatrixCount_t nels;   // Total number of entries (nrows*ncols*nvals)
    double *elements;
    double matrixdata[];
} objectmatrix;

/** Tests whether an object is a matrix */
#define MORPHO_ISMATRIX(val) object_istype(val, OBJECT_MATRIX)

/** Gets the object as an matrix */
#define MORPHO_GETMATRIX(val)   ((objectmatrix *) MORPHO_GETOBJECT(val))

/** @brief Use to create static matrices on the C stack
    @details Intended for small matrices; Caller needs to supply a double array of size nr*nc. */
#define MORPHO_STATICMATRIX(darray, nr, nc)      { .obj.type=OBJECT_MATRIX, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .elements=darray, .nrows=nr, .ncols=nc, .nvals=1, .nels=nr*nc }

/** Macro to decide if a matrix is 'small' or 'large' and hence static or dynamic allocation should be used. */
#define MATRIX_ISSMALL(m) (m->nrows*m->ncols<MORPHO_MAXIMUMSTACKALLOC)

/* -------------------------------------------------------
 * Matrix interface definitions
 * ------------------------------------------------------- */

/** Function that prints a single matrix element */
typedef void (*matrix_printelfn_t) (vm *, double *);

/** Function that prints a single matrix element to a text buffer
 * @param[in] out - buffer to write to
 * @param[in] format - format string
 * @param[in] el - pointer to matrix element data
 * @returns true on success */
typedef bool (*matrix_printeltobufffn_t) (varray_char *out, char *format, double *el);

/** Function that materializes a value from a pointer to an element */
typedef value (*matrix_getelfn_t) (vm *, double *);

/** Function that sets the an element given a value */
typedef linalgError_t (*matrix_setelfn_t) (vm *, value, double *);

typedef enum {
    MATRIX_NORM_MAX,
    MATRIX_NORM_L1,
    MATRIX_NORM_INF,
    MATRIX_NORM_FROBENIUS,
} matrix_norm_t;

/** Convert matrix_norm_t to a character for use with lapack routines */
char matrix_normtolapack(matrix_norm_t norm);

/** Compute various matrix norms */
typedef double (*matrix_normfn_t) (objectmatrix *a, matrix_norm_t nrm);

/** Function that solves the linear system a.x = b
 * @param[in|out] a - lhs; overwritten by LU decomposition
 * @param[in|out] b - rhs; overwritten by solution
 * @param[out] pivot - you must provide an array with the same number of rows as a.
 * @returns a matrix error code */
typedef linalgError_t (*matrix_solvefn_t) (objectmatrix *a, objectmatrix *b, int *pivot);

/** Function that finds the eigenvalues of a matrix
 * @param[in|out] a - matrix to diagonalize; overwritten
 * @param[out] w -  eigenvalues; dimension N
 * @param[out] vec - right eigenvectors. Can be NULL if only eigenvalues requested
 * @returns a matrix error code */
typedef linalgError_t (*matrix_eigenfn_t) (objectmatrix *a, MorphoComplex *w, objectmatrix *vec);

/** Function that finds the svd of a matrix
 * @param[in|out] a - overwritten
 * @param[out] s -  singular values
 * @param[out] u - left singular vectors
 * @param[out] v - right singular vectors (transposed so columns contain singular vectors)
 * @returns a matrix error code */
typedef linalgError_t (*matrix_svdfn_t) (objectmatrix *a, double *s, objectmatrix *u, objectmatrix *vt);

/** Function that finds the QR decomposition of a matrix
 * @param[in|out] a - overwritten with R in upper triangle and reflectors below
 * @param[out] q - orthogonal matrix Q
 * @param[out] r - upper triangular matrix R
 * @returns a matrix error code */
typedef linalgError_t (*matrix_qrfn_t) (objectmatrix *a, objectmatrix *q, objectmatrix *r);

typedef struct {
    matrix_printelfn_t printelfn;
    matrix_printeltobufffn_t printeltobufffn;
    matrix_getelfn_t   getelfn;
    matrix_setelfn_t   setelfn;
    matrix_normfn_t    normfn;
    matrix_solvefn_t   solvefn;
    matrix_eigenfn_t   eigenfn;
    matrix_svdfn_t     svdfn;
    matrix_qrfn_t      qrfn;
} matrixinterfacedefn;

void matrix_addinterface(matrixinterfacedefn *defn);
matrixinterfacedefn *matrix_getinterface(objectmatrix *a);

/* -------------------------------------------------------
 * Matrix veneer class
 * ------------------------------------------------------- */

#define MATRIX_CLASSNAME                   "Matrix"

#define MATRIX_GETCOLUMN_METHOD            "column"
#define MATRIX_DIMENSIONS_METHOD           "dimensions"
#define MATRIX_EIGENVALUES_METHOD          "eigenvalues"
#define MATRIX_EIGENSYSTEM_METHOD          "eigensystem"
#define MATRIX_SVD_METHOD                  "svd"
#define MATRIX_QR_METHOD                   "qr"
#define MATRIX_INNER_METHOD                "inner"
#define MATRIX_INVERSE_METHOD              "inverse"
#define MATRIX_NORM_METHOD                 "norm"
#define MATRIX_OUTER_METHOD                "outer"
#define MATRIX_RESHAPE_METHOD              "reshape"
#define MATRIX_ROLL_METHOD                 "roll"
#define MATRIX_SETCOLUMN_METHOD_DEPRECATED "setcolumn"
#define MATRIX_SETCOLUMN_METHOD            "setColumn"
#define MATRIX_TRACE_METHOD                "trace"
#define MATRIX_TRANSPOSE_METHOD            "transpose"

#define MATRIX_IDENTITYCONSTRUCTOR         "IdentityMatrix"

void matrix_initialize(void);

#define IMPLEMENTATIONFN(fn) value fn (vm *v, int nargs, value *args)

IMPLEMENTATIONFN(matrix_constructor__matrix);

IMPLEMENTATIONFN(Matrix_print);
IMPLEMENTATIONFN(Matrix_format);
IMPLEMENTATIONFN(Matrix_assign);
IMPLEMENTATIONFN(Matrix_clone);

IMPLEMENTATIONFN(Matrix_index__int);
IMPLEMENTATIONFN(Matrix_index__int_int);
IMPLEMENTATIONFN(Matrix_index__x_x);
IMPLEMENTATIONFN(Matrix_index__err);
IMPLEMENTATIONFN(Matrix_setindex__int_x);
IMPLEMENTATIONFN(Matrix_setindex__int_int_x);
IMPLEMENTATIONFN(Matrix_setindex__x_x_matrix);

IMPLEMENTATIONFN(Matrix_getcolumn__int);
IMPLEMENTATIONFN(Matrix_setcolumn__int_matrix);

IMPLEMENTATIONFN(Matrix_add__matrix);
IMPLEMENTATIONFN(Matrix_add__nil);
IMPLEMENTATIONFN(Matrix_add__x);
IMPLEMENTATIONFN(Matrix_sub__matrix);
IMPLEMENTATIONFN(Matrix_sub__x);
IMPLEMENTATIONFN(Matrix_subr__x);
IMPLEMENTATIONFN(Matrix_mul__float);
IMPLEMENTATIONFN(Matrix_div__float);
IMPLEMENTATIONFN(Matrix_div__matrix);
IMPLEMENTATIONFN(Matrix_acc__x_x_matrix);

IMPLEMENTATIONFN(Matrix_norm__x);
IMPLEMENTATIONFN(Matrix_norm);
IMPLEMENTATIONFN(Matrix_sum);
IMPLEMENTATIONFN(Matrix_transpose);
IMPLEMENTATIONFN(Matrix_eigenvalues);
IMPLEMENTATIONFN(Matrix_eigensystem);
IMPLEMENTATIONFN(Matrix_svd);
IMPLEMENTATIONFN(Matrix_qr);
IMPLEMENTATIONFN(Matrix_reshape);
IMPLEMENTATIONFN(Matrix_roll__int);
IMPLEMENTATIONFN(Matrix_roll__int_int);
IMPLEMENTATIONFN(Matrix_enumerate);
IMPLEMENTATIONFN(Matrix_count);
IMPLEMENTATIONFN(Matrix_dimensions);

#undef DECLARE_IMPLEMENTATIONFN

/* -------------------------------------------------------
 * Errors
 * ------------------------------------------------------- */

#define MATRIX_CONSTRUCTOR                "MtrxCns"
#define MATRIX_CONSTRUCTOR_MSG            "Matrix() constructor should be called either with integer dimensions or an array, list, tuple or matrix initializer."

#define MATRIX_IDENTCONSTRUCTOR           "MtrxIdnttyCns"
#define MATRIX_IDENTCONSTRUCTOR_MSG       "IdentityMatrix expects the dimension as its argument."

/* -------------------------------------------------------
 * Interface
 * ------------------------------------------------------- */

bool matrix_isamatrix(value val);

objectmatrix *matrix_newwithtype(objecttype type, MatrixIdx_t nrows, MatrixIdx_t ncols, MatrixIdx_t nvals, bool zero);
objectmatrix *matrix_new(MatrixIdx_t nrows, MatrixIdx_t ncols, bool zero);
objectmatrix *matrix_clone(objectmatrix *in);
objectmatrix *matrix_listconstructor(vm *v, value lst, objecttype type, MatrixIdx_t nvals);
objectmatrix *matrix_arrayconstructor(vm *v, objectarray *a, objecttype type, MatrixIdx_t nvals);

MatrixCount_t matrix_countdof(objectmatrix *a);

linalgError_t matrix_axpy(double alpha, objectmatrix *x, objectmatrix *y);
linalgError_t matrix_copy(objectmatrix *x, objectmatrix *y);
linalgError_t matrix_copyat(objectmatrix *a, objectmatrix *out, int row0, int col0);
void matrix_scale(objectmatrix *x, double scale);
linalgError_t matrix_zero(objectmatrix *x);
linalgError_t matrix_identity(objectmatrix *x);
linalgError_t matrix_mmul(double alpha, objectmatrix *x, objectmatrix *y, double beta, objectmatrix *z);
linalgError_t matrix_mul(objectmatrix *x, objectmatrix *y, objectmatrix *z);
linalgError_t matrix_addscalar(objectmatrix *x, double alpha, double beta);
linalgError_t matrix_transpose(objectmatrix *x, objectmatrix *y);

double matrix_norm(objectmatrix *a, matrix_norm_t norm);
void matrix_sum(objectmatrix *a, double *sum);
linalgError_t matrix_trace(objectmatrix *a, double *out);

linalgError_t matrix_inner(objectmatrix *x, objectmatrix *y, double *out);

linalgError_t matrix_solvesmall(objectmatrix *a, objectmatrix *b);
linalgError_t matrix_solvelarge(objectmatrix *a, objectmatrix *b);
linalgError_t matrix_solve(objectmatrix *a, objectmatrix *b);
linalgError_t matrix_inverse(objectmatrix *a);

linalgError_t matrix_svd(objectmatrix *a, double *s, objectmatrix *u, objectmatrix *vt);

linalgError_t matrix_qr(objectmatrix *a, objectmatrix *q, objectmatrix *r);

linalgError_t matrix_validateindex(MatrixIdx_t *idx, MatrixIdx_t size);
linalgError_t matrix_setelement(objectmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double value);
linalgError_t matrix_getelement(objectmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double *value);
linalgError_t matrix_getelementptr(objectmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double **value);
linalgError_t matrix_getcolumnptr(objectmatrix *matrix, MatrixIdx_t col, double **value);
linalgError_t matrix_setcolumn(objectmatrix *a, MatrixIdx_t col, objectmatrix *b);
linalgError_t matrix_setcolumnptr(objectmatrix *a, MatrixIdx_t col, double *b);
linalgError_t matrix_addtocolumnptr(objectmatrix *a, MatrixIdx_t col, double alpha, double *b);

MatrixCount_t matrix_countdof(objectmatrix *a);

void matrix_print(vm *v, objectmatrix *m);

#endif /* MORPHO_INCLUDE_LINALG */

#endif /* matrix_h */
