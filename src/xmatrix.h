/** @file xmatrix.h
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#ifndef xmatrix_h
#define xmatrix_h

#define LINALG_MAXMATRIXDEFNS 4

/* -------------------------------------------------------
 * Matrix object type
 * ------------------------------------------------------- */

extern objecttype objectxmatrixtype;
#define OBJECT_XMATRIX objectxmatrixtype

extern objecttypedefn objectxmatrixdefn;

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
} objectxmatrix;

/** Tests whether an object is a matrix */
#define MORPHO_ISXMATRIX(val) object_istype(val, OBJECT_XMATRIX)

/** Gets the object as an matrix */
#define MORPHO_GETXMATRIX(val)   ((objectxmatrix *) MORPHO_GETOBJECT(val))

/** @brief Use to create static matrices on the C stack
    @details Intended for small matrices; Caller needs to supply a double array of size nr*nc. */
#define MORPHO_STATICXMATRIX(darray, nr, nc)      { .obj.type=OBJECT_XMATRIX, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .elements=darray, .nrows=nr, .ncols=nc, .nels=nr*nc }

/** Macro to decide if a matrix is 'small' or 'large' and hence static or dynamic allocation should be used. */
#define MATRIX_ISSMALL(m) (m->nrows*m->ncols<MORPHO_MAXIMUMSTACKALLOC)


/* -------------------------------------------------------
 * Matrix interface definitions
 * ------------------------------------------------------- */

/** Function that prints a single matrix element */
typedef void (*xmatrix_printelfn_t) (vm *, double *);

/** Function that prints a single matrix element to a text buffer
 * @param[in] out - buffer to write to
 * @param[in] format - format string
 * @param[in] el - pointer to matrix element data
 * @returns true on success */
typedef bool (*xmatrix_printeltobufffn_t) (varray_char *out, char *format, double *el);

/** Function that materializes a value from a pointer to an element */
typedef value (*xmatrix_getelfn_t) (vm *, double *);

/** Function that sets the an element given a value */
typedef linalgError_t (*xmatrix_setelfn_t) (vm *, value, double *);

typedef enum {
    XMATRIX_NORM_MAX,
    XMATRIX_NORM_L1,
    XMATRIX_NORM_INF,
    XMATRIX_NORM_FROBENIUS,
} xmatrix_norm_t;

/** Convert xmatrix_norm_t to a character for use with lapack routines */
char xmatrix_normtolapack(xmatrix_norm_t norm);

/** Compute various matrix norms */
typedef double (*xmatrix_normfn_t) (objectxmatrix *a, xmatrix_norm_t nrm);

/** Function that solves the linear system a.x = b
 * @param[in|out] a - lhs; overwritten by LU decomposition
 * @param[in|out] b - rhs; overwritten by solution
 * @param[out] pivot - you must provide an array with the same number of rows as a.
 * @returns a matrix error code */
typedef linalgError_t (*xmatrix_solvefn_t) (objectxmatrix *a, objectxmatrix *b, int *pivot);

/** Function that finds the eigenvalues of a matrix
 * @param[in|out] a - matrix to diagonalize; overwritten
 * @param[out] w -  eigenvalues; dimension N
 * @param[out] vec - right eigenvectors. Can be NULL if only eigenvalues requested
 * @returns a matrix error code */
typedef linalgError_t (*xmatrix_eigenfn_t) (objectxmatrix *a, MorphoComplex *w, objectxmatrix *vec);

/** Function that finds the svd of a matrix
 * @param[in|out] a - overwritten
 * @param[out] s -  singular values
 * @param[out] u - left singular vectors
 * @param[out] v - right singular vectors (transposed so columns contain singular vectors)
 * @returns a matrix error code */
typedef linalgError_t (*xmatrix_svdfn_t) (objectxmatrix *a, double *s, objectxmatrix *u, objectxmatrix *vt);

typedef struct {
    xmatrix_printelfn_t printelfn;
    xmatrix_printeltobufffn_t printeltobufffn;
    xmatrix_getelfn_t   getelfn;
    xmatrix_setelfn_t   setelfn;
    xmatrix_normfn_t    normfn;
    xmatrix_solvefn_t   solvefn;
    xmatrix_eigenfn_t   eigenfn;
    xmatrix_svdfn_t     svdfn;
} matrixinterfacedefn;

void xmatrix_addinterface(matrixinterfacedefn *defn);
matrixinterfacedefn *xmatrix_getinterface(objectxmatrix *a);

/* -------------------------------------------------------
 * Matrix veneer class
 * ------------------------------------------------------- */

#define XMATRIX_CLASSNAME                   "XMatrix"

#define XMATRIX_GETCOLUMN_METHOD            "column"
#define XMATRIX_DIMENSIONS_METHOD           "dimensions"
#define XMATRIX_EIGENVALUES_METHOD          "eigenvalues"
#define XMATRIX_EIGENSYSTEM_METHOD          "eigensystem"
#define XMATRIX_SVD_METHOD                  "svd"
#define XMATRIX_INNER_METHOD                "inner"
#define XMATRIX_INVERSE_METHOD              "inverse"
#define XMATRIX_NORM_METHOD                 "norm"
#define XMATRIX_OUTER_METHOD                "outer"
#define XMATRIX_RESHAPE_METHOD              "reshape"
#define XMATRIX_ROLL_METHOD                 "roll"
#define XMATRIX_SETCOLUMN_METHOD            "setColumn"
#define XMATRIX_TRACE_METHOD                "trace"
#define XMATRIX_TRANSPOSE_METHOD            "transpose"

#define XMATRIX_IDENTITYCONSTRUCTOR         "IdentityXMatrix"

void xmatrix_initialize(void);

#define IMPLEMENTATIONFN(fn) value fn (vm *v, int nargs, value *args)

IMPLEMENTATIONFN(xmatrix_constructor__xmatrix);

IMPLEMENTATIONFN(XMatrix_print);
IMPLEMENTATIONFN(XMatrix_format);
IMPLEMENTATIONFN(XMatrix_assign);
IMPLEMENTATIONFN(XMatrix_clone);

IMPLEMENTATIONFN(XMatrix_index__int);
IMPLEMENTATIONFN(XMatrix_index__int_int);
IMPLEMENTATIONFN(XMatrix_index__x_x);
IMPLEMENTATIONFN(XMatrix_setindex__int_x);
IMPLEMENTATIONFN(XMatrix_setindex__int_int_x);
IMPLEMENTATIONFN(XMatrix_setindex__x_x_xmatrix);

IMPLEMENTATIONFN(XMatrix_getcolumn__int);
IMPLEMENTATIONFN(XMatrix_setcolumn__int_xmatrix);

IMPLEMENTATIONFN(XMatrix_add__xmatrix);
IMPLEMENTATIONFN(XMatrix_add__nil);
IMPLEMENTATIONFN(XMatrix_add__x);
IMPLEMENTATIONFN(XMatrix_sub__xmatrix);
IMPLEMENTATIONFN(XMatrix_sub__x);
IMPLEMENTATIONFN(XMatrix_subr__x);
IMPLEMENTATIONFN(XMatrix_mul__float);
IMPLEMENTATIONFN(XMatrix_div__float);
IMPLEMENTATIONFN(XMatrix_div__xmatrix);
IMPLEMENTATIONFN(XMatrix_acc__x_xmatrix);

IMPLEMENTATIONFN(XMatrix_norm__x);
IMPLEMENTATIONFN(XMatrix_norm);
IMPLEMENTATIONFN(XMatrix_sum);
IMPLEMENTATIONFN(XMatrix_transpose);
IMPLEMENTATIONFN(XMatrix_eigenvalues);
IMPLEMENTATIONFN(XMatrix_eigensystem);
IMPLEMENTATIONFN(XMatrix_svd);
IMPLEMENTATIONFN(XMatrix_reshape);
IMPLEMENTATIONFN(XMatrix_roll__int);
IMPLEMENTATIONFN(XMatrix_roll__int_int);
IMPLEMENTATIONFN(XMatrix_enumerate);
IMPLEMENTATIONFN(XMatrix_count);
IMPLEMENTATIONFN(XMatrix_dimensions);

#undef DECLARE_IMPLEMENTATIONFN

/* -------------------------------------------------------
 * Interface
 * ------------------------------------------------------- */

bool xmatrix_isamatrix(value val);

objectxmatrix *xmatrix_newwithtype(objecttype type, MatrixIdx_t nrows, MatrixIdx_t ncols, MatrixIdx_t nvals, bool zero);
objectxmatrix *xmatrix_new(MatrixIdx_t nrows, MatrixIdx_t ncols, bool zero);
objectxmatrix *xmatrix_clone(objectxmatrix *in);
objectxmatrix *xmatrix_listconstructor(vm *v, value lst, objecttype type, MatrixIdx_t nvals);
objectxmatrix *xmatrix_arrayconstructor(vm *v, objectarray *a, objecttype type, MatrixIdx_t nvals);

linalgError_t xmatrix_axpy(double alpha, objectxmatrix *x, objectxmatrix *y);
linalgError_t xmatrix_copy(objectxmatrix *x, objectxmatrix *y);
void xmatrix_scale(objectxmatrix *x, double scale);
linalgError_t xmatrix_identity(objectxmatrix *x);
linalgError_t xmatrix_mmul(double alpha, objectxmatrix *x, objectxmatrix *y, double beta, objectxmatrix *z);
linalgError_t xmatrix_addscalar(objectxmatrix *x, double alpha, double beta);
linalgError_t xmatrix_transpose(objectxmatrix *x, objectxmatrix *y);

linalgError_t xmatrix_solve(objectxmatrix *a, objectxmatrix *b);

linalgError_t xmatrix_svd(objectxmatrix *a, double *s, objectxmatrix *u, objectxmatrix *vt);

linalgError_t xmatrix_setelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double value);
linalgError_t xmatrix_getelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double *value);
linalgError_t xmatrix_getelementptr(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double **value);

void xmatrix_print(vm *v, objectxmatrix *m);

#endif
