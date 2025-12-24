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

typedef int MatrixIdx_t;
typedef size_t MatrixCount_t;

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

/** Function that materializes a value from a pointer to an element */
typedef value (*xmatrix_getelfn_t) (vm *, double *);

/** Function that solves the linear system a.x = b
 * @param[in|out] a - lhs; overwritten by LU decomposition
 * @param[in|out] b - rhs; overwritten by solution
 * @param[out] pivot - you must provide an array with the same number of rows as a.
 * @returns a matrix error code */
typedef linalgError_t (*xmatrix_solvefn_t) (objectxmatrix *a, objectxmatrix *b, int *pivot);

/** Function that finds the eigenvalues of a matrix
 * @param[in|out] a - lhs; overwritten
 * @param[out] w -  eigenvalues; dimension N
 * @param[out] vec - right eigenvectors. Can be NULL if only eigenvalues requested
 * @returns a matrix error code */
typedef linalgError_t (*xmatrix_eigenfn_t) (objectxmatrix *a, MorphoComplex *w, objectxmatrix *vec);

/** Function that sets a given entry given a value */
//typedef void (*xmatrix_setfn_t) (objectxmatrix *, void *, value v);

typedef struct {
    xmatrix_printelfn_t printelfn;
    xmatrix_getelfn_t   getelfn;
    xmatrix_solvefn_t   solvefn;
    xmatrix_eigenfn_t   eigenfn;
//    xmatrix_setfn_t     setfn;
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
#define XMATRIX_INNER_METHOD                "inner"
#define XMATRIX_INVERSE_METHOD              "inverse"
#define XMATRIX_NORM_METHOD                 "norm"
#define XMATRIX_RESHAPE_METHOD              "reshape"
#define XMATRIX_SETCOLUMN_METHOD            "setColumn"
#define XMATRIX_TRACE_METHOD                "trace"
#define XMATRIX_TRANSPOSE_METHOD            "transpose"

#define XMATRIX_IDENTITYCONSTRUCTOR         "IdentityXMatrix"

void xmatrix_initialize(void);

value XMatrix_print(vm *v, int nargs, value *args);
value XMatrix_assign(vm *v, int nargs, value *args);
value XMatrix_clone(vm *v, int nargs, value *args);

value XMatrix_index__int(vm *v, int nargs, value *args);
value XMatrix_index__int_int(vm *v, int nargs, value *args);

value XMatrix_getcolumn__int(vm *v, int nargs, value *args);
value XMatrix_setcolumn__int_xmatrix(vm *v, int nargs, value *args);

value XMatrix_add__xmatrix(vm *v, int nargs, value *args);
value XMatrix_add__nil(vm *v, int nargs, value *args);
value XMatrix_add__x(vm *v, int nargs, value *args);
value XMatrix_sub__xmatrix(vm *v, int nargs, value *args);
value XMatrix_sub__x(vm *v, int nargs, value *args);
value XMatrix_subr__x(vm *v, int nargs, value *args);
value XMatrix_mul__float(vm *v, int nargs, value *args);
value XMatrix_div__float(vm *v, int nargs, value *args);
value XMatrix_div__xmatrix(vm *v, int nargs, value *args);
value XMatrix_acc__x_xmatrix(vm *v, int nargs, value *args);

value XMatrix_transpose(vm *v, int nargs, value *args);

value XMatrix_eigenvalues(vm *v, int nargs, value *args);
value XMatrix_eigensystem(vm *v, int nargs, value *args);

value XMatrix_reshape(vm *v, int nargs, value *args);
value XMatrix_enumerate(vm *v, int nargs, value *args);
value XMatrix_count(vm *v, int nargs, value *args);
value XMatrix_dimensions(vm *v, int nargs, value *args);

/* -------------------------------------------------------
 * Interface
 * ------------------------------------------------------- */

objectxmatrix *xmatrix_newwithtype(objecttype type, MatrixIdx_t nrows, MatrixIdx_t ncols, MatrixIdx_t nvals, bool zero);
objectxmatrix *xmatrix_clone(objectxmatrix *in);

linalgError_t xmatrix_setelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double value);
linalgError_t xmatrix_getelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double *value);
linalgError_t xmatrix_getelementptr(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double **value);

void xmatrix_print(vm *v, objectxmatrix *m);

#endif
