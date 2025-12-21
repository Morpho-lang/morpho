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
typedef void (*xmatrix_printelfn_t) (vm *, objectxmatrix *, MatrixIdx_t, MatrixIdx_t);

/** Function that solves a linear system */
typedef linalgError_t (*xmatrix_solvefn_t) (objectxmatrix *, objectxmatrix *, int *);

typedef struct {
    xmatrix_printelfn_t printelfn;
    xmatrix_solvefn_t   solvefn;
} matrixinterfacedefn;

void xmatrix_addinterface(matrixinterfacedefn *defn);
matrixinterfacedefn *xmatrix_getinterface(objectxmatrix *a);

/* -------------------------------------------------------
 * Matrix veneer class
 * ------------------------------------------------------- */

#define XMATRIX_CLASSNAME                   "XMatrix"

#define XMATRIX_GETCOLUMN_METHOD            "column"
#define XMATRIX_SETCOLUMN_METHOD            "setColumn"
#define XMATRIX_DIMENSIONS_METHOD           "dimensions"
#define XMATRIX_INNER_METHOD                "inner"
#define XMATRIX_NORM_METHOD                 "norm"
#define XMATRIX_RESHAPE_METHOD              "reshape"

#define XMATRIX_IDENTITYCONSTRUCTOR         "IdentityXMatrix"

void xmatrix_initialize(void);

value XMatrix_print(vm *v, int nargs, value *args);
value XMatrix_assign(vm *v, int nargs, value *args);
value XMatrix_clone(vm *v, int nargs, value *args);

value XMatrix_add__xmatrix(vm *v, int nargs, value *args);
value XMatrix_sub__xmatrix(vm *v, int nargs, value *args);
value XMatrix_mul__float(vm *v, int nargs, value *args);
value XMatrix_div__float(vm *v, int nargs, value *args);
value XMatrix_div__xmatrix(vm *v, int nargs, value *args);

value XMatrix_getcolumn__int(vm *v, int nargs, value *args);
value XMatrix_setcolumn__int_xmatrix(vm *v, int nargs, value *args);

value XMatrix_reshape(vm *v, int nargs, value *args);
value XMatrix_dimensions(vm *v, int nargs, value *args);
value XMatrix_count(vm *v, int nargs, value *args);

/* -------------------------------------------------------
 * Interface
 * ------------------------------------------------------- */

objectxmatrix *xmatrix_newwithtype(objecttype type, MatrixIdx_t nrows, MatrixIdx_t ncols, MatrixIdx_t nvals, bool zero);

linalgError_t xmatrix_setelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double value);
linalgError_t xmatrix_getelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double *value);
linalgError_t xmatrix_getelementptr(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double **value);

void xmatrix_print(vm *v, objectxmatrix *m);

#endif
