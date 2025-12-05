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

/* -------------------------------------------------------
 * Matrix callback types
 * ------------------------------------------------------- */

typedef void (*xmatrix_elprintfn) (vm *v, objectxmatrix *m, MatrixIdx_t i, MatrixIdx_t j);

/* -------------------------------------------------------
 * Matrix veneer class
 * ------------------------------------------------------- */

#define XMATRIX_CLASSNAME                   "XMatrix"

void xmatrix_initialize(void);

/* -------------------------------------------------------
 * Interface
 * ------------------------------------------------------- */

objectxmatrix *xmatrix_newwithtype(objecttype type, MatrixIdx_t nrows, MatrixIdx_t ncols, MatrixIdx_t nvals, bool zero);

bool xmatrix_setelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double value);
bool xmatrix_getelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double *value);
bool xmatrix_getelementptr(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double **value);

void xmatrix_print(vm *v, objectxmatrix *m, xmatrix_elprintfn fn);

#endif
