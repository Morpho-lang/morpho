/** @file matrix.c
 *  @author T J Atherton
 *
 *  @brief New matrices
*/

#include "build.h"
#ifdef MORPHO_INCLUDE_LINALG

#include <string.h>
#include "morpho.h"
#include "classes.h"

#include "matrix.h"
#include "sparse.h"
#include "format.h"

/* **********************************************************************
 * Matrix interface definitions
 * ********************************************************************** */

/** Hold the matrix interface definitions as they're created */
static matrixinterfacedefn _matrixdefn[LINALG_MAXMATRIXDEFNS];
objecttype matrixinterfacedefnnext=0; /** Type of the next object definition */

void matrix_addinterface(matrixinterfacedefn *defn) {
    if (matrixinterfacedefnnext<LINALG_MAXMATRIXDEFNS) {
        _matrixdefn[matrixinterfacedefnnext]=*defn;
        matrixinterfacedefnnext++;
    } else UNREACHABLE("Too many matrix interface definitions.");
}

matrixinterfacedefn *matrix_getinterface(objectmatrix *a) {
    int iindx = a->obj.type-OBJECT_MATRIX;
    if (iindx<LINALG_MAXMATRIXDEFNS) return &_matrixdefn[iindx];
    return NULL;
}

/** Checks if a value is a known kind of matrix. */
bool matrix_isamatrix(value val) {
    if (!MORPHO_ISOBJECT(val)) return false;
    int iindx=MORPHO_GETOBJECT(val)->type-OBJECT_MATRIX;
    return iindx>=0 && iindx<matrixinterfacedefnnext;
}

/* **********************************************************************
 * Matrix objects
 * ********************************************************************** */

objecttype objectmatrixtype;

/** Matrix object definitions */
size_t objectmatrix_sizefn(object *obj) {
    return sizeof(objectmatrix)+sizeof(double) * ((objectmatrix *) obj)->nels;
}

void objectmatrix_printfn(object *obj, void *v) {
    objectclass *klass=object_getveneerclass(obj->type);
    morpho_printf(v, "<");
    morpho_printvalue(v, klass->name);
    morpho_printf(v, ">");
}

objecttypedefn objectmatrixdefn = {
    .printfn=objectmatrix_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectmatrix_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/* **********************************************************************
 * Matrix utility functions
 * ********************************************************************** */

/* ----------------------
 * Matrix interface
 * ---------------------- */

static void _printelfn(vm *v, double *el) {
    double val=*el;
    morpho_printf(v, "%g", (fabs(val)<MORPHO_EPS ? 0 : val));
}

static bool _printeltobufffn(varray_char *out, char *format, double *el) {
    return format_printtobuffer(MORPHO_FLOAT(*el), format, out);
}

static value _getelfn(vm *v, double *el) {
    return MORPHO_FLOAT(*el);
}

static linalgError_t _setelfn(vm *v, value in, double *el) {
    if (!morpho_valuetofloat(in, el)) return LINALGERR_NON_NUMERICAL;
    return LINALGERR_OK;
}

/** Convert matrix_norm_t to a character for use with lapack routines */
char matrix_normtolapack(matrix_norm_t norm) {
    switch (norm) {
        case MATRIX_NORM_MAX: return 'M';
        case MATRIX_NORM_L1: return '1';
        case MATRIX_NORM_INF: return 'I';
        case MATRIX_NORM_FROBENIUS: return 'F';
    }
    return '\0';
}

/** Evaluate norms */
static double _normfn(objectmatrix *a, matrix_norm_t nrm) {
    char cnrm = matrix_normtolapack(nrm);
    int nrows=a->nrows, ncols=a->ncols;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    return LAPACKE_dlange(LAPACK_COL_MAJOR, cnrm, a->nrows, a->ncols, a->elements, a->nrows);
#else
    double work[a->nrows];
    return dlange_(&cnrm, &nrows, &ncols, a->elements, &nrows, work);
#endif
}

/** Low level linear solve */
static linalgError_t _solve(objectmatrix *a, objectmatrix *b, int *pivot) {
    int n=a->nrows, nrhs = b->ncols, info;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a->elements, n, pivot, b->elements, n);
#else
    dgesv_(&n, &nrhs, a->elements, &n, pivot, b->elements, &n, &info);
#endif
    
    return (info==0 ? LINALGERR_OK : (info>0 ? LINALGERR_MATRIX_SINGULAR : LINALGERR_LAPACK_INVLD_ARGS));
}

/** Low level eigensolver */
static linalgError_t _eigen(objectmatrix *a, MorphoComplex *w, objectmatrix *vec) {
    int info, n=a->nrows;
    double wr[n], wi[n];
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', (vec ? 'V' : 'N'), n, a->elements, n, wr, wi, NULL, n, (vec ? vec->elements : NULL), n);
#else
    int lwork=4*n; double work[4*n];
    dgeev_("N", (vec ? "V" : "N"), &n, a->elements, &n, wr, wi, NULL, &n, (vec ? vec->elements : NULL), &n, work, &lwork, &info);
#endif
    for (int i=0; i<n; i++) w[i]=MCBuild(wr[i], wi[i]);
    
    return (info==0 ? LINALGERR_OK : (info>0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS));
}

/** Low level SVD */
static linalgError_t _svd(objectmatrix *a, double *s, objectmatrix *u, objectmatrix *vt) {
    int info, m=a->nrows, n=a->ncols;
    int minmn = (m < n) ? m : n;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    double* superb = malloc(minmn * sizeof(double));
    info = LAPACKE_dgesvd(LAPACK_COL_MAJOR,
                          (u ? 'A' : 'N'),      // jobu: 'A' = all U columns, 'N' = no U
                          (vt ? 'A' : 'N'),     // jobvt: 'A' = all VT rows, 'N' = no VT
                          m, n,
                          a->elements, m,       // input matrix A (overwritten)
                          s,                    // singular values (min(m,n))
                          (u ? u->elements : NULL), m,  // U matrix (m×m)
                          (vt ? vt->elements : NULL), n,  // VT matrix (n×n)
                          superb
                         );
#else
    int lwork = -1;
    double work_query;
    // Query optimal work size
    dgesvd_((u ? "A" : "N"), (vt ? "A" : "N"), &m, &n, a->elements, &m, s,
            (u ? u->elements : NULL), &m, (vt ? vt->elements : NULL), &n,
            &work_query, &lwork, &info);
    if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
    
    lwork = (int)work_query;
    double work[lwork];
    dgesvd_((u ? "A" : "N"), (vt ? "A" : "N"), &m, &n, a->elements, &m, s,
            (u ? u->elements : NULL), &m, (vt ? vt->elements : NULL), &n,
            work, &lwork, &info);
#endif
    
    return (info == 0 ? LINALGERR_OK : (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS));
}

/** Low level QR decomposition without pivoting */
static linalgError_t _qr(objectmatrix *a, objectmatrix *q, objectmatrix *r) {
    int info, m=a->nrows, n=a->ncols;
    int minmn = (m < n) ? m : n;
    double tau[minmn];
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a->elements, m, tau);
    if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
#else
    int lwork = -1;
    double work_query;
    
    // Query optimal work size for DGEQRF, which is reused for DORGQR
    dgeqrf_(&m, &n, a->elements, &m, tau, &work_query, &lwork, &info);
    if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
    
    int lwork_geqrf = (int) work_query;
    double work[lwork_geqrf];
    lwork = lwork_geqrf;
    
    // Compute QR factorization without pivoting
    dgeqrf_(&m, &n, a->elements, &m, tau, work, &lwork, &info);
    if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
#endif
    
    // Extract R (upper triangle of a) into r
    // Copy entire matrix first, then zero out below the diagonal
    matrix_copy(a, r);
    // Only process columns where there are rows below the diagonal (j < m - 1)
    for (int j = 0; j < n && j < m - 1; j++) {
        memset(&r->elements[j * m + (j + 1)], 0, (m - j - 1) * sizeof(double));
    }
    
    // Generate Q from reflectors
    if (q) {
        // Copy reflectors from a to q (only first n columns, since a is m×n and q is m×m)
        // DGEQRF stores reflectors in lower triangle and R in upper triangle of first n columns
        for (int j = 0; j < n; j++) cblas_dcopy(m, &a->elements[j * m], 1, &q->elements[j * m], 1);
        
#ifdef MORPHO_LINALG_USE_LAPACKE
        info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, minmn, minmn, q->elements, m, tau);
        if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
#else
        lwork = lwork_geqrf;
        dorgqr_(&m, &minmn, &minmn, q->elements, &m, tau, work, &lwork, &info);
        if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
#endif
        
        // If Q should be m×m, zero out remaining columns if m > minmn
        // DORGQR only generates the first minmn columns, so we zero the rest
        if (m > minmn) memset(&q->elements[minmn * m], 0, (m - minmn) * m * sizeof(double));
    }
    
    return LINALGERR_OK;
}

/* ----------------------
 * Interface definition
 * ---------------------- */

matrixinterfacedefn matrixdefn = {
    .printelfn = _printelfn,
    .printeltobufffn = _printeltobufffn,
    .getelfn = _getelfn,
    .setelfn = _setelfn,
    .normfn = _normfn,
    .solvefn = _solve,
    .eigenfn = _eigen,
    .svdfn = _svd,
    .qrfn = _qr
};

/* ----------------------
 * Constructors
 * ---------------------- */

/** Create a generic matrix with given type and layout */
objectmatrix *matrix_newwithtype(objecttype type, MatrixIdx_t nrows, MatrixIdx_t ncols, MatrixIdx_t nvals, bool zero) {
    MatrixCount_t nels = nrows*ncols*nvals;
    objectmatrix *new = (objectmatrix *) object_new(sizeof(objectmatrix) + nels*sizeof(double), type);
    
    if (new) {
        new->nrows=nrows;
        new->ncols=ncols;
        new->nvals=nvals;
        new->nels=nels;
        new->elements=new->matrixdata;
        if (zero) memset(new->elements, 0, nels*sizeof(double));
    }
    
    return new;
}

/** Create a new real matrix */
objectmatrix *matrix_new(MatrixIdx_t nrows, MatrixIdx_t ncols, bool zero) {
    return matrix_newwithtype(OBJECT_MATRIX, nrows, ncols, 1, zero);
}

/** Clone a matrix */
objectmatrix *matrix_clone(objectmatrix *in) {
    objectmatrix *new = matrix_newwithtype(in->obj.type, in->nrows, in->ncols, in->nvals, false);
    
    if (new) cblas_dcopy((linalg_int_t) in->nels, in->elements, 1, new->elements, 1);
    return new;
}

static bool _getelement(value v, int i, value *out) {
    if (MORPHO_ISLIST(v)) {
        return list_getelement(MORPHO_GETLIST(v), i, out);
    } else if (MORPHO_ISTUPLE(v)) {
        return tuple_getelement(MORPHO_GETTUPLE(v), i, out);
    } else if (MORPHO_ISNUMBER(v) || MORPHO_ISCOMPLEX(v)) {
        if (i==0) { *out = v; return true; }
    }
    return false;
}

static bool _length(value v, int *len) {
    if (MORPHO_ISLIST(v)) {
        *len = list_length(MORPHO_GETLIST(v)); return true;
    } else if (MORPHO_ISTUPLE(v)) {
        *len = tuple_length(MORPHO_GETTUPLE(v)); return true;
    } else if (MORPHO_ISNUMBER(v) || MORPHO_ISCOMPLEX(v)) {
        *len = 1; return true; 
    }
    return false;
}

/** Create a matrix from a list of lists (or tuples) */
objectmatrix *matrix_listconstructor(vm *v, value lst, objecttype type, MatrixIdx_t nvals) {
    value iel, jel;
    
    int nrows=0, ncols=0, rlen;
    if (!_length(lst, &nrows)) return NULL;
    for (int i=0; i<nrows; i++) {
        if (_getelement(lst, i, &iel) &&
            _length(iel, &rlen)) {
            if (rlen>ncols) ncols=rlen;
        } else return NULL;
    }
    
    objectmatrix *new=matrix_newwithtype(type, nrows, ncols, nvals, true);
    if (!new) return NULL;
    
    for (int i=0; i<nrows; i++) {
        _getelement(lst, i, &iel);
        for (int j=0; j<ncols; j++) {
            if (_getelement(iel, j, &jel)) {
                if (matrix_getinterface(new)->setelfn(v, jel, new->elements+(j*nrows + i)*new->nvals)!=LINALGERR_OK) goto matrix_listconstructor_cleanup;
            }
        }
    }
    
    return new;
matrix_listconstructor_cleanup:
    object_free((object *) new);
    return NULL;
}

/** Construct a matrix from an array */
objectmatrix *matrix_arrayconstructor(vm *v, objectarray *a, objecttype type, MatrixIdx_t nvals) {
    int nrows = MORPHO_GETINTEGERVALUE(a->dimensions[0]);
    int ncols = MORPHO_GETINTEGERVALUE(a->dimensions[1]);
    
    objectmatrix *new=matrix_newwithtype(type, nrows, ncols, nvals, true);
    if (!new) return NULL;
    
    for (int i=0; i<nrows; i++) {
        for (int j=0; j<ncols; j++) {
            unsigned int indx[2]={ i, j };
            value el;
            if (array_getelement(a, 2, indx, &el)==ARRAY_OK) {
                matrix_getinterface(new)->setelfn(v, el, new->elements+(j*nrows + i)*new->nvals);
            }
        }
    }
    return new;
}

/* ----------------------
 * Accessing elements
 * ---------------------- */

 /** @brief Validates index bounds, converting negative indices to positive
 *  @param idx Pointer to the index, updated if valid and negative 
 *  @param size The size of the dimension
 *  @returns LINALGERR_OK if conversion successful, LINALGERR_INDX_OUT_OF_BNDS if out of bounds */
linalgError_t matrix_validateindex(MatrixIdx_t *idx, MatrixIdx_t size) {
    if (*idx < 0) {
        if (*idx < -size) return LINALGERR_INDX_OUT_OF_BNDS;
        *idx = size + *idx;
    } else if (*idx >= size) return LINALGERR_INDX_OUT_OF_BNDS;
    return LINALGERR_OK;
}

/** @brief Sets a matrix element.
    @returns LINALGERR_OK if successful, LINALGERR_INDX_OUT_OF_BNDS if index out of bounds */
linalgError_t matrix_setelement(objectmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double value) {
    MatrixIdx_t row_idx = row, col_idx = col;
    LINALG_ERRCHECKRETURN(matrix_validateindex(&row_idx, matrix->nrows));
    LINALG_ERRCHECKRETURN(matrix_validateindex(&col_idx, matrix->ncols));
    matrix->elements[matrix->nvals*(col_idx*matrix->nrows+row_idx)]=value;
    return LINALGERR_OK;
}

/** @brief Gets a matrix element
 *  @returns LINALGERR_OK if successful, LINALGERR_INDX_OUT_OF_BNDS if index out of bounds */
linalgError_t matrix_getelement(objectmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double *value) {
    MatrixIdx_t row_idx = row, col_idx = col;
    LINALG_ERRCHECKRETURN(matrix_validateindex(&row_idx, matrix->nrows));
    LINALG_ERRCHECKRETURN(matrix_validateindex(&col_idx, matrix->ncols));
    if (value) *value=matrix->elements[matrix->nvals*(col_idx*matrix->nrows+row_idx)];
    return LINALGERR_OK;
}

/** @brief Gets a pointer to a matrix element
 *  @returns LINALGERR_OK if successful, LINALGERR_INDX_OUT_OF_BNDS if index out of bounds */
linalgError_t matrix_getelementptr(objectmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double **value) {
    MatrixIdx_t row_idx = row, col_idx = col;
    LINALG_ERRCHECKRETURN(matrix_validateindex(&row_idx, matrix->nrows));
    LINALG_ERRCHECKRETURN(matrix_validateindex(&col_idx, matrix->ncols));
    if (value) *value=matrix->elements+matrix->nvals*(col_idx*matrix->nrows+row_idx);
    return LINALGERR_OK;
}

/** @brief Gets a pointer to a matrix column
 *  @returns true if the column is in the range of the matrix, false otherwise */
linalgError_t matrix_getcolumnptr(objectmatrix *matrix, MatrixIdx_t col, double **value) {
    return matrix_getelementptr(matrix, 0, col, value);
}

/** Copies the column col of matrix a into the column vector b */
linalgError_t matrix_getcolumn(objectmatrix *a, MatrixIdx_t col, objectmatrix *b) {
    MatrixIdx_t col_idx = col;
    LINALG_ERRCHECKRETURN(matrix_validateindex(&col_idx, a->ncols));
    if (b->nels!=a->nrows*a->nvals) return LINALGERR_INCOMPATIBLE_DIM;
    cblas_dcopy((linalg_int_t) b->nels, a->elements+a->nvals*col_idx*a->nrows, 1, b->elements, 1);
    return LINALGERR_OK;
}

/** Copies the column vector b into column col of matrix a */
linalgError_t matrix_setcolumn(objectmatrix *a, MatrixIdx_t col, objectmatrix *b) {
    MatrixIdx_t col_idx = col;
    LINALG_ERRCHECKRETURN(matrix_validateindex(&col_idx, a->ncols));
    if (b->nels!=a->nrows*a->nvals) return LINALGERR_INCOMPATIBLE_DIM;
    cblas_dcopy((linalg_int_t) b->nels, b->elements, 1, a->elements+a->nvals*col_idx*a->nrows, 1);
    return LINALGERR_OK;
}

/** Copies the column vector b as a raw list of doubles into column col of matrix a */
linalgError_t matrix_setcolumnptr(objectmatrix *a, MatrixIdx_t col, double *b) {
    MatrixIdx_t col_idx = col;
    LINALG_ERRCHECKRETURN(matrix_validateindex(&col_idx, a->ncols));
    cblas_dcopy((linalg_int_t) a->nrows*a->nvals, b, 1, a->elements+a->nvals*col_idx*a->nrows, 1);
    return LINALGERR_OK;
}

/** @brief Add a vector to a column in a matrix
 *  @param[in] m - the matrix
 *  @param[in] col - column number
 *  @param[in] alpha - scale
 *  @param[out] v - column entries (matrix->nrows in number) [should have m->nrows entries]
 *  @returns true on success */
linalgError_t matrix_addtocolumnptr(objectmatrix *a, MatrixIdx_t col, double alpha, double *b) {
    if (col<0 || col>=a->ncols) return LINALGERR_INDX_OUT_OF_BNDS;
    
    cblas_daxpy(a->nrows*a->nvals, alpha, b, 1, a->elements+a->nvals*col*a->nrows, 1);
    return LINALGERR_OK;
}

/** Counts the number of dofs in a matrix */
MatrixCount_t matrix_countdof(objectmatrix *a) {
    return a->ncols*a->nrows*a->nvals;
}

/* ----------------------
 * Arithmetic operations
 * ---------------------- */

/** Vector addition: Performs y <- alpha*x + y */
linalgError_t matrix_axpy(double alpha, objectmatrix *x, objectmatrix *y) {
    if (!(x->ncols==y->ncols && x->nrows==y->nrows)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_daxpy((linalg_int_t) x->nels, alpha, x->elements, 1, y->elements, 1);
    return LINALGERR_OK;
}

/** Copies a matrix  y <- x */
linalgError_t matrix_copy(objectmatrix *x, objectmatrix *y) {
    if (!(x->ncols==y->ncols && x->nrows==y->nrows)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_dcopy((linalg_int_t) x->nels, x->elements, 1, y->elements, 1);
    return LINALGERR_OK;
}

/** Copies one matrix into another at an arbitrary point */
linalgError_t matrix_copyat(objectmatrix *a, objectmatrix *out, int row0, int col0) {
    if (!(col0+a->ncols<=out->ncols && row0+a->nrows<=out->nrows)) return LINALGERR_INCOMPATIBLE_DIM;
    
    for (int j=0; j<a->ncols; j++) {
        for (int i=0; i<a->nrows; i++) {
            double *src, *dest;
            LINALG_ERRCHECKRETURN(matrix_getelementptr(a, i, j, &src));
            LINALG_ERRCHECKRETURN(matrix_getelementptr(out, row0+i, col0+j, &dest));
            memcpy(dest, src, sizeof(double)*a->nvals);
        }
    }
    return LINALGERR_OK;
}

/** Scales a matrix x <- scale * x >*/
void matrix_scale(objectmatrix *x, double scale) {
    cblas_dscal((linalg_int_t) x->nels, scale, x->elements, 1);
}

/** Loads the zero matrix a <- 0 */
linalgError_t matrix_zero(objectmatrix *x) {
    memset(x->elements, 0, sizeof(double)*x->nrows*x->ncols*x->nvals);
    return LINALGERR_OK;
}

/** Loads the identity matrix a <- I(n) */
linalgError_t matrix_identity(objectmatrix *x) {
    if (x->ncols!=x->nrows) return LINALGERR_NOT_SQUARE;
    matrix_zero(x);
    for (int i=0; i<x->nrows; i++) x->elements[x->nvals*(i+x->nrows*i)]=1.0;
    return LINALGERR_OK;
}

/** Performs z <- alpha*(x*y) + beta*z */
linalgError_t matrix_mmul(double alpha, objectmatrix *x, objectmatrix *y, double beta, objectmatrix *z) {
    if (!(x->ncols==y->nrows && x->nrows==z->nrows && y->ncols==z->ncols)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, x->nrows, y->ncols, x->ncols, alpha, x->elements, x->nrows, y->elements, y->nrows, beta, z->elements, z->nrows);
    return LINALGERR_OK;
}

linalgError_t matrix_mul(objectmatrix *x, objectmatrix *y, objectmatrix *z) {
    return matrix_mmul(1.0, x, y, 0.0, z);
}

/** Performs x <- alpha*x + beta */
linalgError_t matrix_addscalar(objectmatrix *x, double alpha, double beta) {
    for (MatrixCount_t i=0; i<x->ncols*x->nrows; i++) {
        for (int k=0; k<x->nvals; k++) {
            x->elements[i*x->nvals+k]*=alpha;
            if (k==0) x->elements[i*x->nvals+k]+=beta;
        }
    }
    return LINALGERR_OK;
}

/** Performs y <- x^T>*/
linalgError_t matrix_transpose(objectmatrix *x, objectmatrix *y) {
    if (!(x->ncols==y->nrows && x->nrows==y->ncols)) return LINALGERR_INCOMPATIBLE_DIM;
    
    for (MatrixCount_t i=0; i<x->ncols; i++) {
        for (MatrixCount_t j=0; j<x->nrows; j++) {
            for (int k=0; k<x->nvals; k++) {
                y->elements[j*y->nrows*y->nvals+i*y->nvals+k] = x->elements[i*x->nrows*x->nvals+j*x->nvals+k];
            }
        }
    }
    return LINALGERR_OK;
}

/* ----------------------
 * Unary operations
 * ---------------------- */

/** Computes various matrix norms */
double matrix_norm(objectmatrix *a, matrix_norm_t norm) {
    return matrix_getinterface(a)->normfn(a, norm);
}

/** Computes the sum of all elements in a matrix */
void matrix_sum(objectmatrix *a, double *sum) {
    double c[a->nvals], y, t;
    for (int i=0; i<a->nvals; i++) { sum[i]=0; c[i]=0; }
    
    for (MatrixCount_t i=0; i<a->nels; i+=a->nvals) {
        for (int k=0; k<a->nvals; k++) {
            y=a->elements[i+k]-c[k];
            t=sum[k]+y;
            c[k]=(t-sum[k])-y;
            sum[k]=t;
        }
    }
}

/** Calculate the trace of a matrix */
linalgError_t matrix_trace(objectmatrix *a, double *out) {
    if (a->nrows!=a->ncols) return LINALGERR_NOT_SQUARE;
    *out = 0.0;
    for (int i = 0; i < a->nrows; i++) {
        *out += a->elements[a->nvals * (i * a->nrows + i)];
    }
    
    return LINALGERR_OK;
}

/* ----------------------
 * Binary operations
 * ---------------------- */

/** Finds the Frobenius inner product of two matrices  */
linalgError_t matrix_inner(objectmatrix *x, objectmatrix *y, double *out) {
    if (!(x->ncols==y->ncols && x->nrows==y->nrows)) return LINALGERR_INCOMPATIBLE_DIM;
    
    *out=cblas_ddot((linalg_int_t) x->nels, x->elements, 1, y->elements, 1);
    return LINALGERR_OK;
}

/** Rank 1 update: Performs  c <- alpha*a \outer b + c; a and b are treated as column vectors */
linalgError_t matrix_r1update(double alpha, objectmatrix *a, objectmatrix *b, objectmatrix *c) {
    MatrixIdx_t m=a->nrows*a->ncols, n=b->nrows*b->ncols;
    if (!(m==c->nrows && n==c->ncols)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_dger(CblasColMajor, m, n, alpha, a->elements, 1, b->elements, 1, c->elements, c->nrows);
    return LINALGERR_OK;
}

/** Solve the linear system a.x = b using stack allocated memory for temporary */
linalgError_t matrix_solvesmall(objectmatrix *a, objectmatrix *b) {
    int pivot[a->nrows];
    double els[a->nels];
    objectmatrix A = MORPHO_STATICMATRIX(els, a->nrows, a->ncols);
    matrix_copy(a, &A);
    return (matrix_getinterface(a)->solvefn) (&A, b, pivot);
}

/** Solve the linear system a.x = b using heap allocated memory for temporary */
linalgError_t matrix_solvelarge(objectmatrix *a, objectmatrix *b) {
    int *pivot = MORPHO_MALLOC(sizeof(int)*a->nrows);
    objectmatrix *A = matrix_clone(a);
    linalgError_t out = LINALGERR_ALLOC;
    if (pivot && A) {
        out = (matrix_getinterface(a)->solvefn) (A, b, pivot);
    }
    if (A) object_free((object *) A);
    if (pivot) MORPHO_FREE(pivot);
    return out;
}

/** Solve the linear system a.x = b; automatrically allocates storage depending on size of the matrix
 * @param[in]     a  lhs
 * @param[in|out]  b  rhs — overwritten by the solution
 * @returns linalgError_t indicating the status; MATRIX_OK indicates success. */
linalgError_t matrix_solve(objectmatrix *a, objectmatrix *b) {
    if (MATRIX_ISSMALL(a)) return matrix_solvesmall(a, b);
    else return matrix_solvelarge(a, b);
}

/** Inverts the matrix a
 * @param[in] a  matrix to be inverted
 * @returns linalgError_t indicating the status; MATRIX_OK indicates success. */
linalgError_t matrix_inverse(objectmatrix *a) {
    int nrows=a->nrows, ncols=a->ncols, info;
    int pivot[nrows];
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgetrf(LAPACK_COL_MAJOR, nrows, ncols, a->elements, nrows, pivot);
#else
    dgetrf_(&nrows, &ncols, a->elements, &nrows, pivot, &info);
#endif
    if (info!=0) return (info>0 ? LINALGERR_MATRIX_SINGULAR : LINALGERR_LAPACK_INVLD_ARGS);
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgetri(LAPACK_COL_MAJOR, nrows, a->elements, nrows, pivot);
#else
    int lwork=nrows*ncols; double work[nrows*ncols];
    dgetri_(&nrows, a->elements, &nrows, pivot, work, &lwork, &info);
#endif
    
    return (info==0 ? LINALGERR_OK : (info>0 ? LINALGERR_MATRIX_SINGULAR : LINALGERR_LAPACK_INVLD_ARGS));
}

/** Interface to eigensystem */
linalgError_t matrix_eigen(objectmatrix *a, MorphoComplex *w, objectmatrix *vec) {
    if (a->nrows!=a->ncols) return LINALGERR_NOT_SQUARE;
    if (vec && ((a->nrows!=vec->nrows) || (a->nrows!=vec->ncols))) return LINALGERR_INCOMPATIBLE_DIM;
    
    matrix_eigenfn_t efn = matrix_getinterface(a)->eigenfn;
    if (!efn) return LINALGERR_NOT_SUPPORTED;
    
    objectmatrix *temp = matrix_clone(a);
    if (!temp) return LINALGERR_ALLOC;
        
    return efn(temp, w, vec);
}

/* ----------------------
 * Display
 * ---------------------- */

/** Prints a matrix */
void matrix_print(vm *v, objectmatrix *m) {
    matrixinterfacedefn *interface=matrix_getinterface(m);
    double *elptr;
    for (MatrixIdx_t i=0; i<m->nrows; i++) { // Rows run from 0...m
        morpho_printf(v, "[ ");
        for (MatrixIdx_t j=0; j<m->ncols; j++) { // Columns run from 0...k
            matrix_getelementptr(m, i, j, &elptr);
            (*interface->printelfn) (v, elptr);
            morpho_printf(v, " ");
        }
        morpho_printf(v, "]%s", (i<m->nrows-1 ? "\n" : ""));
    }
}

/** Prints a matrix to a buffer */
bool matrix_printtobuffer(objectmatrix *m, char *format, varray_char *out) {
    matrixinterfacedefn *interface=matrix_getinterface(m);
    double *elptr;
    for (MatrixIdx_t i=0; i<m->nrows; i++) { // Rows run from 0...m
        varray_charadd(out, "[ ", 2);
        
        for (MatrixIdx_t j=0; j<m->ncols; j++) { // Columns run from 0...k
            matrix_getelementptr(m, i, j, &elptr);
            if (!(*interface->printeltobufffn) (out, format, elptr)) return false;
            varray_charadd(out, " ", 1);
        }
        varray_charadd(out, "]", 1);
        if (i<m->nrows-1) varray_charadd(out, "\n", 1);
    }
    return true;
}

/* ----------------------
 * Roll
 * ---------------------- */

/** Rolls the matrix list */
static void _rollflat(objectmatrix *a, objectmatrix *b, int nplaces) {
    MatrixCount_t N=a->nrows*a->ncols*a->nvals;
    MatrixCount_t n = abs(nplaces)*a->nvals;
    if (n>N) n = n % N;
    MatrixCount_t Np = N - n; // Number of elements to roll
    
    if (nplaces<0) {
        memcpy(b->matrixdata, a->matrixdata+n, sizeof(double)*Np);
        memcpy(b->matrixdata+Np, a->matrixdata, sizeof(double)*n);
    } else {
        memcpy(b->matrixdata+n, a->matrixdata, sizeof(double)*Np);
        if (n>0) memcpy(b->matrixdata, a->matrixdata+Np, sizeof(double)*n);
    }
}

/** Copies a arow from matrix a into brow for matrix b */
static void _copyrow(objectmatrix *a, MatrixIdx_t arow, objectmatrix *b, MatrixIdx_t brow) {
    for (MatrixIdx_t i=0; i<a->ncols; i++)
        memcpy(b->elements+b->nvals*(i*b->nrows+brow), a->elements+a->nvals*(i*a->nrows+arow), sizeof(double)*a->nvals);
}

/** Rolls a list by a number of elements along a given axis; stores the result in b */
linalgError_t matrix_roll(objectmatrix *a, int nplaces, int axis, objectmatrix *b) {
    if (!(a->nrows==b->nrows && a->ncols==b->ncols && a->nvals==b->nvals)) return LINALGERR_INCOMPATIBLE_DIM;
    
    switch(axis) {
        case 0:
            for (int i=0; i<a->nrows; i++) {
                int j = (i+nplaces);
                while (j<0) j+=a->nrows;
                _copyrow(a, i, b, j % a->nrows);
            }
            break;
        case 1: _rollflat(a, b, nplaces*a->nrows); break;
        default: return LINALGERR_NOT_SUPPORTED;
    }

    return LINALGERR_OK;
}

/* **********************************************************************
 * Matrix constructors
 * ********************************************************************** */

value matrix_constructor__int_int(vm *v, int nargs, value *args) {
    MatrixIdx_t nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                ncols = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    
    objectmatrix *new=matrix_new(nrows, ncols, true);
    return morpho_wrapandbind(v, (object *) new);
}

value matrix_constructor__int(vm *v, int nargs, value *args) {
    MatrixIdx_t nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    
    objectmatrix *new=matrix_new(nrows, 1, true);
    return morpho_wrapandbind(v, (object *) new);
}

/** Clones a matrix */
value matrix_constructor__matrix(vm *v, int nargs, value *args) {
    objectmatrix *a = MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
    return morpho_wrapandbind(v, (object *) matrix_clone(a));
}

/** Constructs a matrix from a list of lists or tuples */
value matrix_constructor__list(vm *v, int nargs, value *args) {
    objectmatrix *new = matrix_listconstructor(v, MORPHO_GETARG(args, 0), OBJECT_MATRIX, 1);
#ifdef MORPHO_INCLUDE_SPARSE
    if (!new) {
        /** Could this be a concatenation operation? */
        objectsparseerror err = sparse_catmatrix(MORPHO_GETLIST(MORPHO_GETARG(args, 0)), &new);
        if (err==SPARSE_INVLDINIT) {
            morpho_runtimeerror(v, LINALG_INVLDARGS);
        } else if (err!=SPARSE_OK) sparse_raiseerror(v, err);
    }
#else
    if (!new) morpho_runtimeerror(v, LINALG_INVLDARGS);
#endif
    return morpho_wrapandbind(v, (object *) new);
}

/** Constructs a matrix from an array */
value matrix_constructor__array(vm *v, int nargs, value *args) {
    objectarray *a = MORPHO_GETARRAY(MORPHO_GETARG(args, 0));
    if (a->ndim!=2) { morpho_runtimeerror(v, LINALG_INVLDARGS); return MORPHO_NIL; }
    
    objectmatrix *new = matrix_arrayconstructor(v, a, OBJECT_MATRIX, 1);
    return morpho_wrapandbind(v, (object *) new);
}

/** Constructs a matrix from a sparse matrix */
value matrix_constructor__sparse(vm *v, int nargs, value *args) {
    objectmatrix *new = NULL;
    objectsparseerror err=sparse_tomatrix(MORPHO_GETSPARSE(MORPHO_GETARG(args, 0)), &new);
    if (err!=SPARSE_OK) morpho_runtimeerror(v, LINALG_INVLDARGS);
    return morpho_wrapandbind(v, (object *) new);
}

value matrix_constructor__err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, MATRIX_CONSTRUCTOR);
    return MORPHO_NIL;
}

/** Creates an identity matrix */
value matrix_identityconstructor(vm *v, int nargs, value *args) {
    if (nargs!=1) { morpho_runtimeerror(v, MATRIX_IDENTCONSTRUCTOR); return MORPHO_NIL; }
    
    MatrixIdx_t n = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

    objectmatrix *new = matrix_new(n,n,false);
    if (new) matrix_identity(new);
    
    return morpho_wrapandbind(v, (object *) new);
}

/* **********************************************************************
 * Matrix veneer class
 * ********************************************************************** */

/* ----------------------
 * Common utility methods
 * ---------------------- */

/** Prints a matrix */
value Matrix_print(vm *v, int nargs, value *args) {
    if (MORPHO_ISCLASS(MORPHO_SELF(args))) return Object_print(v, nargs, args); // Handle calls on the class
    objectmatrix *m=MORPHO_GETMATRIX(MORPHO_SELF(args));
    matrix_print(v, m);
    return MORPHO_NIL;
}

/** Formatted conversion to a string */
value Matrix_format(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    varray_char str;
    varray_charinit(&str);
    
    if (matrix_printtobuffer(MORPHO_GETMATRIX(MORPHO_SELF(args)),
                              MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)),
                             &str)) {
        out = object_stringfromvarraychar(&str);
        if (MORPHO_ISOBJECT(out)) morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    varray_charclear(&str);
    return out;
}

/** Copies the contents of one matrix into another */
value Matrix_assign(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
    LINALG_ERRCHECKVM(matrix_copy(b, a));
    return MORPHO_NIL;
}

/** Clones a matrix */
value Matrix_clone(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *new=matrix_clone(a);
    return morpho_wrapandbind(v, (object *) new);
}

/* ---------
 * index()
 * --------- */

value Matrix_index__int_int(vm *v, int nargs, value *args) {
    objectmatrix *m = MORPHO_GETMATRIX(MORPHO_SELF(args));
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                j = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    value out=MORPHO_NIL;
    
    double *elptr=NULL;
    LINALG_ERRCHECKVM(matrix_getelementptr(m, i, j, &elptr));

    if (elptr) out=matrix_getinterface(m)->getelfn(v, elptr);
    return out;
}

static linalgError_t _slice_count(value in, MatrixIdx_t *count) {
    if (morpho_isnumber(in)) { *count=1; return LINALGERR_OK; }
    else if (MORPHO_ISRANGE(in)) { *count = (MatrixIdx_t) range_count(MORPHO_GETRANGE(in)); return LINALGERR_OK; }
    else if (MORPHO_ISLIST(in)) { *count = (MatrixIdx_t) list_length(MORPHO_GETLIST(in)); return LINALGERR_OK; }
    else if (MORPHO_ISTUPLE(in)) { *count = (MatrixIdx_t) tuple_length(MORPHO_GETTUPLE(in)); return LINALGERR_OK; }
    return LINALGERR_NON_NUMERICAL;
}

static linalgError_t _slice_iterate(value in, unsigned int i, MatrixIdx_t *ix) {
    value val=in;
    if (MORPHO_ISRANGE(in)) {
        val=range_iterate(MORPHO_GETRANGE(in), i);
    } else if (MORPHO_ISLIST(in)) {
        if (!list_getelement(MORPHO_GETLIST(in), i, &val)) return LINALGERR_INVLD_ARG;
    } else if (MORPHO_ISTUPLE(in)) {
        if (!tuple_getelement(MORPHO_GETTUPLE(in), i, &val)) return LINALGERR_INVLD_ARG;
    }
        
    if (MORPHO_ISINTEGER(val)) { *ix=MORPHO_GETINTEGERVALUE(val); return LINALGERR_OK; }
    else if (MORPHO_ISFLOAT(val)) { *ix=(MatrixIdx_t) MORPHO_GETFLOATVALUE(val); return LINALGERR_OK; }
    return LINALGERR_INVLD_ARG;
}

static linalgError_t _slice_validate(value iv, value jv, MatrixIdx_t *icnt, MatrixIdx_t *jcnt) {
    LINALG_ERRCHECKRETURN(_slice_count(iv, icnt));
    LINALG_ERRCHECKRETURN(_slice_count(jv, jcnt));
    if (*icnt<1 || *jcnt<1) return LINALGERR_INVLD_ARG;
    return LINALGERR_OK;
}

static linalgError_t _slice_copy(value iv, value jv, MatrixIdx_t icnt, MatrixIdx_t jcnt, objectmatrix *a, objectmatrix *b, bool swap) {
    double *ael, *bel;
    for (MatrixIdx_t j=0; j<jcnt; j++) {
        MatrixIdx_t jx;
        LINALG_ERRCHECKRETURN(_slice_iterate(jv, j, &jx));
        LINALG_ERRCHECKRETURN(matrix_validateindex(&jx, a->ncols));
        for (MatrixIdx_t i=0; i<icnt; i++) {
            MatrixIdx_t ix;
            LINALG_ERRCHECKRETURN(_slice_iterate(iv, i, &ix));
            LINALG_ERRCHECKRETURN(matrix_validateindex(&ix, a->nrows));
            LINALG_ERRCHECKRETURN(matrix_getelementptr(a, ix, jx, &ael));
            LINALG_ERRCHECKRETURN(matrix_getelementptr(b, i, j, &bel));
            if (swap) memcpy(ael, bel, sizeof(double)*a->nvals);
            else memcpy(bel, ael, sizeof(double)*b->nvals);
        }
    }
    return LINALGERR_OK;
}

value Matrix_index__x_x(vm *v, int nargs, value *args) {
    objectmatrix *m = MORPHO_GETMATRIX(MORPHO_SELF(args)), *new=NULL;
    value iv=MORPHO_GETARG(args, 0), jv=MORPHO_GETARG(args, 1);
    value out=MORPHO_NIL;
    
    MatrixIdx_t icnt=0, jcnt=0; // Counts become size of new matrix
    LINALG_ERRCHECKVMRETURN(_slice_validate(iv, jv, &icnt, &jcnt), MORPHO_NIL);
    
    new=matrix_newwithtype(MORPHO_GETOBJECTTYPE(MORPHO_SELF(args)), icnt, jcnt, m->nvals, false);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return MORPHO_NIL; }
    
    linalgError_t err=_slice_copy(iv, jv, icnt, jcnt, m, new, false);
    if (err!=LINALGERR_OK) { linalg_raiseerror(v, err); object_free((object *) new); }
    else out = morpho_wrapandbind(v, (object *) new);
    
    return out;
}

value Matrix_index__err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, LINALG_INVLDINDICES);
    return MORPHO_NIL;
}

/* ---------
 * setindex()
 * --------- */

static void _setindex(vm *v, objectmatrix *m, MatrixIdx_t i, MatrixIdx_t j, value in) {
    double *elptr=NULL;
    LINALG_ERRCHECKVM(matrix_getelementptr(m, i, j, &elptr));
    if (elptr) LINALG_ERRCHECKVM(matrix_getinterface(m)->setelfn(v, in, elptr));
}

value Matrix_setindex__int_x(vm *v, int nargs, value *args) {
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    _setindex(v, MORPHO_GETMATRIX(MORPHO_SELF(args)), i, 0, MORPHO_GETARG(args, 1));
    return MORPHO_NIL;
}

value Matrix_setindex__int_int_x(vm *v, int nargs, value *args) {
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    MatrixIdx_t j = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    _setindex(v, MORPHO_GETMATRIX(MORPHO_SELF(args)), i, j, MORPHO_GETARG(args, 2));
    return MORPHO_NIL;
}

value Matrix_setindex__x_x_matrix(vm *v, int nargs, value *args) {
    objectmatrix *m = MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *msrc = MORPHO_GETMATRIX(MORPHO_GETARG(args, 2));
    value iv=MORPHO_GETARG(args, 0), jv=MORPHO_GETARG(args, 1);
    
    MatrixIdx_t icnt=0, jcnt=0;
    LINALG_ERRCHECKVMRETURN(_slice_validate(iv, jv, &icnt, &jcnt), MORPHO_NIL);
    
    LINALG_ERRCHECKVM(_slice_copy(iv, jv, icnt, jcnt, m, msrc, true));
    
    return MORPHO_NIL;
}

/* ---------
 * column
 * --------- */

value Matrix_getcolumn__int(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    value out=MORPHO_NIL;
    
    if (i>=0 && i<a->ncols) {
        objectmatrix *new=matrix_newwithtype(a->obj.type, a->nrows, 1, a->nvals, false);
        if (new) matrix_getcolumn(a, i, new);
        out=morpho_wrapandbind(v, (object *)new);
    } else linalg_raiseerror(v, LINALGERR_INDX_OUT_OF_BNDS);
    
    return out;
}

value Matrix_setcolumn__int_matrix(vm *v, int nargs, value *args) {
    LINALG_ERRCHECKVM(matrix_setcolumn(MORPHO_GETMATRIX(MORPHO_SELF(args)),
                                        MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                                        MORPHO_GETMATRIX(MORPHO_GETARG(args, 1))));
    return MORPHO_NIL;
}

/* ----------
 * Arithmetic
 * ---------- */

/** Add a vector */
static value _axpy(vm *v, int nargs, value *args, double alpha) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));      // Receiver is left hand operand
    objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0)); // Argument is right hand operand
    objectmatrix *new = NULL;
    value out=MORPHO_NIL;
    
    if (a->ncols==b->ncols && a->nrows==b->nrows) {
        new=matrix_clone(a);
        if (new) LINALG_ERRCHECKVM(matrix_axpy(alpha, b, new));
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES);
    
    return out;
}

/** Add a scalar */
static value _xpa(vm *v, int nargs, value *args, double sgna, double sgnb) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *new=NULL;
    value out=MORPHO_NIL;
    
    double beta;
    if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &beta)) {
        new = matrix_clone(a);
        if (new) LINALG_ERRCHECKVM(matrix_addscalar(new, sgna, beta*sgnb));
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, LINALG_INVLDARGS);
    
    return out;
}

value Matrix_add__matrix(vm *v, int nargs, value *args) {
    return _axpy(v,nargs,args,1.0);
}

value Matrix_add__nil(vm *v, int nargs, value *args) {
    return MORPHO_SELF(args);
}

value Matrix_add__x(vm *v, int nargs, value *args) {
    if (matrix_isamatrix(MORPHO_GETARG(args, 0))) return MORPHO_NIL; // Redirect to addr
    return _xpa(v,nargs,args,1.0,1.0);
}

value Matrix_sub__matrix(vm *v, int nargs, value *args) {
    return _axpy(v,nargs,args,-1.0);
}

value Matrix_sub__x(vm *v, int nargs, value *args) {
    if (matrix_isamatrix(MORPHO_GETARG(args, 0))) return MORPHO_NIL; // Redirect to subr
    return _xpa(v,nargs,args,1.0,-1.0);
}

value Matrix_subr__x(vm *v, int nargs, value *args) {
    return _xpa(v,nargs,args,-1.0,1.0);
}

value Matrix_mul__float(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    
    double scale;
    if (!morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale)) return MORPHO_NIL;
    
    objectmatrix *new = matrix_clone(a);
    if (new) matrix_scale(new, scale);
    return morpho_wrapandbind(v, (object *) new);
}

value Matrix_mul__matrix(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
    value out=MORPHO_NIL;
    
    if (a->ncols==b->nrows) {
        objectmatrix *new = matrix_new(a->nrows, b->ncols, false);
        if (new) LINALG_ERRCHECKVM(matrix_mmul(1.0, a, b, 0.0, new));
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES);
    return out;
}

value Matrix_div__float(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    
    double scale;
    if (!morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale)) return MORPHO_NIL;
    scale = 1.0/scale;
    if (isnan(scale)) morpho_runtimeerror(v, VM_DVZR);
    
    objectmatrix *new = matrix_clone(a);
    if (new) matrix_scale(new, scale);
    return morpho_wrapandbind(v, (object *) new);
}

value Matrix_div__matrix(vm *v, int nargs, value *args) {
    objectmatrix *b=MORPHO_GETMATRIX(MORPHO_SELF(args)); // Note that the rhs is the receiver
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0)); // ... the lhs is the argument
    
    objectmatrix *sol = matrix_clone(b);
    if (sol) LINALG_ERRCHECKVM(matrix_solve(a, sol));
    
    return morpho_wrapandbind(v, (object *) sol);
}

/** Accumulate in place */
value Matrix_acc__x_x_matrix(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 1));
    
    double alpha=1.0;
    if (!morpho_valuetofloat(MORPHO_GETARG(args, 0), &alpha)) { morpho_runtimeerror(v, LINALG_ARITHARGS); return MORPHO_NIL; }
    
    LINALG_ERRCHECKVM(matrix_axpy(alpha, b, a));
    return MORPHO_NIL;
}

/* ----------------
 * Unary operations
 * ---------------- */

/** Matrix norm */
value Matrix_norm__x(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    double n;
    
    if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &n)) {
        if (fabs(n-1.0)<MORPHO_EPS) {
            return MORPHO_FLOAT(matrix_norm(a, MATRIX_NORM_L1));
        } else if (isinf(n)) {
            return MORPHO_FLOAT(matrix_norm(a, MATRIX_NORM_INF));
        }
    }
    morpho_runtimeerror(v, LINALG_NORMARGS);
    return MORPHO_NIL;
}

/** Frobenius norm */
value Matrix_norm(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    return MORPHO_FLOAT(matrix_norm(a, MATRIX_NORM_FROBENIUS));
}

/** Sums all matrix values */
value Matrix_sum(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    double sum[a->nvals];
    
    matrix_sum(a, sum);
    return matrix_getinterface(a)->getelfn(v, sum);
}

/** Computes the trace */
value Matrix_trace(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    double out=0.0;
    LINALG_ERRCHECKVM(matrix_trace(a, &out));
    return MORPHO_FLOAT(out);
}

/** Inverts a matrix */
value Matrix_transpose(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *new = matrix_clone(a);
    if (new) {
        new->ncols=a->nrows;
        new->nrows=a->ncols;
        LINALG_ERRCHECKVM(matrix_transpose(a, new));
    }
    return morpho_wrapandbind(v, (object *) new);
}

/** Inverts a matrix */
value Matrix_inverse(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *new = matrix_clone(a);
    if (new) LINALG_ERRCHECKVM(matrix_inverse(new));
    
    return morpho_wrapandbind(v, (object *) new);
}

/* ----------------
 * Eigensystem
 * ---------------- */

static bool _processeigenvalues(vm *v, MatrixIdx_t n, MorphoComplex *w, value *out) {
    value ev[n];
    for (int i=0; i<n; i++) ev[i]=MORPHO_NIL;
    for (int i=0; i<n; i++) {
        double abs = cabs(w[i]);
        if (abs > DBL_MIN ? fabs(cimag(w[i]))/abs <= MORPHO_EPS : fabs(cimag(w[i])) < DBL_MIN) {
            ev[i]=MORPHO_FLOAT(creal(w[i]));
        } else {
            objectcomplex *new = object_newcomplex(creal(w[i]), cimag(w[i]));
            if (new) ev[i]=MORPHO_OBJECT(new);
            else goto _processeigenvalues_cleanup;
        }
    }
    
    objecttuple *new = object_newtuple(n, ev);
    if (!new) goto _processeigenvalues_cleanup;
    
    *out = MORPHO_OBJECT(new);
    return true;
    
_processeigenvalues_cleanup:
    for (int i=0; i<n; i++) morpho_freeobject(ev[i]);
    return false;
}

/** Finds the eigenvalues of a matrix */
value Matrix_eigenvalues(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out = MORPHO_NIL;
    
    MatrixIdx_t n=a->ncols;
    MorphoComplex w[n];
    linalgError_t err=matrix_eigen(a, w, NULL);
    if (err==LINALGERR_OK) {
        if (_processeigenvalues(v, n, w, &out)) {
            morpho_bindobjects(v, 1, &out);
        } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    } else linalg_raiseerror(v, err);
    
    return out;
}

#define _CHK(x) if (!x) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto _eigensystem_cleanup; }

/** Finds the eigenvalues and eigenvectors of a matrix */
value Matrix_eigensystem(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    
    value ev=MORPHO_NIL; // Will hold eigenvalues
    objectmatrix *evec=NULL; // Holds eigenvectors
    objecttuple *otuple=NULL; // Tuple to return everything
    
    MatrixIdx_t n=a->ncols;
    MorphoComplex w[n];
    
    evec=matrix_clone(a);
    _CHK(evec);
    
    linalgError_t err=matrix_eigen(a, w, evec);
    if (err!=LINALGERR_OK) { linalg_raiseerror(v, err); goto _eigensystem_cleanup; }
    
    _CHK(_processeigenvalues(v, n, w, &ev));
    
    value outtuple[2] = { ev, MORPHO_OBJECT(evec) };
    otuple = object_newtuple(2, outtuple);
    _CHK(otuple);
    
    return morpho_wrapandbind(v, (object *) otuple);
    
_eigensystem_cleanup:
    if (evec) object_free((object *) evec);
    if (otuple) object_free((object *) otuple);
    if (MORPHO_ISOBJECT(ev)) {
        value evx;
        objecttuple *t = MORPHO_GETTUPLE(ev);
        for (int i=0; i<tuple_length(t); i++) if (tuple_getelement(t, i, &evx)) morpho_freeobject(evx);
    }
    morpho_freeobject(ev); 
    
    return MORPHO_NIL;
}
#undef _CHK

/* ----------------
 * SVD
 * ---------------- */

/** Interface to SVD */
linalgError_t matrix_svd(objectmatrix *a, double *s, objectmatrix *u, objectmatrix *vt) {
    if (u && ((a->nrows != u->nrows) || (a->nrows != u->ncols))) return LINALGERR_INCOMPATIBLE_DIM;
    if (vt && ((a->ncols != vt->nrows) || (a->ncols != vt->ncols))) return LINALGERR_INCOMPATIBLE_DIM;
    
    objectmatrix *temp = matrix_clone(a);
    if (!temp) return LINALGERR_ALLOC;
    
    linalgError_t err = matrix_getinterface(a)->svdfn (temp, s, u, vt);
    object_free((object *) temp);
    return err;
}

/* ----------------
 * QR decomposition
 * ---------------- */

/** Interface to QR decomposition */
linalgError_t matrix_qr(objectmatrix *a, objectmatrix *q, objectmatrix *r) {
    if (q && ((a->nrows != q->nrows) || (a->nrows != q->ncols))) return LINALGERR_INCOMPATIBLE_DIM;
    if (r && ((a->nrows != r->nrows) || (a->ncols != r->ncols))) return LINALGERR_INCOMPATIBLE_DIM;
    
    objectmatrix *temp = matrix_clone(a);
    if (!temp) return LINALGERR_ALLOC;
    
    linalgError_t err = matrix_getinterface(a)->qrfn (temp, q, r);
    object_free((object *) temp);
    return err;
}

/** Processes singular values into a tuple */
static bool _processsingularvalues(vm *v, MatrixIdx_t n, double *s, value *out) {
    value sv[n];
    for (int i = 0; i < n; i++) sv[i] = MORPHO_FLOAT(s[i]);
    
    objecttuple *new = object_newtuple(n, sv);
    if (!new) return false;
    
    *out = MORPHO_OBJECT(new);
    return true;
}

#define _CHK_SVD(x) if (!x) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto _svd_cleanup; }
/** Singular Value Decomposition */
value Matrix_svd(vm *v, int nargs, value *args) {
    objectmatrix *a = MORPHO_GETMATRIX(MORPHO_SELF(args));
    
    value s = MORPHO_NIL;           // Will hold singular values
    objectmatrix *u = NULL;        // Left singular vectors
    objectmatrix *vt = NULL;       // Right singular vectors (transposed)
    objecttuple *otuple = NULL;     // Tuple to return everything
    
    MatrixIdx_t m = a->nrows, n = a->ncols;
    MatrixIdx_t minmn = (m < n) ? m : n;
    double singular_values[minmn];
    
    // Allocate U (m×m) and VT (n×n) matrices
    u = matrix_newwithtype(MORPHO_GETOBJECTTYPE(MORPHO_SELF(args)), m, m, a->nvals, false);
    _CHK_SVD(u);
    
    vt = matrix_newwithtype(MORPHO_GETOBJECTTYPE(MORPHO_SELF(args)), n, n, a->nvals, false);
    _CHK_SVD(vt);
    
    linalgError_t err = matrix_svd(a, singular_values, u, vt);
    if (err != LINALGERR_OK) { linalg_raiseerror(v, err); goto _svd_cleanup; }
    
    _CHK_SVD(_processsingularvalues(v, minmn, singular_values, &s));
    
    value outtuple[3] = { MORPHO_OBJECT(u), s, MORPHO_OBJECT(vt) };
    otuple = object_newtuple(3, outtuple);
    _CHK_SVD(otuple);
    
    return morpho_wrapandbind(v, (object *) otuple);
    
_svd_cleanup:
    if (u) object_free((object *) u);
    if (vt) object_free((object *) vt);
    if (otuple) object_free((object *) otuple);
    morpho_freeobject(s);
    
    return MORPHO_NIL;
}
#undef _CHK_SVD

/* ----------------
 * QR decomposition
 * ---------------- */

#define _CHK_QR(x) if (!x) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto _qr_cleanup; }
/** QR Decomposition */
value Matrix_qr(vm *v, int nargs, value *args) {
    objectmatrix *a = MORPHO_GETMATRIX(MORPHO_SELF(args));
    
    objectmatrix *q = NULL;        // Orthogonal matrix Q
    objectmatrix *r = NULL;        // Upper triangular matrix R
    objecttuple *otuple = NULL;     // Tuple to return everything
    
    MatrixIdx_t m = a->nrows, n = a->ncols;
    
    // Allocate Q (m×m) and R (m×n) matrices
    q = matrix_newwithtype(MORPHO_GETOBJECTTYPE(MORPHO_SELF(args)), m, m, a->nvals, false);
    _CHK_QR(q);
    
    r = matrix_newwithtype(MORPHO_GETOBJECTTYPE(MORPHO_SELF(args)), m, n, a->nvals, false);
    _CHK_QR(r);
    
    linalgError_t err = matrix_qr(a, q, r);
    if (err != LINALGERR_OK) { linalg_raiseerror(v, err); goto _qr_cleanup; }
    
    value outtuple[2] = { MORPHO_OBJECT(q), MORPHO_OBJECT(r) };
    otuple = object_newtuple(2, outtuple);
    _CHK_QR(otuple);
    
    return morpho_wrapandbind(v, (object *) otuple);
    
_qr_cleanup:
    if (q) object_free((object *) q);
    if (r) object_free((object *) r);
    if (otuple) object_free((object *) otuple);
    
    return MORPHO_NIL;
}
#undef _CHK_QR

/* ---------
 * Products
 * --------- */

/** Frobenius inner product */
value Matrix_inner(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
    double prod=0.0;
    
    LINALG_ERRCHECKVM(matrix_inner(a, b, &prod));
    
    return MORPHO_FLOAT(prod);
}

/** Outer product */
value Matrix_outer(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
    
    objectmatrix *new=matrix_new(a->nrows*a->ncols, b->nrows*b->ncols, true);
    if (new) LINALG_ERRCHECKVM(matrix_r1update(1.0, a, b, new));
    
    return morpho_wrapandbind(v, (object *) new);
}

/* ---------
 * Metadata
 * --------- */

/** Reshape a matrix */
value Matrix_reshape(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    int nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
        ncols = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    
    if (nrows*ncols==a->nrows*a->ncols) {
        a->nrows=nrows;
        a->ncols=ncols;
    } else morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES);
    
    return MORPHO_NIL;
}

static value _roll(vm *v, objectmatrix *a, int roll, int axis) {
    objectmatrix *new = matrix_clone(a);
    if (new) matrix_roll(a, roll, axis, new);
    return morpho_wrapandbind(v, (object *) new);
}

/** Roll a matrix */
value Matrix_roll__int_int(vm *v, int nargs, value *args) {
    objectmatrix *a = MORPHO_GETMATRIX(MORPHO_SELF(args));
    int roll = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
        axis = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    return _roll(v, a, roll, axis);
}

/** Roll a matrix by row */
value Matrix_roll__int(vm *v, int nargs, value *args) {
    objectmatrix *a = MORPHO_GETMATRIX(MORPHO_SELF(args));
    int roll = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    return _roll(v, a, roll, 0);
}

/** Enumerate protocol */
value Matrix_enumerate(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    value out=MORPHO_NIL;
    
    if (i<0) {
        out=MORPHO_INTEGER(a->ncols*a->nrows);
    } else if (i<a->ncols*a->nrows) {
        out=matrix_getinterface(a)->getelfn(v, a->elements+i*a->nvals);
    } else {
        linalg_raiseerror(v, LINALGERR_INDX_OUT_OF_BNDS);
    }
    
    return out;
}

/** Number of matrix elements */
value Matrix_count(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    return MORPHO_INTEGER(a->ncols*a->nrows);
}

/** Matrix dimensions */
value Matrix_dimensions(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    
    value dim[2] = { MORPHO_INTEGER(a->nrows), MORPHO_INTEGER(a->ncols) };
    objecttuple *new=object_newtuple(2, dim);
    
    return morpho_wrapandbind(v, (object *) new);
}

MORPHO_BEGINCLASS(Matrix)
MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "()", Matrix_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_FORMAT_METHOD, "(String)", Matrix_format, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ASSIGN_METHOD, "(Matrix)", Matrix_assign, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, "Matrix ()", Matrix_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Float (Int)", Matrix_enumerate, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Float (Int, Int)", Matrix_index__int_int, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Matrix (_,_)", Matrix_index__x_x, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Matrix (...)", Matrix_index__err, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,_)", Matrix_setindex__int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,Int,_)", Matrix_setindex__int_int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(_,_,Matrix)", Matrix_setindex__x_x_matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_GETCOLUMN_METHOD, "Matrix (Int)", Matrix_getcolumn__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_SETCOLUMN_METHOD, "(Int, Matrix)", Matrix_setcolumn__int_matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_SETCOLUMN_METHOD_DEPRECATED, "(Int, Matrix)", Matrix_setcolumn__int_matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Matrix (Matrix)", Matrix_add__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Matrix (Nil)", Matrix_add__nil, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Matrix (_)", Matrix_add__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "Matrix (_)", Matrix_add__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "Matrix (Nil)", Matrix_add__nil, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "Matrix (Matrix)", Matrix_sub__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "Matrix (Nil)", Matrix_add__nil, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "Matrix (_)", Matrix_sub__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "Matrix (_)", Matrix_subr__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "Matrix (_)", Matrix_mul__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "Matrix (Matrix)", Matrix_mul__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "Matrix (_)", Matrix_mul__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "Matrix (Matrix)", Matrix_div__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "Matrix (_)", Matrix_div__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ACC_METHOD, "(_, Matrix)", Matrix_acc__x_x_matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_INVERSE_METHOD, "Matrix ()", Matrix_inverse, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_NORM_METHOD, "Float (_)", Matrix_norm__x, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MATRIX_NORM_METHOD, "Float ()", Matrix_norm, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_SUM_METHOD, "Float ()", Matrix_sum, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MATRIX_TRACE_METHOD, "Float ()", Matrix_trace, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MATRIX_TRANSPOSE_METHOD, "Matrix ()", Matrix_transpose, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_INNER_METHOD, "Float (Matrix)", Matrix_inner, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MATRIX_OUTER_METHOD, "Matrix (Matrix)", Matrix_outer, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_EIGENVALUES_METHOD, "Tuple ()", Matrix_eigenvalues, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_EIGENSYSTEM_METHOD, "Tuple ()", Matrix_eigensystem, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_SVD_METHOD, "Tuple ()", Matrix_svd, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_QR_METHOD, "Tuple ()", Matrix_qr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_RESHAPE_METHOD, "(Int,Int)", Matrix_reshape, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_ROLL_METHOD, "Matrix (Int)", Matrix_roll__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_ROLL_METHOD, "Matrix (Int,Int)", Matrix_roll__int_int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ENUMERATE_METHOD, "(Int)", Matrix_enumerate, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int ()", Matrix_count, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MATRIX_DIMENSIONS_METHOD, "Tuple ()", Matrix_dimensions, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void matrix_initialize(void) {
    objectmatrixtype=object_addtype(&objectmatrixdefn);
    matrix_addinterface(&matrixdefn);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    value matrixclass=builtin_addclass(MATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(Matrix), objclass);
    object_setveneerclass(OBJECT_MATRIX, matrixclass);
    
    morpho_addfunction(MATRIX_CLASSNAME, "Matrix (Int, Int)", matrix_constructor__int_int, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(MATRIX_CLASSNAME, "Matrix (Int)", matrix_constructor__int, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(MATRIX_CLASSNAME, "Matrix (Matrix)", matrix_constructor__matrix, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(MATRIX_CLASSNAME, "Matrix (List)", matrix_constructor__list, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(MATRIX_CLASSNAME, "Matrix (Tuple)", matrix_constructor__list, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(MATRIX_CLASSNAME, "Matrix (Array)", matrix_constructor__array, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(MATRIX_CLASSNAME, "Matrix (Sparse)", matrix_constructor__sparse, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(MATRIX_CLASSNAME, "(...)", matrix_constructor__err, MORPHO_FN_CONSTRUCTOR, NULL);
    
    morpho_addfunction(MATRIX_IDENTITYCONSTRUCTOR, "Matrix (Int)", matrix_identityconstructor, MORPHO_FN_CONSTRUCTOR, NULL);
    
    morpho_defineerror(MATRIX_CONSTRUCTOR,             ERROR_HALT, MATRIX_CONSTRUCTOR_MSG);
    morpho_defineerror(MATRIX_IDENTCONSTRUCTOR,        ERROR_HALT, MATRIX_IDENTCONSTRUCTOR_MSG);
    
    complexmatrix_initialize();
}

#endif /* MORPHO_INCLUDE_LINALG */
