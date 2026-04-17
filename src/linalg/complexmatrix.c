/** @file complexmatrix.c
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#include <platform.h>

#include "linalg.h"
#include "format.h"
#include "cmplx.h"

objecttype objectcomplexmatrixtype;

/* **********************************************************************
 * ComplexMatrix utility functions
 * ********************************************************************** */

/* ----------------------
 * Callbacks
 * ---------------------- */

static void _printelfn(vm *v, double *el) {
    objectcomplex cmplx = MORPHO_STATICCOMPLEX(el[0], el[1]);
    complex_print(v, &cmplx);
}

static bool _printeltobufffn(varray_char *out, char *format, double *el) {
    if (!format_printtobuffer(MORPHO_FLOAT(el[0]), format, out)) return false;
    varray_charadd(out, " ", 1);
    varray_charadd(out, (el[1]<0 ? "-" : "+"), 1);
    if (!format_printtobuffer(MORPHO_FLOAT(fabs(el[1])), format, out)) return false;
    varray_charadd(out, "im", 2);
    return true;
}

static value _getelfn(vm *v, double *el) {
    objectcomplex *new = object_newcomplex(el[0], el[1]);
    return morpho_wrapandbind(v, (object *) new);
}

static linalgError_t _setelfn(vm *v, value in, double *el) {
    if (MORPHO_ISCOMPLEX(in)) {
        *((MorphoComplex *) el) = MORPHO_GETCOMPLEX(in)->Z;
    } else if (morpho_valuetofloat(in, el)) {
        el[1] = 0.0; // Set imaginary part to zero
    } else return LINALGERR_NON_NUMERICAL;
    return LINALGERR_OK;
}

/** Evaluate norms */
static double _normfn(objectmatrix *a, matrix_norm_t nrm) {
    char cnrm = matrix_normtolapack(nrm);
    int nrows=a->nrows, ncols=a->ncols;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    return LAPACKE_zlange(LAPACK_COL_MAJOR, cnrm, a->nrows, a->ncols, (linalg_complexdouble_t *) a->elements, a->nrows);
#else
    double work[a->nrows];
    return zlange_(&cnrm, &nrows, &ncols, (linalg_complexdouble_t *) a->elements, &nrows, work);
#endif
}

/** Low level linear solve */
static linalgError_t _solve(objectmatrix *a, objectmatrix *b, int *pivot) {
    int n=a->nrows, nrhs = b->ncols, info;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_zgesv(LAPACK_COL_MAJOR, n, nrhs, (linalg_complexdouble_t *) a->elements, n, pivot, (linalg_complexdouble_t *) b->elements, n);
#else
    zgesv_(&n, &nrhs, (linalg_complexdouble_t *) a->elements,
           &n, pivot, (linalg_complexdouble_t *) b->elements, &n, &info);
#endif
    
    return (info==0 ? LINALGERR_OK : (info>0 ? LINALGERR_MATRIX_SINGULAR : LINALGERR_LAPACK_INVLD_ARGS));
}

/** Low level eigensolver */
static linalgError_t _eigen(objectmatrix *a, MorphoComplex *w, objectmatrix *vec) {
    int info, n=a->nrows;

#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', (vec ? 'V' : 'N'), n, (linalg_complexdouble_t *) a->elements, n, (linalg_complexdouble_t *) w, NULL, n, (linalg_complexdouble_t *) (vec ? vec->elements : NULL), n);
#else
    int lwork=4*n; MorphoComplex work[4*n]; double rwork[2*n];
    zgeev_("N", (vec ? "V" : "N"), &n, (linalg_complexdouble_t *) a->elements, &n, (linalg_complexdouble_t *) w, NULL, &n, (linalg_complexdouble_t *) (vec ? vec->elements : NULL), &n, work, &lwork, rwork, &info);
#endif
    
    return (info==0 ? LINALGERR_OK : (info>0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS));
}

/** Low level SVD */
static linalgError_t _svd(objectmatrix *a, double *s, objectmatrix *u, objectmatrix *vt) {
    int info, m=a->nrows, n=a->ncols;
    int minmn = (m < n) ? m : n;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    double* superb = malloc(minmn * sizeof(double));
    if (!superb) return LINALGERR_ALLOC;
    info = LAPACKE_zgesvd(LAPACK_COL_MAJOR,
                          (u ? 'A' : 'N'),      // jobu: 'A' = all U columns, 'N' = no U
                          (vt ? 'A' : 'N'),     // jobvt: 'A' = all VT rows, 'N' = no VT
                          m, n,
                          (linalg_complexdouble_t *) a->elements, m,  // input matrix A (overwritten)
                          s,                    // singular values (min(m,n))
                          (linalg_complexdouble_t *) (u ? u->elements : NULL), m,  // U matrix (m×m)
                          (linalg_complexdouble_t *) (vt ? vt->elements : NULL), n,  // VT matrix (n×n)
                          superb
                         );
    free(superb);
#else
    int lwork = -1;
    linalg_complexdouble_t work_query;
    double rwork[5 * minmn];  // rwork needs at least 5*min(m,n) for zgesvd
    
    // Query optimal work size
    zgesvd_((u ? "A" : "N"), (vt ? "A" : "N"), &m, &n, 
            (linalg_complexdouble_t *) a->elements, &m, s,
            (linalg_complexdouble_t *) (u ? u->elements : NULL), &m,
            (linalg_complexdouble_t *) (vt ? vt->elements : NULL), &n,
            &work_query, &lwork, rwork, &info);
    
    if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
    
    lwork = (int)creal(work_query);
    linalg_complexdouble_t work[lwork];
    zgesvd_((u ? "A" : "N"), (vt ? "A" : "N"), &m, &n,
            (linalg_complexdouble_t *) a->elements, &m, s,
            (linalg_complexdouble_t *) (u ? u->elements : NULL), &m,
            (linalg_complexdouble_t *) (vt ? vt->elements : NULL), &n,
            work, &lwork, rwork, &info);
#endif
    
    return (info == 0 ? LINALGERR_OK : (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS));
}

/** Low level QR decomposition */
static linalgError_t _qr(objectmatrix *a, objectmatrix *q, objectmatrix *r) {
    int info, m=a->nrows, n=a->ncols;
    int minmn = (m < n) ? m : n;
    
    if (q) {
        linalg_complexdouble_t *qelems = (linalg_complexdouble_t *) q->elements;
        memset(q->elements, 0, q->nels*sizeof(double));
        for (int i = 0; i < m; i++) qelems[i*m+i] = 1.0 + 0.0*I;
    }
    if (minmn==0) {
        if (r) matrix_copy(a, r);
        return LINALGERR_OK;
    }
    
    // Compute QR factorization without pivoting: A = Q*R
#ifdef MORPHO_LINALG_USE_LAPACKE
    linalg_complexdouble_t tau[minmn];
    info = LAPACKE_zgeqrf(LAPACK_COL_MAJOR, m, n, (linalg_complexdouble_t *) a->elements, m, tau);
    if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
#else
    linalg_complexdouble_t tau[minmn];
    int lwork = -1;
    int lwork_geqrf;
    linalg_complexdouble_t work_query;
    
    // Query optimal work size for ZGEQRF
    zgeqrf_(&m, &n, (linalg_complexdouble_t *) a->elements, &m, tau, &work_query, &lwork, &info);
    if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
    
    lwork_geqrf = (int) creal(work_query);
    linalg_complexdouble_t work[lwork_geqrf];
    lwork = lwork_geqrf;
    
    // Compute QR factorization without pivoting
    zgeqrf_(&m, &n, (linalg_complexdouble_t *) a->elements, &m, tau, work, &lwork, &info);
    if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
#endif
    
    // Extract R (upper triangle of a) into r
    // Copy entire matrix first then zero out below the diagonal
    matrix_copy(a, r);
    linalg_complexdouble_t *relems = (linalg_complexdouble_t *) r->elements;
    for (int j = 0; j < n && j < m - 1; j++) {
        memset(&relems[j * m + (j + 1)], 0, (m - j - 1) * sizeof(linalg_complexdouble_t));
    }
    
    // Generate Q by applying the Householder product to the identity matrix
    if (q) {
#ifdef MORPHO_LINALG_USE_LAPACKE
        info = LAPACKE_zunmqr(LAPACK_COL_MAJOR, 'L', 'N', m, m, minmn, (linalg_complexdouble_t *) a->elements, m, tau, (linalg_complexdouble_t *) q->elements, m);
        if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
#else
        int lwork_unmqr = -1;
        linalg_complexdouble_t unmqr_work_query;
        zunmqr_("L", "N", &m, &m, &minmn, (linalg_complexdouble_t *) a->elements, &m, tau, (linalg_complexdouble_t *) q->elements, &m, &unmqr_work_query, &lwork_unmqr, &info);
        if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);

        lwork_unmqr = (int) creal(unmqr_work_query);
        linalg_complexdouble_t unmqr_work[lwork_unmqr];
        zunmqr_("L", "N", &m, &m, &minmn, (linalg_complexdouble_t *) a->elements, &m, tau, (linalg_complexdouble_t *) q->elements, &m, unmqr_work, &lwork_unmqr, &info);
        if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
#endif
    }
    
    return LINALGERR_OK;
}

/* ----------------------
 * Interface definition
 * ---------------------- */

matrixinterfacedefn complexmatrixdefn = {
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
 * Constructor
 * ---------------------- */

/** Create a new complex matrix */
objectcomplexmatrix *complexmatrix_new(MatrixIdx_t nrows, MatrixIdx_t ncols, bool zero) {
    return (objectcomplexmatrix *) matrix_newwithtype(OBJECT_COMPLEXMATRIX, nrows, ncols, 2, zero);
}

/* ----------------------
 * Element access
 * ---------------------- */

/** Sets a matrix element. */
linalgError_t complexmatrix_setelement(objectcomplexmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, MorphoComplex value) {
    MatrixIdx_t row_idx = row, col_idx = col;
    LINALG_ERRCHECKRETURN(matrix_validateindex(&row_idx, matrix->nrows));
    LINALG_ERRCHECKRETURN(matrix_validateindex(&col_idx, matrix->ncols));
    MatrixCount_t ix = matrix->nvals*(col_idx*matrix->nrows+row_idx);
    matrix->elements[ix]=creal(value);
    matrix->elements[ix+1]=cimag(value);
    return LINALGERR_OK;
}

/** Gets a matrix element */
linalgError_t complexmatrix_getelement(objectcomplexmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, MorphoComplex *value) {
    MatrixIdx_t row_idx = row, col_idx = col;
    LINALG_ERRCHECKRETURN(matrix_validateindex(&row_idx, matrix->nrows));
    LINALG_ERRCHECKRETURN(matrix_validateindex(&col_idx, matrix->ncols));
    MatrixCount_t ix = matrix->nvals*(col_idx*matrix->nrows+row_idx);
    if (value) *value=MCBuild(matrix->elements[ix],matrix->elements[ix+1]);
    return LINALGERR_OK;
}

/** Copies a real matrix x into a complex matrix y */
static linalgError_t _stridedcopy(objectmatrix *x, objectmatrix *y, int offset) {
    if (!(x->ncols==y->ncols && x->nrows==y->nrows)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_dcopy((linalg_int_t) x->ncols*x->nrows, x->elements+offset, x->nvals, y->elements, y->nvals);
    return LINALGERR_OK;
}

linalgError_t complexmatrix_promote(objectmatrix *x, objectcomplexmatrix *y) {
    return _stridedcopy(x, y, 0);
}

/** Copies the real part of a complex matrix y into  */
linalgError_t complexmatrix_demote(objectcomplexmatrix *x, objectmatrix *y, bool imag) {
    return _stridedcopy(x, y, (imag?1:0));
}

/* ----------------------
 * Complex arithmetic
 * ---------------------- */

/** Performs c <- alpha*(a*b) + beta*c with complex matrices */
linalgError_t complexmatrix_mmul(MorphoComplex alpha, objectmatrix *a, objectmatrix *b, MorphoComplex beta, objectmatrix *c) {
    if (!(a->ncols==b->nrows && a->nrows==c->nrows && b->ncols==c->ncols)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                a->nrows, b->ncols, a->ncols,
                &alpha, (linalg_complexdouble_t *) a->elements,
                a->nrows, (linalg_complexdouble_t *) b->elements, b->nrows,
                &beta, (linalg_complexdouble_t *) c->elements, c->nrows);
    return LINALGERR_OK;
}

/** Scales a matrix x <- scale * x >*/
void complexmatrix_scale(objectmatrix *a, MorphoComplex scale) {
    cblas_zscal(a->nrows * a->ncols, (linalg_complexdouble_t *) &scale, (linalg_complexdouble_t *) a->elements, 1);
}

/** Finds the Frobenius inner product of two complex matrices (a, b) = \sum_{i,j} conj(a)_ij * b_ij */
linalgError_t complexmatrix_inner(objectcomplexmatrix *a, objectcomplexmatrix *b, MorphoComplex *out) {
    if (!(a->ncols==b->ncols && a->nrows==b->nrows)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_zdotc_sub(a->nrows * a->ncols, (linalg_complexdouble_t *) a->elements, 1,
                     (linalg_complexdouble_t *) b->elements, 1,
                     (linalg_complexdouble_t *) out);
    return LINALGERR_OK;
}

/** Rank 1 update: Performs  c <- alpha*a \outer b + c; a and b are treated as column vectors */
linalgError_t complexmatrix_r1update(MorphoComplex alpha, objectcomplexmatrix *a, objectcomplexmatrix *b, objectcomplexmatrix *c) {
    MatrixIdx_t m=a->nrows*a->ncols, n=b->nrows*b->ncols;
    if (!(m==c->nrows && n==c->ncols)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_zgeru(CblasColMajor, m, n, (linalg_complexdouble_t *) &alpha, (linalg_complexdouble_t *) a->elements, 1,
                (linalg_complexdouble_t *) b->elements, 1,
                (linalg_complexdouble_t *) c->elements, c->nrows);
    return LINALGERR_OK;
}

/** Calculate the trace of a matrix */
linalgError_t complexmatrix_trace(objectcomplexmatrix *a, MorphoComplex *out) {
    if (a->nrows!=a->ncols) return LINALGERR_NOT_SQUARE;
    MorphoComplex one = MCBuild(1.0, 0.0);
    cblas_zdotu_sub(a->nrows, (linalg_complexdouble_t *) a->elements, a->ncols+1, (linalg_complexdouble_t *) &one, 0, (linalg_complexdouble_t *) out);
    return LINALGERR_OK;
}

/** Inverts the matrix a
 * @param[in] a  matrix to be inverted
 * @returns linalgError_t indicating the status; MATRIX_OK indicates success. */
linalgError_t complexmatrix_inverse(objectcomplexmatrix *a) {
    int nrows=a->nrows, ncols=a->ncols, info;
    int pivot[nrows];
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_zgetrf(LAPACK_COL_MAJOR, nrows, ncols, (linalg_complexdouble_t *) a->elements, nrows, pivot);
#else
    zgetrf_(&nrows, &ncols, (linalg_complexdouble_t *) a->elements, &nrows, pivot, &info);
#endif
    if (info!=0) return (info>0 ? LINALGERR_MATRIX_SINGULAR : LINALGERR_LAPACK_INVLD_ARGS);
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_zgetri(LAPACK_COL_MAJOR, nrows, (linalg_complexdouble_t *) a->elements, nrows, pivot);
#else
    int lwork=nrows*ncols; linalg_complexdouble_t work[nrows*ncols];
    zgetri_(&nrows, (linalg_complexdouble_t *) a->elements, &nrows, pivot, work, &lwork, &info);
#endif
    
    return (info==0 ? LINALGERR_OK : (info>0 ? LINALGERR_MATRIX_SINGULAR : LINALGERR_LAPACK_INVLD_ARGS));
}

/* **********************************************************************
 * ComplexMatrix constructors
 * ********************************************************************** */

value complexmatrix_constructor__int_int(vm *v, int nargs, value *args) {
    MatrixIdx_t nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    MatrixIdx_t ncols = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));

    LINALG_ERRCHECKVMRETURN(LINALG_VALIDATECONSTRUCTORDIMS(nrows, ncols), MORPHO_NIL);
    objectcomplexmatrix *new=complexmatrix_new(nrows, ncols, true);
    return morpho_wrapandbind(v, (object *) new);
}

value complexmatrix_constructor__int(vm *v, int nargs, value *args) {
    MatrixIdx_t nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

    LINALG_ERRCHECKVMRETURN(LINALG_VALIDATECONSTRUCTORDIMS(nrows, 1), MORPHO_NIL);
    objectcomplexmatrix *new=complexmatrix_new(nrows, 1, true);
    return morpho_wrapandbind(v, (object *) new);
}

value complematrix_constructor__matrix(vm *v, int nargs, value *args) {
    objectmatrix *a = MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
    
    objectcomplexmatrix *new=complexmatrix_new(a->nrows, a->ncols, true);
    if (new) complexmatrix_promote(a, new);
    return morpho_wrapandbind(v, (object *) new);
}

/** Constructs a complexmatrix from a list of lists or tuples */
value complexmatrix_constructor__list(vm *v, int nargs, value *args) {
    objectmatrix *new = matrix_listconstructor(v, MORPHO_GETARG(args, 0), OBJECT_COMPLEXMATRIX, 2);
    return morpho_wrapandbind(v, (object *) new);
}

/** Constructs a matrix from an array */
value complexmatrix_constructor__array(vm *v, int nargs, value *args) {
    objectarray *a = MORPHO_GETARRAY(MORPHO_GETARG(args, 0));
    if (a->ndim!=2) { morpho_runtimeerror(v, LINALG_INVLDARGS); return MORPHO_NIL; }
    
    objectmatrix *new = matrix_arrayconstructor(v, a, OBJECT_COMPLEXMATRIX, 2);
    if (!new) return MORPHO_NIL;
    return morpho_wrapandbind(v, (object *) new);
}

/* **********************************************************************
 * ComplexMatrix veneer class
 * ********************************************************************** */

/* ----------------------
 * Arithmetic
 * ---------------------- */

/** Add a vector */
static value _axpy(vm *v, int nargs, value *args, double alpha) {
    objectcomplexmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
    value out=MORPHO_NIL;
    
    if (a->ncols==b->ncols && a->nrows==b->nrows) {
        objectcomplexmatrix *new=complexmatrix_new(a->nrows, a->ncols, true);
        if (new) {
            complexmatrix_promote(b, new);
            matrix_axpy(alpha, a, new);
        }
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES);
    return out;
}

value ComplexMatrix_add__matrix(vm *v, int nargs, value *args) {
    return _axpy(v, nargs, args, 1.0);
}

value ComplexMatrix_sub__matrix(vm *v, int nargs, value *args) {
    value out = _axpy(v, nargs, args, -1.0);
    if (matrix_isamatrix(out)) matrix_scale(MORPHO_GETMATRIX(out), -1.0); // -(-A + B)
    return out;
}

value ComplexMatrix_subr__matrix(vm *v, int nargs, value *args) {
    return _axpy(v, nargs, args, -1.0);
}

value ComplexMatrix_mul__complex(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    
    objectmatrix *new = matrix_clone(a);
    if (new) complexmatrix_scale(new, MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0))->Z);
    return morpho_wrapandbind(v, (object *) new);
}

/** Multiplication by a complexmatrix or a regular matrix */
static bool _promote(vm *v, objectmatrix *b, objectmatrix **bp) { // Promotes b to a complexmatrix
    *bp=complexmatrix_new(b->nrows, b->ncols, true);
    if (!*bp) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }
    return complexmatrix_promote(b, *bp)==LINALGERR_OK;
}

static value _axb(vm *v, objectmatrix *a, objectmatrix *b) { // Performs a*b returning a wrapped value
    if (a->ncols!=b->nrows) { morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES); return MORPHO_NIL; }
    objectcomplexmatrix *new=complexmatrix_new(a->nrows, b->ncols, false);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return MORPHO_NIL; }
    complexmatrix_mmul(MCBuild(1.0, 0.0), a, b, MCBuild(0.0, 0.0), new);
    return morpho_wrapandbind(v, (object *) new);
}

static value _mul(vm *v, value a, value b, bool promoteb, bool swap) { // Driver routine for a*b
    objectmatrix *A=MORPHO_GETMATRIX(a), *B=MORPHO_GETMATRIX(b), *bp=NULL;
    if (promoteb) { if (_promote(v, B, &bp)) { B=bp; } else { return MORPHO_NIL; } } // Promote b if requested
    value out = (swap ? _axb(v, B, A) : _axb(v, A, B)); // Multiply, swapping arguments if requested
    if (bp) object_free((object *) bp);
    return out;
}

value ComplexMatrix_mul__complexmatrix(vm *v, int nargs, value *args) {
    return _mul(v, MORPHO_SELF(args), MORPHO_GETARG(args, 0), false, false);
}

value ComplexMatrix_mul__matrix(vm *v, int nargs, value *args) {
    return _mul(v, MORPHO_SELF(args), MORPHO_GETARG(args, 0), true, false);
}

value ComplexMatrix_mulr__matrix(vm *v, int nargs, value *args) {
    return _mul(v, MORPHO_SELF(args), MORPHO_GETARG(args, 0), true, true);
}

value ComplexMatrix_div__matrix(vm *v, int nargs, value *args) {
    objectmatrix *b=MORPHO_GETMATRIX(MORPHO_SELF(args)), *A=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0)), *ap=NULL;
    objectmatrix *new=matrix_clone(b);
    
    if (!new) return morpho_wrapandbind(v, NULL);
    if (!_promote(v, A, &ap)) goto ComplexMatrix_div__matrix_cleanup;
    
    LINALG_ERRCHECKVMGOTO(matrix_solve(ap, new), ComplexMatrix_div__matrix_cleanup);
    
    object_free((object *) ap);
    return morpho_wrapandbind(v, (object *) new);
    
ComplexMatrix_div__matrix_cleanup:
    if (new) object_free((object *) new);
    if (ap) object_free((object *) ap);
    return MORPHO_NIL;
}

value ComplexMatrix_divr__matrix(vm *v, int nargs, value *args) {
    objectmatrix *A=MORPHO_GETMATRIX(MORPHO_SELF(args)), *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0)), *bp=NULL;
    
    if (!_promote(v, b, &bp)) goto ComplexMatrix_divr__matrix_cleanup;
    LINALG_ERRCHECKVMGOTO(matrix_solve(A, bp), ComplexMatrix_divr__matrix_cleanup);
    
    return morpho_wrapandbind(v, (object *) bp);
    
ComplexMatrix_divr__matrix_cleanup:
    if (bp) object_free((object *) bp);
    return MORPHO_NIL;
}

/** Computes the trace */
value ComplexMatrix_trace(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    MorphoComplex tr=MCBuild(0,0);
    LINALG_ERRCHECKVM(complexmatrix_trace(a, &tr));
    objectcomplex *new = object_newcomplex(creal(tr), cimag(tr));
    return morpho_wrapandbind(v, (object *) new);
}

/** Inverts a matrix */
value ComplexMatrix_inverse(vm *v, int nargs, value *args) {
    objectcomplexmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    objectcomplexmatrix *new = matrix_clone(a);
    out = morpho_wrapandbind(v, (object *) new);
    if (new) LINALG_ERRCHECKVM(complexmatrix_inverse(new));
    
    return out;
}

static value _realimag(vm *v, int nargs, value *args, bool imag) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *new=matrix_new(a->nrows, a->ncols, false);
    if (new) complexmatrix_demote(a, new, imag);
    return morpho_wrapandbind(v, (object *) new);
}

/** Extract real part */
value ComplexMatrix_real(vm *v, int nargs, value *args) {
    return _realimag(v, nargs, args, false);
}

/** Extract imaginary part */
value ComplexMatrix_imag(vm *v, int nargs, value *args) {
    return _realimag(v, nargs, args, true);
}

static value _conj(vm *v, int nargs, value *args, bool trans) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *new=matrix_clone(a);
    if (new) {
        if (trans) matrix_transpose(a, new);
        cblas_dscal(a->nrows*a->ncols, -1.0, new->elements+1, new->nvals);
    }
    return morpho_wrapandbind(v, (object *) new);
}

/** Extract imaginary part */
value ComplexMatrix_conj(vm *v, int nargs, value *args) {
    return _conj(v, nargs, args, false);
}

/** Return conjugate transpose */
value ComplexMatrix_conjTranspose(vm *v, int nargs, value *args) {
    return _conj(v, nargs, args, true);
}

/* ---------
 * Products
 * --------- */

/** Frobenius inner product */
value ComplexMatrix_inner(vm *v, int nargs, value *args) {
    objectcomplexmatrix *a=MORPHO_GETCOMPLEXMATRIX(MORPHO_SELF(args)), *b = NULL;
    MorphoComplex prod=MCBuild(0.0, 0.0);
    value arg = MORPHO_GETARG(args, 0), out = MORPHO_NIL;
    objectmatrix *bp = NULL;
    
    if (MORPHO_ISCOMPLEXMATRIX(arg)) {
        b=MORPHO_GETCOMPLEXMATRIX(arg);
    } else if (_promote(v, MORPHO_GETMATRIX(arg), &bp)) {
        b=bp;
    } else goto ComplexMatrix_inner_cleanup;
    
    if (complexmatrix_inner(a, b, &prod)==LINALGERR_OK) {
        objectcomplex *new = object_newcomplex(creal(prod), cimag(prod));
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES);
    
ComplexMatrix_inner_cleanup:
    if (bp) object_free((object *) bp);
    return out;
}

/** Outer product */
value ComplexMatrix_outer(vm *v, int nargs, value *args) {
    objectcomplexmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectcomplexmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
    
    objectcomplexmatrix *new=complexmatrix_new(a->nrows*a->ncols, b->nrows*b->ncols, true);
    if (new) LINALG_ERRCHECKVM(complexmatrix_r1update(MCBuild(1.0,0.0), a, b, new));
    
    return morpho_wrapandbind(v, (object *) new);
}

MORPHO_BEGINCLASS(ComplexMatrix)
MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "()", Matrix_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_FORMAT_METHOD, "(String)", Matrix_format, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ASSIGN_METHOD, "(ComplexMatrix)", Matrix_assign, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, "ComplexMatrix ()", Matrix_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Complex (Int)", Matrix_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "ComplexMatrix (Range)", Matrix_index__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "ComplexMatrix (List)", Matrix_index__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "ComplexMatrix (Tuple)", Matrix_index__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Complex (Int, Int)", Matrix_index__int_int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "ComplexMatrix (_,_)", Matrix_index__x_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Matrix (...)", Matrix_index__err, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,_)", Matrix_setindex__int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,Int,_)", Matrix_setindex__int_int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(_,_,ComplexMatrix)", Matrix_setindex__x_x_matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_GETCOLUMN_METHOD, "ComplexMatrix (Int)", Matrix_getcolumn__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_SETCOLUMN_METHOD, "(Int, ComplexMatrix)", Matrix_setcolumn__int_matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "ComplexMatrix (ComplexMatrix)", Matrix_add__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "ComplexMatrix (Matrix)", ComplexMatrix_add__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "ComplexMatrix (_)", Matrix_add__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "ComplexMatrix (Nil)", Matrix_add__nil, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "ComplexMatrix (Matrix)", ComplexMatrix_add__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "ComplexMatrix (_)", Matrix_add__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "ComplexMatrix (Nil)", Matrix_add__nil, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "ComplexMatrix (ComplexMatrix)", Matrix_sub__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "ComplexMatrix (Matrix)", ComplexMatrix_sub__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "ComplexMatrix (Nil)", Matrix_add__nil, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "ComplexMatrix (_)", Matrix_sub__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "ComplexMatrix (_)", Matrix_subr__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "ComplexMatrix (Matrix)", ComplexMatrix_subr__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "ComplexMatrix (_)", Matrix_mul__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "ComplexMatrix (Complex)", ComplexMatrix_mul__complex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "ComplexMatrix (ComplexMatrix)", ComplexMatrix_mul__complexmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "ComplexMatrix (Matrix)", ComplexMatrix_mul__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "ComplexMatrix (Matrix)", ComplexMatrix_mulr__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "ComplexMatrix (Complex)", ComplexMatrix_mul__complex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "ComplexMatrix (_)", Matrix_mul__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "ComplexMatrix (_)", Matrix_div__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "ComplexMatrix (Matrix)", ComplexMatrix_div__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "ComplexMatrix (ComplexMatrix)", Matrix_div__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIVR_METHOD, "ComplexMatrix (Matrix)", ComplexMatrix_divr__matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ACC_METHOD, "(_, ComplexMatrix)", Matrix_acc__x_x_matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_INVERSE_METHOD, "ComplexMatrix ()", ComplexMatrix_inverse, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_NORM_METHOD, "Float (_)", Matrix_norm__x, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MATRIX_NORM_METHOD, "Float ()", Matrix_norm, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_SUM_METHOD, "Complex ()", Matrix_sum, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_TRACE_METHOD, "Complex ()", ComplexMatrix_trace, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_TRANSPOSE_METHOD, "ComplexMatrix ()", Matrix_transpose, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(COMPLEX_REAL_METHOD, "Matrix ()", ComplexMatrix_real, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(COMPLEX_IMAG_METHOD, "Matrix ()", ComplexMatrix_imag, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(COMPLEX_CONJUGATE_METHOD, "ComplexMatrix ()", ComplexMatrix_conj, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(COMPLEXMATRIX_CONJTRANSPOSE_METHOD, "ComplexMatrix ()", ComplexMatrix_conjTranspose, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_INNER_METHOD, "Complex (Matrix)", ComplexMatrix_inner, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_OUTER_METHOD, "ComplexMatrix (ComplexMatrix)", ComplexMatrix_outer, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_EIGENVALUES_METHOD, "Tuple ()", Matrix_eigenvalues, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_EIGENSYSTEM_METHOD, "Tuple ()", Matrix_eigensystem, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_SVD_METHOD, "Tuple ()", Matrix_svd, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_QR_METHOD, "Tuple ()", Matrix_qr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_RESHAPE_METHOD, "(Int,Int)", Matrix_reshape, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_ROLL_METHOD, "ComplexMatrix (Int)", Matrix_roll__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MATRIX_ROLL_METHOD, "ComplexMatrix (Int,Int)", Matrix_roll__int_int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ENUMERATE_METHOD, "(Int)", Matrix_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int ()", Matrix_count, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MATRIX_DIMENSIONS_METHOD, "Tuple ()", Matrix_dimensions, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void complexmatrix_initialize(void) {
    objectcomplexmatrixtype=object_addtype(&objectmatrixdefn);
    matrix_addinterface(&complexmatrixdefn);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    value complexmatrixclass=builtin_addclass(COMPLEXMATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(ComplexMatrix), objclass);
    object_setveneerclass(OBJECT_COMPLEXMATRIX, complexmatrixclass);
    
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (Int, Int)", complexmatrix_constructor__int_int, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (Int)", complexmatrix_constructor__int, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (ComplexMatrix)", matrix_constructor__matrix, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (Matrix)", complematrix_constructor__matrix, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (List)", complexmatrix_constructor__list, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (Tuple)", complexmatrix_constructor__list, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (Array)", complexmatrix_constructor__array, MORPHO_FN_CONSTRUCTOR, NULL);
}
