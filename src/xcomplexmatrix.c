/** @file xcomplexmatrix.c
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#define ACCELERATE_NEW_LAPACK
#define MORPHO_INCLUDE_LINALG

#include <platform.h>

#include "newlinalg.h"
#include "xmatrix.h"
#include "xcomplexmatrix.h"
#include "format.h"
#include "cmplx.h"

objecttype objectcomplexmatrixtype;
#define OBJECT_COMPLEXMATRIX objectcomplexmatrixtype

typedef objectxmatrix objectcomplexmatrix;

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
static double _normfn(objectxmatrix *a, xmatrix_norm_t nrm) {
    char cnrm = xmatrix_normtolapack(nrm);
    int nrows=a->nrows, ncols=a->ncols;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    return LAPACKE_zlange(LAPACK_COL_MAJOR, cnrm, a->nrows, a->ncols, a->elements, a->nrows);
#else
    double work[a->nrows];
    return zlange_(&cnrm, &nrows, &ncols, (__LAPACK_double_complex *) a->elements, &nrows, work);
#endif
}

/** Low level linear solve */
static linalgError_t _solve(objectxmatrix *a, objectxmatrix *b, int *pivot) {
    int n=a->nrows, nrhs = b->ncols, info;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_zgesv(LAPACK_COL_MAJOR, n, nrhs, a->elements, n, pivot, b->elements, n);
#else
    zgesv_(&n, &nrhs, (__LAPACK_double_complex *) a->elements,
           &n, pivot, (__LAPACK_double_complex *) b->elements, &n, &info);
#endif
    
    return (info==0 ? LINALGERR_OK : (info>0 ? LINALGERR_MATRIX_SINGULAR : LINALGERR_LAPACK_INVLD_ARGS));
}

/** Low level eigensolver */
static linalgError_t _eigen(objectxmatrix *a, MorphoComplex *w, objectxmatrix *vec) {
    int info, n=a->nrows;

#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', (vec ? 'V' : 'N'), n, a->elements, n, (__LAPACK_double_complex *) w, NULL, n, (vec ? vec->elements : NULL), n);
#else
    int lwork=4*n; MorphoComplex work[4*n]; double rwork[2*n];
    zgeev_("N", (vec ? "V" : "N"), &n, (__LAPACK_double_complex *) a->elements, &n, (__LAPACK_double_complex *) w, NULL, &n, (__LAPACK_double_complex *) (vec ? vec->elements : NULL), &n, work, &lwork, rwork, &info);
#endif
    
    return (info==0 ? LINALGERR_OK : (info>0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS));
}

/** Low level SVD */
static linalgError_t _svd(objectxmatrix *a, double *s, objectxmatrix *u, objectxmatrix *vt) {
    int info, m=a->nrows, n=a->ncols;
    int minmn = (m < n) ? m : n;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info = LAPACKE_zgesvd(LAPACK_COL_MAJOR,
                          (u ? 'A' : 'N'),      // jobu: 'A' = all U columns, 'N' = no U
                          (vt ? 'A' : 'N'),     // jobvt: 'A' = all VT rows, 'N' = no VT
                          m, n,
                          (__LAPACK_double_complex *) a->elements, m,  // input matrix A (overwritten)
                          s,                    // singular values (min(m,n))
                          (__LAPACK_double_complex *) (u ? u->elements : NULL), m,  // U matrix (m×m)
                          (__LAPACK_double_complex *) (vt ? vt->elements : NULL), n  // VT matrix (n×n)
                         );
#else
    int lwork = -1;
    __LAPACK_double_complex work_query;
    double rwork[5 * minmn];  // rwork needs at least 5*min(m,n) for zgesvd
    
    // Query optimal work size
    zgesvd_((u ? "A" : "N"), (vt ? "A" : "N"), &m, &n, 
            (__LAPACK_double_complex *) a->elements, &m, s,
            (__LAPACK_double_complex *) (u ? u->elements : NULL), &m,
            (__LAPACK_double_complex *) (vt ? vt->elements : NULL), &n,
            &work_query, &lwork, rwork, &info);
    
    if (info != 0) return (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS);
    
    lwork = (int)creal(work_query);
    __LAPACK_double_complex work[lwork];
    zgesvd_((u ? "A" : "N"), (vt ? "A" : "N"), &m, &n,
            (__LAPACK_double_complex *) a->elements, &m, s,
            (__LAPACK_double_complex *) (u ? u->elements : NULL), &m,
            (__LAPACK_double_complex *) (vt ? vt->elements : NULL), &n,
            work, &lwork, rwork, &info);
#endif
    
    return (info == 0 ? LINALGERR_OK : (info > 0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS));
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
    .svdfn = _svd
};

/* ----------------------
 * Constructor
 * ---------------------- */

/** Create a new complex matrix */
objectcomplexmatrix *complexmatrix_new(MatrixIdx_t nrows, MatrixIdx_t ncols, bool zero) {
    return (objectcomplexmatrix *) xmatrix_newwithtype(OBJECT_COMPLEXMATRIX, nrows, ncols, 2, zero);
}

/* ----------------------
 * Element access
 * ---------------------- */

/** Sets a matrix element. */
linalgError_t complexmatrix_setelement(objectcomplexmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, MorphoComplex value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return LINALGERR_INDX_OUT_OF_BNDS;
        
    MatrixCount_t ix = matrix->nvals*(col*matrix->nrows+row);
    matrix->elements[ix]=creal(value);
    matrix->elements[ix+1]=cimag(value);
    return LINALGERR_OK;
}

/** Gets a matrix element */
linalgError_t complexmatrix_getelement(objectcomplexmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, MorphoComplex *value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return LINALGERR_INDX_OUT_OF_BNDS;
    
    MatrixCount_t ix = matrix->nvals*(col*matrix->nrows+row);
    if (value) *value=MCBuild(matrix->elements[ix],matrix->elements[ix+1]);
    return LINALGERR_OK;
}

/** Copies a real matrix x into a complex matrix y */
static linalgError_t _stridedcopy(objectxmatrix *x, objectxmatrix *y, int offset) {
    if (!(x->ncols==y->ncols && x->nrows==y->nrows)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_dcopy((__LAPACK_int) x->ncols*x->nrows, x->elements+offset, x->nvals, y->elements, y->nvals);
    return LINALGERR_OK;
}

linalgError_t complexmatrix_promote(objectxmatrix *x, objectcomplexmatrix *y) {
    return _stridedcopy(x, y, 0);
}

/** Copies the real part of a complex matrix y into  */
linalgError_t complexmatrix_demote(objectcomplexmatrix *x, objectxmatrix *y, bool imag) {
    return _stridedcopy(x, y, (imag?1:0));
}

/* ----------------------
 * Complex arithmetic
 * ---------------------- */

/** Performs c <- alpha*(a*b) + beta*c with complex matrices */
linalgError_t complexmatrix_mmul(MorphoComplex alpha, objectxmatrix *a, objectxmatrix *b, MorphoComplex beta, objectxmatrix *c) {
    if (!(a->ncols==b->nrows && a->nrows==c->nrows && b->ncols==c->ncols)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                a->nrows, b->ncols, a->ncols,
                &alpha, (__LAPACK_double_complex *) a->elements,
                a->nrows, (__LAPACK_double_complex *) b->elements, b->nrows,
                &beta, (__LAPACK_double_complex *) c->elements, c->nrows);
    return LINALGERR_OK;
}

/** Scales a matrix x <- scale * x >*/
void complexmatrix_scale(objectxmatrix *a, MorphoComplex scale) {
    cblas_zscal(a->nrows * a->ncols, (__LAPACK_double_complex *) &scale, (__LAPACK_double_complex *) a->elements, 1);
}

/** Finds the Frobenius inner product of two complex matrices (a, b) = \sum_{i,j} conj(a)_ij * b_ij */
linalgError_t complexmatrix_inner(objectcomplexmatrix *a, objectcomplexmatrix *b, MorphoComplex *out) {
    if (!(a->ncols==b->ncols && a->nrows==b->nrows)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_zdotc_sub(a->nrows * a->ncols, (__LAPACK_double_complex *) a->elements, 1,
                     (__LAPACK_double_complex *) b->elements, 1,
                     (__LAPACK_double_complex *) out);
    return LINALGERR_OK;
}

/** Rank 1 update: Performs  c <- alpha*a \outer b + c; a and b are treated as column vectors */
linalgError_t complexmatrix_r1update(MorphoComplex alpha, objectcomplexmatrix *a, objectcomplexmatrix *b, objectcomplexmatrix *c) {
    MatrixIdx_t m=a->nrows*a->ncols, n=b->nrows*b->ncols;
    if (!(m==c->nrows && n==c->ncols)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_zgeru(CblasColMajor, m, n, (__LAPACK_double_complex *) &alpha, (__LAPACK_double_complex *) a->elements, 1,
                (__LAPACK_double_complex *) b->elements, 1,
                (__LAPACK_double_complex *) c->elements, c->nrows);
    return LINALGERR_OK;
}

/** Calculate the trace of a matrix */
linalgError_t complexmatrix_trace(objectcomplexmatrix *a, MorphoComplex *out) {
    if (a->nrows!=a->ncols) return LINALGERR_NOT_SQUARE;
    MorphoComplex one = MCBuild(1.0, 0.0);
    cblas_zdotu_sub(a->nrows, (__LAPACK_double_complex *) a->elements, a->ncols+1, (__LAPACK_double_complex *) &one, 0, (__LAPACK_double_complex *) out);
    return LINALGERR_OK;
}

/** Inverts the matrix a
 * @param[in] a  matrix to be inverted
 * @returns linalgError_t indicating the status; MATRIX_OK indicates success. */
linalgError_t complexmatrix_inverse(objectcomplexmatrix *a) {
    int nrows=a->nrows, ncols=a->ncols, info;
    int pivot[nrows];
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_zgetrf(LAPACK_COL_MAJOR, nrows, ncols, a->elements, nrows, pivot);
#else
    zgetrf_(&nrows, &ncols, (__LAPACK_double_complex *) a->elements, &nrows, pivot, &info);
#endif
    if (info!=0) return (info>0 ? LINALGERR_MATRIX_SINGULAR : LINALGERR_LAPACK_INVLD_ARGS);
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_zgetri(LAPACK_COL_MAJOR, nrows, a->elements, nrows, pivot);
#else
    int lwork=nrows*ncols; __LAPACK_double_complex work[nrows*ncols];
    zgetri_(&nrows, (__LAPACK_double_complex *) a->elements, &nrows, pivot, work, &lwork, &info);
#endif
    
    return (info==0 ? LINALGERR_OK : (info>0 ? LINALGERR_MATRIX_SINGULAR : LINALGERR_LAPACK_INVLD_ARGS));
}

/* **********************************************************************
 * ComplexMatrix constructors
 * ********************************************************************** */

value complexmatrix_constructor__int_int(vm *v, int nargs, value *args) {
    MatrixIdx_t nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    MatrixIdx_t ncols = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    
    objectcomplexmatrix *new=complexmatrix_new(nrows, ncols, true);
    return morpho_wrapandbind(v, (object *) new);
}

value complexmatrix_constructor__int(vm *v, int nargs, value *args) {
    MatrixIdx_t nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    
    objectcomplexmatrix *new=complexmatrix_new(nrows, 1, true);
    return morpho_wrapandbind(v, (object *) new);
}

value complexmatrix_constructor__xmatrix(vm *v, int nargs, value *args) {
    objectxmatrix *a = MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0));
    
    objectcomplexmatrix *new=complexmatrix_new(a->nrows, a->ncols, true);
    if (new) complexmatrix_promote(a, new);
    return morpho_wrapandbind(v, (object *) new);
}

/** Constructs a complexmatrix from a list of lists or tuples */
value complexmatrix_constructor__list(vm *v, int nargs, value *args) {
    objectxmatrix *new = xmatrix_listconstructor(v, MORPHO_GETARG(args, 0), OBJECT_COMPLEXMATRIX, 2);
    return morpho_wrapandbind(v, (object *) new);
}

/** Constructs a matrix from an array */
value complexmatrix_constructor__array(vm *v, int nargs, value *args) {
    objectarray *a = MORPHO_GETARRAY(MORPHO_GETARG(args, 0));
    if (a->ndim!=2) { morpho_runtimeerror(v, LINALG_INVLDARGS); return MORPHO_NIL; }
    
    objectxmatrix *new = xmatrix_arrayconstructor(v, a, OBJECT_COMPLEXMATRIX, 2);
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
    objectcomplexmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectxmatrix *b=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0));
    value out=MORPHO_NIL;
    
    if (a->ncols==b->ncols && a->nrows==b->nrows) {
        objectcomplexmatrix *new=complexmatrix_new(a->nrows, a->ncols, true);
        if (new) {
            complexmatrix_promote(b, new);
            xmatrix_axpy(alpha, a, new);
        }
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES);
    return out;
}

value ComplexMatrix_add__xmatrix(vm *v, int nargs, value *args) {
    return _axpy(v, nargs, args, 1.0);
}

value ComplexMatrix_sub__xmatrix(vm *v, int nargs, value *args) {
    value out = _axpy(v, nargs, args, -1.0);
    if (xmatrix_isamatrix(out)) xmatrix_scale(MORPHO_GETXMATRIX(out), -1.0); // -(-A + B)
    return out;
}

value ComplexMatrix_subr__xmatrix(vm *v, int nargs, value *args) {
    return _axpy(v, nargs, args, -1.0);
}

value ComplexMatrix_mul__complex(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    
    objectxmatrix *new = xmatrix_clone(a);
    if (new) complexmatrix_scale(new, MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0))->Z);
    return morpho_wrapandbind(v, (object *) new);
}

/** Multiplication by a complexmatrix or a regular matrix */
static bool _promote(vm *v, objectxmatrix *b, objectxmatrix **bp) { // Promotes b to a complexmatrix
    *bp=complexmatrix_new(b->nrows, b->ncols, true);
    if (!*bp) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }
    return complexmatrix_promote(b, *bp)==LINALGERR_OK;
}

static value _axb(vm *v, objectxmatrix *a, objectxmatrix *b) { // Performs a*b returning a wrapped value
    if (a->ncols!=b->nrows) { morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES); return MORPHO_NIL; }
    objectcomplexmatrix *new=complexmatrix_new(a->nrows, b->ncols, false);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return MORPHO_NIL; }
    complexmatrix_mmul(MCBuild(1.0, 0.0), a, b, MCBuild(0.0, 0.0), new);
    return morpho_wrapandbind(v, (object *) new);
}

static value _mul(vm *v, value a, value b, bool promoteb, bool swap) { // Driver routine for a*b
    objectxmatrix *A=MORPHO_GETXMATRIX(a), *B=MORPHO_GETXMATRIX(b), *bp=NULL;
    if (promoteb) { if (_promote(v, B, &bp)) { B=bp; } else { return MORPHO_NIL; } } // Promote b if requested
    value out = (swap ? _axb(v, B, A) : _axb(v, A, B)); // Multiply, swapping arguments if requested
    if (bp) object_free((object *) bp);
    return out;
}

value ComplexMatrix_mul__complexmatrix(vm *v, int nargs, value *args) {
    return _mul(v, MORPHO_SELF(args), MORPHO_GETARG(args, 0), false, false);
}

value ComplexMatrix_mul__xmatrix(vm *v, int nargs, value *args) {
    return _mul(v, MORPHO_SELF(args), MORPHO_GETARG(args, 0), true, false);
}

value ComplexMatrix_mulr__xmatrix(vm *v, int nargs, value *args) {
    return _mul(v, MORPHO_SELF(args), MORPHO_GETARG(args, 0), true, true);
}

value ComplexMatrix_div__xmatrix(vm *v, int nargs, value *args) {
    objectxmatrix *b=MORPHO_GETXMATRIX(MORPHO_SELF(args)), *A=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0)), *ap=NULL;
    objectxmatrix *new=xmatrix_clone(b);
    if (new && _promote(v, A, &ap)) xmatrix_solve(ap, new);
    return morpho_wrapandbind(v, (object *) new);
}

value ComplexMatrix_divr__xmatrix(vm *v, int nargs, value *args) {
    objectxmatrix *A=MORPHO_GETXMATRIX(MORPHO_SELF(args)), *b=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0)), *bp=NULL;
    if (_promote(v, b, &bp)) xmatrix_solve(A, bp); // Promote the matrix that will contain the solution anyway
    return morpho_wrapandbind(v, (object *) bp);
}

/** Computes the trace */
value ComplexMatrix_trace(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    MorphoComplex tr=MCBuild(0,0);
    LINALG_ERRCHECKVM(complexmatrix_trace(a, &tr));
    objectcomplex *new = object_newcomplex(creal(tr), cimag(tr));
    return morpho_wrapandbind(v, (object *) new);
}

/** Inverts a matrix */
value ComplexMatrix_inverse(vm *v, int nargs, value *args) {
    objectcomplexmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    objectcomplexmatrix *new = xmatrix_clone(a);
    out = morpho_wrapandbind(v, (object *) new);
    if (new) LINALG_ERRCHECKVM(complexmatrix_inverse(new));
    
    return out;
}

static value _realimag(vm *v, int nargs, value *args, bool imag) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectxmatrix *new=xmatrix_new(a->nrows, a->ncols, false);
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
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectxmatrix *new=xmatrix_clone(a);
    if (new) {
        if (trans) xmatrix_transpose(a, new);
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
    objectcomplexmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectcomplexmatrix *b=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0));
    MorphoComplex prod=MCBuild(0.0, 0.0);
    value out = MORPHO_NIL;
    
    if (complexmatrix_inner(a, b, &prod)==LINALGERR_OK) {
        objectcomplex *new = object_newcomplex(creal(prod), cimag(prod));
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES);
    
    return out;
}

/** Outer product */
value ComplexMatrix_outer(vm *v, int nargs, value *args) {
    objectcomplexmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectcomplexmatrix *b=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0));
    
    objectcomplexmatrix *new=complexmatrix_new(a->nrows*a->ncols, b->nrows*b->ncols, true);
    if (new) LINALG_ERRCHECKVM(complexmatrix_r1update(MCBuild(1.0,0.0), a, b, new));
    
    return morpho_wrapandbind(v, (object *) new);
}

MORPHO_BEGINCLASS(ComplexMatrix)
MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "()", XMatrix_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_FORMAT_METHOD, "(String)", XMatrix_format, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ASSIGN_METHOD, "(ComplexMatrix)", XMatrix_assign, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, "ComplexMatrix ()", XMatrix_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Complex (Int)", XMatrix_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Complex (Int, Int)", XMatrix_index__int_int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "ComplexMatrix (_,_)", XMatrix_index__x_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,_)", XMatrix_setindex__int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,Int,_)", XMatrix_setindex__int_int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(_,_,ComplexMatrix)", XMatrix_setindex__x_x_xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_GETCOLUMN_METHOD, "ComplexMatrix (Int)", XMatrix_getcolumn__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_SETCOLUMN_METHOD, "(Int, ComplexMatrix)", XMatrix_setcolumn__int_xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "ComplexMatrix (ComplexMatrix)", XMatrix_add__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "ComplexMatrix (XMatrix)", ComplexMatrix_add__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "ComplexMatrix (_)", XMatrix_add__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "ComplexMatrix (Nil)", XMatrix_add__nil, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "ComplexMatrix (XMatrix)", ComplexMatrix_add__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "ComplexMatrix (_)", XMatrix_add__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "ComplexMatrix (Nil)", XMatrix_add__nil, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "ComplexMatrix (ComplexMatrix)", XMatrix_sub__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "ComplexMatrix (XMatrix)", ComplexMatrix_sub__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "ComplexMatrix (Nil)", XMatrix_add__nil, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "ComplexMatrix (_)", XMatrix_sub__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "ComplexMatrix (_)", XMatrix_subr__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "ComplexMatrix (XMatrix)", ComplexMatrix_subr__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "ComplexMatrix (_)", XMatrix_mul__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "ComplexMatrix (Complex)", ComplexMatrix_mul__complex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "ComplexMatrix (ComplexMatrix)", ComplexMatrix_mul__complexmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "ComplexMatrix (XMatrix)", ComplexMatrix_mul__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "ComplexMatrix (XMatrix)", ComplexMatrix_mulr__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "ComplexMatrix (Complex)", ComplexMatrix_mul__complex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "ComplexMatrix (_)", XMatrix_mul__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "ComplexMatrix (_)", XMatrix_div__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "ComplexMatrix (XMatrix)", ComplexMatrix_div__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "ComplexMatrix (ComplexMatrix)", XMatrix_div__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIVR_METHOD, "ComplexMatrix (XMatrix)", ComplexMatrix_divr__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ACC_METHOD, "(_, ComplexMatrix)", XMatrix_acc__x_xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_INVERSE_METHOD, "ComplexMatrix ()", ComplexMatrix_inverse, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_NORM_METHOD, "Float (_)", XMatrix_norm__x, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(XMATRIX_NORM_METHOD, "Float ()", XMatrix_norm, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_SUM_METHOD, "Complex ()", XMatrix_sum, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_TRACE_METHOD, "Complex ()", ComplexMatrix_trace, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_TRANSPOSE_METHOD, "ComplexMatrix ()", XMatrix_transpose, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(COMPLEX_REAL_METHOD, "XMatrix ()", ComplexMatrix_real, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(COMPLEX_IMAG_METHOD, "XMatrix ()", ComplexMatrix_imag, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(COMPLEX_CONJUGATE_METHOD, "ComplexMatrix ()", ComplexMatrix_conj, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(COMPLEXMATRIX_CONJTRANSPOSE_METHOD, "ComplexMatrix ()", ComplexMatrix_conjTranspose, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_INNER_METHOD, "Complex (XMatrix)", ComplexMatrix_inner, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_OUTER_METHOD, "ComplexMatrix (ComplexMatrix)", ComplexMatrix_outer, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_EIGENVALUES_METHOD, "Tuple ()", XMatrix_eigenvalues, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_EIGENSYSTEM_METHOD, "Tuple ()", XMatrix_eigensystem, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_SVD_METHOD, "Tuple ()", XMatrix_svd, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_RESHAPE_METHOD, "(Int,Int)", XMatrix_reshape, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_ROLL_METHOD, "ComplexMatrix (Int)", XMatrix_roll__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_ROLL_METHOD, "ComplexMatrix (Int,Int)", XMatrix_roll__int_int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ENUMERATE_METHOD, "(Int)", XMatrix_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int ()", XMatrix_count, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(XMATRIX_DIMENSIONS_METHOD, "Tuple ()", XMatrix_dimensions, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void complexmatrix_initialize(void) {
    objectcomplexmatrixtype=object_addtype(&objectxmatrixdefn);
    xmatrix_addinterface(&complexmatrixdefn);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    value complexmatrixclass=builtin_addclass(COMPLEXMATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(ComplexMatrix), objclass);
    object_setveneerclass(OBJECT_COMPLEXMATRIX, complexmatrixclass);
    
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (Int, Int)", complexmatrix_constructor__int_int, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (Int)", complexmatrix_constructor__int, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (ComplexMatrix)", xmatrix_constructor__xmatrix, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (XMatrix)", complexmatrix_constructor__xmatrix, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (List)", complexmatrix_constructor__list, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (Tuple)", complexmatrix_constructor__list, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (Array)", complexmatrix_constructor__array, MORPHO_FN_CONSTRUCTOR, NULL);
}
