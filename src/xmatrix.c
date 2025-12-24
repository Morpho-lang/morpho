/** @file xmatrix.c
 *  @author T J Atherton
 *
 *  @brief New matrices
*/

#define ACCELERATE_NEW_LAPACK
#define MORPHO_INCLUDE_LINALG

#include "newlinalg.h"
#include "format.h"

/* **********************************************************************
 * Matrix interface definitions
 * ********************************************************************** */

/** Hold the matrix interface definitions as they're created */
static matrixinterfacedefn _matrixdefn[LINALG_MAXMATRIXDEFNS];
objecttype matrixinterfacedefnnext=0; /** Type of the next object definition */

void xmatrix_addinterface(matrixinterfacedefn *defn) {
    if (matrixinterfacedefnnext<LINALG_MAXMATRIXDEFNS) {
        _matrixdefn[matrixinterfacedefnnext]=*defn;
        matrixinterfacedefnnext++;
    } else UNREACHABLE("Too many matrix interface definitions.");
}

matrixinterfacedefn *xmatrix_getinterface(objectxmatrix *a) {
    int iindx = a->obj.type-OBJECT_XMATRIX;
    if (iindx<LINALG_MAXMATRIXDEFNS) return &_matrixdefn[iindx];
    return NULL;
}

/* **********************************************************************
 * XMatrix objects
 * ********************************************************************** */

objecttype objectxmatrixtype;

/** Matrix object definitions */
size_t objectxmatrix_sizefn(object *obj) {
    return sizeof(objectxmatrix)+sizeof(double) * ((objectxmatrix *) obj)->nels;
}

void objectxmatrix_printfn(object *obj, void *v) {
    objectclass *klass=object_getveneerclass(obj->type);
    morpho_printf(v, "<");
    morpho_printvalue(v, klass->name);
    morpho_printf(v, ">");
}

objecttypedefn objectxmatrixdefn = {
    .printfn=objectxmatrix_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectxmatrix_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/* **********************************************************************
 * XMatrix utility functions
 * ********************************************************************** */

/* ----------------------
 * XMatrix interface
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

/** Low level linear solve */
static linalgError_t _solve(objectxmatrix *a, objectxmatrix *b, int *pivot) {
    int n=a->nrows, nrhs = b->ncols, info;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a->elements, n, pivot, b->elements, n);
#else
    dgesv_(&n, &nrhs, a->elements, &n, pivot, b->elements, &n, &info);
#endif
    
    return (info==0 ? LINALGERR_OK : (info>0 ? LINALGERR_MATRIX_SINGULAR : LINALGERR_LAPACK_INVLD_ARGS));
}

/** Low level eigensolver */
static linalgError_t _eigen(objectxmatrix *a, MorphoComplex *w, objectxmatrix *vec) {
    int info, n=a->nrows;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', (vec ? 'V' : 'N'), n, a->elements, n, wr, wi, NULL, n, (vec ? vec->elements : NULL), n);
#else
    int lwork=4*n; double work[4*n], wr[n], wi[n];
    dgeev_("N", (vec ? "V" : "N"), &n, a->elements, &n, wr, wi, NULL, &n, (vec ? vec->elements : NULL), &n, work, &lwork, &info);
    for (int i=0; i<n; i++) w[i]=MCBuild(wr[i], wi[i]);
#endif
    
    return (info==0 ? LINALGERR_OK : (info>0 ? LINALGERR_OP_FAILED : LINALGERR_LAPACK_INVLD_ARGS));
}

/* ----------------------
 * Interface definition
 * ---------------------- */

matrixinterfacedefn xmatrixdefn = {
    .printelfn = _printelfn,
    .printeltobufffn = _printeltobufffn,
    .getelfn = _getelfn,
    .setelfn = _setelfn,
    .solvefn = _solve,
    .eigenfn = _eigen
};

/* ----------------------
 * Constructors
 * ---------------------- */

/** Create a generic matrix with given type and layout */
objectxmatrix *xmatrix_newwithtype(objecttype type, MatrixIdx_t nrows, MatrixIdx_t ncols, MatrixIdx_t nvals, bool zero) {
    MatrixCount_t nels = nrows*ncols*nvals;
    objectxmatrix *new = (objectxmatrix *) object_new(sizeof(objectxmatrix) + nels*sizeof(double), type);
    
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
objectxmatrix *xmatrix_new(MatrixIdx_t nrows, MatrixIdx_t ncols, bool zero) {
    return xmatrix_newwithtype(OBJECT_XMATRIX, nrows, ncols, 1, zero);
}

/** Clone a matrix */
objectxmatrix *xmatrix_clone(objectxmatrix *in) {
    objectxmatrix *new = xmatrix_newwithtype(in->obj.type, in->nrows, in->ncols, in->nvals, false);
    
    if (new) cblas_dcopy((__LAPACK_int) in->nels, in->elements, 1, new->elements, 1);
    return new;
}

/* ----------------------
 * Accessing elements
 * ---------------------- */

/** @brief Sets a matrix element.
    @returns true if the element is in the range of the matrix, false otherwise */
linalgError_t xmatrix_setelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return LINALGERR_INDX_OUT_OF_BNDS;
        
    matrix->elements[matrix->nvals*(col*matrix->nrows+row)]=value;
    return LINALGERR_OK;
}

/** @brief Gets a matrix element
 *  @returns true if the element is in the range of the matrix, false otherwise */
linalgError_t xmatrix_getelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double *value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return LINALGERR_INDX_OUT_OF_BNDS;
    
    if (value) *value=matrix->elements[matrix->nvals*(col*matrix->nrows+row)];
    return LINALGERR_OK;
}

/** @brief Gets a pointer to a matrix element
 *  @returns true if the element is in the range of the matrix, false otherwise */
linalgError_t xmatrix_getelementptr(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double **value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return LINALGERR_INDX_OUT_OF_BNDS;
    
    if (value) *value=matrix->elements+matrix->nvals*(col*matrix->nrows+row);
    return LINALGERR_OK;
}

/** Copies the column col of matrix a into the column vector b */
linalgError_t xmatrix_getcolumn(objectxmatrix *a, MatrixIdx_t col, objectxmatrix *b) {
    if (col<0 || col>=a->ncols) return LINALGERR_INDX_OUT_OF_BNDS;
    if (b->nels!=a->nrows*a->nvals) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_dcopy((__LAPACK_int) b->nels, a->elements+a->nvals*col*a->nrows, 1, b->elements, 1);
    return LINALGERR_OK;
}

/** Copies the column vector b into column col of matrix a */
linalgError_t xmatrix_setcolumn(objectxmatrix *a, MatrixIdx_t col, objectxmatrix *b) {
    if (col<0 || col>=a->ncols) return LINALGERR_INDX_OUT_OF_BNDS;
    if (b->nels!=a->nrows*a->nvals) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_dcopy((__LAPACK_int) b->nels, b->elements, 1, a->elements+a->nvals*col*a->nrows, 1);
    return LINALGERR_OK;
}

/* ----------------------
 * Arithmetic operations
 * ---------------------- */

/** Vector addition: Performs y <- alpha*x + y */
linalgError_t xmatrix_axpy(double alpha, objectxmatrix *x, objectxmatrix *y) {
    if (!(x->ncols==y->ncols && x->nrows==y->nrows)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_daxpy((__LAPACK_int) x->nels, alpha, x->elements, 1, y->elements, 1);
    return LINALGERR_OK;
}

/** Copies a matrix  y <- a */
linalgError_t xmatrix_copy(objectxmatrix *x, objectxmatrix *y) {
    if (!(x->ncols==y->ncols && x->nrows==y->nrows)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_dcopy((__LAPACK_int) x->nels, x->elements, 1, y->elements, 1);
    return LINALGERR_OK;
}

/** Scales a matrix x <- scale * x >*/
void xmatrix_scale(objectxmatrix *x, double scale) {
    cblas_dscal((__LAPACK_int) x->nels, scale, x->elements, 1);
}

/** Loads the identity matrix a <- I(n) */
linalgError_t xmatrix_identity(objectxmatrix *x) {
    if (x->ncols!=x->nrows) return LINALGERR_NOT_SQUARE;
    memset(x->elements, 0, sizeof(double)*x->nrows*x->ncols);
    for (int i=0; i<x->nrows; i++) x->elements[x->nvals*(i+x->nrows*i)]=1.0;
    return LINALGERR_OK;
}

/** Performs z <- alpha*(x*y) + beta*z */
linalgError_t xmatrix_mmul(double alpha, objectxmatrix *x, objectxmatrix *y, double beta, objectxmatrix *z) {
    if (!(x->ncols==y->nrows && x->nrows==z->nrows && y->ncols==z->ncols)) return LINALGERR_INCOMPATIBLE_DIM;
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, x->nrows, y->ncols, x->ncols, alpha, x->elements, x->nrows, y->elements, y->nrows, beta, z->elements, z->nrows);
    return LINALGERR_OK;
}

/** Performs x <- alpha*x + beta */
linalgError_t xmatrix_addscalar(objectxmatrix *x, double alpha, double beta) {
    for (MatrixCount_t i=0; i<x->ncols*x->nrows; i++) {
        for (int k=0; k<x->nvals; k++) {
            x->elements[i*x->nvals+k]*=alpha;
            if (k==0) x->elements[i*x->nvals+k]+=beta;
        }
    }
    return LINALGERR_OK;
}

/** Performs y <- x^T>*/
linalgError_t xmatrix_transpose(objectxmatrix *x, objectxmatrix *y) {
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

// TODO: Fix with correct norms!

/** Computes the Frobenius norm of a matrix */
double xmatrix_norm(objectxmatrix *a) {
    return cblas_dnrm2((__LAPACK_int) a->nels, a->elements, 1);
}

/** Computes the L1 norm of a matrix */
double xmatrix_l1norm(objectxmatrix *a) {
    return cblas_dasum((__LAPACK_int) a->nels, a->elements, 1);
}

/** Computes the Ln norm of a matrix */
double xmatrix_lnnorm(objectxmatrix *a, double n) {
    double sum=0.0, c=0.0, y,t;
    
    for (MatrixCount_t i=0; i<a->nels; i++) {
        y=pow(a->elements[i],n)-c; // Kahan summation
        t=sum+y;
        c=(t-sum)-y;
        sum=t;
    }
    return pow(sum,1.0/n);
}

/** Computes the infinity norm of a matrix */
double xmatrix_linfnorm(objectxmatrix *a) {
    int imax=cblas_idamax((__LAPACK_int) a->nels, a->elements, 1);
    return a->elements[imax];
}

/** Computes the sum of all elements in a matrix */
void xmatrix_sum(objectxmatrix *a, double *sum) {
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
linalgError_t xmatrix_trace(objectxmatrix *a, double *out) {
    if (a->nrows!=a->ncols) return LINALGERR_NOT_SQUARE;
    *out=1.0;
    *out=cblas_ddot(a->nrows, a->elements, a->ncols+1, out, 0);
    
    return LINALGERR_OK;
}

/* ----------------------
 * Binary operations
 * ---------------------- */

/** Finds the Frobenius inner product of two matrices  */
linalgError_t xmatrix_inner(objectxmatrix *x, objectxmatrix *y, double *out) {
    if (!(x->ncols==y->ncols && x->nrows==y->nrows)) return LINALGERR_INCOMPATIBLE_DIM;
    
    *out=cblas_ddot((__LAPACK_int) x->nels, x->elements, 1, y->elements, 1);
    return LINALGERR_OK;
}

/** Solve the linear system a.x = b using stack allocated memory for temporary */
linalgError_t xmatrix_solvesmall(objectxmatrix *a, objectxmatrix *b) {
    int pivot[a->nrows];
    double els[a->nels];
    objectxmatrix A = MORPHO_STATICXMATRIX(els, a->nrows, a->ncols);
    xmatrix_copy(a, &A);
    return (xmatrix_getinterface(a)->solvefn) (&A, b, pivot);
}

/** Solve the linear system a.x = b using heap allocated memory for temporary */
linalgError_t xmatrix_solvelarge(objectxmatrix *a, objectxmatrix *b) {
    int *pivot = MORPHO_MALLOC(sizeof(int)*a->nrows);
    objectxmatrix *A = xmatrix_clone(a);
    linalgError_t out = LINALGERR_ALLOC;
    if (pivot && A) {
        out = (xmatrix_getinterface(a)->solvefn) (A, b, pivot);
    }
    if (A) object_free((object *) A);
    if (pivot) MORPHO_FREE(pivot);
    return out;
}

/** Solve the linear system a.x = b; automatrically allocates storage depending on size of the matrix
 * @param[in]     a  lhs
 * @param[in|out]  b  rhs — overwritten by the solution
 * @returns linalgError_t indicating the status; MATRIX_OK indicates success. */
linalgError_t xmatrix_solve(objectxmatrix *a, objectxmatrix *b) {
    if (MATRIX_ISSMALL(a)) return xmatrix_solvesmall(a, b);
    else return xmatrix_solvelarge(a, b);
}

/** Inverts the matrix a
 * @param[in] a  matrix to be inverted
 * @returns linalgError_t indicating the status; MATRIX_OK indicates success. */
linalgError_t xmatrix_inverse(objectxmatrix *a) {
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
linalgError_t xmatrix_eigen(objectxmatrix *a, MorphoComplex *w, objectxmatrix *vec) {
    if (a->nrows!=a->ncols) return LINALGERR_NOT_SQUARE;
    if (vec && ((a->nrows!=vec->nrows) || (a->nrows!=vec->ncols))) return LINALGERR_INCOMPATIBLE_DIM;
    
    xmatrix_eigenfn_t efn = xmatrix_getinterface(a)->eigenfn;
    if (!efn) return LINALGERR_NOT_SUPPORTED;
    
    objectxmatrix *temp = xmatrix_clone(a);
    if (!temp) return LINALGERR_ALLOC;
        
    return efn(temp, w, vec);
}

/* ----------------------
 * Display
 * ---------------------- */

/** Prints a matrix */
void xmatrix_print(vm *v, objectxmatrix *m) {
    matrixinterfacedefn *interface=xmatrix_getinterface(m);
    double *elptr;
    for (MatrixIdx_t i=0; i<m->nrows; i++) { // Rows run from 0...m
        morpho_printf(v, "[ ");
        for (MatrixIdx_t j=0; j<m->ncols; j++) { // Columns run from 0...k
            xmatrix_getelementptr(m, i, j, &elptr);
            (*interface->printelfn) (v, elptr);
            morpho_printf(v, " ");
        }
        morpho_printf(v, "]%s", (i<m->nrows-1 ? "\n" : ""));
    }
}

/** Prints a matrix to a buffer */
bool xmatrix_printtobuffer(objectxmatrix *m, char *format, varray_char *out) {
    matrixinterfacedefn *interface=xmatrix_getinterface(m);
    double *elptr;
    for (MatrixIdx_t i=0; i<m->nrows; i++) { // Rows run from 0...m
        varray_charadd(out, "[ ", 2);
        
        for (MatrixIdx_t j=0; j<m->ncols; j++) { // Columns run from 0...k
            xmatrix_getelementptr(m, i, j, &elptr);
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
static void _rollflat(objectxmatrix *a, objectxmatrix *b, int nplaces) {
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
static void _copyrow(objectxmatrix *a, MatrixIdx_t arow, objectxmatrix *b, MatrixIdx_t brow) {
    for (MatrixIdx_t i=0; i<a->ncols; i++)
        memcpy(b->elements+b->nvals*(i*b->nrows+brow), a->elements+a->nvals*(i*a->nrows+arow), sizeof(double)*a->nvals);
}

/** Rolls a list by a number of elements along a given axis; stores the result in b */
linalgError_t xmatrix_roll(objectxmatrix *a, int nplaces, int axis, objectxmatrix *b) {
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
 * XMatrix constructors
 * ********************************************************************** */

value xmatrix_constructor__int_int(vm *v, int nargs, value *args) {
    MatrixIdx_t nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                ncols = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    
    objectxmatrix *new=xmatrix_new(nrows, ncols, true);
    return morpho_wrapandbind(v, (object *) new);
}

value xmatrix_constructor__list(vm *v, int nargs, value *args) {
    return MORPHO_NIL;
}

value xmatrix_constructor__err(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, MATRIX_CONSTRUCTOR);
    return MORPHO_NIL;
}

/** Creates an identity matrix */
value xmatrix_identityconstructor(vm *v, int nargs, value *args) {
    MatrixIdx_t n = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    
    objectxmatrix *new = xmatrix_new(n,n,false);
    if (new) xmatrix_identity(new);
    
    return morpho_wrapandbind(v, (object *) new);
}

/* **********************************************************************
 * XMatrix veneer class
 * ********************************************************************** */

/* ----------------------
 * Common utility methods
 * ---------------------- */

/** Prints a matrix */
value XMatrix_print(vm *v, int nargs, value *args) {
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    xmatrix_print(v, m);
    return MORPHO_NIL;
}

/** Formatted conversion to a string */
value XMatrix_format(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    varray_char str;
    varray_charinit(&str);
    
    if (xmatrix_printtobuffer(MORPHO_GETXMATRIX(MORPHO_SELF(args)),
                              MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)),
                             &str)) {
        out = object_stringfromvarraychar(&str);
        if (MORPHO_ISOBJECT(out)) morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    varray_charclear(&str);
    return out;
}

/** Copies the contents of one matrix into another */
value XMatrix_assign(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectxmatrix *b=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0));
    LINALG_ERRCHECKVM(xmatrix_copy(b, a));
    return MORPHO_NIL;
}

/** Clones a matrix */
value XMatrix_clone(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectxmatrix *new=xmatrix_clone(a);
    return morpho_wrapandbind(v, (object *) new);
}

/* ---------
 * index()
 * --------- */

static value _getindex(vm *v, objectxmatrix *m, MatrixIdx_t i, MatrixIdx_t j) {
    value out=MORPHO_NIL;
    
    double *elptr=NULL;
    LINALG_ERRCHECKVM(xmatrix_getelementptr(m, i, j, &elptr));

    if (elptr) out=xmatrix_getinterface(m)->getelfn(v, elptr);
    return out;
}

value XMatrix_index__int(vm *v, int nargs, value *args) {
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    return _getindex(v, MORPHO_GETXMATRIX(MORPHO_SELF(args)), i, 0);
}

value XMatrix_index__int_int(vm *v, int nargs, value *args) {
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                j = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    return _getindex(v, MORPHO_GETXMATRIX(MORPHO_SELF(args)), i, j);
}

/* ---------
 * setindex()
 * --------- */

static void _setindex(vm *v, objectxmatrix *m, MatrixIdx_t i, MatrixIdx_t j, value in) {
    double *elptr=NULL;
    LINALG_ERRCHECKVM(xmatrix_getelementptr(m, i, j, &elptr));
    if (elptr) LINALG_ERRCHECKVM(xmatrix_getinterface(m)->setelfn(v, in, elptr));
}

value XMatrix_setindex__int_x(vm *v, int nargs, value *args) {
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    _setindex(v, MORPHO_GETXMATRIX(MORPHO_SELF(args)), i, 0, MORPHO_GETARG(args, 1));
}

value XMatrix_setindex__int_int_x(vm *v, int nargs, value *args) {
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    MatrixIdx_t j = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    _setindex(v, MORPHO_GETXMATRIX(MORPHO_SELF(args)), i, j, MORPHO_GETARG(args, 2));
}

/* ---------
 * column
 * --------- */

value XMatrix_getcolumn__int(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    value out=MORPHO_NIL;
    
    if (i>=0 && i<a->ncols) {
        objectxmatrix *new=xmatrix_newwithtype(a->obj.type, a->nrows, 1, a->nvals, false);
        if (new) xmatrix_getcolumn(a, i, new);
        out=morpho_wrapandbind(v, (object *)new);
    } else linalg_raiseerror(v, LINALGERR_INDX_OUT_OF_BNDS);
    
    return out;
}

value XMatrix_setcolumn__int_xmatrix(vm *v, int nargs, value *args) {
    LINALG_ERRCHECKVM(xmatrix_setcolumn(MORPHO_GETXMATRIX(MORPHO_SELF(args)),
                                        MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                                        MORPHO_GETXMATRIX(MORPHO_GETARG(args, 1))));
    return MORPHO_NIL;
}

/* ----------
 * Arithmetic
 * ---------- */

/** Add a vector */
static value _axpy(vm *v, int nargs, value *args, double alpha) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectxmatrix *b=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0));
    objectxmatrix *new = NULL;
    value out=MORPHO_NIL;
    
    if (a->ncols==b->ncols && a->nrows==b->nrows) {
        new=xmatrix_clone(b);
        if (new) xmatrix_axpy(alpha, a, new); // TODO: Error check
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES);
    
    return out;
}

/** Add a scalar */
static value _xpa(vm *v, int nargs, value *args, double sgna, double sgnb) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    double beta;
    if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &beta)) {
        objectxmatrix *new = xmatrix_clone(a);
        if (new) xmatrix_addscalar(new, sgna, beta*sgnb);
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, LINALG_INVLDARGS);
    
    return out;
}

value XMatrix_add__xmatrix(vm *v, int nargs, value *args) {
    return _axpy(v,nargs,args,1.0);
}

value XMatrix_add__nil(vm *v, int nargs, value *args) {
    return MORPHO_SELF(args);
}

value XMatrix_add__x(vm *v, int nargs, value *args) {
    return _xpa(v,nargs,args,1.0,1.0);
}

value XMatrix_sub__xmatrix(vm *v, int nargs, value *args) {
    return _axpy(v,nargs,args,-1.0);
}

value XMatrix_sub__x(vm *v, int nargs, value *args) {
    return _xpa(v,nargs,args,1.0,-1.0);
}

value XMatrix_subr__x(vm *v, int nargs, value *args) {
    return _xpa(v,nargs,args,-1.0,1.0);
}

value XMatrix_mul__float(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    double scale;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale);
    
    objectxmatrix *new = xmatrix_clone(a);
    if (new) xmatrix_scale(new, scale);
    return morpho_wrapandbind(v, (object *) new);
}

value XMatrix_mul__xmatrix(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectxmatrix *b=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0));
    value out=MORPHO_NIL;
    
    if (a->ncols==b->nrows) {
        objectxmatrix *new = xmatrix_new(a->nrows, b->ncols, false);
        if (new) xmatrix_mmul(1.0, a, b, 0.0, new);
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES);
    return out;
}

value XMatrix_div__float(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    double scale;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale);
    scale = 1.0/scale;
    if (isnan(scale)) morpho_runtimeerror(v, VM_DVZR);
    
    objectxmatrix *new = xmatrix_clone(a);
    if (new) xmatrix_scale(new, scale);
    return morpho_wrapandbind(v, (object *) new);
}

value XMatrix_div__xmatrix(vm *v, int nargs, value *args) {
    objectxmatrix *b=MORPHO_GETXMATRIX(MORPHO_SELF(args)); // Note that the rhs is the receiver
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0)); // ... the lhs is the argument
    value out=MORPHO_NIL;
    
    objectxmatrix *sol = xmatrix_clone(b);
    out = morpho_wrapandbind(v, (object *) sol);
    if (sol) LINALG_ERRCHECKVM(xmatrix_solve(a, sol));
    
    return out;
}

/** Accumulate in place */
value XMatrix_acc__x_xmatrix(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectxmatrix *b=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 1));
    
    double alpha=1.0;
    if (!morpho_valuetofloat(MORPHO_GETARG(args, 0), &alpha)) morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    LINALG_ERRCHECKVM(xmatrix_axpy(alpha, b, a));
    return MORPHO_NIL;
}

/* ----------------
 * Unary operations
 * ---------------- */

/** Matrix norm */
value XMatrix_norm__x(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    value out = MORPHO_NIL;
    double n;
    
    if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &n)) {
        if (fabs(n-1.0)<MORPHO_EPS) {
            out=MORPHO_FLOAT(xmatrix_l1norm(a));
        } else if (fabs(n-2.0)<MORPHO_EPS) {
            out=MORPHO_FLOAT(xmatrix_norm(a));
        } else if (isinf(n)) {
            out=MORPHO_FLOAT(xmatrix_linfnorm(a));
        } else {
            out=MORPHO_FLOAT(xmatrix_lnnorm(a, n));
        }
    } else morpho_runtimeerror(v, MATRIX_NORMARGS);
    
    return out;
}

value XMatrix_norm(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    return MORPHO_FLOAT(xmatrix_norm(a));
}

/** Sums all matrix values */
value XMatrix_sum(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    double sum[a->nvals];
    
    xmatrix_sum(a, sum);
    return xmatrix_getinterface(a)->getelfn(v, sum);
}

/** Computes the trace */
value XMatrix_trace(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    double out=0.0;
    LINALG_ERRCHECKVM(xmatrix_trace(a, &out));
    return MORPHO_FLOAT(out);
}

/** Inverts a matrix */
value XMatrix_transpose(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    objectxmatrix *new = xmatrix_clone(a);
    if (new) {
        new->ncols=a->nrows;
        new->nrows=a->ncols;
        LINALG_ERRCHECKVM(xmatrix_transpose(a, new));
    }
    out = morpho_wrapandbind(v, (object *) new);
    
    return out;
}

/** Inverts a matrix */
value XMatrix_inverse(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    objectxmatrix *new = xmatrix_clone(a);
    out = morpho_wrapandbind(v, (object *) new);
    if (new) LINALG_ERRCHECKVM(xmatrix_inverse(new));
    
    return out;
}

static bool _processeigenvalues(vm *v, MatrixIdx_t n, MorphoComplex *w, value *out) {
    value ev[n];
    for (int i=0; i<n; i++) ev[i]=MORPHO_NIL;
    for (int i=0; i<n; i++) {
        if (fabs(cimag(w[i])) < MORPHO_EPS*cabs(w[i])) {
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
value XMatrix_eigenvalues(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    value out = MORPHO_NIL;
    
    MatrixIdx_t n=a->ncols;
    MorphoComplex w[n];
    linalgError_t err=xmatrix_eigen(a, w, NULL);
    if (err==LINALGERR_OK) {
        if (_processeigenvalues(v, n, w, &out)) {
            morpho_bindobjects(v, 1, &out); // TODO: Correctly bind subsidiary values
        } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    } else linalg_raiseerror(v, err);
    
    return out;
}

#define _CHK(x) if (!x) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto _eigensystem_cleanup; }

/** Finds the eigenvalues and eigenvectors of a matrix */
value XMatrix_eigensystem(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    
    value ev=MORPHO_NIL; // Will hold eigenvalues
    objectxmatrix *evec=NULL; // Holds eigenvectors
    objecttuple *otuple=NULL; // Tuple to return everything
    
    MatrixIdx_t n=a->ncols;
    MorphoComplex w[n];
    
    evec=xmatrix_clone(a);
    _CHK(evec);
    
    linalgError_t err=xmatrix_eigen(a, w, evec);
    if (err!=LINALGERR_OK) { linalg_raiseerror(v, err); goto _eigensystem_cleanup; }
    
    _CHK(_processeigenvalues(v, n, w, &ev));
    
    value outtuple[2] = { ev, MORPHO_OBJECT(evec) };
    otuple = object_newtuple(2, outtuple);
    _CHK(otuple);
    
    return morpho_wrapandbind(v, (object *) otuple); // TODO: Correctly bind subsidiary values
    
_eigensystem_cleanup:
    if (evec) object_free((object *) evec);
    if (otuple) object_free((object *) otuple);
    morpho_freeobject(ev); // TODO: Free contents? 
    
    return MORPHO_NIL;
}
#undef _CHK

/* ---------
 * Products
 * --------- */

/** Frobenius inner product */
value XMatrix_inner(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectxmatrix *b=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0));
    double prod=0.0;
    
    LINALG_ERRCHECKVM(xmatrix_inner(a, b, &prod));
    
    return MORPHO_FLOAT(prod);
}

/* ---------
 * Metadata
 * --------- */

/** Reshape a matrix */
value XMatrix_reshape(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    int nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
        ncols = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    
    if (nrows*ncols==a->nrows*a->ncols) {
        a->nrows=nrows;
        a->ncols=ncols;
    } else morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES);
    
    return MORPHO_NIL;
}

static value _roll(vm *v, objectxmatrix *a, int roll, int axis) {
    objectxmatrix *new = xmatrix_clone(a);
    if (new) xmatrix_roll(a, roll, axis, new);
    return morpho_wrapandbind(v, (object *) new);
}

/** Roll a matrix */
value XMatrix_roll__int_int(vm *v, int nargs, value *args) {
    objectxmatrix *a = MORPHO_GETXMATRIX(MORPHO_SELF(args));
    int roll = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
        axis = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    return _roll(v, a, roll, axis);
}

/** Roll a matrix by row */
value XMatrix_roll__int(vm *v, int nargs, value *args) {
    objectxmatrix *a = MORPHO_GETXMATRIX(MORPHO_SELF(args));
    int roll = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    return _roll(v, a, roll, 0);
}

/** Enumerate protocol */
value XMatrix_enumerate(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    value out=MORPHO_NIL;
    
    if (i<0) {
        out=MORPHO_INTEGER(a->ncols*a->nrows);
    } else if (i<a->ncols*a->nrows) {
        out=xmatrix_getinterface(a)->getelfn(v, a->elements+i*a->nvals);
    } else {
        linalg_raiseerror(v, LINALGERR_INDX_OUT_OF_BNDS);
    }
    
    return out;
}

/** Number of matrix elements */
value XMatrix_count(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    return MORPHO_INTEGER(a->ncols*a->nrows);
}

/** Matrix dimensions */
value XMatrix_dimensions(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    
    value dim[2] = { MORPHO_INTEGER(a->nrows), MORPHO_INTEGER(a->ncols) };
    objecttuple *new=object_newtuple(2, dim);
    
    return morpho_wrapandbind(v, (object *) new);
}


MORPHO_BEGINCLASS(XMatrix)
MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "()", XMatrix_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_FORMAT_METHOD, "(String)", XMatrix_format, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ASSIGN_METHOD, "(XMatrix)", XMatrix_assign, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, "XMatrix ()", XMatrix_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Float (Int)", XMatrix_index__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Float (Int, Int)", XMatrix_index__int_int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,_)", XMatrix_setindex__int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,Int,_)", XMatrix_setindex__int_int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_GETCOLUMN_METHOD, "XMatrix (Int)", XMatrix_getcolumn__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_SETCOLUMN_METHOD, "(Int, XMatrix)", XMatrix_setcolumn__int_xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "XMatrix (XMatrix)", XMatrix_add__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "XMatrix (Nil)", XMatrix_add__nil, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "XMatrix (_)", XMatrix_add__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "XMatrix (_)", XMatrix_add__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "XMatrix (Nil)", XMatrix_add__nil, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "XMatrix (XMatrix)", XMatrix_sub__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "XMatrix (Nil)", XMatrix_add__nil, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "XMatrix (_)", XMatrix_sub__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "XMatrix (_)", XMatrix_subr__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "XMatrix (Float)", XMatrix_mul__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "XMatrix (XMatrix)", XMatrix_mul__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "XMatrix (Float)", XMatrix_mul__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "XMatrix (XMatrix)", XMatrix_div__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "XMatrix (Float)", XMatrix_div__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ACC_METHOD, "(_, XMatrix)", XMatrix_acc__x_xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_INVERSE_METHOD, "XMatrix ()", XMatrix_inverse, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUM_METHOD, "Float ()", XMatrix_sum, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_TRACE_METHOD, "Float ()", XMatrix_trace, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_TRANSPOSE_METHOD, "XMatrix ()", XMatrix_transpose, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_INNER_METHOD, "Float (XMatrix)", XMatrix_inner, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_EIGENVALUES_METHOD, "Tuple ()", XMatrix_eigenvalues, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_EIGENSYSTEM_METHOD, "Tuple ()", XMatrix_eigensystem, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_NORM_METHOD, "Float (_)", XMatrix_norm__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_NORM_METHOD, "Float ()", XMatrix_norm, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_RESHAPE_METHOD, "(Int,Int)", XMatrix_reshape, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_ROLL_METHOD, "XMatrix (Int)", XMatrix_roll__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_ROLL_METHOD, "XMatrix (Int,Int)", XMatrix_roll__int_int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ENUMERATE_METHOD, "(Int)", XMatrix_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int ()", XMatrix_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_DIMENSIONS_METHOD, "Tuple ()", XMatrix_dimensions, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void xmatrix_initialize(void) {
    objectxmatrixtype=object_addtype(&objectxmatrixdefn);
    xmatrix_addinterface(&xmatrixdefn);
    
    value xmatrixclass=builtin_addclass(XMATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(XMatrix), MORPHO_NIL);
    object_setveneerclass(OBJECT_XMATRIX, xmatrixclass);
    
    morpho_addfunction(XMATRIX_CLASSNAME, "XMatrix (Int, Int)", xmatrix_constructor__int_int, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(XMATRIX_CLASSNAME, "XMatrix (List)", xmatrix_constructor__list, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(XMATRIX_CLASSNAME, "(...)", xmatrix_constructor__err, MORPHO_FN_CONSTRUCTOR, NULL);
    
    morpho_addfunction(XMATRIX_IDENTITYCONSTRUCTOR, "XMatrix (Int)", xmatrix_identityconstructor, MORPHO_FN_CONSTRUCTOR, NULL);
    
    complexmatrix_initialize();
}
