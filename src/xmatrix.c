/** @file xmatrix.c
 *  @author T J Atherton
 *
 *  @brief New matrices
*/

#define ACCELERATE_NEW_LAPACK
#define MORPHO_INCLUDE_LINALG

#include "newlinalg.h"

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
 * XMatrix callbacks
 * ---------------------- */

static void _printelfn(vm *v, objectxmatrix *m, MatrixIdx_t i, MatrixIdx_t j) {
    double val;
    xmatrix_getelement(m, i, j, &val);
    morpho_printf(v, "%g", (fabs(val)<MORPHO_EPS ? 0 : val));
}

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

/* ----------------------
 * Products
 * ---------------------- */

/** Finds the Frobenius inner product of two matrices  */
linalgError_t xmatrix_inner(objectxmatrix *x, objectxmatrix *y, double *out) {
    if (!(x->ncols==y->ncols && x->nrows==y->nrows)) return LINALGERR_INCOMPATIBLE_DIM;
    
    *out=cblas_ddot((__LAPACK_int) x->nels, x->elements, 1, y->elements, 1);
    return LINALGERR_OK;
}

/** Low level solve for linear system a.x = b
 * @param[in|out] a - lhs; overwritten by LU decomposition
 * @param[in|out] b - rhs; overwritten by solution
 * @param[out] pivot - you must provide an array with the same number of rows as a.
 * @returns a matrix error code */
static linalgError_t _solve(objectxmatrix *a, objectxmatrix *b, int *pivot) {
    int n=a->nrows, nrhs = b->ncols, info;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, x->elements, n, pivot, y->elements, n);
#else
    dgesv_(&n, &nrhs, a->elements, &n, pivot, b->elements, &n, &info);
#endif
    
    return (info==0 ? LINALGERR_OK : (info>0 ? LINALGERR_MATRIX_SINGULAR : LINALGERR_LAPACK_INVLD_ARGS));
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
 * @returns objectmatrixerror indicating the status; MATRIX_OK indicates success. */
linalgError_t xmatrix_solve(objectxmatrix *a, objectxmatrix *b) {
    if (MATRIX_ISSMALL(a)) return xmatrix_solvesmall(a, b);
    else return xmatrix_solvelarge(a, b);
}

/* ----------------------
 * Display
 * ---------------------- */

/** Prints a matrix */
void xmatrix_print(vm *v, objectxmatrix *m) {
    matrixinterfacedefn *interface=xmatrix_getinterface(m);
    for (MatrixIdx_t i=0; i<m->nrows; i++) { // Rows run from 0...m
        morpho_printf(v, "[ ");
        for (MatrixIdx_t j=0; j<m->ncols; j++) { // Columns run from 0...k
            (*interface->printelfn) (v, m, i, j);
            morpho_printf(v, " ");
        }
        morpho_printf(v, "]%s", (i<m->nrows-1 ? "\n" : ""));
    }
}

/* **********************************************************************
 * Interface definition
 * ********************************************************************** */

matrixinterfacedefn xmatrixdefn = {
    .printelfn = _printelfn,
    .solvefn = _solve
};

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

/* ----------
 * Arithmetic
 * ---------- */

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

value XMatrix_add__xmatrix(vm *v, int nargs, value *args) {
    return _axpy(v,nargs,args,1.0);
}

value XMatrix_sub__xmatrix(vm *v, int nargs, value *args) {
    return _axpy(v,nargs,args,-1.0);
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
    if (sol) {
        xmatrix_solve(a, sol); // TODO: Error check
        out = morpho_wrapandbind(v, (object *) sol);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
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
 * index()
 * --------- */

static value _getindex(vm *v, objectxmatrix *m, MatrixIdx_t i, MatrixIdx_t j) {
    double out=0.0;
    
    LINALG_ERRCHECKVM(xmatrix_getelement(m, i, j, &out));
    return MORPHO_FLOAT(out);
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

value _setindex(vm *v, objectxmatrix *m, MatrixIdx_t i, MatrixIdx_t j, value in) {
    double val=0.0;
    if (!morpho_valuetofloat(in, &val)) true; // TODO: Should raise an error (Matrix doesn't!)
    LINALG_ERRCHECKVM(xmatrix_setelement(m, i, j, val));
    return MORPHO_NIL;
}

value XMatrix_setindex__int_x(vm *v, int nargs, value *args) {
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    return _setindex(v, MORPHO_GETXMATRIX(MORPHO_SELF(args)), i, 0, MORPHO_GETARG(args, 1));
}

value XMatrix_setindex__int_int_x(vm *v, int nargs, value *args) {
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    MatrixIdx_t j = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    return _setindex(v, MORPHO_GETXMATRIX(MORPHO_SELF(args)), i, j, MORPHO_GETARG(args, 2));
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

/* ---------
 * Metadata
 * --------- */

/** Reshape a matrix */
value XMatrix_reshape(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    int nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
        ncols = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    
    if (nrows*ncols==a->nrows*a->ncols) {
        a->nrows=nrows;
        a->ncols=ncols;
    } else morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES);
    
    return MORPHO_NIL;
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
MORPHO_METHOD_SIGNATURE(MORPHO_ASSIGN_METHOD, "(XMatrix)", XMatrix_assign, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, "XMatrix ()", XMatrix_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "XMatrix (XMatrix)", XMatrix_add__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "XMatrix (XMatrix)", XMatrix_sub__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "XMatrix (Float)", XMatrix_mul__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "XMatrix (XMatrix)", XMatrix_mul__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "XMatrix (Float)", XMatrix_mul__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "XMatrix (XMatrix)", XMatrix_div__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "XMatrix (Float)", XMatrix_div__float, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_INNER_METHOD, "Float (XMatrix)", XMatrix_inner, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Float (Int)", XMatrix_index__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Float (Int, Int)", XMatrix_index__int_int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,_)", XMatrix_setindex__int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,Int,_)", XMatrix_setindex__int_int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_GETCOLUMN_METHOD, "XMatrix (Int)", XMatrix_getcolumn__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_SETCOLUMN_METHOD, "(Int, XMatrix)", XMatrix_setcolumn__int_xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_NORM_METHOD, "(_)", XMatrix_norm__x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_NORM_METHOD, "()", XMatrix_norm, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_RESHAPE_METHOD, "(Int,Int)", XMatrix_reshape, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(XMATRIX_DIMENSIONS_METHOD, "Tuple ()", XMatrix_dimensions, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int ()", XMatrix_count, BUILTIN_FLAGSEMPTY)
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
