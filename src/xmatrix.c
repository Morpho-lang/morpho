/** @file xmatrix.c
 *  @author T J Atherton
 *
 *  @brief New matrices
*/

#define ACCELERATE_NEW_LAPACK
#define MORPHO_INCLUDE_LINALG

#include "newlinalg.h"
#include "xmatrix.h"
#include "xcomplexmatrix.h"

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
bool xmatrix_setelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
        
    matrix->elements[matrix->nvals*(col*matrix->nrows+row)]=value;
    return true;
}

/** @brief Gets a matrix element
 *  @returns true if the element is in the range of the matrix, false otherwise */
bool xmatrix_getelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double *value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
    
    if (value) *value=matrix->elements[matrix->nvals*(col*matrix->nrows+row)];
    return true;
}

/** @brief Gets a pointer to a matrix element
 *  @returns true if the element is in the range of the matrix, false otherwise */
bool xmatrix_getelementptr(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double **value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
    
    if (value) *value=matrix->elements+matrix->nvals*(col*matrix->nrows+row);
    return true;
}

/* ----------------------
 * Arithmetic operations
 * ---------------------- */

/** Vector addition: Performs y <- alpha*x + y */
objectmatrixerror xmatrix_axpy(double alpha, objectxmatrix *x, objectxmatrix *y) {
    if (x->ncols==y->ncols && x->nrows==y->nrows) {
        cblas_daxpy((__LAPACK_int) x->nels, alpha, x->elements, 1, y->elements, 1);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Copies a matrix  y <- a */
objectmatrixerror xmatrix_copy(objectxmatrix *x, objectxmatrix *y) {
    if (x->ncols==y->ncols && x->nrows==y->nrows) {
        cblas_dcopy((__LAPACK_int) x->nels, x->elements, 1, y->elements, 1);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Scales a matrix x <- scale * x >*/
objectmatrixerror xmatrix_scale(objectxmatrix *x, double scale) {
    cblas_dscal((__LAPACK_int) x->nels, scale, x->elements, 1);
    return MATRIX_OK;
}

/** Loads the identity matrix a <- I(n) */
objectmatrixerror xmatrix_identity(objectxmatrix *x) {
    if (x->ncols!=x->nrows) return MATRIX_NSQ;
    memset(x->elements, 0, sizeof(double)*x->nrows*x->ncols);
    for (int i=0; i<x->nrows; i++) x->elements[x->nvals*(i+x->nrows*i)]=1.0;
    return MATRIX_OK;
}

/** Performs z <- alpha*(x*y) + beta*z */
objectmatrixerror xmatrix_mmul(double alpha, objectxmatrix *x, objectxmatrix *y, double beta, objectxmatrix *z) {
    if (x->ncols==y->nrows && x->nrows==z->nrows && y->ncols==z->ncols) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, x->nrows, y->ncols, x->ncols, alpha, x->elements, x->nrows, y->elements, y->nrows, beta, z->elements, z->nrows);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Finds the Frobenius inner product of two matrices  */
objectmatrixerror xmatrix_inner(objectxmatrix *x, objectxmatrix *y, double *out) {
    if (x->ncols==y->ncols && x->nrows==y->nrows) {
        *out=cblas_ddot((__LAPACK_int) x->nels, x->elements, 1, y->elements, 1);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Solve the linear system a.x = b;
 * @param[in|out] a  lhs — overwritten by the LU decomposition
 * @param[in|out] b  rhs — overwritten by the solution
 * @param[out] pivot - you must provide an array with the same number of rows as a.
 * @returns objectmatrixerror indicating the status; MATRIX_OK indicates success. */
static objectmatrixerror _solve(objectxmatrix *a, objectxmatrix *b, int *pivot) {
    int n=a->nrows, nrhs = b->ncols, info;
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, x->elements, n, pivot, y->elements, n);
#else
    dgesv_(&n, &nrhs, a->elements, &n, pivot, b->elements, &n, &info);
#endif
    
    return (info==0 ? MATRIX_OK : (info>0 ? MATRIX_SING : MATRIX_INVLD));
}

/** Solve the linear system a.x = b using stack allocated memory for temporary */
objectmatrixerror xmatrix_solvesmall(objectxmatrix *a, objectxmatrix *b) {
    int pivot[a->nrows];
    double els[a->nels];
    objectxmatrix A = MORPHO_STATICXMATRIX(els, a->nrows, a->ncols);
    xmatrix_copy(a, &A);
    return _solve(&A, b, pivot);
}

/** Solve the linear system a.x = b using heap allocated memory for temporary */
objectmatrixerror xmatrix_solvelarge(objectxmatrix *a, objectxmatrix *b) {
    int *pivot = MORPHO_MALLOC(sizeof(double)*a->nrows);
    objectxmatrix *A = xmatrix_clone(a);
    objectmatrixerror out = MATRIX_ALLOC;
    if (pivot && A) {
        out = _solve(A, b, pivot);
    }
    if (A) object_free((object *) A);
    if (pivot) MORPHO_FREE(pivot);
    return out;
}

objectmatrixerror xmatrix_solve(objectxmatrix *a, objectxmatrix *b) {
    if (MATRIX_ISSMALL(a)) return xmatrix_solvesmall(a, b);
    else return xmatrix_solvelarge(a, b);
}

/* ----------------------
 * Display
 * ---------------------- */

/** Prints a matrix */
void xmatrix_print(vm *v, objectxmatrix *m, xmatrix_elprintfn fn) {
    for (MatrixIdx_t i=0; i<m->nrows; i++) { // Rows run from 0...m
        morpho_printf(v, "[ ");
        for (MatrixIdx_t j=0; j<m->ncols; j++) { // Columns run from 0...k
            (*fn) (v, m, i, j);
            morpho_printf(v, " ");
        }
        morpho_printf(v, "]%s", (i<m->nrows-1 ? "\n" : ""));
    }
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
    xmatrix_print(v, m, _printelfn);
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
        if (new) xmatrix_axpy(alpha, a, new);
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    
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
    } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
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
        xmatrix_solve(a, sol); // TODO: Check for errors
        out = morpho_wrapandbind(v, (object *) sol);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

/* ---------
 * Products
 * --------- */

/** Frobenius inner product */
value XMatrix_inner(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectxmatrix *b=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0));
    double prod=0.0;
    
    if (xmatrix_inner(a, b, &prod)!=MATRIX_OK) {
        morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    }
    
    return MORPHO_FLOAT(prod);
}

/* ---------
 * index()
 * --------- */

static value _getindex(vm *v, objectxmatrix *m, MatrixIdx_t i, MatrixIdx_t j) {
    double out;
    if (xmatrix_getelement(m, i, j, &out)) return MORPHO_FLOAT(out);
    //morpho_runtimeerror(v, XMATRIX_INDICESOUTSIDEBOUNDS);
    return MORPHO_NIL;
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
    if (!morpho_valuetofloat(in, &val)) true; // Should raise an error (Matrix doesn't!)
    if (!xmatrix_setelement(m, i, j, val)) true; //morpho_runtimeerror(v, XMATRIX_INDICESOUTSIDEBOUNDS);
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
 * Metadata
 * --------- */

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
MORPHO_METHOD_SIGNATURE(XMATRIX_DIMENSIONS_METHOD, "Tuple ()", XMatrix_dimensions, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int ()", XMatrix_count, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void xmatrix_initialize(void) {
    objectxmatrixtype=object_addtype(&objectxmatrixdefn);
    
    value xmatrixclass=builtin_addclass(XMATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(XMatrix), MORPHO_NIL);
    object_setveneerclass(OBJECT_XMATRIX, xmatrixclass);
    
    morpho_addfunction(XMATRIX_CLASSNAME, "XMatrix (Int, Int)", xmatrix_constructor__int_int, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(XMATRIX_CLASSNAME, "XMatrix (List)", xmatrix_constructor__list, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(XMATRIX_CLASSNAME, "(...)", xmatrix_constructor__err, MORPHO_FN_CONSTRUCTOR, NULL);
    
    morpho_addfunction(XMATRIX_IDENTITYCONSTRUCTOR, "XMatrix (Int)", xmatrix_identityconstructor, MORPHO_FN_CONSTRUCTOR, NULL);
    
    complexmatrix_initialize();
}
