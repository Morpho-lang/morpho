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
    morpho_printf(v, "<" XMATRIX_CLASSNAME ">");
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
 * Constructors
 * ---------------------- */

/** Create a new matrix */
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

objectxmatrix *xmatrix_new(MatrixIdx_t nrows, MatrixIdx_t ncols, bool zero) {
    return xmatrix_newwithtype(OBJECT_XMATRIX, nrows, ncols, 1, zero);
}

/** Clone a matrix */
objectxmatrix *xmatrix_clone(objectxmatrix *in) {
    objectxmatrix *new = xmatrix_new(in->nrows, in->ncols, false);
    
    if (new) cblas_dcopy(in->ncols * in->nrows, in->elements, 1, new->elements, 1);
    return new;
}

/* ----------------------
 * Accessing elements
 * ---------------------- */

/** @brief Sets a matrix element.
    @returns true if the element is in the range of the matrix, false otherwise */
bool xmatrix_setelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
        
    matrix->elements[col*matrix->nrows+row]=value;
    return true;
}

/** @brief Gets a matrix element
 *  @returns true if the element is in the range of the matrix, false otherwise */
bool xmatrix_getelement(objectxmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, double *value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
    
    if (value) *value=matrix->elements[col*matrix->nrows+row];
    return true;
}

/* ----------------------
 * Arithmetic operations
 * ---------------------- */

/** Performs out <- alpha*x + y */
objectmatrixerror xmatrix_axpy(double alpha, objectxmatrix *x, objectxmatrix *y, objectxmatrix *out) {
    if (x->ncols==y->ncols && y->ncols==out->ncols &&
        x->nrows==y->nrows && y->nrows==out->nrows) {
        if (x!=out) cblas_dcopy(x->ncols * x->nrows, x->elements, 1, out->elements, 1);
        cblas_daxpy(x->ncols * x->nrows, alpha, y->elements, 1, out->elements, 1);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/* ----------------------
 * Display
 * ---------------------- */

/** Prints a matrix */
void xmatrix_print(vm *v, objectxmatrix *m) {
    for (MatrixIdx_t i=0; i<m->nrows; i++) { // Rows run from 0...m
        morpho_printf(v, "[ ");
        for (MatrixIdx_t j=0; j<m->ncols; j++) { // Columns run from 0...k
            double val=0.0;
            xmatrix_getelement(m, i, j, &val);
            morpho_printf(v, "%g ", (fabs(val)<MORPHO_EPS ? 0 : val));
        }
        morpho_printf(v, "]%s", (i<m->nrows-1 ? "\n" : ""));
    }
}

/* **********************************************************************
 * XMatrix constructors
 * ********************************************************************** */

value xmatrix_constructor__int_int(vm *v, int nargs, value *args) {
    int nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    int ncols = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    
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

/** Clones a matrix */
value XMatrix_clone(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectxmatrix *new=xmatrix_clone(a);
    if (new) {
        out = morpho_wrapandbind(v, (object *) new);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    return out;
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
        new=xmatrix_new(a->nrows, a->ncols, false);
        if (new) {
            xmatrix_axpy(alpha, a, b, new);
            out = morpho_wrapandbind(v, (object *) new);
        }
    } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    
    return out;
}

value XMatrix_add__xmatrix(vm *v, int nargs, value *args) {
    return _axpy(v,nargs,args,1.0);
}

value XMatrix_sub__xmatrix(vm *v, int nargs, value *args) {
    return _axpy(v,nargs,args,-1.0);
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
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    return _getindex(v, m, i, 0);
}

value XMatrix_index__int_int(vm *v, int nargs, value *args) {
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    MatrixIdx_t j = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    return _getindex(v, m, i, j);
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
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    return _setindex(v, m, i, 0, MORPHO_GETARG(args, 1));
}

value XMatrix_setindex__int_int_x(vm *v, int nargs, value *args) {
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    MatrixIdx_t j = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    return _setindex(v, m, i, j, MORPHO_GETARG(args, 2));
}

MORPHO_BEGINCLASS(XMatrix)
MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "()", XMatrix_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, "XMatrix ()", XMatrix_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "(XMatrix)", XMatrix_add__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "(XMatrix)", XMatrix_sub__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Float (Int)", XMatrix_index__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Float (Int, Int)", XMatrix_index__int_int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,_)", XMatrix_setindex__int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,Int,_)", XMatrix_setindex__int_int_x, BUILTIN_FLAGSEMPTY)
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
    
    complexmatrix_initialize();
}
