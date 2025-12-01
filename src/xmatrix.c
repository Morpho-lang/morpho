/** @file xmatrix.c
 *  @author T J Atherton
 *
 *  @brief New matrices
*/

#define MORPHO_INCLUDE_LINALG

#include "newlinalg.h"
#include "xmatrix.h" 

/* **********************************************************************
 * XMatrix objects
 * ********************************************************************** */

objecttype objectxmatrixtype;

/** Matrix object definitions */
size_t objectxmatrix_sizefn(object *obj) {
    return sizeof(objectxmatrix)+sizeof(double) *
            ((objectxmatrix *) obj)->nels;
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

objectxmatrix *xmatrix_new(int nrows, int ncols, bool zero) {
    int nels = nrows*ncols;
    objectxmatrix *new = (objectxmatrix *) object_new(sizeof(objectxmatrix) + nels*sizeof(double), OBJECT_XMATRIX);
    
    if (new) {
        new->nrows=nrows;
        new->ncols=ncols;
        new->nels=nels;
        new->elements=new->matrixdata;
        if (zero) memset(new->elements, 0, nels*sizeof(double));
    }
    
    return new;
}

/* ----------------------
 * Accessing elements
 * ---------------------- */

/** @brief Sets a matrix element.
    @returns true if the element is in the range of the matrix, false otherwise */
bool xmatrix_setelement(objectxmatrix *matrix, unsigned int row, unsigned int col, double value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
        
    matrix->elements[col*matrix->nrows+row]=value;
    return true;
}

/** @brief Gets a matrix element
 *  @returns true if the element is in the range of the matrix, false otherwise */
bool xmatrix_getelement(objectxmatrix *matrix, unsigned int row, unsigned int col, double *value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
    
    if (value) *value=matrix->elements[col*matrix->nrows+row];
    return true;
}

/* ----------------------
 * Arithmetic operations
 * ---------------------- */

/** Performs a + b -> out. */
objectmatrixerror xmatrix_add(objectxmatrix *a, objectxmatrix *b, objectxmatrix *out) {
    if (a->ncols==b->ncols && a->ncols==out->ncols &&
        a->nrows==b->nrows && a->nrows==out->nrows) {
        if (a!=out) cblas_dcopy(a->ncols * a->nrows, a->elements, 1, out->elements, 1);
        cblas_daxpy(a->ncols * a->nrows, 1.0, b->elements, 1, out->elements, 1);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Performs lambda*a + beta -> out. */
objectmatrixerror xmatrix_addscalar(objectxmatrix *a, double lambda, double beta, objectxmatrix *out) {
    if (a->ncols==out->ncols && a->nrows==out->nrows) {
        for (unsigned int i=0; i<out->nrows*out->ncols; i++) {
            out->elements[i]=lambda*a->elements[i]+beta;
        }
        return MATRIX_OK;
    }

    return MATRIX_INCMPTBLDIM;
}

/* ----------------------
 * Display
 * ---------------------- */

/** Prints a matrix */
void xmatrix_print(vm *v, objectxmatrix *m) {
    for (int i=0; i<m->nrows; i++) { // Rows run from 0...m
        morpho_printf(v, "[ ");
        for (int j=0; j<m->ncols; j++) { // Columns run from 0...k
            double val=0.0;
            xmatrix_getelement(m, i, j, &val);
            morpho_printf(v, "%g ", (fabs(val)<MORPHO_EPS ? 0 : val));
        }
        morpho_printf(v, "]%s", (i<m->nrows-1 ? "\n" : ""));
    }
}

/* **********************************************************************
 * XMatrix constructor
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

/** Prints a matrix */
value XMatrix_print(vm *v, int nargs, value *args) {
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    xmatrix_print(v, m);
    return MORPHO_NIL;
}

/* ---------
 * add()
 * --------- */

value XMatrix_add__xmatrix(vm *v, int nargs, value *args) {
    objectxmatrix *a=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    objectxmatrix *b=MORPHO_GETXMATRIX(MORPHO_GETARG(args, 0));
    objectxmatrix *new = NULL;
    value out=MORPHO_NIL;
    
    if (a->ncols==b->ncols && a->nrows==b->nrows) {
        new=xmatrix_new(a->nrows, a->ncols, false);
        if (new) {
            xmatrix_add(a, b, new);
            out = morpho_wrapandbind(v, (object *) new);
        }
    } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    
    return out;
}

/* ---------
 * index()
 * --------- */

value _getindex(vm *v, objectxmatrix *m, unsigned int i, unsigned int j) {
    double out;
    if (xmatrix_getelement(m, i, j, &out)) return MORPHO_FLOAT(out);
    //morpho_runtimeerror(v, XMATRIX_INDICESOUTSIDEBOUNDS);
    return MORPHO_NIL;
}

value XMatrix_index__int(vm *v, int nargs, value *args) {
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    unsigned int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    return _getindex(v, m, i, 0);
}

value XMatrix_index__int_int(vm *v, int nargs, value *args) {
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    unsigned int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    unsigned int j = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    return _getindex(v, m, i, j);
}

/* ---------
 * setindex()
 * --------- */

value _setindex(vm *v, objectxmatrix *m, unsigned int i, unsigned int j, value in) {
    double val=0.0;
    if (!morpho_valuetofloat(in, &val)) true; // Should raise an error (Matrix doesn't!)
    if (!xmatrix_setelement(m, i, j, val)) true; //morpho_runtimeerror(v, XMATRIX_INDICESOUTSIDEBOUNDS);
    return MORPHO_NIL;
}

value XMatrix_setindex__int_x(vm *v, int nargs, value *args) {
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    unsigned int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    return _setindex(v, m, i, 0, MORPHO_GETARG(args, 1));
}

value XMatrix_setindex__int_int_x(vm *v, int nargs, value *args) {
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    unsigned int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    unsigned int j = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    return _setindex(v, m, i, j, MORPHO_GETARG(args, 2));
}

MORPHO_BEGINCLASS(XMatrix)
MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "()", XMatrix_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "(XMatrix)", XMatrix_add__xmatrix, BUILTIN_FLAGSEMPTY),
//MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "(_)", XMatrix_add__x, BUILTIN_FLAGSEMPTY),
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
}

