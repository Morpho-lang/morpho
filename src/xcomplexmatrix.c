/** @file xcomplexmatrix.c
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#include "xcomplexmatrix.h"
#include <platform.h>

objecttype objectcomplexmatrixtype;
#define OBJECT_COMPLEXMATRIX objectcomplexmatrixtype

typedef objectxmatrix objectcomplexmatrix;

/* **********************************************************************
 * ComplexMatrix utility functions
 * ********************************************************************** */

/* ----------------------
 * Callbacks
 * ---------------------- */

static void _printelfn(vm *v, objectxmatrix *m, MatrixIdx_t i, MatrixIdx_t j) {
    double *elptr;
    xmatrix_getelementptr(m, i, j, &elptr);
    objectcomplex cmplx = MORPHO_STATICCOMPLEX(elptr[0], elptr[1]);
    complex_print(v, &cmplx);
}

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
bool complexmatrix_setelement(objectcomplexmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, MorphoComplex value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
        
    MatrixCount_t ix = matrix->nvals*(col*matrix->nrows+row);
    matrix->elements[ix]=creal(value);
    matrix->elements[ix+1]=cimag(value);
    return true;
}

/** Gets a matrix element */
bool complexmatrix_getelement(objectcomplexmatrix *matrix, MatrixIdx_t row, MatrixIdx_t col, MorphoComplex *value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
    
    MatrixCount_t ix = 2*(col*matrix->nrows+row);
    if (value) *value=MCBuild(matrix->elements[ix],matrix->elements[ix+1]);
    return true;
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

/* **********************************************************************
 * ComplexMatrix veneer class
 * ********************************************************************** */

/* ----------------------
 * Common utility methods
 * ---------------------- */

/** Prints a matrix */
value ComplexMatrix_print(vm *v, int nargs, value *args) {
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    xmatrix_print(v, m, _printelfn);
    return MORPHO_NIL;
}

/* ---------
 * index()
 * --------- */

static value _getindex(vm *v, objectxmatrix *m, MatrixIdx_t i, MatrixIdx_t j) {
    double *el;
    xmatrix_getelementptr(m, i, j, &el); //morpho_runtimeerror(v, XMATRIX_INDICESOUTSIDEBOUNDS);
    objectcomplex *new = object_newcomplex(el[0], el[1]);
    return morpho_wrapandbind(v, (object *) new);
    return MORPHO_NIL;
}

value ComplexMatrix_index__int(vm *v, int nargs, value *args) {
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    return _getindex(v, m, i, 0);
}

value ComplexMatrix_index__int_int(vm *v, int nargs, value *args) {
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    unsigned int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    unsigned int j = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    return _getindex(v, m, i, j);
}

/* ---------
 * setIndex()
 * --------- */

static value _setindex(vm *v, objectcomplexmatrix *m, MatrixIdx_t i, MatrixIdx_t j, value in) {
    if (MORPHO_ISCOMPLEX(in) &&
        !complexmatrix_setelement(m, i, j, MORPHO_GETCOMPLEX(in)->Z)) {
        // Should raise an error
    }
    return MORPHO_NIL;
}

value ComplexMatrix_setindex__int_x(vm *v, int nargs, value *args) {
    objectcomplexmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    return _setindex(v, m, i, 0, MORPHO_GETARG(args, 1));
}

value ComplexMatrix_setindex__int_int_x(vm *v, int nargs, value *args) {
    objectcomplexmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    MatrixIdx_t i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    MatrixIdx_t j = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    return _setindex(v, m, i, j, MORPHO_GETARG(args, 2));
}

MORPHO_BEGINCLASS(ComplexMatrix)
MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "()", ComplexMatrix_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Complex (Int)", ComplexMatrix_index__int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Complex (Int, Int)", ComplexMatrix_index__int_int, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int, Complex)", ComplexMatrix_setindex__int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int,Int, Complex)", ComplexMatrix_setindex__int_int_x, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "ComplexMatrix (ComplexMatrix)", XMatrix_add__xmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "ComplexMatrix (ComplexMatrix)", XMatrix_sub__xmatrix, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void complexmatrix_initialize(void) {
    objectcomplexmatrixtype=object_addtype(&objectxmatrixdefn);
    
    value complexmatrixclass=builtin_addclass(COMPLEXMATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(ComplexMatrix), MORPHO_NIL);
    object_setveneerclass(OBJECT_COMPLEXMATRIX, complexmatrixclass);
    
    morpho_addfunction(COMPLEXMATRIX_CLASSNAME, "ComplexMatrix (Int, Int)", complexmatrix_constructor__int_int, MORPHO_FN_CONSTRUCTOR, NULL);
}
