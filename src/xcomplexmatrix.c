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

/** Sets a matrix element. */
bool complexmatrix_setelement(objectcomplexmatrix *matrix, unsigned int row, unsigned int col, MorphoComplex value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
        
    int ix = 2*(col*matrix->nrows+row);
    matrix->elements[ix]=creal(value);
    matrix->elements[ix+1]=cimag(value);
    return true;
}

/** Gets a matrix element */
bool complexmatrix_getelement(objectxmatrix *matrix, unsigned int row, unsigned int col, MorphoComplex *value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
    
    int ix = 2*(col*matrix->nrows+row);
    if (value) *value=MCBuild(0.0,0.0);
        //matrix->elements[col*matrix->nrows+row];
    return true;
}

/* **********************************************************************
 * ComplexMatrix constructors
 * ********************************************************************** */

value complexmatrix_constructor__int_int(vm *v, int nargs, value *args) {
    int nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    int ncols = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    
    objectcomplexmatrix *new=NULL; //xmatrix_new(nrows, ncols, true);
    
    return morpho_wrapandbind(v, (object *) new);
}

/* **********************************************************************
 * ComplexMatrix veneer class
 * ********************************************************************** */

value ComplexMatrix_index__int_int(vm *v, int nargs, value *args) {
    objectxmatrix *m=MORPHO_GETXMATRIX(MORPHO_SELF(args));
    unsigned int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    unsigned int j = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    //return _getindex(v, m, i, j);
    return MORPHO_NIL; 
}

MORPHO_BEGINCLASS(ComplexMatrix)
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Complex (Int, Int)", ComplexMatrix_index__int_int, BUILTIN_FLAGSEMPTY)
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
