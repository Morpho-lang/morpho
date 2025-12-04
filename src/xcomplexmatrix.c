/** @file xcomplexmatrix.c
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#include "xcomplexmatrix.h"
#include <platform.h>

objecttype objectcomplexmatrixtype;

typedef objectxmatrix objectcomplexmatrix;

/** Sets a matrix element. */
bool complexmatrix_setelement(objectcomplexmatrix *matrix, unsigned int row, unsigned int col, MorphoComplex value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
        
    unsigned int ix = col*matrix->nrows+row;
    matrix->elements[ix]=creal(value);
    matrix->elements[ix+1]=cimag(value);
    return true;
}

/** Gets a matrix element */
bool complexmatrix_getelement(objectxmatrix *matrix, unsigned int row, unsigned int col, MorphoComplex *value) {
    if (!(col<matrix->ncols && row<matrix->nrows)) return false;
    
    if (value) *value=MCBuild(0.0,0.0);
        //matrix->elements[col*matrix->nrows+row];
    return true;
}

/* **********************************************************************
 * ComplexMatrix veneer class
 * ********************************************************************** */

/*
 MORPHO_BEGINCLASS(ComplexMatrix)
 MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "()", XMatrix_print, BUILTIN_FLAGSEMPTY)
 MORPHO_ENDCLASS
 */

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void complexmatrix_initialize(void) {
    objectcomplexmatrixtype=object_addtype(&objectxmatrixdefn);
}
