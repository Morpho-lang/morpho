/** @file xcomplexmatrix.c
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#include "xcomplexmatrix.h"

objecttype objectcomplexmatrixtype;

//typedef objectxmatrix objectcomplexmatrix;

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
