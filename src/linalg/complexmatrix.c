/** @file newlinalg.c
 *  @author T J Atherton
 *
 *  @brief New linear algebra
 */

#include <string.h>
#include "morpho.h"
#include "classes.h"
#include "newlinalg.h"
#include "complexmatrix.h"

/* **********************************************************************
 * Matrix objects
 * ********************************************************************** */

objecttype objectcomplexmatrixtype;

/** Matrix object definitions */
size_t objectcomplexmatrix_sizefn(object *obj) {
    return sizeof(objectxmatrix)+sizeof(complex double) *
            ((objectxmatrix *) obj)->nels;
}

void objectcomplexmatrix_printfn(object *obj, void *v) {
    morpho_printf(v, "<" COMPLEXMATRIX_CLASSNAME ">");
}

objecttypedefn objectcomplexmatrixdefn = {
    .printfn=objectcomplexmatrix_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectcomplexmatrix_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/* **********************************************************************
 * Veneer class
 * ********************************************************************** */

/** ComplexMatrix add */
value ComplexMatrix_add_Matrix(vm *v, int nargs, value *args) {
    return MORPHO_NIL;
}

MORPHO_BEGINCLASS(ComplexMatrix)
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "(XMatrix)", ComplexMatrix_add_Matrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "(ComplexMatrix)", ComplexMatrix_add_Matrix, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void complexmatrix_initialize(void) {
    objectcomplexmatrixtype=object_addtype(&objectcomplexmatrixdefn);
    
    objectstring matrixname = MORPHO_STATICSTRING(MATRIX_CLASSNAME);
    value matrixclass = builtin_findclass(MORPHO_OBJECT(&matrixname));
    
    value complexmatrixclass=builtin_addclass(COMPLEXMATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(ComplexMatrix), matrixclass);
    object_setveneerclass(OBJECT_COMPLEXMATRIX, complexmatrixclass);
}
