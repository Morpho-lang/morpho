/** @file newlinalg.c
 *  @author T J Atherton
 *
 *  @brief New linear algebra
 */

#include <string.h>
#include "morpho.h"
#include "classes.h"
#include "builtin.h"
#include "newlinalg.h"

/* **********************************************************************
 * Matrix objects
 * ********************************************************************** */

objecttype objectxmatrixtype;

/** Matrix object definitions */
size_t objectxmatrix_sizefn(object *obj) {
    return sizeof(objectxmatrix)+sizeof(double) *
            ((objectxmatrix *) obj)->nels;
}

void objectxmatrix_printfn(object *obj, void *v) {
    morpho_printf(v, "<XMatrix>");
}

objecttypedefn objectxmatrixdefn = {
    .printfn=objectxmatrix_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectxmatrix_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/** Creates a new matrix object */
objectmatrix *object_newxmatrix(int nrows, int ncols, long nels, bool zero) {
    objectxmatrix *new = (objectxmatrix *) object_new(sizeof(objectxmatrix)+nels*sizeof(double), OBJECT_MATRIX);
    
    if (new) {
        new->ncols=ncols;
        new->nrows=nrows;
        new->nels=nels;
        new->elements=new->matrixdata;
        if (zero) memset(new->elements, 0, sizeof(double)*nels);
    }
    
    return new;
}

/* **********************************************************************
 * Constructors
 * ********************************************************************** */

value xmatrix_constructor(vm *v, int nargs, value *args) {
    
}

/* **********************************************************************
 * Veneer classes
 * ********************************************************************** */

/** Matrix add */
value Matrix_add_Matrix(vm *v, int nargs, value *args) {
    return MORPHO_NIL;
}

MORPHO_BEGINCLASS(XMatrix)
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "(Matrix)", Matrix_add_Matrix, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void xmatrix_initialize(void) {
    objectxmatrixtype=object_addtype(&objectxmatrixdefn);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    value xmatrixclass=builtin_addclass(XMATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(XMatrix), objclass);
    object_setveneerclass(OBJECT_XMATRIX, xmatrixclass);
    
    value consfn = builtin_addfunction(XMATRIX_CLASSNAME, xmatrix_constructor, MORPHO_FN_CONSTRUCTOR);
    builtin_setsignature(consfn, "(Int, Int)");
}
