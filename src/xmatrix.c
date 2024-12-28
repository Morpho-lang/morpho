/** @file xmatrix.c
 *  @author T J Atherton
 *
 *  @brief New matrices
*/

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

/* **********************************************************************
 * XMatrix constructor
 * ********************************************************************** */

value xmatrix_constructor(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    
    int nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    int ncols = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    
    objectxmatrix *new=xmatrix_new(nrows, ncols, true);
    
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

value xmatrix_list_constructor(vm *v, int nargs, value *args) {
    
    
    return MORPHO_NIL;
}

value xmatrix_invalid_constructor(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, MATRIX_CONSTRUCTOR);
    return MORPHO_NIL;
}

/* **********************************************************************
 * XMatrix veneer class
 * ********************************************************************** */

value XMatrix_add(vm *v, int nargs, value *args) {
    return MORPHO_NIL;
}

MORPHO_BEGINCLASS(XMatrix)
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "(XMatrix)", XMatrix_add, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void xmatrix_initialize(void) {
    objectxmatrixtype=object_addtype(&objectxmatrixdefn);
    
    value xmatrixclass=builtin_addclass(XMATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(XMatrix), MORPHO_NIL);
    
    object_setveneerclass(OBJECT_XMATRIX, xmatrixclass);
    
    morpho_addfunction(XMATRIX_CLASSNAME, "XMatrix (Int, Int)", xmatrix_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(XMATRIX_CLASSNAME, "XMatrix (List)", xmatrix_list_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(XMATRIX_CLASSNAME, "(...)", xmatrix_invalid_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
}

