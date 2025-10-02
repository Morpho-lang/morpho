/** @file cfunction.c
 *  @author T J Atherton
 *
 *  @brief Veneer class for C functions
 */

#include "morpho.h"
#include "classes.h"
#include "common.h"

/* **********************************************************************
 * CFunction veneer class
 * ********************************************************************** */

value CFunction_tostring(vm *v, int nargs, value *args) {
    objectbuiltinfunction *func=MORPHO_GETBUILTINFUNCTION(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    varray_char buffer;
    varray_charinit(&buffer);

    varray_charadd(&buffer, "<fn ", 4);
    morpho_printtobuffer(v, func->name, &buffer);
    varray_charwrite(&buffer, '>');

    out = object_stringfromvarraychar(&buffer);
    if (MORPHO_ISSTRING(out)) {
        morpho_bindobjects(v, 1, &out);
    }
    varray_charclear(&buffer);

    return out;
}

MORPHO_BEGINCLASS(CFunction)
MORPHO_METHOD(MORPHO_TOSTRING_METHOD, CFunction_tostring, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, MORPHO_FN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

void cfunction_initialize(void) {
    // Locate the Object class to use as the parent class of CFunction
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // Create CFunction veneer class
    value cfunctionclass=builtin_addclass(CFUNCTION_CLASSNAME, MORPHO_GETCLASSDEFINITION(CFunction), objclass);
    object_setveneerclass(OBJECT_BUILTINFUNCTION, cfunctionclass);
}
