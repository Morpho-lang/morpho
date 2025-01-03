/** @file bool.c
 *  @author T J Atherton
 *
 *  @brief Veneer class for bool values
 */

#include "morpho.h"
#include "classes.h"

/* **********************************************************************
 * Bool veneer class
 * ********************************************************************** */

MORPHO_BEGINCLASS(Bool)
MORPHO_METHOD(MORPHO_CLASS_METHOD, Object_class, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_RESPONDSTO_METHOD, Object_respondsto, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_INVOKE_METHOD, Object_invoke, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, MORPHO_FN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

void bool_initialize(void) {
    // Create Bool veneer class
    value boolclass=builtin_addclass(BOOL_CLASSNAME, MORPHO_GETCLASSDEFINITION(Bool), MORPHO_NIL);
    value_setveneerclass(MORPHO_TRUE, boolclass);
}
