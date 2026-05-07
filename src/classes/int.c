/** @file int.c
 *  @author T J Atherton
 *
 *  @brief Veneer class for float values
 */

#include "morpho.h"
#include "classes.h"

/* **********************************************************************
 * int utility functions
 * ********************************************************************** */

/* **********************************************************************
 * Int veneer class
 * ********************************************************************** */

MORPHO_BEGINCLASS(Int)
MORPHO_METHOD(MORPHO_CLASS_METHOD, Object_class, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_RESPONDSTO_METHOD, Object_respondsto, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_INVOKE_METHOD, Object_invoke, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, MORPHO_FN_IO),
MORPHO_METHOD_SIGNATURE(MORPHO_FORMAT_METHOD, "String (_)", Value_format, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

void int_initialize(void) {
    // Create Int veneer class
    value intclass=builtin_addclass(INT_CLASSNAME, MORPHO_GETCLASSDEFINITION(Int), MORPHO_NIL);
    value_setveneerclass(MORPHO_INTEGER(1), intclass);
}
