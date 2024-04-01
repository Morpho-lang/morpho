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
MORPHO_METHOD(MORPHO_FORMAT_METHOD, Value_format, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

void int_initialize(void) {
    // Create Int veneer class
    value intclass=builtin_addclass(INT_CLASSNAME, MORPHO_GETCLASSDEFINITION(Int), MORPHO_NIL);
    value_setveneerclass(MORPHO_INTEGER(1), intclass);
}
