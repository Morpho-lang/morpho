/** @file flt.c
 *  @author T J Atherton
 *
 *  @brief Veneer class for float values
 */

#include "morpho.h"
#include "classes.h"

/* **********************************************************************
 * float utility functions
 * ********************************************************************** */

/* **********************************************************************
 * Float veneer class
 * ********************************************************************** */

value Float_format(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;

    return out;
}

MORPHO_BEGINCLASS(Float)
MORPHO_METHOD(MORPHO_FORMAT_METHOD, Float_format, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

void float_initialize(void) {
    // Create Float veneer class
    value floatclass=builtin_addclass(FLOAT_CLASSNAME, MORPHO_GETCLASSDEFINITION(Float), NULL);
    value_setveneerclass(MORPHO_FLOAT(0.0), floatclass);
}

void float_finalize(void) {
}
