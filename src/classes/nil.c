/** @file nil.c
 *  @author T J Atherton
 *
 *  @brief Veneer class for float values
 */

#include "morpho.h"
#include "classes.h"

/* **********************************************************************
 * Float veneer class
 * ********************************************************************** */

MORPHO_BEGINCLASS(Nil)
MORPHO_METHOD(MORPHO_CLASS_METHOD, Object_class, MORPHO_FN_PUREFN)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

void nil_initialize(void) {
    // Create Nil veneer class
    value nilclass=builtin_addclass(NIL_CLASSNAME, MORPHO_GETCLASSDEFINITION(Nil), MORPHO_NIL);
    value_setveneerclass(MORPHO_NIL, nilclass);
}
