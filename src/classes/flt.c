/** @file flt.c
 *  @author T J Atherton
 *
 *  @brief Veneer class for float values
 */

#include "morpho.h"
#include "classes.h"
#include "format.h"

value Value_format(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;

    if (nargs==1 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        varray_char str;
        varray_charinit(&str);
        
        if (format_printtobuffer(MORPHO_SELF(args),
                            MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)),
                                 &str)) {
            
            out = object_stringfromvarraychar(&str);
            if (MORPHO_ISOBJECT(out)) morpho_bindobjects(v, 1, &out);
        } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        
        varray_charclear(&str);
    } else {
        morpho_runtimeerror(v, VALUE_FRMTARG);
    }
    
    return out;
}

/* **********************************************************************
 * Float veneer class
 * ********************************************************************** */

MORPHO_BEGINCLASS(Float)
MORPHO_METHOD(MORPHO_FORMAT_METHOD, Value_format, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

void float_initialize(void) {
    // Create Float veneer class
    value floatclass=builtin_addclass(FLOAT_CLASSNAME, MORPHO_GETCLASSDEFINITION(Float), MORPHO_NIL);
    value_setveneerclass(MORPHO_FLOAT(0.0), floatclass);
    
    morpho_defineerror(VALUE_FRMTARG, ERROR_HALT, VALUE_FRMTARG_MSG);
}

void float_finalize(void) {
}
