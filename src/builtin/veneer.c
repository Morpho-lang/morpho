/** @file veneer.c
 *  @author T J Atherton
 *
 *  @brief Veneer classes over built in objects
 */

#include "morpho.h"
#include "veneer.h"
#include "object.h"
#include "common.h"
#include "parse.h"

/* **********************************************************************
 * Error
 * ********************************************************************** */

static value error_tagproperty;
static value error_messageproperty;

/** Initializer
 * In: 1. Error tag
 *   2. Default error message
 */
value Error_init(vm *v, int nargs, value *args) {

    if ((nargs==2) &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 1))) {

        objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), error_tagproperty, MORPHO_GETARG(args, 0));
        objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), error_messageproperty, MORPHO_GETARG(args, 1));

    } else MORPHO_RAISE(v, ERROR_ARGS);

    return MORPHO_NIL;
}

/** Throw an error */
value Error_throw(vm *v, int nargs, value *args) {
    objectinstance *slf = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    value tag=MORPHO_NIL, msg=MORPHO_NIL;

    if (slf) {
        objectinstance_getpropertyinterned(slf, error_tagproperty, &tag);
        if (nargs==0) {
            objectinstance_getpropertyinterned(slf, error_messageproperty, &msg);
        } else {
            msg=MORPHO_GETARG(args, 0);
        }

        morpho_usererror(v, MORPHO_GETCSTRING(tag), MORPHO_GETCSTRING(msg));
    }

    return MORPHO_NIL;
}

/** Print errors */
value Error_print(vm *v, int nargs, value *args) {
    object_print(MORPHO_SELF(args));

    return MORPHO_SELF(args);
}

MORPHO_BEGINCLASS(Error)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Error_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_THROW_METHOD, Error_throw, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Error_print, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void veneer_initialize(void) {
    /* Error */
    //builtin_addclass(ERROR_CLASSNAME, MORPHO_GETCLASSDEFINITION(Error), objclass);
    //error_tagproperty=builtin_internsymbolascstring(ERROR_TAG_PROPERTY);
    //error_messageproperty=builtin_internsymbolascstring(ERROR_MESSAGE_PROPERTY);

    
    morpho_defineerror(ISMEMBER_ARG, ERROR_HALT, ISMEMBER_ARG_MSG);
    morpho_defineerror(CLASS_INVK, ERROR_HALT, CLASS_INVK_MSG);
    morpho_defineerror(ERROR_ARGS, ERROR_HALT, ERROR_ARGS_MSG);
}
