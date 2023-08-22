/** @file err.c
 *  @author T J Atherton
 *
 *  @brief Implements the Error class
 */

#include "morpho.h"
#include "classes.h"

static value error_tagproperty;
static value error_messageproperty;

/* **********************************************************************
 * Error class
 * ********************************************************************** */

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

/** Extract the tag and message for an error */
bool _err_extract(vm *v, int nargs, value *args, value *tag, value *msg) {
    objectinstance *slf = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    if (!slf) return false;
    
    objectinstance_getpropertyinterned(slf, error_tagproperty, tag);
    if (nargs==0) {
        objectinstance_getpropertyinterned(slf, error_messageproperty, msg);
    } else {
        *msg=MORPHO_GETARG(args, 0);
    }
    
    return true;
}

/** Throw an error */
value Error_throw(vm *v, int nargs, value *args) {
    value tag=MORPHO_NIL, msg=MORPHO_NIL;

    if (_err_extract(v, nargs, args, &tag, &msg)) {
        error err;
        error_init(&err);
        morpho_writeusererror(&err, MORPHO_GETCSTRING(tag), MORPHO_GETCSTRING(msg));
        morpho_error(v, &err);
        error_clear(&err);
    }

    return MORPHO_NIL;
}

/** Raise a warning */
value Error_warning(vm *v, int nargs, value *args) {
    value tag=MORPHO_NIL, msg=MORPHO_NIL;

    if (_err_extract(v, nargs, args, &tag, &msg)) {
        error err;
        error_init(&err);
        morpho_writeusererror(&err, MORPHO_GETCSTRING(tag), MORPHO_GETCSTRING(msg));
        morpho_warning(v, &err);
        error_clear(&err);
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
MORPHO_METHOD(MORPHO_WARNING_METHOD, Error_warning, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Error_print, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

void err_initialize(void) {
    // Locate the Object class to use as the parent class of Error
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // Create Error class
    builtin_addclass(ERROR_CLASSNAME, MORPHO_GETCLASSDEFINITION(Error), objclass);
    
    // Create labels for Error property names
    error_tagproperty=builtin_internsymbolascstring(ERROR_TAG_PROPERTY);
    error_messageproperty=builtin_internsymbolascstring(ERROR_MESSAGE_PROPERTY);
    
    // Error error messages
    morpho_defineerror(ERROR_ARGS, ERROR_HALT, ERROR_ARGS_MSG);
}

void err_finalize(void) {
}
