/** @file invocation.c
 *  @author T J Atherton
 *
 *  @brief Implements the Invocation class
 */

#include "morpho.h"
#include "classes.h"
#include "common.h"

/* **********************************************************************
 * objectinvocation definitions
 * ********************************************************************** */

/** Invocation object definitions */
void objectinvocation_printfn(object *obj, void *v) {
    objectinvocation *c = (objectinvocation *) obj;
#ifndef MORPHO_LOXCOMPATIBILITY
    morpho_printvalue(v, c->receiver);
    morpho_printf(v, ".");
#endif
    morpho_printvalue(v, c->method);
}

void objectinvocation_markfn(object *obj, void *v) {
    objectinvocation *c = (objectinvocation *) obj;
    morpho_markvalue(v, c->receiver);
    morpho_markvalue(v, c->method);
}

size_t objectinvocation_sizefn(object *obj) {
    return sizeof(objectinvocation);
}

objecttypedefn objectinvocationdefn = {
    .printfn=objectinvocation_printfn,
    .markfn=objectinvocation_markfn,
    .freefn=NULL,
    .sizefn=objectinvocation_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/* **********************************************************************
 * objectinvocation utility functions
 * ********************************************************************** */

/** Create a new invocation */
objectinvocation *object_newinvocation(value receiver, value method) {
    objectinvocation *new = (objectinvocation *) object_new(sizeof(objectinvocation), OBJECT_INVOCATION);

    if (new) {
        new->receiver=receiver;
        new->method=method;
    }

    return new;
}

/* **********************************************************************
 * Invocation veneer class
 * ********************************************************************** */

/** Creates a new invocation object */
value invocation_constructor(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;

    if (nargs==2) {
        value receiver = MORPHO_GETARG(args, 0);
        value selector = MORPHO_GETARG(args, 1);
        
        if (!MORPHO_ISOBJECT(receiver) || !MORPHO_ISSTRING(selector)) {
            morpho_runtimeerror(v, INVOCATION_ARGS);
            return MORPHO_NIL;
        }
        
        value method = MORPHO_NIL;
        
        objectclass *klass=morpho_lookupclass(receiver);
        
        if (dictionary_get(&klass->methods, selector, &method)) {
            objectinvocation *new = object_newinvocation(receiver, method);

            if (new) {
                out = MORPHO_OBJECT(new);
                morpho_bindobjects(v, 1, &out);
            } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        }
        
    } else morpho_runtimeerror(v, INVOCATION_ARGS);

    return out;
}

/** Converts to a string for string interpolation */
value Invocation_tostring(vm *v, int nargs, value *args) {
    objectinvocation *inv=MORPHO_GETINVOCATION(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    varray_char buffer;
    varray_charinit(&buffer);
    
    morpho_printtobuffer(v, inv->receiver, &buffer);
    varray_charwrite(&buffer, '.');
    morpho_printtobuffer(v, inv->method, &buffer);
    
    out = object_stringfromvarraychar(&buffer);
    if (MORPHO_ISSTRING(out)) {
        morpho_bindobjects(v, 1, &out);
    }
    varray_charclear(&buffer);

    return out;
}

/** Clones a range */
value Invocation_clone(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    value out = MORPHO_NIL;
    if (!MORPHO_ISINVOCATION(self)) return MORPHO_NIL;
    
    objectinvocation *slf = MORPHO_GETINVOCATION(self);
    objectinvocation *new = object_newinvocation(slf->receiver, slf->method);
    
    if (new) {
        out = MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

MORPHO_BEGINCLASS(Invocation)
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_TOSTRING_METHOD, Invocation_tostring, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Invocation_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

objecttype objectinvocationtype;

void invocation_initialize(void) {
    // Create invocation object type
    objectinvocationtype=object_addtype(&objectinvocationdefn);
    
    // Locate the Object class to use as the parent class of Invocation
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // Invocation constructor function
    builtin_addfunction(INVOCATION_CLASSNAME, invocation_constructor, BUILTIN_FLAGSEMPTY);
    
    // Create invocation veneer class
    value invocationclass=builtin_addclass(INVOCATION_CLASSNAME, MORPHO_GETCLASSDEFINITION(Invocation), objclass);
    object_setveneerclass(OBJECT_INVOCATION, invocationclass);
    
    // Invocation error messages
    morpho_defineerror(INVOCATION_ARGS, ERROR_HALT, INVOCATION_ARGS_MSG);
    morpho_defineerror(INVOCATION_METHOD, ERROR_HALT, INVOCATION_METHOD_MSG);
}
