/** @file clss.c
 *  @author T J Atherton
 *
 *  @brief Defines class object type
 */

#include "morpho.h"
#include "classes.h"

/* **********************************************************************
 * objectclass definitions
 * ********************************************************************** */

/** Class object definitions */
void objectclass_printfn(object *obj, void *v) {
    morpho_printf(v, "@%s", MORPHO_GETCSTRING(((objectclass *) obj)->name));
}

void objectclass_markfn(object *obj, void *v) {
    objectclass *c = (objectclass *) obj;
    morpho_markvalue(v, c->name);
    morpho_markdictionary(v, &c->methods);
}

void objectclass_freefn(object *obj) {
    objectclass *klass = (objectclass *) obj;
    morpho_freeobject(klass->name);
    dictionary_clear(&klass->methods);
}

size_t objectclass_sizefn(object *obj) {
    return sizeof(objectclass);
}

objecttypedefn objectclassdefn = {
    .printfn=objectclass_printfn,
    .markfn=objectclass_markfn,
    .freefn=objectclass_freefn,
    .sizefn=objectclass_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

objectclass *object_newclass(value name) {
    objectclass *newclass = (objectclass *) object_new(sizeof(objectclass), OBJECT_CLASS);

    if (newclass) {
        newclass->name=object_clonestring(name);
        dictionary_init(&newclass->methods);
        newclass->superclass=NULL;
        newclass->uid=0;
    }

    return newclass;
}

/* **********************************************************************
 * objectclass utility functions
 * ********************************************************************** */

/* **********************************************************************
 * (Future) Class veneer class
 * ********************************************************************** */

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

objecttype objectclasstype;

void class_initialize(void) {
    // objectclass is a core type so is intialized earlier
    
    // TODO: Add Class veneer class
    // Locate the Object class to use as the parent class of Class
    //objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    //value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // No constructor function; classes are generated by the compiler
    
    // Class error messages
    morpho_defineerror(CLASS_INVK, ERROR_HALT, CLASS_INVK_MSG);
}
