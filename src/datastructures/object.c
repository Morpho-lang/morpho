/** @file object.c
 *  @author T J Atherton
 *
 *  @brief Provide functionality for extended and mutable data types.
*/

#include <string.h>
#include <stdio.h>

#include "morpho.h"
#include "classes.h"

/* **********************************************************************
 * Object definitions
 * ********************************************************************** */

/** Hold the object type definitions as they're created */
objecttypedefn objectdefns[MORPHO_MAXIMUMOBJECTDEFNS];
objecttype objectdefnnext = 0; /** Type of the next object definition */

/** Adds a new object type with a given definition.
 @returns: the objecttype identifier to be used henceforth */
objecttype object_addtype(objecttypedefn *def) {
    if (!def->printfn || !def->sizefn) {
        UNREACHABLE("Object definition must provide a print and size function.");
    }

    if (objectdefnnext>=MORPHO_MAXIMUMOBJECTDEFNS) {
        UNREACHABLE("Too many object definitions (increase MORPHO_MAXIMUMOBJECTDEFNS).");
    }

    objectdefns[objectdefnnext]=*def;
    objectdefns[objectdefnnext].veneer = NULL;
    objectdefnnext+=1;

    return objectdefnnext-1;
}

/** Gets the appropriate definition given an object */
objecttypedefn *object_getdefn(object *obj) {
    return &objectdefns[obj->type];
}

/* **********************************************************************
 * Objects
 * ********************************************************************** */

/** @brief Initializes an object to be a certain type
 *  @param obj    object to initialize
 *  @param type   type to initialize with */
void object_init(object *obj, objecttype type) {
    obj->next=NULL;
    obj->hsh=HASH_EMPTY;
    obj->status=OBJECT_ISUNMANAGED;
    obj->type=type;
}

/** Frees an object */
void object_free(object *obj) {
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    if (obj) {
        printf("Free object %p of type %d ", (void *) obj, obj->type);
        object_print(MORPHO_OBJECT(obj));
        printf("\n");
    }
#endif
    if (object_getdefn(obj)->freefn) object_getdefn(obj)->freefn(obj);
    MORPHO_FREE(obj);
}

/** Free an object if it is unmanaged */
void object_freeifunmanaged(object *obj) {
    if (obj->status==OBJECT_ISUNMANAGED) object_free(obj);
}

/** Prints an object */
void object_print(value v) {
    object *obj = MORPHO_GETOBJECT(v);
    object_getdefn(obj)->printfn(obj);
}

/** Gets the total size of an object */
size_t object_size(object *obj) {
    return object_getdefn(obj)->sizefn(obj);
}

/** @brief Allocates an object
 *  @param size   size of memory to reserve
 *  @param type   type to initialize with */
object *object_new(size_t size, objecttype type) {
    object *new = MORPHO_MALLOC(size);

    if (new) object_init(new, type);

#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("Create object %p of size %ld with type %d.\n", (void *) new, size, type);
#endif

    return new;
}

/** Checks if an object is of a particular type */
bool object_istype(value val, objecttype type) {
    return (MORPHO_ISOBJECT(val) && MORPHO_GETOBJECTTYPE(val)==type);
}

/** Free any object that may be contained in a value. */
void morpho_freeobject(value val) {
    if (MORPHO_ISOBJECT(val)) object_free(MORPHO_GETOBJECT(val));
}

/* **********************************************************************
 * Veneer classes
 * ********************************************************************** */

/** @brief Sets the veneer class for a particular object type */
void object_setveneerclass(objecttype type, value class) {
    if (objectdefns[type].veneer!=NULL) {
        UNREACHABLE("Veneer class redefined.\n");
    }
    objectdefns[type].veneer=(object *) MORPHO_GETCLASS(class);
}

/** @brief Gets the veneer for a particular object type */
objectclass *object_getveneerclass(objecttype type) {
    return (objectclass *) objectdefns[type].veneer;
}
