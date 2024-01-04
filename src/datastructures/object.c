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
objecttypedefn _objectdefns[MORPHO_MAXIMUMOBJECTDEFNS];
objecttype objectdefnnext; /** Type of the next object definition */

/** Adds a new object type with a given definition.
 @returns: the objecttype identifier to be used henceforth */
objecttype object_addtype(objecttypedefn *def) {
    if (!def->printfn || !def->sizefn) {
        UNREACHABLE("Object definition must provide a print and size function.");
    }

    if (objectdefnnext>=MORPHO_MAXIMUMOBJECTDEFNS) {
        UNREACHABLE("Too many object definitions (increase MORPHO_MAXIMUMOBJECTDEFNS).");
    }

    _objectdefns[objectdefnnext]=*def;
    _objectdefns[objectdefnnext].veneer = NULL;
    objectdefnnext+=1;

    return objectdefnnext-1;
}

/** Gets the appropriate definition given an object */
objecttypedefn *object_getdefn(object *obj) {
    return &_objectdefns[obj->type];
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
        fprintf(stderr, "Free object %p of type %d ", (void *) obj, obj->type);
        morpho_printvalue(NULL, MORPHO_OBJECT(obj));
        fprintf(stderr, "\n");
    }
#endif
    if (object_getdefn(obj)->freefn) object_getdefn(obj)->freefn(obj);
    MORPHO_FREE(obj);
}

/** Free an object if it is unmanaged */
void object_freeifunmanaged(object *obj) {
    if (obj->status==OBJECT_ISUNMANAGED) object_free(obj);
}

/** Calls an object's print function */
void object_print(void *v, value val) {
    object *obj = MORPHO_GETOBJECT(val);
    object_getdefn(obj)->printfn(obj, v);
}

/** Gets the total size of an object */
size_t object_size(object *obj) {
    return object_getdefn(obj)->sizefn(obj);
}

/** Hash an object, either by calling its hash function or by hashing its pointer */
hash object_hash(object *obj) {
    objecttypedefn *defn = object_getdefn(obj);
    if (defn->hashfn) return (defn->hashfn) (obj);
    
    return dictionary_hashpointer(obj);
}

/** Compare two objects */
int object_cmp(object *a, object *b) {
    objecttypedefn *defn = object_getdefn(a);
    
    if (defn->cmpfn) return (defn->cmpfn) (a, b);
    
    return (a == b? MORPHO_EQUAL: MORPHO_NOTEQUAL);
}

/** @brief Allocates an object
 *  @param size   size of memory to reserve
 *  @param type   type to initialize with */
object *object_new(size_t size, objecttype type) {
    object *new = MORPHO_MALLOC(size);

    if (new) object_init(new, type);

#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    fprintf(stderr, "Create object %p of size %ld with type %d.\n", (void *) new, size, type);
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
    if (_objectdefns[type].veneer!=NULL) {
        UNREACHABLE("Veneer class redefined.\n");
    }
    _objectdefns[type].veneer=(object *) MORPHO_GETCLASS(class);
}

/** @brief Gets the veneer for a particular object type */
objectclass *object_getveneerclass(objecttype type) {
    return (objectclass *) _objectdefns[type].veneer;
}

/* **********************************************************************
 * Initialization
 * ********************************************************************* */

void object_initialize(void) {
    objectdefnnext=0;
}

void object_finalize(void) {
    objectdefnnext=0;
}
