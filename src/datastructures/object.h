/** @file object.h
 *  @author T J Atherton
 *
 *  @brief Provide functionality for extended and mutable data types.
*/

#ifndef object_h
#define object_h

#include <stddef.h>
#include "value.h"
#include "dictionary.h"

typedef ptrdiff_t indx;

void object_initialize(void);
void object_finalize(void);

/* ---------------------------
 * Generic objects
 * --------------------------- */

/** Categorizes the type of an object */
typedef int objecttype;

/** Simplest object */
struct sobject {
    objecttype type;
    enum {
        OBJECT_ISUNMANAGED,
        OBJECT_ISUNMARKED,
        OBJECT_ISMARKED
    } status;
    hash hsh;
    struct sobject *next; 
};

/** Gets the type of the object associated with a value */
#define MORPHO_GETOBJECTTYPE(val)           (MORPHO_GETOBJECT(val)->type)

/** Gets an objects key */
#define MORPHO_GETOBJECTHASH(val)           (MORPHO_GETOBJECT(val)->hsh)

/** Sets an objects key */
#define MORPHO_SETOBJECTHASH(val, newhash)  (MORPHO_GETOBJECT(val)->hsh = newhash)

/* ---------------------------
 * Generic object functions
 * --------------------------- */

/** Tests whether an object is of a specified type */
static inline bool object_istype(value val, objecttype type) {
    return (MORPHO_ISOBJECT(val) && MORPHO_GETOBJECTTYPE(val)==type);
}

void object_init(object *obj, objecttype type);
void object_free(object *obj);
void object_freeunmanaged(object *obj);
void object_print(value v);
void object_printtobuffer(value v, varray_char *buffer);
object *object_new(size_t size, objecttype type);
size_t object_size(object *obj);

static inline void morpho_freeobject(value val) {
    if (MORPHO_ISOBJECT(val)) object_free(MORPHO_GETOBJECT(val));
}

/* --------------------------------------
 * Custom object types can be defined
 * by providing a few interface functions
 * -------------------------------------- */

/** Prints a short identifier for the object */
typedef void (*objectprintfn) (object *obj);

/** Mark the contents of an object */
typedef void (*objectmarkfn) (object *obj, void *v);

/** Frees any unmanaged subsidiary data structures for an object */
typedef void (*objectfreefn) (object *obj);

/** Returns the size of an object and allocated data */
typedef size_t (*objectsizefn) (object *obj);

/** Define a custom object type */
typedef struct {
    object *veneer; // Veneer class
    objectfreefn freefn;
    objectmarkfn markfn;
    objectsizefn sizefn;
    objectprintfn printfn;
} objecttypedefn;

DECLARE_VARRAY(objecttypedefn, objecttypedefn)

void object_nullfn(object *obj);

objecttype object_addtype(objecttypedefn *def);

objecttypedefn *object_getdefn(object *obj);

#endif /* object_h */
