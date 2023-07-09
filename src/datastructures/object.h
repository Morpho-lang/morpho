/** @file object.h
 *  @author T J Atherton
 *
 *  @brief Implement objects, heap-allocated data structures with type information
*/

#ifndef object_h
#define object_h

#include <stddef.h>
#include "value.h"
#include "dictionary.h"

/* -------------------------------------------------------
 * Fundamental object type
 * ------------------------------------------------------- */

/** Objects are heap-allocated data structures that have a common header allowing type identification.
    The header should never be accessed directly, but through macros as provided below */

/** The objecttype identifies the type of an object. */
typedef int objecttype;

/** Fundamental object structure */
struct sobject {
    objecttype type;            // Type
    enum {                      // Memory management status for the object:
        OBJECT_ISUNMANAGED,     // - UNMANAGED means the object is manually alloc'd/dealloc'd
        OBJECT_ISUNMARKED,      // - UNMARKED means the object is managed by the GC
        OBJECT_ISMARKED         // - MARKED is used internally by the GC
    } status;
    hash hsh;                   // hash value
    struct sobject *next;       // All objects can be chained together (e.g. to attach to the VM that created them)
};

/** These macros access the object structure's fields. */

/** Gets the type of the object associated with a value
    @warning: Do not use this to compare types, use an appropriate macro like MORPHO_ISXXX  */
#define MORPHO_GETOBJECTTYPE(val)           (MORPHO_GETOBJECT(val)->type)

/** Gets an object's key */
#define MORPHO_GETOBJECTHASH(val)           (MORPHO_GETOBJECT(val)->hsh)

/** Sets an objects key */
#define MORPHO_SETOBJECTHASH(val, newhash)  (MORPHO_GETOBJECT(val)->hsh = newhash)

/* -------------------------------------------------------
 * object definitions
 * ------------------------------------------------------- */

/** To define a new object type, you must provide several functions that enable the morpho runtime to interact with it.
    These object definition functions are collected together in an objecttypedefn.
    The object type is assigned at initialization by calling object_addtype */

/** Called to print a short identifier for the object */
typedef void (*objectprintfn) (object *obj);

/** Called to mark the contents of an object; called by the garbage collector to identify subsidiary objects */
typedef void (*objectmarkfn) (object *obj, void *v);

/** Called to free any unmanaged subsidiary data structures for an object */
typedef void (*objectfreefn) (object *obj);

/** Called to return the size of an object and attached data (anything NOT stored in a value) */
typedef size_t (*objectsizefn) (object *obj);

/** Defines a custom object type. */
typedef struct {
    object *veneer; // Veneer class
    objectfreefn freefn;
    objectmarkfn markfn;
    objectsizefn sizefn;
    objectprintfn printfn;
} objecttypedefn;

/* -------------------------------------------------------
 * Object creation and management
 * ------------------------------------------------------- */

// Define new object types
objecttype object_addtype(objecttypedefn *def);
objecttypedefn *object_getdefn(object *obj);

// Management of object structures
void object_init(object *obj, objecttype type);
void object_free(object *obj);
void object_freeifunmanaged(object *obj);
void object_print(value v);
void object_printtobuffer(value v, varray_char *buffer);
size_t object_size(object *obj);

bool object_istype(value val, objecttype type);

// Create a new object with a specified allocation size and type
object *object_new(size_t size, objecttype type);

// Recommended interface to free an object from a value
void morpho_freeobject(value val);

// Index type
typedef ptrdiff_t indx;

#endif /* object_h */
