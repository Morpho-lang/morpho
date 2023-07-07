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

/* *************************************
 * We now define essential object types
 * ************************************* */

/* -------------------------------------------------------
 * Upvalue structure
 * ------------------------------------------------------- */

/** Each upvalue */
typedef struct {
    bool islocal; /** Set if the upvalue is local to this function */
    indx reg; /** An index that either:
                  if islocal - refers to the register
               OR otherwise  - refers to the upvalue array in the current closure */
} upvalue;

DECLARE_VARRAY(upvalue, upvalue)

DECLARE_VARRAY(varray_upvalue, varray_upvalue)

/* ---------------------------
 * Classes
 * --------------------------- */

extern objecttype objectclasstype;
#define OBJECT_CLASS objectclasstype

typedef struct sobjectclass {
    object obj;
    struct sobjectclass *superclass;
    value name;
    dictionary methods;
} objectclass;

/** Tests whether an object is a class */
#define MORPHO_ISCLASS(val) object_istype(val, OBJECT_CLASS)

/** Gets the object as a class */
#define MORPHO_GETCLASS(val)   ((objectclass *) MORPHO_GETOBJECT(val))

/** Gets the superclass */
#define MORPHO_GETSUPERCLASS(val)   (MORPHO_GETCLASS(val)->superclass)

objectclass *object_newclass(value name);

objectclass *morpho_lookupclass(value obj);

/* ---------------------------
 * Upvalue objects
 * --------------------------- */

extern objecttype objectupvaluetype;
#define OBJECT_UPVALUE objectupvaluetype

typedef struct sobjectupvalue {
    object obj;
    value* location; /** Pointer to the location of the upvalue */
    value  closed; /** Closed value of the upvalue */
    struct sobjectupvalue *next;
} objectupvalue;

void object_upvalueinit(objectupvalue *c);
objectupvalue *object_newupvalue(value *reg);

/** Gets an upvalue from a value */
#define MORPHO_GETUPVALUE(val)   ((objectupvalue *) MORPHO_GETOBJECT(val))

/** Tests whether an object is an upvalue */
#define MORPHO_ISUPVALUE(val) object_istype(val, object_upvaluetype)

/* ---------------------------
 * Instances
 * --------------------------- */

extern objecttype objectinstancetype;
#define OBJECT_INSTANCE objectinstancetype

typedef struct {
    object obj;
    objectclass *klass;
    dictionary fields;
} objectinstance;

/** Tests whether an object is a class */
#define MORPHO_ISINSTANCE(val) object_istype(val, OBJECT_INSTANCE)

/** Gets the object as a class */
#define MORPHO_GETINSTANCE(val)   ((objectinstance *) MORPHO_GETOBJECT(val))

objectinstance *object_newinstance(objectclass *klass);

bool objectinstance_setproperty(objectinstance *obj, value key, value val);
bool objectinstance_getproperty(objectinstance *obj, value key, value *val);
bool objectinstance_getpropertyinterned(objectinstance *obj, value key, value *val);

/* ---------------------------
 * Bound methods
 * --------------------------- */

extern objecttype objectinvocationtype;
#define OBJECT_INVOCATION objectinvocationtype

typedef struct {
    object obj;
    value receiver;
    value method;
} objectinvocation;

/** Tests whether an object is an invocation */
#define MORPHO_ISINVOCATION(val) object_istype(val, OBJECT_INVOCATION)

/** Gets the object as an invocation */
#define MORPHO_GETINVOCATION(val)   ((objectinvocation *) MORPHO_GETOBJECT(val))

objectinvocation *object_newinvocation(value receiver, value method);

/* -------------------------------------------------------
 * Veneer classes
 * ------------------------------------------------------- */

void object_setveneerclass(objecttype type, value class);
objectclass *object_getveneerclass(objecttype type);

#endif /* object_h */
