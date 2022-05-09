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
 * Functions
 * --------------------------- */

extern objecttype objectfunctiontype;
#define OBJECT_FUNCTION objectfunctiontype

typedef struct {
    value symbol; /** Symbol associated with the variable */
    indx def; /** Default value as constant */
    indx reg; /** Associated register */
} optionalparam;

DECLARE_VARRAY(optionalparam, optionalparam)

/** A function object */
typedef struct sobjectfunction {
    object obj;
    int nargs;
    int vararg; // The parameter number of a variadic parameter.
    value name;
    indx entry;
    struct sobjectfunction *parent;
    int nupvalues;
    int nregs;
    varray_value konst;
    varray_varray_upvalue prototype;
    varray_optionalparam opt;
} objectfunction;

/** Gets an objectfunction from a value */
#define MORPHO_GETFUNCTION(val)   ((objectfunction *) MORPHO_GETOBJECT(val))

/** Tests whether an object is a function */
#define MORPHO_ISFUNCTION(val) object_istype(val, OBJECT_FUNCTION)

void object_functioninit(objectfunction *func);
void object_functionclear(objectfunction *func);
bool object_functionaddprototype(objectfunction *func, varray_upvalue *v, indx *ix);
objectfunction *object_getfunctionparent(objectfunction *func);
value object_getfunctionname(objectfunction *func);
varray_value *object_functiongetconstanttable(objectfunction *func);
objectfunction *object_newfunction(indx entry, value name, objectfunction *parent, unsigned int nargs);

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

/** Gets an objectfunction from a value */
#define MORPHO_GETUPVALUE(val)   ((objectupvalue *) MORPHO_GETOBJECT(val))

/** Tests whether an object is a function */
#define MORPHO_ISUPVALUE(val) object_istype(val, object_upvaluetype)

/* ---------------------------
 * Closures
 * --------------------------- */

extern objecttype objectclosuretype;
#define OBJECT_CLOSURE objectclosuretype

typedef struct {
    object obj;
    objectfunction *func;
    int nupvalues;
    objectupvalue *upvalues[];
} objectclosure;

objectclosure *object_newclosure(objectfunction *sf, objectfunction *func, indx np);

/** Tests whether an object is a closure */
#define MORPHO_ISCLOSURE(val) object_istype(val, OBJECT_CLOSURE)

/** Gets the object as a closure */
#define MORPHO_GETCLOSURE(val)   ((objectclosure *) MORPHO_GETOBJECT(val))

/** Retrieve the function object from a closure */
#define MORPHO_GETCLOSUREFUNCTION(val)  (((objectclosure *) MORPHO_GETOBJECT(val))->func)

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

bool objectinstance_insertpropertybycstring(objectinstance *obj, char *property, value val);

bool objectinstance_getpropertybycstring(objectinstance *obj, char *property, value *val);

/* ---------------------------
 * Strings
 * --------------------------- */

extern objecttype objectstringtype;
#define OBJECT_STRING objectstringtype

/** A string object */
typedef struct {
    object obj;
    size_t length;
    char *string;
    char stringdata[];
} objectstring;

#define MORPHO_GETSTRING(val)             ((objectstring *) MORPHO_GETOBJECT(val))
#define MORPHO_GETCSTRING(val)            (((objectstring *) MORPHO_GETOBJECT(val))->string)
#define MORPHO_GETSTRINGLENGTH(val)       (((objectstring *) MORPHO_GETOBJECT(val))->length)

/** Tests whether an object is a string */
#define MORPHO_ISSTRING(val) object_istype(val, OBJECT_STRING)

/** Use to create static strings on the C stack */
#define MORPHO_STATICSTRING(cstring)      { .obj.type=OBJECT_STRING, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .string=cstring, .length=strlen(cstring) }

/** Use to create static strings on the C stack */
#define MORPHO_STATICSTRINGWITHLENGTH(cstring, len)      { .obj.type=OBJECT_STRING, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .string=cstring, .length=len }


#define OBJECT_STRINGLABEL "string"
#define OBJECT_SYMBOLLABEL "symbol"

value object_stringfromcstring(const char *in, size_t length);
value object_stringfromvarraychar(varray_char *in);
value object_clonestring(value val);
value object_concatenatestring(value a, value b);

/* -------------------------------------------------------
 * Dictionaries
 * ------------------------------------------------------- */

extern objecttype objectdictionarytype;
#define OBJECT_DICTIONARY objectdictionarytype

typedef struct {
    object obj;
    dictionary dict;
} objectdictionary;

/** Tests whether an object is a dictionary */
#define MORPHO_ISDICTIONARY(val) object_istype(val, OBJECT_DICTIONARY)

/** Gets the object as a dictionary */
#define MORPHO_GETDICTIONARY(val)   ((objectdictionary *) MORPHO_GETOBJECT(val))

/** Extract the dictionary from an objectdictionary */
/*static dictionary *object_dictionaryfromobject(objectdictionary *dict) {
    return &dict->dict;
}*/

objectdictionary *object_newdictionary(void);

/* -------------------------------------------------------
 * Lists
 * ------------------------------------------------------- */

extern objecttype objectlisttype;
#define OBJECT_LIST objectlisttype

typedef struct {
    object obj;
    varray_value val;
} objectlist;

/** Tests whether an object is a list */
#define MORPHO_ISLIST(val) object_istype(val, OBJECT_LIST)

/** Gets the object as a list */
#define MORPHO_GETLIST(val)   ((objectlist *) MORPHO_GETOBJECT(val))

/** Create a static list - you must initialize the list separately */
#define MORPHO_STATICLIST      { .obj.type=OBJECT_LIST, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .val.count=0, .val.capacity=0, .val.data=NULL }

objectlist *object_newlist(unsigned int nval, value *val);

/* -------------------------------------------------------
 * Arrays
 * ------------------------------------------------------- */

extern objecttype objectarraytype;
#define OBJECT_ARRAY objectarraytype

typedef struct {
    object obj;
    unsigned int ndim;
    unsigned int nelements;
    value *values;
    value *dimensions;
    value *multipliers;
    value data[];
} objectarray;

/** Tests whether an object is an array */
#define MORPHO_ISARRAY(val) object_istype(val, OBJECT_ARRAY)

/** Gets the object as an array */
#define MORPHO_GETARRAY(val)   ((objectarray *) MORPHO_GETOBJECT(val))

/** Creates an array object */
objectarray *object_newarray(unsigned int dimension, unsigned int *dim);

/** Creates a new array from a list of values */
objectarray *object_arrayfromvaluelist(unsigned int n, value *v);

/** Creates a new 1D array from a list of varray_value */
objectarray *object_arrayfromvarrayvalue(varray_value *v);

/** Creates a new array object with the dimensions given as a list of values */
objectarray *object_arrayfromvalueindices(unsigned int ndim, value *dim);

/* -------------------------------------------------------
 * Ranges
 * ------------------------------------------------------- */

extern objecttype objectrangetype;
#define OBJECT_RANGE objectrangetype

typedef struct {
    object obj;
    unsigned int nsteps;
    value start;
    value end;
    value step;
} objectrange;

/** Tests whether an object is a range */
#define MORPHO_ISRANGE(val) object_istype(val, OBJECT_RANGE)

/** Gets the object as a range */
#define MORPHO_GETRANGE(val)   ((objectrange *) MORPHO_GETOBJECT(val))

objectrange *object_newrange(value start, value end, value step);

/* -------------------------------------------------------
 * Veneer classes
 * ------------------------------------------------------- */

void object_setveneerclass(objecttype type, value class);
objectclass *object_getveneerclass(objecttype type);

#endif /* object_h */
