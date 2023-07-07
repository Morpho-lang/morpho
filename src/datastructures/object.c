/** @file object.c
 *  @author T J Atherton
 *
 *  @brief Provide functionality for extended and mutable data types.
*/

#include <string.h>
#include <stdio.h>
#include "value.h"
#include "object.h"
#include "builtin.h"
#include "memory.h"
#include "error.h"
#include "strng.h"
#include "sparse.h"
#include "selection.h"
#include "common.h"

/* **********************************************************************
 * Reuse pool
 * ********************************************************************** */

#ifdef MORPHO_REUSEPOOL
#define POOLMAX 1000
int npool;
object *pool;
#endif

/* **********************************************************************
 * Object definitions
 * ********************************************************************** */

/** Hold the object type definitions as they're created */
objecttypedefn objectdefns[MORPHO_MAXIMUMOBJECTDEFNS];
objecttype objectdefnnext = 0; /** Type of the next object definition */

/** Adds a new object type */
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
void object_freeunmanaged(object *obj) {
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

/* **********************************************************************
 * Upvalues
 * ********************************************************************** */

DEFINE_VARRAY(upvalue, upvalue);
DEFINE_VARRAY(varray_upvalue, varray_upvalue);

/** Upvalue object definitions */
void objectupvalue_printfn(object *obj) {
    printf("upvalue");
}

void objectupvalue_markfn(object *obj, void *v) {
    morpho_markvalue(v, ((objectupvalue *) obj)->closed);
}

size_t objectupvalue_sizefn(object *obj) {
    return sizeof(objectupvalue);
}

objecttypedefn objectupvaluedefn = {
    .printfn=objectupvalue_printfn,
    .markfn=objectupvalue_markfn,
    .freefn=NULL,
    .sizefn=objectupvalue_sizefn
};


/** Initializes a new upvalue object. */
void object_upvalueinit(objectupvalue *c) {
    object_init(&c->obj, OBJECT_UPVALUE);
    c->location=NULL;
    c->closed=MORPHO_NIL;
    c->next=NULL;
}

/** Creates a new upvalue for the register pointed to by reg. */
objectupvalue *object_newupvalue(value *reg) {
    objectupvalue *new = (objectupvalue *) object_new(sizeof(objectupvalue), OBJECT_UPVALUE);

    if (new) {
        object_upvalueinit(new);
        new->location=reg;
    }

    return new;
}

/* **********************************************************************
 * Classes
 * ********************************************************************** */

/** Class object definitions */
void objectclass_printfn(object *obj) {
#ifndef MORPHO_LOXCOMPATIBILITY
    printf("@");
#endif
    printf("%s", MORPHO_GETCSTRING(((objectclass *) obj)->name));
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
};

objectclass *object_newclass(value name) {
    objectclass *newclass = (objectclass *) object_new(sizeof(objectclass), OBJECT_CLASS);

    if (newclass) {
        newclass->name=object_clonestring(name);
        dictionary_init(&newclass->methods);
        newclass->superclass=NULL;
    }

    return newclass;
}

/* **********************************************************************
 * Instances
 * ********************************************************************** */

/** Instance object definitions */
void objectinstance_printfn(object *obj) {
#ifndef MORPHO_LOXCOMPATIBILITY
    printf("<");
#endif
    printf("%s", MORPHO_GETCSTRING(((objectinstance *) obj)->klass->name));
#ifndef MORPHO_LOXCOMPATIBILITY
    printf(">");
#else
    printf(" instance");
#endif
}

void objectinstance_markfn(object *obj, void *v) {
    objectinstance *c = (objectinstance *) obj;
    morpho_markdictionary(v, &c->fields);
}

void objectinstance_freefn(object *obj) {
    objectinstance *instance = (objectinstance *) obj;

#ifdef MORPHO_REUSEPOOL
    if (npool<POOLMAX) {
        obj->next=pool;
        pool=obj;
        npool++;
        return;
    }
#endif

    dictionary_clear(&instance->fields);
}

size_t objectinstance_sizefn(object *obj) {
    return sizeof(objectinstance);
}

objecttypedefn objectinstancedefn = {
    .printfn=objectinstance_printfn,
    .markfn=objectinstance_markfn,
    .freefn=objectinstance_freefn,
    .sizefn=objectinstance_sizefn
};

/** Create an instance */
objectinstance *object_newinstance(objectclass *klass) {
    objectinstance *new;

#ifdef MORPHO_REUSEPOOL
    if (npool>0) {
        new = (objectinstance *) pool;
        pool = new->obj.next;
        npool--;

        new->obj.next=NULL;
        new->obj.hsh=HASH_EMPTY;
        new->obj.status=OBJECT_ISUNMANAGED;
        dictionary_wipe(&new->fields);

        new->klass=klass;
        return new;
    }
#endif

    new = (objectinstance *) object_new(sizeof(objectinstance), OBJECT_INSTANCE);

    if (new) {
        new->klass=klass;
        dictionary_init(&new->fields);
    }

    return new;
}

/* @brief Inserts a value into a property
 * @param obj   the object
 * @param key   key to use @warning: This MUST have been previously interned into a symboltable
 *                                   e.g. with builtin_internsymbol
 * @param val   value to use
 * @returns true on success  */
bool objectinstance_setproperty(objectinstance *obj, value key, value val) {
    return dictionary_insertintern(&obj->fields, key, val);
}

/* @brief Gets a value into a property
 * @param obj   the object
 * @param key   key to use
 * @param[out] val   stores the value
 * @returns true on success  */
bool objectinstance_getproperty(objectinstance *obj, value key, value *val) {
    return dictionary_get(&obj->fields, key, val);
}

/* @brief Interned property lookup
 * @param obj   the object
 * @param key   key to use @warning: This MUST have been previously interned into a symboltable
 *                                   e.g. with builtin_internsymbol
 * @param[out] val   stores the value
 * @returns true on success  */
bool objectinstance_getpropertyinterned(objectinstance *obj, value key, value *val) {
    return dictionary_getintern(&obj->fields, key, val);
}

/* **********************************************************************
 * Invocations
 * ********************************************************************** */

/** Invocation object definitions */
void objectinvocation_printfn(object *obj) {
    objectinvocation *c = (objectinvocation *) obj;
#ifndef MORPHO_LOXCOMPATIBILITY
    object_print(c->receiver);
    printf(".");
#endif
    object_print(c->method);
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
    .sizefn=objectinvocation_sizefn
};

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
 * Initialization
 * ********************************************************************** */

objecttype objectupvaluetype;
objecttype objectclasstype;
objecttype objectinstancetype;
objecttype objectinvocationtype;

void object_initialize(void) {
#ifdef MORPHO_REUSEPOOL
    pool=NULL;
    npool=0;
#endif

    objectupvaluetype=object_addtype(&objectupvaluedefn);
    objectclasstype=object_addtype(&objectclassdefn);
    objectinstancetype=object_addtype(&objectinstancedefn);
    objectinvocationtype=object_addtype(&objectinvocationdefn);
}

void object_finalize(void) {
}
