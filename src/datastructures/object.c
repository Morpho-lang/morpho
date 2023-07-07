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
 * Functions
 * ********************************************************************** */

DEFINE_VARRAY(upvalue, upvalue);
DEFINE_VARRAY(varray_upvalue, varray_upvalue);

/** Function object definitions */
void objectfunction_freefn(object *obj) {
    objectfunction *func = (objectfunction *) obj;
    morpho_freeobject(func->name);
    varray_optionalparamclear(&func->opt);
    object_functionclear(func);
}

void objectfunction_markfn(object *obj, void *v) {
    objectfunction *f = (objectfunction *) obj;
    morpho_markvalue(v, f->name);
    morpho_markvarrayvalue(v, &f->konst);
}

size_t objectfunction_sizefn(object *obj) {
    return sizeof(objectfunction);
}

void objectfunction_printfn(object *obj) {
    objectfunction *f = (objectfunction *) obj;
    if (f) printf("<fn %s>", (MORPHO_ISNIL(f->name) ? "" : MORPHO_GETCSTRING(f->name)));
}

objecttypedefn objectfunctiondefn = {
    .printfn=objectfunction_printfn,
    .markfn=objectfunction_markfn,
    .freefn=objectfunction_freefn,
    .sizefn=objectfunction_sizefn
};

/** @brief Initializes a new function */
void object_functioninit(objectfunction *func) {
    func->entry=0;
    func->name=MORPHO_NIL;
    func->nargs=0;
    func->parent=NULL;
    func->nupvalues=0;
    func->nregs=0;
    varray_valueinit(&func->konst);
    varray_varray_upvalueinit(&func->prototype);
}

/** @brief Clears a function */
void object_functionclear(objectfunction *func) {
    varray_valueclear(&func->konst);
    /** Clear the upvalue prototypes */
    for (unsigned int i=0; i<func->prototype.count; i++) {
        varray_upvalueclear(&func->prototype.data[i]);
    }
    varray_varray_upvalueclear(&func->prototype);
}

/** @brief Creates a new function */
objectfunction *object_newfunction(indx entry, value name, objectfunction *parent, unsigned int nargs) {
    objectfunction *new = (objectfunction *) object_new(sizeof(objectfunction), OBJECT_FUNCTION);

    if (new) {
        object_functioninit(new);
        new->entry=entry;
        new->name=object_clonestring(name);
        new->nargs=nargs;
        new->varg=-1; // No vargs
        new->parent=parent;
        new->klass=NULL; 
        varray_optionalparaminit(&new->opt);
    }

    return new;
}

/** Gets the parent of a function */
objectfunction *object_getfunctionparent(objectfunction *func) {
    return func->parent;
}

/** Gets the name of a function */
value object_getfunctionname(objectfunction *func) {
    return func->name;
}

/** Gets the constant table associated with a function */
varray_value *object_functiongetconstanttable(objectfunction *func) {
    if (func) {
        return &func->konst;
    }
    return NULL;
}

/** Does a function have variadic args? */
bool object_functionhasvargs(objectfunction *func) {
    return (func->varg>=0);
}

/** Sets the parameter number of a variadic argument */
void object_functionsetvarg(objectfunction *func, unsigned int varg) {
    func->varg=varg;
}

/** Adds an upvalue prototype to a function
 * @param[in]  func   function object to add to
 * @param[in]  v      a varray of upvalues that will be copied into the function
 *                    definition.
 * @param[out] ix     index of the closure created
 * @returns true on success */
bool object_functionaddprototype(objectfunction *func, varray_upvalue *v, indx *ix) {
    bool success=false;
    varray_upvalue new;
    varray_upvalueinit(&new);
    success=varray_upvalueadd(&new, v->data, v->count);
    if (success) varray_varray_upvalueadd(&func->prototype, &new, 1);
    if (success && ix) *ix = (indx) func->prototype.count-1;
    return success;
}

/* **********************************************************************
 * Upvalues
 * ********************************************************************** */

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
 * Closures
 * ********************************************************************** */

/** Closure object definitions */
void objectclosure_printfn(object *obj) {
    objectclosure *f = (objectclosure *) obj;
    printf("<");
    objectfunction_printfn((object *) f->func);
    printf(">");
}

void objectclosure_markfn(object *obj, void *v) {
    objectclosure *c = (objectclosure *) obj;
    morpho_markobject(v, (object *) c->func);
    for (unsigned int i=0; i<c->nupvalues; i++) {
        morpho_markobject(v, (object *) c->upvalues[i]);
    }
}

size_t objectclosure_sizefn(object *obj) {
    return sizeof(objectclosure)+sizeof(objectupvalue *)*((objectclosure *) obj)->nupvalues;
}

objecttypedefn objectclosuredefn = {
    .printfn=objectclosure_printfn,
    .markfn=objectclosure_markfn,
    .freefn=NULL,
    .sizefn=objectclosure_sizefn
};

/** Closure functions */
void object_closureinit(objectclosure *c) {
    c->func=NULL;
}

/** @brief Creates a new closure
 *  @param sf       the objectfunction of the current environment
 *  @param func     a function object to enclose
 *  @param np       the prototype number to use */
objectclosure *object_newclosure(objectfunction *sf, objectfunction *func, indx np) {
    objectclosure *new = NULL;
    varray_upvalue *up = NULL;

    if (np<sf->prototype.count) {
        up = &sf->prototype.data[np];
    }

    if (up) {
        new = (objectclosure *) object_new(sizeof(objectclosure) + sizeof(objectupvalue*)*up->count, OBJECT_CLOSURE);
        if (new) {
            object_closureinit(new);
            new->func=func;
            for (unsigned int i=0; i<up->count; i++) {
                new->upvalues[i]=NULL;
            }
            new->nupvalues=up->count;
        }
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
 * Dictionaries
 * ********************************************************************** */

/** Dictionary object definitions */
void objectdictionary_printfn(object *obj) {
    printf("<Dictionary>");
}

void objectdictionary_freefn(object *obj) {
    objectdictionary *dict = (objectdictionary *) obj;
    dictionary_clear(&dict->dict);
}

void objectdictionary_markfn(object *obj, void *v) {
    objectdictionary *c = (objectdictionary *) obj;
    morpho_markdictionary(v, &c->dict);
}

size_t objectdictionary_sizefn(object *obj) {
    return sizeof(objectdictionary)+(((objectdictionary *) obj)->dict.capacity)*sizeof(dictionaryentry);
}

objecttypedefn objectdictionarydefn = {
    .printfn=objectdictionary_printfn,
    .markfn=objectdictionary_markfn,
    .freefn=objectdictionary_freefn,
    .sizefn=objectdictionary_sizefn
};

/** Creates a new dictionary */
objectdictionary *object_newdictionary(void) {
    objectdictionary *new = (objectdictionary *) object_new(sizeof(objectdictionary), OBJECT_DICTIONARY);

    if (new) dictionary_init(&new->dict);

    return new;
}

/** Extracts the dictionary from an objectdictionary. */
dictionary *object_dictionary(objectdictionary *dict) {
    return &dict->dict;
}

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

objecttype objectfunctiontype;
objecttype objectupvaluetype;
objecttype objectclosuretype;
objecttype objectclasstype;
objecttype objectinstancetype;
objecttype objectinvocationtype;

objecttype objectdictionarytype;

void object_initialize(void) {
#ifdef MORPHO_REUSEPOOL
    pool=NULL;
    npool=0;
#endif

    objectfunctiontype=object_addtype(&objectfunctiondefn);
    objectupvaluetype=object_addtype(&objectupvaluedefn);
    objectclosuretype=object_addtype(&objectclosuredefn);
    objectclasstype=object_addtype(&objectclassdefn);
    objectinstancetype=object_addtype(&objectinstancedefn);
    objectinvocationtype=object_addtype(&objectinvocationdefn);
    objectdictionarytype=object_addtype(&objectdictionarydefn);
}

void object_finalize(void) {
}
