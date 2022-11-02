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
 * @param key   key to use @warning: This MUST have been previously interned into a symboltable
 *                                   e.g. with builtin_internsymbol
 * @param[out] val   stores the value
 * @returns true on success  */
bool objectinstance_getproperty(objectinstance *obj, value key, value *val) {
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
 * Strings
 * ********************************************************************** */

/** String object definitions */
void objectstring_printfn(object *obj) {
    printf("%s", ((objectstring *) obj)->string);
}

size_t objectstring_sizefn(object *obj) {
    return sizeof(objectstring)+((objectstring *) obj)->length+1;
}

objecttypedefn objectstringdefn = {
    .printfn = objectstring_printfn,
    .markfn = NULL,
    .freefn = NULL,
    .sizefn = objectstring_sizefn
};

/** @brief Creates a string from an existing character array with given length
 *  @param in     the string to copy
 *  @param length length of string to copy
 *  @returns the object (as a value) which will be MORPHO_NIL on failure */
value object_stringfromcstring(const char *in, size_t length) {
    value out = MORPHO_NIL;
    objectstring *new = (objectstring *) object_new(sizeof(objectstring) + sizeof(char) * (length + 1), OBJECT_STRING);

    if (new) {
        new->string=new->stringdata;
        new->string[length] = '\0'; /* Zero terminate the string to be compatible with C */
        memcpy(new->string, in, length);
        new->length=strlen(new->string);
        out = MORPHO_OBJECT(new);
    }
    return out;
}

/** @brief Converts a varray_char into a string.
 *  @param in  the varray to convert
 *  @returns the object (as a value) which will be MORPHO_NIL on failure */
value object_stringfromvarraychar(varray_char *in) {
    return object_stringfromcstring(in->data, in->count);
}


/* Clones a string object */
value object_clonestring(value val) {
    value out = MORPHO_NIL;
    if (MORPHO_ISSTRING(val)) {
        objectstring *s = MORPHO_GETSTRING(val);
        out=object_stringfromcstring(s->string, s->length);
    }
    return out;
}

/** @brief Concatenates strings together
 *  @param a      first string
 *  @param b      second string
 *  @returns the object (as a value) which will be MORPHO_NIL on failure  */
value object_concatenatestring(value a, value b) {
    objectstring *astring = MORPHO_GETSTRING(a);
    objectstring *bstring = MORPHO_GETSTRING(b);
    size_t length = (astring ? astring->length : 0) + (bstring ? bstring->length : 0);
    value out = MORPHO_NIL;

    objectstring *new = (objectstring *) object_new(sizeof(objectstring) + sizeof(char) * (length + 1), OBJECT_STRING);

    if (new) {
        new->string=new->stringdata;
        new->length=length;
        /* Copy across old strings */
        if (astring) memcpy(new->string, astring->string, astring->length);
        if (bstring) memcpy(new->string+(astring ? astring->length : 0), bstring->string, bstring->length);
        new->string[length]='\0';
        out = MORPHO_OBJECT(new);
    }
    return out;
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
 * Lists
 * ********************************************************************** */

/** List object definitions */
void objectlist_printfn(object *obj) {
    printf("<List>");
}

void objectlist_freefn(object *obj) {
    objectlist *list = (objectlist *) obj;
    varray_valueclear(&list->val);
}

void objectlist_markfn(object *obj, void *v) {
    objectlist *c = (objectlist *) obj;
    morpho_markvarrayvalue(v, &c->val);
}

size_t objectlist_sizefn(object *obj) {
    return sizeof(objectlist)+sizeof(value) *
            ((objectlist *) obj)->val.capacity;
}

objecttypedefn objectlistdefn = {
    .printfn=objectlist_printfn,
    .markfn=objectlist_markfn,
    .freefn=objectlist_freefn,
    .sizefn=objectlist_sizefn
};

/** Creates a new list */
objectlist *object_newlist(unsigned int nval, value *val) {
    objectlist *new = (objectlist *) object_new(sizeof(objectlist), OBJECT_LIST);

    if (new) {
        varray_valueinit(&new->val);
        if (val) varray_valueadd(&new->val, val, nval);
        else varray_valueresize(&new->val, nval);
    }

    return new;
}

/* **********************************************************************
 * Arrays
 * ********************************************************************** */

/** Array object definitions */
void objectarray_printfn(object *obj) {
    printf("<Array>");
}

void objectarray_markfn(object *obj, void *v) {
    objectarray *c = (objectarray *) obj;
    for (unsigned int i=0; i<c->nelements; i++) {
        morpho_markvalue(v, c->values[i]);
    }
}

size_t objectarray_sizefn(object *obj) {
    return sizeof(objectarray) +
        sizeof(value) * ( ((objectarray *) obj)->nelements+2*((objectarray *) obj)->ndim );
}

objecttypedefn objectarraydefn = {
    .printfn=objectarray_printfn,
    .markfn=objectarray_markfn,
    .freefn=NULL,
    .sizefn=objectarray_sizefn
};

/** Initializes an array given the size */
void object_arrayinit(objectarray *array, unsigned int ndim, unsigned int *dim) {
    object_init((object *) array, OBJECT_ARRAY);
    unsigned int nel = (ndim==0 ? 0 : 1);

    /* Store pointers into the data array */
    array->dimensions=array->data;
    array->multipliers=array->data+ndim;
    array->values=array->data+2*ndim;

    /* Store the description of array dimensions */
    array->ndim=ndim;
    for (unsigned int i=0; i<ndim; i++) {
        array->dimensions[i]=MORPHO_INTEGER(dim[i]);
        array->multipliers[i]=MORPHO_INTEGER(nel);
        nel*=dim[i];
    }

    /* Store the size of the object for convenient access */
    array->nelements=nel;

    /* Arrays are initialized to nil. */
#ifdef MORPHO_NAN_BOXING
    memset(array->values, 0, sizeof(value)*nel);
#else
    for (unsigned int i=0; i<nel; i++) array->values[i]=MORPHO_FLOAT(0.0);
#endif
}

/** @brief Creates an array object
 * @details Arrays are stored in memory as follows:
 *          objectarray structure with flexible array member value
 *          value [0..dim-1] the dimensions of the array
 *          value [dim..2*dim-1] stores multipliers for each dimension to translate to the index
 *          value [2*dim..] array elements in column major order, i.e. the matrix
 *          [ [ 1, 2],
 *           [ 3, 4] ] is stored as:
 *          <structure> // the structure
 *          2, 2, // the dimensions
 *          1, 2, // multipliers for each index to access elements
 *          1, 3, 2, 4 // the elements in column major order */
objectarray *object_newarray(unsigned int ndim, unsigned int *dim) {
    /* Calculate the number of elements */
    unsigned int nel=(ndim==0 ? 0 : dim[0]);
    for (unsigned int i=1; i<ndim; i++) nel*=dim[i];

    size_t size = sizeof(objectarray)+sizeof(value)*(2*ndim + nel);

    objectarray *new = (objectarray *) object_new(size, OBJECT_ARRAY);
    if (new) object_arrayinit(new, ndim, dim);

    return new;
}

/* **********************************************************************
 * Ranges
 * ********************************************************************** */

/** Array object definitions */
void objectrange_printfn(object *obj) {
    objectrange *r = (objectrange *) obj;
    morpho_printvalue(r->start);
    printf("..");
    morpho_printvalue(r->end);
    if (!MORPHO_ISNIL(r->step)) {
        printf(":");
        morpho_printvalue(r->step);
    }
}

size_t objectrange_sizefn(object *obj) {
    return sizeof(objectrange);
}

objecttypedefn objectrangedefn = {
    .printfn=objectrange_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectrange_sizefn
};

/** Create a new range. Step may be set to MORPHO_NIL to use the default value of 1 */
objectrange *object_newrange(value start, value end, value step) {
    value v[3]={start, end, step};

    /* Ensure all three values are either integer or floating point */
    if (!value_promotenumberlist((MORPHO_ISNIL(step) ? 2 : 3), v)) return NULL;

    objectrange *new = (objectrange *) object_new(sizeof(objectrange), OBJECT_RANGE);

    if (new) {
        new->start=v[0];
        new->end=v[1];
        new->step=v[2];
        new->nsteps=range_count(new);
    }

    return new;
}

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

objecttype objectstringtype;
objecttype objectfunctiontype;
objecttype objectupvaluetype;
objecttype objectclosuretype;
objecttype objectclasstype;
objecttype objectinstancetype;
objecttype objectinvocationtype;

objecttype objectdictionarytype;
objecttype objectarraytype;
objecttype objectlisttype;

objecttype objectrangetype;

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

    objectstringtype=object_addtype(&objectstringdefn);
    objectarraytype=object_addtype(&objectarraydefn);
    objectlisttype=object_addtype(&objectlistdefn);
    objectdictionarytype=object_addtype(&objectdictionarydefn);
    objectrangetype=object_addtype(&objectrangedefn);
}

void object_finalize(void) {
}
