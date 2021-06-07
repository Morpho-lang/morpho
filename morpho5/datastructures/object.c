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
    /* We must free any private unmanaged data */
    switch (obj->type) {
        case OBJECT_FUNCTION: {
            objectfunction *func = (objectfunction *) obj;
            morpho_freeobject(func->name);
            varray_optionalparamclear(&func->opt);
            object_functionclear(func);
        }
            break;
        case OBJECT_BUILTINFUNCTION: {
            objectbuiltinfunction *func = (objectbuiltinfunction *) obj;
            morpho_freeobject(func->name);
        }
            break;
        case OBJECT_CLASS: {
            objectclass *klass = (objectclass *) obj;
            morpho_freeobject(klass->name);
            dictionary_clear(&klass->methods);
        }
            break;
        case OBJECT_INSTANCE: {
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
            break;
        case OBJECT_ARRAY: {
            // Should free children of unmanaged arrays
        }
            break;
        case OBJECT_DICTIONARY: {
            objectdictionary *dict = (objectdictionary *) obj;
            dictionary_clear(&dict->dict);
        }
            break;
        case OBJECT_LIST: {
            objectlist *list = (objectlist *) obj;
            varray_valueclear(&list->val);
        }
            break;
        case OBJECT_MATRIX: {
        }
            break;
        case OBJECT_SPARSE: {
            objectsparse *s = (objectsparse *) obj;
            sparse_clear(s);
        }
            break;
        case OBJECT_MESH: {
            objectmesh *m = (objectmesh *) obj;
            if (m->link) {
                object *next=NULL;
                for (object *obj=m->link; obj!=NULL; obj=next) {
                    next=obj->next;
                    object_free(obj);
                }
            }
            if (m->conn) object_free((object *) m->conn);
        }
            break;
        case OBJECT_SELECTION: {
            objectselection *s = (objectselection *) obj;
            selection_clear(s);
        }
            break;
        case OBJECT_FIELD: {
            objectfield *f = (objectfield *) obj;
            
            if (f->dof) MORPHO_FREE(f->dof);
            if (f->offset) MORPHO_FREE(f->offset);
            if (f->pool) MORPHO_FREE(f->pool);
        }
        default:
            break;
    }
    
    MORPHO_FREE(obj);
}

/** Free an object if it is unmanaged */
void object_freeunmanaged(object *obj) {
    if (obj->status==OBJECT_ISUNMANAGED) object_free(obj);
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
 * Strings
 * ********************************************************************** */

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
        new->length=length;
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
 * Functions
 * ********************************************************************** */

DEFINE_VARRAY(upvalue, upvalue);

DEFINE_VARRAY(varray_upvalue, varray_upvalue);

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
        new->parent=parent;
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
 * Closures
 * ********************************************************************** */

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
 * Upvalues
 * ********************************************************************** */

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
 * Ranges
 * ********************************************************************** */

/* **********************************************************************
 * Dictionaries
 * ********************************************************************** */

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

/** Initializes an array given the size */
void object_arrayinit(objectarray *array, unsigned int ndim, unsigned int *dim) {
    object_init((object *) array, OBJECT_ARRAY);
    unsigned int nel = (ndim==0 ? 0 : 1);
    
    /* Store the description of array dimensions */
    array->dimensions=ndim;
    for (unsigned int i=0; i<ndim; i++) {
        array->data[i]=MORPHO_INTEGER(dim[i]);
        nel*=dim[i];
    }

    /* Store the size of the object for convenient access */
    array->nelements=nel;
 
    /* Arrays are initialized to nil. */
#ifdef MORPHO_NAN_BOXING
    memset(array->data+ndim, 0, sizeof(value)*nel);
#else
    for (unsigned int i=0; i<nel; i++) array->data[ndim+i]=MORPHO_FLOAT(0.0);
#endif
}

/** @brief Creates an array object
 * @details Arrays are stored in memory as follows:
 *          objectarray structure with flexible array member value
 *          value [0..dim-1] the dimensions of the array
 *          value [dim..] array elements in column major order, i.e. the matrix
 *          [ [ 1, 2],
 *           [ 3, 4] ] is stored as:
 *          <structure> // the structure
 *          2, 2, // the dimensions
 *          1, 3, 2, 4 // the elements in column major order */
objectarray *object_newarray(unsigned int ndim, unsigned int *dim) {
    /* Calculate the number of elements */
    unsigned int nel=(ndim==0 ? 0 : dim[0]);
    for (unsigned int i=1; i<ndim; i++) nel*=dim[i];
    
    /* Hence determine the memory required */
    size_t size = sizeof(objectarray)+sizeof(value)*(ndim + nel);
        
    objectarray *new = (objectarray *) object_new(size, OBJECT_ARRAY);
    
    if (new) {
        object_arrayinit(new, ndim, dim);
    }
    
    return new;
}

/* **********************************************************************
 * Utility functions
 * ********************************************************************** */

/* ---------------------------
 * Printing
 * --------------------------- */

static void object_printstring(value v) {
    printf("%s", MORPHO_GETCSTRING(v));
}

static void object_printfunction(objectfunction *f) {
    if (f) printf("<fn %s>", (MORPHO_ISNIL(f->name) ? "" : MORPHO_GETCSTRING(f->name)));
}

/** Prints an object */
void object_print(value v) {
    switch(MORPHO_GETOBJECTTYPE(v)) {
        case OBJECT_STRING: object_printstring(v); break;
        case OBJECT_FUNCTION: object_printfunction(MORPHO_GETFUNCTION(v)); break;
        case OBJECT_BUILTINFUNCTION: builtin_printfunction(MORPHO_GETBUILTINFUNCTION(v)); break;
        case OBJECT_CLOSURE:
            printf("<");
            object_printfunction(MORPHO_GETCLOSUREFUNCTION(v));
            printf(">");
            break;
        case OBJECT_UPVALUE:
            printf("upvalue");
            break;
        case OBJECT_CLASS:
#ifndef MORPHO_LOXCOMPATIBILITY
            printf("@");
#endif
            object_printstring(MORPHO_GETCLASS(v)->name);
            break;
        case OBJECT_INSTANCE:
#ifndef MORPHO_LOXCOMPATIBILITY
            printf("<");
#endif
            object_printstring(MORPHO_GETINSTANCE(v)->klass->name);
#ifndef MORPHO_LOXCOMPATIBILITY
            printf(">");
#else
            printf(" instance");
#endif
            break;
        case OBJECT_INVOCATION:
#ifndef MORPHO_LOXCOMPATIBILITY
            object_print(MORPHO_GETINVOCATION(v)->receiver);
            printf(".");
#endif
            object_print(MORPHO_GETINVOCATION(v)->method);
            break;
        case OBJECT_RANGE: {
            objectrange *r = MORPHO_GETRANGE(v);
            morpho_printvalue(r->start);
            printf("..");
            morpho_printvalue(r->end);
            if (!MORPHO_ISNIL(r->step)) {
                printf(":");
                morpho_printvalue(r->step);
            }
        }
            break;
        case OBJECT_ARRAY:
            printf("<Array>");
            break;
        case OBJECT_MATRIX:
            printf("<Matrix>");
            break;
        case OBJECT_SPARSE:
            printf("<Sparse>");
            break;
        case OBJECT_DICTIONARY:
            printf("<Dictionary>");
            break;
        case OBJECT_MESH:
            printf("<Mesh>");
            break;
        case OBJECT_SELECTION:
            printf("<Selection>");
            break;
        case OBJECT_FIELD:
            printf("<Field>");
            break;
        case OBJECT_LIST:
            printf("<List>");
            break;
        default:
            UNREACHABLE("unhandled object type [Check object_print()]");
    }
}

/** Prints an object to a string buffer
 *  @param[in] v    Object to convert to a buffer
 *  @param[in] buffer   Buffer to output to */
void object_printtobuffer(value v, varray_char *buffer) {
    switch(MORPHO_GETOBJECTTYPE(v)) {
        case OBJECT_STRING:
            varray_charadd(buffer, MORPHO_GETCSTRING(v), (int) MORPHO_GETSTRINGLENGTH(v));
            break;
        default:
            UNREACHABLE("unhandled object type [Check object_printtobuffer()]");
    }
}

/** Gets the total size of an object */
size_t object_size(object *obj) {
    switch (obj->type) {
        case OBJECT_STRING:
            return sizeof(objectstring)+((objectstring *) obj)->length+1;
        case OBJECT_CLOSURE:
            return sizeof(objectclosure)+ sizeof(objectupvalue *)*((objectclosure *) obj)->nupvalues;
        case OBJECT_UPVALUE:
            return sizeof(objectupvalue);
        case OBJECT_FUNCTION:
            return sizeof(objectfunction);
        case OBJECT_BUILTINFUNCTION:
            return sizeof(objectbuiltinfunction);
        case OBJECT_CLASS:
            return sizeof(objectclass);
        case OBJECT_INSTANCE:
            return sizeof(objectinstance);
        case OBJECT_INVOCATION:
            return sizeof(objectinvocation);
        case OBJECT_RANGE:
            return sizeof(objectrange);
        case OBJECT_DICTIONARY:
            return sizeof(objectdictionary)+(((objectdictionary *) obj)->dict.capacity)*sizeof(dictionaryentry); 
        case OBJECT_ARRAY:
            return sizeof(objectarray) +
            sizeof(value) * ( ((objectarray *) obj)->nelements+((objectarray *) obj)->dimensions );
        case OBJECT_LIST:
            return sizeof(objectlist)+sizeof(value) *
                    ((objectlist *) obj)->val.capacity;
        case OBJECT_MATRIX:
            return sizeof(objectmatrix)+sizeof(double) *
                    ((objectmatrix *) obj)->ncols *
                    ((objectmatrix *) obj)->nrows;
        case OBJECT_DOKKEY:
            return sizeof(objectdokkey);
        case OBJECT_SPARSE:
            return sparse_size((objectsparse *) obj);
        case OBJECT_MESH:
            return sizeof(objectmesh);
        case OBJECT_SELECTION:
            return sizeof(objectselection)+sizeof(objectsparse *)*((objectselection *) obj)->ngrades;
        case OBJECT_FIELD:
            return sizeof(objectfield)+(((objectfield *) obj)->ngrades * sizeof(int));
        case OBJECT_EXTERN:
            return sizeof(object);
    }
}

void object_initialize(void) {
#ifdef MORPHO_REUSEPOOL
    pool=NULL;
    npool=0;
#endif
}

void object_finalize(void) {
}
