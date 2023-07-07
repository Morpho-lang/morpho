/** @file veneer.c
 *  @author T J Atherton
 *
 *  @brief Veneer classes over built in objects
 */

#include "morpho.h"
#include "veneer.h"
#include "object.h"
#include "common.h"
#include "parse.h"

/* **********************************************************************
 * Object
 * ********************************************************************** */

/** Sets an object property */
value Object_getindex(vm *v, int nargs, value *args) {
    value self=MORPHO_SELF(args);
    value out=MORPHO_NIL;
    
    if (nargs==1 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISINSTANCE(self)) {
        if (!dictionary_get(&MORPHO_GETINSTANCE(self)->fields, MORPHO_GETARG(args, 0), &out)) {
            morpho_runtimeerror(v, VM_OBJECTLACKSPROPERTY, MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)));
        }
    }

    return out;
}

/** Gets an object property */
value Object_setindex(vm *v, int nargs, value *args) {
    value self=MORPHO_SELF(args);

    if (MORPHO_ISINSTANCE(self)) {
        if (nargs==2 &&
            MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
            dictionary_insert(&MORPHO_GETINSTANCE(self)->fields, MORPHO_GETARG(args, 0), MORPHO_GETARG(args, 1));
        } else morpho_runtimeerror(v, SETINDEX_ARGS);
    } else {
        morpho_runtimeerror(v, OBJECT_IMMUTABLE);
    }

    return MORPHO_NIL;
}

/** Given an object attempts to find its class */
objectclass *object_getclass(value v) {
    objectclass *klass=NULL;
    if (MORPHO_ISINSTANCE(v)) klass=MORPHO_GETINSTANCE(v)->klass;
    else if (MORPHO_ISCLASS(v)) klass=MORPHO_GETCLASS(v);
    else if (MORPHO_ISOBJECT(v)) klass=object_getveneerclass(MORPHO_GETOBJECTTYPE(v));
    return klass;
}

/** Find the object's class */
value Object_class(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    value out = MORPHO_NIL;

    objectclass *klass=object_getclass(self);
    if (klass) out = MORPHO_OBJECT(klass);
    
    return out;
}

/** Find the object's superclass */
value Object_super(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    
    objectclass *klass=object_getclass(self);
    
    return (klass && klass->superclass ? MORPHO_OBJECT(klass->superclass) : MORPHO_NIL);
}

/** Checks if an object responds to a method */
value Object_respondsto(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    
    objectclass *klass=object_getclass(self);

    if (nargs == 0) {
        value out = MORPHO_NIL;
        objectlist *new = object_newlist(0, NULL);
        if (new) {
            list_resize(new, klass->methods.count);
            for (unsigned int i=0; i<klass->methods.capacity; i++) {
                if (MORPHO_ISSTRING(klass->methods.contents[i].key)) {
                    list_append(new, klass->methods.contents[i].key);
                }
            }
            out = MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        return out;

    } else if (nargs==1 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        return MORPHO_BOOL(dictionary_get(&klass->methods, MORPHO_GETARG(args, 0), NULL));
    } else MORPHO_RAISE(v, RESPONDSTO_ARG);

    return MORPHO_FALSE;
}

/** Checks if an object has a property */
value Object_has(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    if (!MORPHO_ISINSTANCE(self)) return MORPHO_FALSE;

    if (nargs == 0) {
        value out = MORPHO_NIL;
        objectlist *new = object_newlist(0, NULL);
        if (new) {
            objectinstance *slf = MORPHO_GETINSTANCE(self);
            list_resize(new, MORPHO_GETINSTANCE(self)->fields.count);
            for (unsigned int i=0; i<slf->fields.capacity; i++) {
                if (MORPHO_ISSTRING(slf->fields.contents[i].key)) {
                    list_append(new, slf->fields.contents[i].key);
                }
            }
            out = MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        return out;

    } else if (nargs==1 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        return MORPHO_BOOL(dictionary_get(&MORPHO_GETINSTANCE(self)->fields, MORPHO_GETARG(args, 0), NULL));
        
    } else MORPHO_RAISE(v, HAS_ARG);
    
    return MORPHO_FALSE;
}

/** Invoke a method */
value Object_invoke(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    value out=MORPHO_NIL;

    objectclass *klass=object_getclass(self);
    
    if (klass && nargs>0 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        value fn;
        if (dictionary_get(&klass->methods, MORPHO_GETARG(args, 0), &fn)) {
            morpho_invoke(v, self, fn, nargs-1, &MORPHO_GETARG(args, 1), &out);
        } else morpho_runtimeerror(v, VM_OBJECTLACKSPROPERTY, MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)));
    } else morpho_runtimeerror(v, VM_INVALIDARGS, 1, 0);

    return out;
}

/** Generic print */
value Object_print(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    objectclass *klass=NULL;
    if (MORPHO_ISCLASS(self)) {
        klass=MORPHO_GETCLASS(self);
#ifndef MORPHO_LOXCOMPATIBILITY
        printf("@%s", (MORPHO_ISSTRING(klass->name) ? MORPHO_GETCSTRING(klass->name): "Object"));
#else
        printf("%s", (MORPHO_ISSTRING(klass->name) ? MORPHO_GETCSTRING(klass->name): "Object"));
#endif
    } else if (MORPHO_ISINSTANCE(self)) {
        klass=MORPHO_GETINSTANCE(self)->klass;
#ifndef MORPHO_LOXCOMPATIBILITY
        if (klass) printf("<%s>", (MORPHO_ISSTRING(klass->name) ? MORPHO_GETCSTRING(klass->name): "Object") );
#else
        if (klass) printf("%s instance", (MORPHO_ISSTRING(klass->name) ? MORPHO_GETCSTRING(klass->name): "Object") );
#endif
    } else {
        morpho_printvalue(self);
    }
    return MORPHO_NIL;
}

/** Count number of properties */
value Object_count(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);

    if (MORPHO_ISINSTANCE(self)) {
        objectinstance *obj = MORPHO_GETINSTANCE(self);
        return MORPHO_INTEGER(obj->fields.count);
    } else if (MORPHO_ISCLASS(self)) {
        return MORPHO_INTEGER(0);
    }

    return MORPHO_NIL;
}

/** Enumerate protocol */
value Object_enumerate(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    value out = MORPHO_NIL;

    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

        if (MORPHO_ISINSTANCE(self)) {
            dictionary *dict= &MORPHO_GETINSTANCE(self)->fields;

            if (n<0) {
                out=MORPHO_INTEGER(dict->count);
            } else if (n<dict->count) {
                unsigned int k=0;
                for (unsigned int i=0; i<dict->capacity; i++) {
                    if (!MORPHO_ISNIL(dict->contents[i].key)) {
                        if (k==n) return dict->contents[i].key;
                        k++;
                    }
                }
            } else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        } else if (MORPHO_ISCLASS(self)) {
            if (n<0) out = MORPHO_INTEGER(0);
        }
    } else MORPHO_RAISE(v, ENUMERATE_ARGS);

     return out;
}

/** Generic serializer */
value Object_serialize(vm *v, int nargs, value *args) {
    return MORPHO_NIL;
}

/** Generic clone */
value Object_clone(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    value out = MORPHO_NIL;

    if (MORPHO_ISINSTANCE(self)) {
        objectinstance *instance = MORPHO_GETINSTANCE(self);
        objectinstance *new = object_newinstance(instance->klass);
        if (new) {
            dictionary_copy(&instance->fields, &new->fields);
            out = MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        }
    } else {
        morpho_runtimeerror(v, OBJECT_CANTCLONE);
    }

    return out;
}

MORPHO_BEGINCLASS(Object)
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, Object_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, Object_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLASS_METHOD, Object_class, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUPER_METHOD, Object_super, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_RESPONDSTO_METHOD, Object_respondsto, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_HAS_METHOD, Object_has, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_INVOKE_METHOD, Object_invoke, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Object_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, Object_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SERIALIZE_METHOD, Object_serialize, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Object_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Dictionary
 * ********************************************************************** */

/** Create a dictionary object */
value dictionary_constructor(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectdictionary *new=object_newdictionary();

    if (new) {
        out=MORPHO_OBJECT(new);

        for (unsigned int i=0; i+1<nargs; i+=2) {
            dictionary_insert(&new->dict, MORPHO_GETARG(args, i), MORPHO_GETARG(args, i+1));
        }

        morpho_bindobjects(v, 1, &out);
    }

    return out;
}

/** Gets a dictionary entry */
value Dictionary_getindex(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (nargs==1) {
        if(!dictionary_get(&slf->dict, MORPHO_GETARG(args, 0), &out)) {
            morpho_runtimeerror(v, DICT_DCTKYNTFND);
        }
    }

    return out;
}

/** Sets a dictionary entry */
value Dictionary_setindex(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));

    if (nargs==2) {
        unsigned int capacity = slf->dict.capacity;

        dictionary_insert(&slf->dict, MORPHO_GETARG(args, 0), MORPHO_GETARG(args, 1));

        if (slf->dict.capacity!=capacity) morpho_resizeobject(v, (object *) slf, capacity*sizeof(dictionaryentry)+sizeof(objectdictionary), slf->dict.capacity*sizeof(dictionaryentry)+sizeof(objectdictionary));
    } else morpho_runtimeerror(v, SETINDEX_ARGS);

    return MORPHO_NIL;
}

/** Returns a Bool value for whether the Dictionary contains a given key */
value Dictionary_contains(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));
    value out=MORPHO_FALSE;

    if (nargs==1) {
        if (dictionary_get(&slf->dict, MORPHO_GETARG(args, 0), &out)) out=MORPHO_TRUE;
    }

    return out;
}

/** Removes a dictionary entry with a given key */
value Dictionary_remove(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));
    
    if (nargs==1) {
        dictionary_remove(&slf->dict, MORPHO_GETARG(args, 0));
    }
    
    return MORPHO_NIL;
}

/** Prints a dictionary */
value Dictionary_print(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    if (!MORPHO_ISDICTIONARY(self)) return Object_print(v, nargs, args);
    
    objectdictionary *slf = MORPHO_GETDICTIONARY(self);

    printf("{ ");
    unsigned int k=0;
    for (unsigned int i=0; i<slf->dict.capacity; i++) {
        if (!MORPHO_ISNIL(slf->dict.contents[i].key)) {
            if (k>0) printf(" , ");
            morpho_printvalue(slf->dict.contents[i].key);
            printf(" : ");
            morpho_printvalue(slf->dict.contents[i].val);
            k++;
        }
    }
    printf(" }");

    return MORPHO_NIL;
}

/** Counts number of items in dictionary */
value Dictionary_count(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));

    return MORPHO_INTEGER(slf->dict.count);
}

/** Iterates over dictionary; current implementation returns a sequence of keys */
value dictionary_iterate(objectdictionary *dict, unsigned int n) {
    unsigned int k=0;
    for (unsigned int i=0; i<dict->dict.capacity; i++) {
        if (!MORPHO_ISNIL(dict->dict.contents[i].key)) {
            if (k==n) return dict->dict.contents[i].key;
            k++;
        }
    }
    return MORPHO_NIL;
}

/** Enumerate protocol */
value Dictionary_enumerate(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

        if (n<0) out=MORPHO_INTEGER(slf->dict.count);
        else out=dictionary_iterate(slf, n);
    } else MORPHO_RAISE(v, ENUMERATE_ARGS);

    return out;
}

/** Gets a list of keys */
value Dictionary_keys(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));
    objectlist *list = object_newlist(slf->dict.count, NULL);
    value out=MORPHO_NIL;

    if (list) {
        for (unsigned int i=0; i<slf->dict.capacity; i++) {
            if (!MORPHO_ISNIL(slf->dict.contents[i].key)) {
                list_append(list, slf->dict.contents[i].key);
            }
        }
        out=MORPHO_OBJECT(list);
        morpho_bindobjects(v, 1, &out);
    }

    return out;
}

/** Clones a dictionary */
value Dictionary_clone(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));
    objectdictionary *new = object_newdictionary();
    if (!new) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    value out=MORPHO_OBJECT(new);

    dictionary_copy(&slf->dict, &new->dict);
    morpho_bindobjects(v, 1, &out);

    return out;
}

/** Clears a Dictionary */
value Dictionary_clear(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));
    
    dictionary_clear(&slf->dict);
    
    return MORPHO_NIL;
}

#define DICTIONARY_SETOP(op) \
value Dictionary_##op(vm *v, int nargs, value *args) { \
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args)); \
    value out=MORPHO_NIL; \
    \
    if (nargs>0 && MORPHO_ISDICTIONARY(MORPHO_GETARG(args, 0))) { \
        objectdictionary *new = object_newdictionary(); \
        \
        if (new) { \
            objectdictionary *b =MORPHO_GETDICTIONARY(MORPHO_GETARG(args, 0)); \
            dictionary_##op(&slf->dict, &b->dict, &new->dict); \
            out=MORPHO_OBJECT(new); \
            morpho_bindobjects(v, 1, &out); \
        } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); \
    } else morpho_runtimeerror(v, DICT_DCTSTARG); \
    \
    return out; \
}

DICTIONARY_SETOP(union)
DICTIONARY_SETOP(intersection)
DICTIONARY_SETOP(difference)

MORPHO_BEGINCLASS(Dictionary)
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, Dictionary_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, Dictionary_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CONTAINS_METHOD, Dictionary_contains, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(DICTIONARY_REMOVE_METHOD, Dictionary_remove, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(DICTIONARY_CLEAR_METHOD, Dictionary_clear, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Dictionary_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Dictionary_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, Dictionary_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(DICTIONARY_KEYS_METHOD, Dictionary_keys, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Dictionary_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_UNION_METHOD, Dictionary_union, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_INTERSECTION_METHOD, Dictionary_intersection, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_DIFFERENCE_METHOD, Dictionary_difference, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADD_METHOD, Dictionary_union, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUB_METHOD, Dictionary_difference, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Closure
 * ********************************************************************** */

value Closure_tostring(vm *v, int nargs, value *args) {
    objectclosure *self=MORPHO_GETCLOSURE(MORPHO_SELF(args));
    value out = MORPHO_NIL;
    
    varray_char buffer;
    varray_charinit(&buffer);

    if (self->func) {
        varray_charadd(&buffer, "<<fn ", 5);
        morpho_printtobuffer(v, self->func->name, &buffer);
        varray_charadd(&buffer, ">>", 2);
    }

    out = object_stringfromvarraychar(&buffer);
    if (MORPHO_ISSTRING(out)) {
        morpho_bindobjects(v, 1, &out);
    }
    varray_charclear(&buffer);

    return out;
}

MORPHO_BEGINCLASS(Closure)
MORPHO_METHOD(MORPHO_TOSTRING_METHOD, Closure_tostring, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Function
 * ********************************************************************** */

value Function_tostring(vm *v, int nargs, value *args) {
    objectfunction *func=MORPHO_GETFUNCTION(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    varray_char buffer;
    varray_charinit(&buffer);

    varray_charadd(&buffer, "<fn ", 4);
    morpho_printtobuffer(v, func->name, &buffer);
    varray_charwrite(&buffer, '>');

    out = object_stringfromvarraychar(&buffer);
    if (MORPHO_ISSTRING(out)) {
        morpho_bindobjects(v, 1, &out);
    }
    varray_charclear(&buffer);

    return out;
}

MORPHO_BEGINCLASS(Function)
MORPHO_METHOD(MORPHO_TOSTRING_METHOD, Function_tostring, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Invocation
 * ********************************************************************** */

/** Creates a new invocation object */
value invocation_constructor(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;

    if (nargs==2) {
        value receiver = MORPHO_GETARG(args, 0);
        value selector = MORPHO_GETARG(args, 1);
        
        if (!MORPHO_ISOBJECT(receiver) || !MORPHO_ISSTRING(selector)) {
            morpho_runtimeerror(v, INVOCATION_ARGS);
            return MORPHO_NIL;
        }
        
        value method = MORPHO_NIL;
        
        objectclass *klass=object_getclass(receiver);
        
        if (dictionary_get(&klass->methods, selector, &method)) {
            objectinvocation *new = object_newinvocation(receiver, method);

            if (new) {
                out = MORPHO_OBJECT(new);
                morpho_bindobjects(v, 1, &out);
            } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        }
        
    } else morpho_runtimeerror(v, INVOCATION_ARGS);

    return out;
}

/** Converts to a string for string interpolation */
value Invocation_tostring(vm *v, int nargs, value *args) {
    objectinvocation *inv=MORPHO_GETINVOCATION(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    varray_char buffer;
    varray_charinit(&buffer);
    
    morpho_printtobuffer(v, inv->receiver, &buffer);
    varray_charwrite(&buffer, '.');
    morpho_printtobuffer(v, inv->method, &buffer);
    
    out = object_stringfromvarraychar(&buffer);
    if (MORPHO_ISSTRING(out)) {
        morpho_bindobjects(v, 1, &out);
    }
    varray_charclear(&buffer);

    return out;
}

/** Clones a range */
value Invocation_clone(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    value out = MORPHO_NIL;
    if (!MORPHO_ISINVOCATION(self)) return MORPHO_NIL;
    
    objectinvocation *slf = MORPHO_GETINVOCATION(self);
    objectinvocation *new = object_newinvocation(slf->receiver, slf->method);
    
    if (new) {
        out = MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

MORPHO_BEGINCLASS(Invocation)
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_TOSTRING_METHOD, Invocation_tostring, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Invocation_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Error
 * ********************************************************************** */

static value error_tagproperty;
static value error_messageproperty;

/** Initializer
 * In: 1. Error tag
 *   2. Default error message
 */
value Error_init(vm *v, int nargs, value *args) {

    if ((nargs==2) &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 1))) {

        objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), error_tagproperty, MORPHO_GETARG(args, 0));
        objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), error_messageproperty, MORPHO_GETARG(args, 1));

    } else MORPHO_RAISE(v, ERROR_ARGS);

    return MORPHO_NIL;
}

/** Throw an error */
value Error_throw(vm *v, int nargs, value *args) {
    objectinstance *slf = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    value tag=MORPHO_NIL, msg=MORPHO_NIL;

    if (slf) {
        objectinstance_getpropertyinterned(slf, error_tagproperty, &tag);
        if (nargs==0) {
            objectinstance_getpropertyinterned(slf, error_messageproperty, &msg);
        } else {
            msg=MORPHO_GETARG(args, 0);
        }

        morpho_usererror(v, MORPHO_GETCSTRING(tag), MORPHO_GETCSTRING(msg));
    }

    return MORPHO_NIL;
}

/** Print errors */
value Error_print(vm *v, int nargs, value *args) {
    object_print(MORPHO_SELF(args));

    return MORPHO_SELF(args);
}

MORPHO_BEGINCLASS(Error)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Error_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_THROW_METHOD, Error_throw, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Error_print, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void veneer_initialize(void) {
    /* Object */
    value objclass=builtin_addclass(OBJECT_CLASSNAME, MORPHO_GETCLASSDEFINITION(Object), MORPHO_NIL);
    morpho_setbaseclass(objclass);

    /* Dictionary */
    builtin_addfunction(DICTIONARY_CLASSNAME, dictionary_constructor, BUILTIN_FLAGSEMPTY);
    value dictionaryclass=builtin_addclass(DICTIONARY_CLASSNAME, MORPHO_GETCLASSDEFINITION(Dictionary), objclass);
    object_setveneerclass(OBJECT_DICTIONARY, dictionaryclass);

    /* Closure */
    value closureclass=builtin_addclass(CLOSURE_CLASSNAME, MORPHO_GETCLASSDEFINITION(Closure), objclass);
    object_setveneerclass(OBJECT_CLOSURE, closureclass);
    
    /* Function */
    value functionclass=builtin_addclass(FUNCTION_CLASSNAME, MORPHO_GETCLASSDEFINITION(Function), objclass);
    object_setveneerclass(OBJECT_FUNCTION, functionclass);
    
    /* Invocation */
    builtin_addfunction(INVOCATION_CLASSNAME, invocation_constructor, BUILTIN_FLAGSEMPTY);
    value invocationclass=builtin_addclass(INVOCATION_CLASSNAME, MORPHO_GETCLASSDEFINITION(Invocation), objclass);
    object_setveneerclass(OBJECT_INVOCATION, invocationclass);
    
    /* Error */
    builtin_addclass(ERROR_CLASSNAME, MORPHO_GETCLASSDEFINITION(Error), objclass);
    error_tagproperty=builtin_internsymbolascstring(ERROR_TAG_PROPERTY);
    error_messageproperty=builtin_internsymbolascstring(ERROR_MESSAGE_PROPERTY);

    morpho_defineerror(ENUMERATE_ARGS, ERROR_HALT, ENUMERATE_ARGS_MSG);
    morpho_defineerror(DICT_DCTKYNTFND, ERROR_HALT, DICT_DCTKYNTFND_MSG);
    morpho_defineerror(DICT_DCTSTARG, ERROR_HALT, DICT_DCTSTARG_MSG);
    morpho_defineerror(SETINDEX_ARGS, ERROR_HALT, SETINDEX_ARGS_MSG);
    morpho_defineerror(RESPONDSTO_ARG, ERROR_HALT, RESPONDSTO_ARG_MSG);
    morpho_defineerror(HAS_ARG, ERROR_HALT, HAS_ARG_MSG);
    morpho_defineerror(ISMEMBER_ARG, ERROR_HALT, ISMEMBER_ARG_MSG);
    morpho_defineerror(CLASS_INVK, ERROR_HALT, CLASS_INVK_MSG);
    morpho_defineerror(ERROR_ARGS, ERROR_HALT, ERROR_ARGS_MSG);
    
    morpho_defineerror(OBJECT_CANTCLONE, ERROR_HALT, OBJECT_CANTCLONE_MSG);
    morpho_defineerror(OBJECT_IMMUTABLE, ERROR_HALT, OBJECT_IMMUTABLE_MSG);
}
