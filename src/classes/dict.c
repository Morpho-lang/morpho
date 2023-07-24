/** @file dict.c
 *  @author T J Atherton
 *
 *  @brief Defines dictionary object type and Dictionary veneer class
 */

#include "morpho.h"
#include "classes.h"

/* **********************************************************************
 * objectdictionary definitions
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

/* **********************************************************************
 * objectdictionary utility functions
 * ********************************************************************** */

/** Extracts the dictionary from an objectdictionary. */
dictionary *object_dictionary(objectdictionary *dict) {
    return &dict->dict;
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

/* **********************************************************************
 * Dictionary veneer class
 * ********************************************************************** */

/** Dictionary constructor function */
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

/** Clears a Dictionary */
value Dictionary_clear(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));
    
    dictionary_clear(&slf->dict);
    
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
 * Initialization and finalization
 * ********************************************************************** */

objecttype objectdictionarytype;

void dict_initialize(void) {
    // Create dictionary object type
    objectdictionarytype=object_addtype(&objectdictionarydefn);
    
    // Locate the Object class to use as the parent class of Dictionary
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // Dictionary constructor function
    builtin_addfunction(DICTIONARY_CLASSNAME, dictionary_constructor, BUILTIN_FLAGSEMPTY);
    
    // Create dictionary veneer class
    value dictionaryclass=builtin_addclass(DICTIONARY_CLASSNAME, MORPHO_GETCLASSDEFINITION(Dictionary), objclass);
    object_setveneerclass(OBJECT_DICTIONARY, dictionaryclass);
    
    // Dictionary error messages
    morpho_defineerror(DICT_DCTKYNTFND, ERROR_HALT, DICT_DCTKYNTFND_MSG);
    morpho_defineerror(DICT_DCTSTARG, ERROR_HALT, DICT_DCTSTARG_MSG);
}

void dict_finalize(void) {
}
