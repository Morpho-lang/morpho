/** @file instance.c
 *  @author T J Atherton
 *
 *  @brief Implements objectinstance and the Object base class
 */

#include "morpho.h"
#include "classes.h"

/* **********************************************************************
 * objectinstance definitions
 * ********************************************************************** */

/** Instance object definitions */
void objectinstance_printfn(object *obj, void *v) {
#ifndef MORPHO_LOXCOMPATIBILITY
    morpho_printf(v, "<");
#endif
    morpho_printf(v, "%s", MORPHO_GETCSTRING(((objectinstance *) obj)->klass->name));
#ifndef MORPHO_LOXCOMPATIBILITY
    morpho_printf(v, ">");
#else
    morpho_printf(v, " instance");
#endif
}

void objectinstance_markfn(object *obj, void *v) {
    objectinstance *c = (objectinstance *) obj;
    morpho_markdictionary(v, &c->fields);
}

void objectinstance_freefn(object *obj) {
    objectinstance *instance = (objectinstance *) obj;
    dictionary_clear(&instance->fields);
}

size_t objectinstance_sizefn(object *obj) {
    return sizeof(objectinstance);
}

objecttypedefn objectinstancedefn = {
    .printfn=objectinstance_printfn,
    .markfn=objectinstance_markfn,
    .freefn=objectinstance_freefn,
    .sizefn=objectinstance_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/** Create an instance */
objectinstance *object_newinstance(objectclass *klass) {
    objectinstance *new= (objectinstance *) object_new(sizeof(objectinstance), OBJECT_INSTANCE);

    if (new) {
        new->klass=klass;
        dictionary_init(&new->fields);
    }

    return new;
}

/* **********************************************************************
 * objectinstance utility functions
 * ********************************************************************** */

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
 * Object veneer class
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

/** Find the object's class */
value Object_class(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    value out = MORPHO_NIL;

    objectclass *klass=morpho_lookupclass(self);
    if (klass) out = MORPHO_OBJECT(klass);
    
    return out;
}

/** Find the object's superclass */
value Object_super(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    
    objectclass *klass=morpho_lookupclass(self);
    
    return (klass && klass->superclass ? MORPHO_OBJECT(klass->superclass) : MORPHO_NIL);
}

/** Checks if an object responds to a method */
value Object_respondsto(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    
    objectclass *klass=morpho_lookupclass(self);

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

    objectclass *klass=morpho_lookupclass(self);
    
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
        morpho_printf(v, "@%s", (MORPHO_ISSTRING(klass->name) ? MORPHO_GETCSTRING(klass->name): "Object"));
#else
        morpho_printf(v, "%s", (MORPHO_ISSTRING(klass->name) ? MORPHO_GETCSTRING(klass->name): "Object"));
#endif
    } else if (MORPHO_ISINSTANCE(self)) {
        klass=MORPHO_GETINSTANCE(self)->klass;
#ifndef MORPHO_LOXCOMPATIBILITY
        if (klass) morpho_printf(v, "<%s>", (MORPHO_ISSTRING(klass->name) ? MORPHO_GETCSTRING(klass->name): "Object") );
#else
        if (klass) morpho_printf(v, "%s instance", (MORPHO_ISSTRING(klass->name) ? MORPHO_GETCSTRING(klass->name): "Object") );
#endif
    } else {
        morpho_printvalue(v, self);
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
MORPHO_METHOD(MORPHO_RESPONDSTO_METHOD, Object_respondsto, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_HAS_METHOD, Object_has, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_INVOKE_METHOD, Object_invoke, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Object_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, Object_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SERIALIZE_METHOD, Object_serialize, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Object_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

objecttype objectinstancetype;

void instance_initialize(void) {
    // Create instance object type
    objectinstancetype=object_addtype(&objectinstancedefn);
    
    // Create Object and set as base class
    value objclass=builtin_addclass(OBJECT_CLASSNAME, MORPHO_GETCLASSDEFINITION(Object), MORPHO_NIL);
    morpho_setbaseclass(objclass);
    
    // Object error messages
    morpho_defineerror(OBJECT_CANTCLONE, ERROR_HALT, OBJECT_CANTCLONE_MSG);
    morpho_defineerror(OBJECT_IMMUTABLE, ERROR_HALT, OBJECT_IMMUTABLE_MSG);
    
    morpho_defineerror(ENUMERATE_ARGS, ERROR_HALT, ENUMERATE_ARGS_MSG);
    morpho_defineerror(SETINDEX_ARGS, ERROR_HALT, SETINDEX_ARGS_MSG);
    morpho_defineerror(RESPONDSTO_ARG, ERROR_HALT, RESPONDSTO_ARG_MSG);
    morpho_defineerror(HAS_ARG, ERROR_HALT, HAS_ARG_MSG);
    morpho_defineerror(ISMEMBER_ARG, ERROR_HALT, ISMEMBER_ARG_MSG);
}
