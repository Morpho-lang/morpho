/** @file veneer.c
 *  @author T J Atherton
 *
 *  @brief Veneer classes over built in objects 
 */

#include "morpho.h"
#include "veneer.h"
#include "object.h"
#include "common.h"

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
    
    if (nargs==2 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISINSTANCE(self)) {
        dictionary_insert(&MORPHO_GETINSTANCE(self)->fields, MORPHO_GETARG(args, 0), MORPHO_GETARG(args, 1));
    } else morpho_runtimeerror(v, SETINDEX_ARGS);
    
    return MORPHO_NIL;
}

/** Find the object's class */
value Object_class(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    
    return MORPHO_OBJECT(MORPHO_GETINSTANCE(self)->klass);
}

/** Find the object's superclass */
value Object_super(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    objectclass *klass=MORPHO_GETINSTANCE(self)->klass;
    
    return (klass->superclass ? MORPHO_OBJECT(klass->superclass) : MORPHO_NIL);
}

/** Checks if an object responds to a method */
value Object_respondsto(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    objectclass *klass=MORPHO_GETINSTANCE(self)->klass;

    if (nargs==1 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        return MORPHO_BOOL(dictionary_get(&klass->methods, MORPHO_GETARG(args, 0), NULL));
    } else MORPHO_RAISE(v, RESPONDSTO_ARG);
    
    return MORPHO_FALSE;
}

/** Checks if an object has a property */
value Object_has(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);

    if (nargs==1 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        return MORPHO_BOOL(dictionary_get(&MORPHO_GETINSTANCE(self)->fields, MORPHO_GETARG(args, 0), NULL));
        
    } else MORPHO_RAISE(v, RESPONDSTO_ARG);
    
    return MORPHO_FALSE;
}

/** Invoke a method */
value Object_invoke(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    objectclass *klass=MORPHO_GETINSTANCE(self)->klass;
    value out=MORPHO_NIL;
    
    if (nargs>0 &&
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
 * String
 * ********************************************************************** */

/** Count number of characters in a string */
int string_countchars(objectstring *s) {
    int n=0;
    for (uint8_t *c = (uint8_t *) s->string; *c!='\0'; ) {
        c+=morpho_utf8numberofbytes(c);
        n++;
    }
    return n;
}

/** Get a pointer to the i'th character of a string */
char *string_index(objectstring *s, int i) {
    int n=0;
    for (uint8_t *c = (uint8_t *) s->string; *c!='\0'; ) {
        if (i==n) return (char *) c;
        c+=morpho_utf8numberofbytes(c);
        n++;
    }
    return NULL;
}

/** Constructor */
value string_constructor(vm *v, int nargs, value *args) {
    value out=morpho_concatenate(v, nargs, args+1);
    if (MORPHO_ISOBJECT(out)) morpho_bindobjects(v, 1, &out);
    return out; 
}

/** Find a string's length */
value String_count(vm *v, int nargs, value *args) {
    objectstring *slf = MORPHO_GETSTRING(MORPHO_SELF(args));
    
    return MORPHO_INTEGER(string_countchars(slf));
}

/** Prints a string */
value String_print(vm *v, int nargs, value *args) {
    morpho_printvalue(MORPHO_SELF(args));
    
    return MORPHO_SELF(args);
}

/** Clones a string */
value String_clone(vm *v, int nargs, value *args) {
    objectstring *slf = MORPHO_GETSTRING(MORPHO_SELF(args));
    value out = object_stringfromcstring(slf->string, slf->length);
    if (MORPHO_ISNIL(out)) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    morpho_bindobjects(v, 1, &out);
    return out;
}

/** Sets an index */
/*value String_setindex(vm *v, int nargs, value *args) {
    objectstring *slf = MORPHO_GETSTRING(MORPHO_SELF(args));
    
    morpho_runtimeerror(v, STRING_IMMTBL);
    
    if (nargs==2 &&
        MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 1))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        objectstring *set = MORPHO_GETSTRING(MORPHO_GETARG(args, 1));
        
        if (n>=0 && n<slf->length) {
            for (unsigned int i=0; i<set->length && n+i<slf->length; i++) {
                slf->stringdata[n+i]=set->stringdata[i];
            }
        } else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
    } else morpho_runtimeerror(v, SETINDEX_ARGS);
    
    return MORPHO_NIL;
}*/

/** Enumerate members of a string */
value String_enumerate(vm *v, int nargs, value *args) {
    objectstring *slf = MORPHO_GETSTRING(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        
        if (n<0) {
            out=MORPHO_INTEGER(string_countchars(slf));
        } else {
            char *c=string_index(slf, n);
            if (c) {
                out=object_stringfromcstring(c, morpho_utf8numberofbytes((uint8_t *) c));
                morpho_bindobjects(v, 1, &out);
            } else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        }
    } else MORPHO_RAISE(v, ENUMERATE_ARGS);
    
    return out;
}

value String_split(vm *v, int nargs, value *args) {
    objectstring *slf = MORPHO_GETSTRING(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (nargs==1 && MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        objectstring *split = MORPHO_GETSTRING(MORPHO_GETARG(args, 0));
        objectlist *new = object_newlist(0, NULL);
        
        if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return MORPHO_NIL; }
        
        char *last = slf->string;
        for (char *c = slf->string; *c!='\0'; c+=morpho_utf8numberofbytes((uint8_t *) c)) { // Loop over string
            for (char *s = split->string; *s!='\0';) { // Loop over split chars
                int nbytes = morpho_utf8numberofbytes((uint8_t *) s);
                if (strncmp(c, s, nbytes)==0) {
                    value newstring = object_stringfromcstring(last, c-last);
                    if (MORPHO_ISNIL(newstring)) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
                    list_append(new, newstring);
                    last=c+nbytes;
                }
                s+=nbytes;
            }
        }
        
        value newstring = object_stringfromcstring(last, slf->string+slf->length-last);
        if (MORPHO_ISNIL(newstring)) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        list_append(new, newstring);
        
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
        morpho_bindobjects(v, new->val.count, new->val.data);
    }
    
    return out;
}

MORPHO_BEGINCLASS(String)
MORPHO_METHOD(MORPHO_COUNT_METHOD, String_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, String_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, String_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, String_enumerate, BUILTIN_FLAGSEMPTY),
//MORPHO_METHOD(MORPHO_SETINDEX_METHOD, String_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, String_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(STRING_SPLIT_METHOD, String_split, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Array
 * ********************************************************************** */

/** Creates a new 1D array from a list of values */
objectarray *object_arrayfromvaluelist(unsigned int n, value *v) {
    objectarray *new = object_newarray(1, &n);
    
    if (new) {
        memcpy(new->data+1, v, sizeof(value)*n);
    }
    
    return new;
}

/** Creates a new 1D array from a list of varray_value */
objectarray *object_arrayfromvarrayvalue(varray_value *v) {
    return object_arrayfromvaluelist(v->count, v->data);
}

/** Creates a new array object with the dimensions given as a list of values */
objectarray *object_arrayfromvalueindices(unsigned int ndim, value *dim) {
    unsigned int indx[ndim];
    if (array_valuestoindices(ndim, dim, indx)) {
        return object_newarray(ndim, indx);
    }
    return NULL;
}

/** Clones an array. Does *not* clone the contents. */
objectarray *object_clonearray(objectarray *array) {
    objectarray *new = object_arrayfromvalueindices(array->dimensions, array->data);
    
    if (new) memcpy(new->data, array->data, sizeof(value)*(array->nelements+array->dimensions));
    
    return new;
}

/** Converts a list of indices into a list of unsigned ints
 *  @param ndim - number of dimensions
 *  @param indx - the indices to evaluate
 *  @param iout - the indices as integers
 *  @returns true on success, or false if an unexpected type was encountered */
inline bool array_valuestoindices(unsigned int ndim, value *indx, unsigned int *iout) {
    for (unsigned int i=0; i<ndim; i++) {
        if (MORPHO_ISINTEGER(indx[i])) {
            iout[i] = (unsigned int) MORPHO_GETINTEGERVALUE(indx[i]);
        } else if (MORPHO_ISFLOAT(indx[i])) {
            iout[i] = (unsigned int) MORPHO_GETFLOATVALUE(indx[i]);
        } else {
            return false;
        }
    }
    return true;
}

/** @brief Calculates the correct element from a set of array indices
 *  @param[in] array - the array
 *  @param[in] ndim   - number of dimensions
 *  @param[in] indx   - list of indices
 *  @param[out] ixout - the element number to use
 *  @returns true on success, false if indices are out of bounds */
inline bool array_indicestoelement(objectarray *array, unsigned int ndim, unsigned int *indx, unsigned int *ixout) {
    unsigned int ix=0, mul=1;
    
    for (unsigned int i=0; i<ndim; i++) {
        int dim = MORPHO_GETINTEGERVALUE(array->data[i]);
        if (indx[i]<dim) {
            ix+=mul*indx[i];
        } else return false;
        mul*=dim;
    }
    if (ixout) *ixout = ndim+ix;
    return true;
}

/** Gets an array element */
objectarrayerror array_getelement(objectarray *array, unsigned int ndim, value *indx, value *out) {
    if (ndim!=array->dimensions) return ARRAY_WRONGDIM;
    
    if (ndim==1 && MORPHO_ISINTEGER(*indx)) {
        int val = MORPHO_GETINTEGERVALUE(*indx);
        if (val>=MORPHO_GETINTEGERVALUE(array->data[0])) return ARRAY_OUTOFBOUNDS;
        *out=array->data[ndim+val];
        return ARRAY_OK;
    } else if (ndim==2 && MORPHO_ISINTEGER(indx[0]) && MORPHO_ISINTEGER(indx[1])) {
        int i1 = MORPHO_GETINTEGERVALUE(indx[0]);
        int i2 = MORPHO_GETINTEGERVALUE(indx[1]);
        int nrows = MORPHO_GETINTEGERVALUE(array->data[0]);
        if (i1<0 || i1>=nrows) return ARRAY_OUTOFBOUNDS;
        if (i2<0 || i2>=MORPHO_GETINTEGERVALUE(array->data[1])) return ARRAY_OUTOFBOUNDS;
        *out=array->data[ndim+i1+i2*nrows];
        return ARRAY_OK;
    }
    
    unsigned int ix[ndim], iel;
    if (array_valuestoindices(ndim, indx, ix)) {
        if (array_indicestoelement(array, ndim, ix, &iel)) {
            *out=array->data[iel];
            return ARRAY_OK;
        } else return ARRAY_OUTOFBOUNDS;
    }
    
    return ARRAY_NONNUMERICALINDX;
}

/** Sets an array element */
objectarrayerror array_setelement(objectarray *array, unsigned int ndim, value *indx, value set) {
    if (ndim!=array->dimensions) return ARRAY_WRONGDIM;
    
    if (ndim==1 && MORPHO_ISINTEGER(*indx)) {
        int val = MORPHO_GETINTEGERVALUE(*indx);
        if (val>=MORPHO_GETINTEGERVALUE(array->data[0])) return ARRAY_OUTOFBOUNDS;
        array->data[ndim+val]=set;

        return ARRAY_OK;
    } else if (ndim==2 && MORPHO_ISINTEGER(indx[0]) && MORPHO_ISINTEGER(indx[1])) {
        int i1 = MORPHO_GETINTEGERVALUE(indx[0]);
        int i2 = MORPHO_GETINTEGERVALUE(indx[1]);
        int nrows = MORPHO_GETINTEGERVALUE(array->data[0]);
        if (i1<0 || i1>=nrows) return ARRAY_OUTOFBOUNDS;
        if (i2<0 || i2>=MORPHO_GETINTEGERVALUE(array->data[1])) return ARRAY_OUTOFBOUNDS;
        array->data[ndim+i1+i2*nrows]=set;
        return ARRAY_OK;
    }
    
    unsigned int ix[ndim], iel;
    if (array_valuestoindices(ndim, indx, ix)) {
        if (array_indicestoelement(array, ndim, ix, &iel)) {
            array->data[iel]=set;
            return ARRAY_OK;
        } else return ARRAY_OUTOFBOUNDS;
    }
    
    return ARRAY_NONNUMERICALINDX;
}

/** Converts an array error into an error code */
errorid array_error(objectarrayerror err) {
    switch (err) {
        case ARRAY_OUTOFBOUNDS: return VM_OUTOFBOUNDS;
        case ARRAY_WRONGDIM: return VM_ARRAYWRONGDIM;
        case ARRAY_NONNUMERICALINDX: return VM_NONNUMINDX;
        case ARRAY_OK: UNREACHABLE("array_error called incorrectly.");
    }
    UNREACHABLE("Unhandled array error.");
    return VM_OUTOFBOUNDS;
}


/** Create an array */
value array_constructor(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    unsigned int dim[nargs];
    
    if (nargs==0) { morpho_runtimeerror(v, ARRAY_ARGS); return MORPHO_NIL; }
    for (unsigned int i=0; i<nargs; i++) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, i))) dim[i]=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, i));
        else { morpho_runtimeerror(v, ARRAY_ARGS); return MORPHO_NIL; }
    }
        
    objectarray *new = object_newarray(nargs, dim);
        
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}


/** Gets the array element with given indices */
value Array_getindex(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectarrayerror err=array_getelement(MORPHO_GETARRAY(MORPHO_SELF(args)), nargs, &MORPHO_GETARG(args, 0), &out);
    if (err!=ARRAY_OK) MORPHO_RAISE(v, array_error(err) );
    
    return out;
}

/** Sets the matrix element with given indices */
value Array_setindex(vm *v, int nargs, value *args) {
    objectarrayerror err=array_setelement(MORPHO_GETARRAY(MORPHO_SELF(args)), nargs-1, &MORPHO_GETARG(args, 0), MORPHO_GETARG(args, nargs-1));
    if (err!=ARRAY_OK) MORPHO_RAISE(v, array_error(err) );
    
    return MORPHO_NIL;
}

/** Print an array */
value Array_print(vm *v, int nargs, value *args) {
    printf("<%s>", ARRAY_CLASSNAME);
    
    return MORPHO_NIL;
}

/** Find an array's size */
value Array_count(vm *v, int nargs, value *args) {
    objectarray *slf = MORPHO_GETARRAY(MORPHO_SELF(args));
    
    return MORPHO_INTEGER(slf->nelements);
}

/** Array dimensions */
value Array_dimensions(vm *v, int nargs, value *args) {
    objectarray *a=MORPHO_GETARRAY(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    objectlist *new=object_newlist(a->dimensions, a->data);
    
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

/** Enumerate members of an array */
value Array_enumerate(vm *v, int nargs, value *args) {
    objectarray *slf = MORPHO_GETARRAY(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
       
        if (n<0) {
            out=MORPHO_INTEGER(slf->nelements);
        } else if (n<slf->nelements) {
            out=slf->data[n+slf->dimensions];
        } else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
    } else MORPHO_RAISE(v, ENUMERATE_ARGS);
    
    return out;
}

/** Clone an array */
value Array_clone(vm *v, int nargs, value *args) {
    objectarray *slf = MORPHO_GETARRAY(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    objectarray *new = object_clonearray(slf);
    if (new) {
        out = MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

MORPHO_BEGINCLASS(Array)
MORPHO_METHOD(MORPHO_PRINT_METHOD, Array_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Array_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(ARRAY_DIMENSIONS_METHOD, Array_dimensions, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, Array_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, Array_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, Array_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Array_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * List
 * ********************************************************************** */

/** Resizes a list */
bool list_resize(objectlist *list, int size) {
    return varray_valueresize(&list->val, size);
}

/** Appends an item to a list */
void list_append(objectlist *list, value v) {
    varray_valuewrite(&list->val, v);
}

/** Appends an item to a list */
unsigned int list_length(objectlist *list) {
    return list->val.count;
}

/** Removes an element from a list
 * @param[in] list a list object
 * @param[in] indx position to insert
 * @param[in] nval number of values to insert
 * @param[in] vals the entries to insert
 * @returns true on success */
bool list_insert(objectlist *list, int indx, int nval, value *vals) {
    int i = indx;
    while (i<0) i+=list->val.count+1;
    if (i>list->val.count) return false;
    if (nval>list->val.capacity-list->val.count) if (!list_resize(list, list->val.count+nval)) return false;
    
    memmove(list->val.data+i+nval, list->val.data+i, sizeof(value)*(list->val.count-i));
    memcpy(list->val.data+i, vals, sizeof(value)*nval);

    list->val.count+=nval;
    
    return true;
}

/** Removes an element from a list
 * @param[in] list a list object
 * @param[in] val the entry to remove
 * @returns true on success */
bool list_remove(objectlist *list, value val) {
    /* Find the element */
    for (unsigned int i=0; i<list->val.count; i++) {
        if (MORPHO_ISEQUAL(list->val.data[i], val)) { /* Remove it if we're not at the end of the list */
            if (i<list->val.count-1) memmove(list->val.data+i, list->val.data+i+1, sizeof(value)*(list->val.count-i-1));
            list->val.count--;
            return true;
        }
    }
    
    return false;
}

/** Gets an element from a list
 * @param[in] list a list object
 * @param[in] i the index (may be negative)
 * @param[in] out filled out on exit if index is in bounds
 * @returns true on success */
bool list_getelement(objectlist *list, int i, value *out) {
    if (!(i>=-(int) list->val.count && i<(int) list->val.count)) return false;
    if (i>=0) *out=list->val.data[i];
    else *out=list->val.data[list->val.count+i];
    return true;
}

/** Sort function for list_sort */
int list_sortfunction(const void *a, const void *b) {
    return -morpho_comparevalue(*(value *) a, *(value *) b);
}

/** Sort the contents of a list */
void list_sort(objectlist *list) {
    qsort(list->val.data, list->val.count, sizeof(value), list_sortfunction);
}

static vm *list_sortwithfn_vm;
static value list_sortwithfn_fn;
static bool list_sortwithfn_err;

/** Sort function for list_sort */
int list_sortfunctionwfn(const void *a, const void *b) {
    value args[2] = {*(value *) a, *(value *) b};
    value ret;
    
    if (morpho_call(list_sortwithfn_vm, list_sortwithfn_fn, 2, args, &ret)) {
        if (MORPHO_ISINTEGER(ret)) return MORPHO_GETINTEGERVALUE(ret);
        if (MORPHO_ISFLOAT(ret)) return morpho_comparevalue(MORPHO_FLOAT(0), ret);
    }
    
    list_sortwithfn_err=true;
    return 0;
}

/** Sort the contents of a list */
bool list_sortwithfn(vm *v, value fn, objectlist *list) {
    list_sortwithfn_vm=v;
    list_sortwithfn_fn=fn;
    list_sortwithfn_err=false;
    qsort(list->val.data, list->val.count, sizeof(value), list_sortfunctionwfn);
    return !list_sortwithfn_err;
}

/** Sort function for list_order */
typedef struct {
    unsigned int indx;
    value val;
} listorderstruct;

/** Sort function for list_order */
int list_orderfunction(const void *a, const void *b) {
    return -morpho_comparevalue(((listorderstruct *) a)->val, ((listorderstruct *) b)->val);
}

/* Returns a list of indices giving the ordering of a list */
objectlist *list_order(objectlist *list) {
    listorderstruct *order = MORPHO_MALLOC(list->val.count*sizeof(listorderstruct));
    objectlist *new = NULL;
    
    if (order) {
        for (unsigned int i=0; i<list->val.count; i++) {
            order[i].indx=i;
            order[i].val=list->val.data[i];
        }
        qsort(order, list->val.count, sizeof(listorderstruct), list_orderfunction);
        
        new=object_newlist(list->val.count, NULL);
        if (new) {
            for (unsigned int i=0; i<list->val.count; i++) {
                new->val.data[i]=MORPHO_INTEGER(order[i].indx);
            }
            new->val.count=list->val.count;
        }
        
        MORPHO_FREE(order);
    }
    return new;
}

/** Tests if a value is a member of a list */
bool list_ismember(objectlist *list, value v) {
    for (unsigned int i=0; i<list->val.count; i++) {
        if (MORPHO_ISEQUAL(list->val.data[i], v)) return true;
    }
    return false; 
}

/** Clones a list */
objectlist *list_clone(objectlist *list) {
    return object_newlist(list->val.count, list->val.data);
}

/** Concatenates two lists */
objectlist *list_concatenate(objectlist *a, objectlist *b) {
    objectlist *new=object_newlist(a->val.count+b->val.count, NULL);
    
    if (new) {
        memcpy(new->val.data, a->val.data, sizeof(value)*a->val.count);
        memcpy(new->val.data+a->val.count, b->val.data, sizeof(value)*b->val.count);
        new->val.count=a->val.count+b->val.count;
    }
    
    return new;
}

/** Loop function for enumerable initializers */
static bool list_enumerableinitializer(vm *v, indx i, value val, void *ref) {
    objectlist *list = (objectlist *) ref;
    list_append(list, val);
    return true;
}

/** Create a list */
value list_constructor(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    value init=MORPHO_NIL;
    objectlist *new=NULL;
    
    if (nargs==1 && MORPHO_ISRANGE(MORPHO_GETARG(args, 0))) {
        init = MORPHO_GETARG(args, 0);
        new = object_newlist(0, NULL);
    } else new = object_newlist(nargs, args+1);
    
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
        
        if (!MORPHO_ISNIL(init)) {
            builtin_enumerateloop(v, init, list_enumerableinitializer, new);
        }
    }
    
    return out;
}

/** Get an element */
value List_append(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    
    unsigned int capacity = slf->val.capacity;
    
    varray_valueadd(&slf->val, args+1, nargs);
    
    if (slf->val.capacity!=capacity) morpho_resizeobject(v, (object *) slf, capacity*sizeof(value), slf->val.capacity*sizeof(value));
    
    return MORPHO_SELF(args);
}

/** Get an element */
value List_pop(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (slf->val.count>0) {
        if (nargs>0 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            int indx = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
            out=slf->val.data[indx];
            memmove(slf->val.data+indx, slf->val.data+indx+1, sizeof(value)*(slf->val.count-indx-1));
        } else {
            out=slf->val.data[slf->val.count-1];
        }
        slf->val.count--;
    }
    
    return out;
}

/** inserts an element */
value List_insert(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    
    if (nargs>=2) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            int indx = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
            if (!list_insert(slf, indx, nargs-1, &MORPHO_GETARG(args, 1))) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        }
    } else morpho_runtimeerror(v, VM_INVALIDARGS, 2, nargs);
    
    return MORPHO_NIL;
}

/** Get an element */
value List_remove(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    
    if (nargs==1) {
        if (!list_remove(slf, MORPHO_GETARG(args, 0))) morpho_runtimeerror(v, LIST_ENTRYNTFND);
    } else morpho_runtimeerror(v, VM_INVALIDARGS, 1, nargs);
    
    return MORPHO_NIL;
}

/** Get an element */
value List_getindex(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (nargs==1) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
            
            if (!list_getelement(slf, i, &out)) {
                morpho_runtimeerror(v, VM_OUTOFBOUNDS);
            }
        } else {
	        morpho_runtimeerror(v, VM_GETINDEXARGS);
        }
    }
    
    return out;
}

/** Get an element */
value List_setindex(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    
    if (nargs==2) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
            if (i<slf->val.count) slf->val.data[i]=MORPHO_GETARG(args, 1);
            else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        } else morpho_runtimeerror(v, SETINDEX_ARGS);
    } else morpho_runtimeerror(v, SETINDEX_ARGS);
    
    return MORPHO_SELF(args);
}

/** Get number of entries */
value List_count(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    
    return MORPHO_INTEGER(slf->val.count);
}

/** Print a list */
value List_print(vm *v, int nargs, value *args) {
    objectlist *lst=MORPHO_GETLIST(MORPHO_SELF(args));
    
    printf("[ ");
    for (unsigned int i=0; i<lst->val.count; i++) {
        morpho_printvalue(lst->val.data[i]);
        if (i<lst->val.count-1) printf(", ");
    }
    printf(" ]");
    
    return MORPHO_NIL;
}

/** Convert a list to a string */
value List_tostring(vm *v, int nargs, value *args) {
    objectlist *lst=MORPHO_GETLIST(MORPHO_SELF(args));
    value out = MORPHO_NIL;
    
    varray_char buffer;
    varray_charinit(&buffer);
    
    varray_charadd(&buffer, "[ ", 2);
    for (unsigned int i=0; i<lst->val.count; i++) {
        morpho_printtobuffer(v, lst->val.data[i], &buffer);
        if (i<lst->val.count-1) varray_charadd(&buffer, ", ", 2);
    }
    varray_charadd(&buffer, " ]", 2);
    
    out = object_stringfromvarraychar(&buffer);
    if (MORPHO_ISSTRING(out)) {
        morpho_bindobjects(v, 1, &out);
    }
    varray_charclear(&buffer);
    
    return out;
}
    
/** Enumerate members of a list */
value List_enumerate(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        
        if (n<0) {
            out=MORPHO_INTEGER(slf->val.count);
        } else if (n<slf->val.count) {
            return slf->val.data[n];
        } else {
            morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        }
    } else MORPHO_RAISE(v, ENUMERATE_ARGS);
    
    return out;
}

/** Enumerate members of a list */
value List_sort(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    
    if (nargs==0) {
        list_sort(slf);
    } else if (nargs==1 && MORPHO_ISCALLABLE(MORPHO_GETARG(args, 0))) {
        if (!list_sortwithfn(v, MORPHO_GETARG(args, 0), slf)) {
            morpho_runtimeerror(v, LIST_SRTFN);
        }
    }
    
    return MORPHO_NIL;
}

/** Enumerate members of a list */
value List_order(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    objectlist *new=list_order(slf);
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

/** Tests if a list has a value as a member */
value List_ismember(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    
    if (nargs==1) {
        return MORPHO_BOOL(list_ismember(slf, MORPHO_GETARG(args, 0)));
    } else morpho_runtimeerror(v, VM_INVALIDARGS, 1, nargs);
    
    return MORPHO_NIL;
}

/** Clones a list */
value List_clone(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    objectlist *new = list_clone(slf);
    if (!new) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    value out = MORPHO_OBJECT(new);
    morpho_bindobjects(v, 1, &out);
    return out;
}

/** Joins two lists together  */
value List_add(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    value out = MORPHO_NIL;
    
    if (nargs==1 && MORPHO_ISLIST(MORPHO_GETARG(args, 0))) {
        objectlist *operand = MORPHO_GETLIST(MORPHO_GETARG(args, 0));
        objectlist *new = list_concatenate(slf, operand);
        
        if (new) {
            out = MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        }
        
    } else morpho_runtimeerror(v, LIST_ADDARGS);
    
    return out;
}

MORPHO_BEGINCLASS(List)
MORPHO_METHOD(MORPHO_APPEND_METHOD, List_append, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_REMOVE_METHOD, List_remove, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_INSERT_METHOD, List_insert, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_POP_METHOD, List_pop, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, List_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, List_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, List_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_TOSTRING_METHOD, List_tostring, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, List_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, List_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, List_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADD_METHOD, List_add, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_SORT_METHOD, List_sort, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_ORDER_METHOD, List_order, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_ISMEMBER_METHOD, List_ismember, BUILTIN_FLAGSEMPTY)
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

/** Sets a dictionary entry */
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

/** Gets a dictionary entry */
value Dictionary_setindex(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));
    
    if (nargs==2) {
        unsigned int capacity = slf->dict.capacity;
        
        dictionary_insert(&slf->dict, MORPHO_GETARG(args, 0), MORPHO_GETARG(args, 1));
        
        if (slf->dict.capacity!=capacity) morpho_resizeobject(v, (object *) slf, capacity*sizeof(value), slf->dict.capacity*sizeof(value));
    } else morpho_runtimeerror(v, SETINDEX_ARGS);
    
    return MORPHO_NIL;
}

/** Sets a dictionary entry */
value Dictionary_contains(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));
    value out=MORPHO_FALSE;
    
    if (nargs==1) {
        if (dictionary_get(&slf->dict, MORPHO_GETARG(args, 0), &out)) out=MORPHO_TRUE;
    }
    
    return out;
}

/** Prints a dictionary */
value Dictionary_print(vm *v, int nargs, value *args) {
    objectdictionary *slf = MORPHO_GETDICTIONARY(MORPHO_SELF(args));
    
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
MORPHO_METHOD(DICTIONARY_CONTAINS_METHOD, Dictionary_contains, BUILTIN_FLAGSEMPTY),
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

/* Range */

/** Calculate the number of steps in a range */
int range_count(objectrange *range) {
    int out=0;
    if (MORPHO_ISFLOAT(range->start)) {
        double diff=MORPHO_GETFLOATVALUE(range->end)-MORPHO_GETFLOATVALUE(range->start);
        double stp=(MORPHO_ISNIL(range->step) ? 1 : MORPHO_GETFLOATVALUE(range->step));
        out=diff/stp;
    } else {
        int diff=MORPHO_GETINTEGERVALUE(range->end)-MORPHO_GETINTEGERVALUE(range->start);
        int stp=(MORPHO_ISNIL(range->step) ? 1 : MORPHO_GETINTEGERVALUE(range->step));
        out=diff/stp;
    }
    if (out>=0) out+=1;
    else out=0;
    
    return out;
}

/** Find the ith value of a range object */
value range_iterate(objectrange *range, unsigned int i) {
    if (MORPHO_ISFLOAT(range->start)) {
        return MORPHO_FLOAT( MORPHO_GETFLOATVALUE(range->start) +
                            i*(MORPHO_ISNIL(range->step) ? 1.0 : MORPHO_GETFLOATVALUE(range->step)));
    } else {
        return MORPHO_INTEGER( MORPHO_GETINTEGERVALUE(range->start) +
                            i*(MORPHO_ISNIL(range->step) ? 1 : MORPHO_GETINTEGERVALUE(range->step)));
    }
}

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

/** Constructor function for ranges */
value range_constructor(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectrange *new=NULL;
    
    /* Check args are numerical */
    for (unsigned int i=0; i<nargs; i++) {
        if (!(MORPHO_ISINTEGER(MORPHO_GETARG(args, i)) || MORPHO_ISFLOAT(MORPHO_GETARG(args, i)))) {
            MORPHO_RAISE(v, RANGE_ARGS);
        }
    }
    
    if (nargs==2) {
        new=object_newrange(MORPHO_GETARG(args, 0), MORPHO_GETARG(args, 1), MORPHO_NIL);
    } else if (nargs==3) {
        new=object_newrange(MORPHO_GETARG(args, 0), MORPHO_GETARG(args, 1), MORPHO_GETARG(args, 2));
    } else MORPHO_RAISE(v, RANGE_ARGS);
    
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

/** Print ranges */
value Range_getindex(vm *v, int nargs, value *args) {
    objectrange *slf = MORPHO_GETRANGE(MORPHO_SELF(args));
    
    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        
        if (n<slf->nsteps) return range_iterate(slf, n);
        else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
    }
    
    return MORPHO_SELF(args);
}

/** Print ranges */
value Range_print(vm *v, int nargs, value *args) {
    object_print(MORPHO_SELF(args));
    
    return MORPHO_SELF(args);
}

/** Enumerate members of a range */
value Range_enumerate(vm *v, int nargs, value *args) {
    objectrange *slf = MORPHO_GETRANGE(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        
        if (n<0) return MORPHO_INTEGER(slf->nsteps);
        else return range_iterate(slf, n);
    } else MORPHO_RAISE(v, ENUMERATE_ARGS);
    
    return out;
}

/** Count number of items in a range */
value Range_count(vm *v, int nargs, value *args) {
    objectrange *slf = MORPHO_GETRANGE(MORPHO_SELF(args));
    
    return MORPHO_INTEGER(slf->nsteps);
}

/** Clones a range */
value Range_clone(vm *v, int nargs, value *args) {
    objectrange *slf = MORPHO_GETRANGE(MORPHO_SELF(args));
    objectrange *new = object_newrange(slf->start, slf->end, slf->step);
    if (!new) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    value out = MORPHO_OBJECT(new);
    morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(Range)
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, Range_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Range_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, Range_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Range_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Range_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* Error */

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
        objectinstance_getproperty(slf, error_tagproperty, &tag);
        if (nargs==0) {
            objectinstance_getproperty(slf, error_messageproperty, &msg);
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
    
    /* String */
    builtin_addfunction(STRING_CLASSNAME, string_constructor, BUILTIN_FLAGSEMPTY);
    value stringclass=builtin_addclass(STRING_CLASSNAME, MORPHO_GETCLASSDEFINITION(String), MORPHO_NIL);
    builtin_setveneerclass(OBJECT_STRING, stringclass);

    /* Array */
    builtin_addfunction(ARRAY_CLASSNAME, array_constructor, BUILTIN_FLAGSEMPTY);
    value arrayclass=builtin_addclass(ARRAY_CLASSNAME, MORPHO_GETCLASSDEFINITION(Array), MORPHO_NIL);
    builtin_setveneerclass(OBJECT_ARRAY, arrayclass);
    
    /* List */
    builtin_addfunction(LIST_CLASSNAME, list_constructor, BUILTIN_FLAGSEMPTY);
    value listclass=builtin_addclass(LIST_CLASSNAME, MORPHO_GETCLASSDEFINITION(List), MORPHO_NIL);
    builtin_setveneerclass(OBJECT_LIST, listclass);
    
    /* Dictionary */
    builtin_addfunction(DICTIONARY_CLASSNAME, dictionary_constructor, BUILTIN_FLAGSEMPTY);
    value dictionaryclass=builtin_addclass(DICTIONARY_CLASSNAME, MORPHO_GETCLASSDEFINITION(Dictionary), MORPHO_NIL);
    builtin_setveneerclass(OBJECT_DICTIONARY, dictionaryclass);
    
    /* Range */
    builtin_addfunction(RANGE_CLASSNAME, range_constructor, BUILTIN_FLAGSEMPTY);
    value rangeclass=builtin_addclass(RANGE_CLASSNAME, MORPHO_GETCLASSDEFINITION(Range), MORPHO_NIL);
    builtin_setveneerclass(OBJECT_RANGE, rangeclass);
    
    /* Error */
    builtin_addclass(ERROR_CLASSNAME, MORPHO_GETCLASSDEFINITION(Error), MORPHO_NIL);
    error_tagproperty=builtin_internsymbolascstring(ERROR_TAG_PROPERTY);
    error_messageproperty=builtin_internsymbolascstring(ERROR_MESSAGE_PROPERTY);
    
    morpho_defineerror(ARRAY_ARGS, ERROR_HALT, ARRAY_ARGS_MSG);
    morpho_defineerror(STRING_IMMTBL, ERROR_HALT, STRING_IMMTBL_MSG);
    morpho_defineerror(RANGE_ARGS, ERROR_HALT, RANGE_ARGS_MSG);
    morpho_defineerror(ENUMERATE_ARGS, ERROR_HALT, ENUMERATE_ARGS_MSG);
    morpho_defineerror(DICT_DCTKYNTFND, ERROR_HALT, DICT_DCTKYNTFND_MSG);
    morpho_defineerror(DICT_DCTSTARG, ERROR_HALT, DICT_DCTSTARG_MSG);
    morpho_defineerror(SETINDEX_ARGS, ERROR_HALT, SETINDEX_ARGS_MSG);
    morpho_defineerror(RESPONDSTO_ARG, ERROR_HALT, RESPONDSTO_ARG_MSG);
    morpho_defineerror(ISMEMBER_ARG, ERROR_HALT, ISMEMBER_ARG_MSG);
    morpho_defineerror(CLASS_INVK, ERROR_HALT, CLASS_INVK_MSG);
    morpho_defineerror(LIST_ENTRYNTFND, ERROR_HALT, LIST_ENTRYNTFND_MSG);
    morpho_defineerror(LIST_ADDARGS, ERROR_HALT, LIST_ADDARGS_MSG);
    morpho_defineerror(LIST_SRTFN, ERROR_HALT, LIST_SRTFN_MSG);
    morpho_defineerror(ERROR_ARGS, ERROR_HALT, ERROR_ARGS_MSG);
}
