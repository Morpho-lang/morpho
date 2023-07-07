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
 * String
 * ********************************************************************** */

/** Convert a string to a number */
bool string_tonumber(objectstring *string, value *out) {
    bool minus=false;
    lexer l;
    token tok;
    error err;
    lex_init(&l, string->string, 0);

    if (lex(&l, &tok, &err)) {
        if (tok.type==TOKEN_MINUS) { // Check for leading minus
            minus=true;
            if (!lex(&l, &tok, &err)) return false;
        } else if (tok.type==TOKEN_PLUS) { // or plus
            if (!lex(&l, &tok, &err)) return false;
        }

        if (tok.type==TOKEN_INTEGER) {
            long i = strtol(tok.start, NULL, 10);
            if (minus) i=-i;
            *out = MORPHO_INTEGER((int) i);
            return true;
        } else if (tok.type==TOKEN_NUMBER) {
            double f = strtod(tok.start, NULL);
            if (minus) f=-f;
            *out = MORPHO_FLOAT(f);
            return true;
        }
    }

    return false;
}

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

value String_isnumber(vm *v, int nargs, value *args) {
    objectstring *slf = MORPHO_GETSTRING(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (string_tonumber(slf, &out)) return MORPHO_TRUE;

    return MORPHO_FALSE;
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
        list_append(new, out);
        morpho_bindobjects(v, new->val.count, new->val.data);
        new->val.count-=1;
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
MORPHO_METHOD(STRING_ISNUMBER_METHOD, String_isnumber, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(STRING_SPLIT_METHOD, String_split, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Array
 * ********************************************************************** */

/** Converts a list of values to a list of integers */
inline bool array_valuelisttoindices(unsigned int ndim, value *in, unsigned int *out) {

    for (unsigned int i=0; i<ndim; i++) {
        if (MORPHO_ISINTEGER(in[i])) out[i]=MORPHO_GETINTEGERVALUE(in[i]);
        else if(MORPHO_ISFLOAT(in[i])) out[i]=round(MORPHO_GETFLOATVALUE(in[i]));
        else return false;
    }

    return true;
}

/** Creates a new 1D array from a list of values */
objectarray *object_arrayfromvaluelist(unsigned int n, value *v) {
    objectarray *new = object_newarray(1, &n);

    if (new) memcpy(new->values, v, sizeof(value)*n);

    return new;
}

/** Creates a new 1D array from a list of varray_value */
objectarray *object_arrayfromvarrayvalue(varray_value *v) {
    return object_arrayfromvaluelist(v->count, v->data);
}

/** Creates a new array object with the dimensions given as a list of values */
objectarray *object_arrayfromvalueindices(unsigned int ndim, value *dim) {
    unsigned int indx[ndim];
    if (array_valuelisttoindices(ndim, dim, indx)) {
        return object_newarray(ndim, indx);
    }
    return NULL;
}

/** Clones an array. Does *not* clone the contents. */
objectarray *object_clonearray(objectarray *array) {
    objectarray *new = object_arrayfromvalueindices(array->ndim, array->data);

    if (new) memcpy(new->data, array->data, sizeof(value)*(array->nelements+2*array->ndim));

    return new;
}

/** Recursively print a slice of an array */
bool array_print_recurse(vm *v, objectarray *a, unsigned int *indx, unsigned int dim, varray_char *out) {
    unsigned int bnd = MORPHO_GETINTEGERVALUE(a->dimensions[dim]);
    value val=MORPHO_NIL;

    varray_charadd(out, "[ ", 2);
    for (indx[dim]=0; indx[dim]<bnd; indx[dim]++) {
        if (dim==a->ndim-1) { // Print if innermost element
            if (array_getelement(a, a->ndim, indx, &val)==ARRAY_OK) {
                morpho_printtobuffer(v, val, out);
            } else return false;
        } else if (!array_print_recurse(v, a, indx, dim+1, out)) return false; // Otherwise recurse

        if (indx[dim]<bnd-1) { // Separators between items
            varray_charadd(out, ", ", 2);
        }
    }
    varray_charadd(out, " ]", 2);

    return true;
}

/* Print the contents of an array */
void array_print(vm *v, objectarray *a) {
    varray_char out;
    varray_charinit(&out);

    unsigned int indx[a->ndim];
    if (array_print_recurse(v, a, indx, 0, &out)) {
        varray_charwrite(&out, '\0'); // Ensure zero terminated
        printf("%s", out.data);
    }

    varray_charclear(&out);
}

/** Converts an array error into an error code */
errorid array_error(objectarrayerror err) {
    switch (err) {
        case ARRAY_OUTOFBOUNDS: return VM_OUTOFBOUNDS;
        case ARRAY_WRONGDIM: return VM_ARRAYWRONGDIM;
		case ARRAY_NONINTINDX: return VM_NONNUMINDX;
        case ARRAY_OK: UNREACHABLE("array_error called incorrectly.");
    }
    UNREACHABLE("Unhandled array error.");
    return VM_OUTOFBOUNDS;
}
/** Converts an array error into an matrix error code for use in slices*/
errorid array_to_matrix_error(objectarrayerror err) {
    switch (err) {
        case ARRAY_OUTOFBOUNDS: return MATRIX_INDICESOUTSIDEBOUNDS;
        case ARRAY_WRONGDIM: return MATRIX_INVLDNUMINDICES;
		case ARRAY_NONINTINDX: return MATRIX_INVLDINDICES;
        case ARRAY_OK: UNREACHABLE("array_to_matrix_error called incorrectly.");
    }
    UNREACHABLE("Unhandled array error.");
    return VM_OUTOFBOUNDS;
}

/** Converts an array error into an list error code for use in slices*/
errorid array_to_list_error(objectarrayerror err) {
    switch (err) {
        case ARRAY_OUTOFBOUNDS: return VM_OUTOFBOUNDS;
        case ARRAY_WRONGDIM: return LIST_NUMARGS;
		case ARRAY_NONINTINDX: return LIST_ARGS;
        case ARRAY_OK: UNREACHABLE("array_to_list_error called incorrectly.");
    }
    UNREACHABLE("Unhandled array error.");
    return VM_OUTOFBOUNDS;
}

/** Gets an array element */
objectarrayerror array_getelement(objectarray *a, unsigned int ndim, unsigned int *indx, value *out) {
    unsigned int k=0;

    if (ndim!=a->ndim) return ARRAY_WRONGDIM;

    for (unsigned int i=0; i<ndim; i++) {
        if (indx[i]>=MORPHO_GETINTEGERVALUE(a->dimensions[i])) return ARRAY_OUTOFBOUNDS;
        k+=indx[i]*MORPHO_GETINTEGERVALUE(a->multipliers[i]);
    }

    *out = a->values[k];
    return ARRAY_OK;
}

/** Creates a slice from a slicable object A.
 * @param[in] a - the sliceable object (array, list, matrix, etc..).
 * @param[in] dimFcn - a function that checks if the number of indecies is compatabile with the slicable object.
 * @param[in] constuctor - a function that create the a new object of the type of a.
 * @param[in] copy - a function that can copy information from a to out.
 * @param[in] ndim - the number of dimensions being indexed.
 * @param[in] slices - a set of indices that can be lists ranges or ints.
 * @param[out] out - returns the requested slice of a.
*/
objectarrayerror getslice(value *a, bool dimFcn(value *,unsigned int),\
						  void constuctor(unsigned int *,unsigned int,value *),\
						  objectarrayerror copy(value * ,value *, unsigned int, unsigned int *,unsigned int *),\
						  unsigned int ndim, value *slices, value *out){
	//dimension checking
	if (!(*dimFcn)(a,ndim)) return ARRAY_WRONGDIM;

	unsigned int slicesize[ndim];
	for (unsigned int i=0; i<ndim; i++) {
		if (MORPHO_ISINTEGER(slices[i])||MORPHO_ISFLOAT(slices[i])) {// if this is an number
			slicesize[i] = 1; // it only has one element
		} else if (MORPHO_ISLIST(slices[i])) { // if this is a list
			objectlist * s = MORPHO_GETLIST(slices[i]);
			slicesize[i] = s->val.count; // get the number of elements
		} else if (MORPHO_ISRANGE(slices[i])) { //if its a range
			objectrange * s = MORPHO_GETRANGE(slices[i]);
			slicesize[i] = range_count(s);
		} else return ARRAY_NONINTINDX; // by returning array a VM_NONNUMIDX will be thrown
	}

	// initalize out with the right size
	(*constuctor)(slicesize,ndim,out);

	// fill it out recurivly
	unsigned int indx[ndim];
	unsigned int newindx[ndim];
	return setslicerecursive(a, out, copy, ndim, 0, indx, newindx, slices);

}

/** Iterates though the a ndim number of provided slices recursivly and copies the data from a to out.
 * @param[in] a - the sliceable object (array, list, matrix, etc..).
 * @param[out] out - returns the requeted slice of a.
 * @param[in] copy - a function that can copy information from a to out.
 * @param[in] ndim - the total number of dimentions being indexed.
 * @param[in] curdim - the current dimention being indexed.
 * @param[in] indx - an ndim list of indices that builds up to a locataion in a to copy data from.
 * @param[in] newindx - the place in out to put the data copied from a
 * @param[in] slices - a set of indices that can be lists ranges or ints.
*/
objectarrayerror setslicerecursive(value* a, value* out,objectarrayerror copy(value * ,value *, unsigned int, unsigned int *,unsigned int *),\
								   unsigned int ndim, unsigned int curdim, unsigned int *indx,unsigned int *newindx, value *slices){
	// this gets given an array and out and a list of slices,
	// we resolve the top slice to a number and add it to a list
	objectarrayerror arrayerr;

	if (curdim == ndim) { // we've resolved all the indices we can now use the list
		arrayerr = (*copy)(a,out,ndim,indx,newindx);
		if (arrayerr!=ARRAY_OK) return arrayerr;
	} else { // we need to iterate though the current object
		if (MORPHO_ISINTEGER(slices[curdim])) {
			indx[curdim] = MORPHO_GETINTEGERVALUE(slices[curdim]);
			newindx[curdim] = 0;

			arrayerr = setslicerecursive(a, out, copy, ndim, curdim+1, indx, newindx, slices);
			if (arrayerr!=ARRAY_OK) return arrayerr;

		} else if (MORPHO_ISLIST(slices[curdim])) { // if this is a list

			objectlist * s = MORPHO_GETLIST(slices[curdim]);
			for (unsigned int  i = 0; i<s->val.count; i++ ){ // iterate through the list
				if (MORPHO_ISINTEGER(s->val.data[i])) {
					indx[curdim] = MORPHO_GETINTEGERVALUE(s->val.data[i]);
					newindx[curdim] = i;
				} else return ARRAY_NONINTINDX;

				arrayerr = setslicerecursive(a, out, copy, ndim, curdim+1, indx, newindx, slices);
				if (arrayerr!=ARRAY_OK) return arrayerr;

			}
		} else if (MORPHO_ISRANGE(slices[curdim])) { //if its a range
			objectrange * s = MORPHO_GETRANGE(slices[curdim]);
			value rangeValue;
			for (unsigned int  i = 0; i<range_count(s); i++) { // iterate though the range
				rangeValue=range_iterate(s,i);
				if (MORPHO_ISINTEGER(rangeValue)) {
					indx[curdim] = MORPHO_GETINTEGERVALUE(rangeValue);
					newindx[curdim] = i;
				} else return ARRAY_NONINTINDX;
				arrayerr = setslicerecursive(a, out, copy, ndim, curdim+1, indx, newindx, slices);
				if (arrayerr!=ARRAY_OK) return arrayerr;
			}
		} else return ARRAY_NONINTINDX;
			//if (!(*dimFcn)(a,ndim)) return ARRAY_WRONGDIM;

	}
	return ARRAY_OK;
}


/** Sets an array element */
objectarrayerror array_setelement(objectarray *a, unsigned int ndim, unsigned int *indx, value in) {
    unsigned int k=0;

    if (ndim!=a->ndim) return ARRAY_WRONGDIM;

    for (unsigned int i=0; i<ndim; i++) {
        if (indx[i]>=MORPHO_GETINTEGERVALUE(a->dimensions[i])) return ARRAY_OUTOFBOUNDS;
        k+=indx[i]*MORPHO_GETINTEGERVALUE(a->multipliers[i]);
    }

    a->values[k]=in;
    return ARRAY_OK;
}

/* ---------------------------
 * Array constructor functions
 * --------------------------- */

/** Returns the maximum nesting depth in a list, including this one.
 * @param[in] list - the list to examine
 * @param[out] out - optionally return the dimensions of the nested lists.
 * To get dimension information:
 * Call list_nestingdepth with out set to NULL; this returns the size of the array needed.
 * Initialize the dimension array to zero.
 * Call list_nestingdepth again with out set to an output array */
unsigned int list_nestingdepth(objectlist *list, unsigned int *out) {
    unsigned int dim=0;
    for (unsigned int i=0; i<list->val.count; i++) {
        if (MORPHO_ISLIST(list->val.data[i])) {
            unsigned int sdim=list_nestingdepth(MORPHO_GETLIST(list->val.data[i]), ( out ? out+1 : NULL));
            if (sdim>dim) dim=sdim;
        }
    }
    if (out && list->val.count>*out) *out=list->val.count;
    return dim+1;
}

/* Internal function that recursively copied a nested list into an array.
   Use public interface array_copyfromnestedlist */
static void array_copyfromnestedlistrecurse(objectlist *list, unsigned int ndim, unsigned int *indx, unsigned int depth, objectarray *out) {
    for (unsigned int i=0; i<list->val.count; i++) {
        indx[depth] = i;
        value val = list->val.data[i];
        if (MORPHO_ISLIST(val)) array_copyfromnestedlistrecurse(MORPHO_GETLIST(val), ndim, indx, depth+1, out);
        else array_setelement(out, ndim, indx, val);
    }
}

/** Copies a nested list into an array.*/
void array_copyfromnestedlist(objectlist *in, objectarray *out) {
    unsigned int indx[out->ndim];
    for (unsigned int i=0; i<out->ndim; i++) indx[i]=0;
    array_copyfromnestedlistrecurse(in, out->ndim, indx, 0, out);
}

/** Constructs an array from a list initializer or returns NULL if the initializer isn't compatible with the requested array */
objectarray *array_constructfromlist(unsigned int ndim, unsigned int *dim, objectlist *initializer) {
    // Establish the dimensions of the nested list
    unsigned int nldim = list_nestingdepth(initializer, NULL);
    unsigned int ldim[nldim];
    for (unsigned int i=0; i<nldim; i++) ldim[i]=0;
    list_nestingdepth(initializer, ldim);

    if (ndim>0) { // Check compatibility
        if (ndim!=nldim) return NULL;
        for (unsigned int i=0; i<ndim; i++) if (ldim[i]!=dim[i]) return NULL;
    }

    objectarray *new = object_newarray(nldim, ldim);
    array_copyfromnestedlist(initializer, new);

    return new;
}

/** Constructs an array from an initializer or returns NULL if the initializer isn't compatible with the requested array */
objectarray *array_constructfromarray(unsigned int ndim, unsigned int *dim, objectarray *initializer) {
    if (ndim>0) { // Check compatibility
        if (ndim!=initializer->ndim) return NULL;
        for (unsigned int i=0; i<ndim; i++) {
            if (dim[i]!=MORPHO_GETINTEGERVALUE(initializer->dimensions[i])) return NULL;
        }
    }

    return object_clonearray(initializer);
}

/** Array constructor function */
value array_constructor(vm *v, int nargs, value *args) {
    unsigned int ndim; // Number of dimensions
    unsigned int dim[nargs+1]; // Size of each dimension
    value initializer=MORPHO_NIL; // An initializer if provided

    // Check that args are present
    if (nargs==0) { morpho_runtimeerror(v, ARRAY_ARGS); return MORPHO_NIL; }

    for (ndim=0; ndim<nargs; ndim++) { // Loop over arguments
        if (!MORPHO_ISNUMBER(MORPHO_GETARG(args, ndim))) break; // Stop once a non-numerical argument is encountered
    }

    // Get dimensions
    if (ndim>0) array_valuelisttoindices(ndim, &MORPHO_GETARG(args, 0), dim);
    // Initializer is the first non-numerical argument; anything after is ignored
    if (ndim<nargs) initializer=MORPHO_GETARG(args, ndim);

    objectarray *new=NULL;

    // Now construct the array
    if (MORPHO_ISNIL(initializer)) {
        new = object_newarray(ndim, dim);
    } else if (MORPHO_ISARRAY(initializer)) {
        new = array_constructfromarray(ndim, dim, MORPHO_GETARRAY(initializer));
        if (!new) morpho_runtimeerror(v, ARRAY_CMPT);
    } else if (MORPHO_ISLIST(initializer)) {
        new = array_constructfromlist(ndim, dim, MORPHO_GETLIST(initializer));
        if (!new) morpho_runtimeerror(v, ARRAY_CMPT);
    } else {
        morpho_runtimeerror(v, ARRAY_ARGS);
    }

    // Bind the new array to the VM
    value out=MORPHO_NIL;
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }

    return out;
}

/** Checks that an array is being indexed with the correct number of indices with a generic interface */
bool array_slicedim(value * a, unsigned int ndim){
	objectarray * array= MORPHO_GETARRAY(*a);
	if (ndim>array->ndim) return false;
	return true;
}

/** Constructsan array is with a generic interface */
void array_sliceconstructor(unsigned int *slicesize,unsigned int ndim,value* out){
	*out = MORPHO_OBJECT(object_newarray(ndim,slicesize));
}

/** Copies data from array a to array out with a generic interface */
objectarrayerror array_slicecopy(value * a,value * out, unsigned int ndim, unsigned int *indx,unsigned int *newindx){
	value data;
	objectarrayerror arrayerr;
	arrayerr = array_getelement(MORPHO_GETARRAY(*a),ndim,indx,&data); // read the data
	if (arrayerr!=ARRAY_OK) return arrayerr;

	arrayerr=array_setelement(MORPHO_GETARRAY(*out), ndim, newindx, data); // write the data
	return arrayerr;

}

/** Gets the array element with given indices */
value Array_getindex(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectarray *array=MORPHO_GETARRAY(MORPHO_SELF(args));
    unsigned int indx[nargs];

    if (array_valuelisttoindices(nargs, &MORPHO_GETARG(args, 0), indx)) {
        objectarrayerror err=array_getelement(array, nargs, indx, &out);
        if (err!=ARRAY_OK) MORPHO_RAISE(v, array_error(err) );

    } else {
		// these aren't simple indices, lets try to make a slice
		objectarrayerror err = getslice(&MORPHO_SELF(args),&array_slicedim,&array_sliceconstructor,&array_slicecopy,nargs,&MORPHO_GETARG(args, 0),&out);
		if (err!=ARRAY_OK) MORPHO_RAISE(v, array_error(err) );
		if (!MORPHO_ISNIL(out)){
			morpho_bindobjects(v,1,&out);
		} else MORPHO_RAISE(v, VM_NONNUMINDX);
	}

    return out;
}

/** Sets the matrix element with given indices */
value Array_setindex(vm *v, int nargs, value *args) {
    objectarray *array=MORPHO_GETARRAY(MORPHO_SELF(args));
    unsigned int indx[nargs-1];

    if (array_valuelisttoindices(nargs-1, &MORPHO_GETARG(args, 0), indx)) {
        objectarrayerror err=array_setelement(array, nargs-1, indx, MORPHO_GETARG(args, nargs-1));
        if (err!=ARRAY_OK) MORPHO_RAISE(v, array_error(err) );
    } else MORPHO_RAISE(v, VM_NONNUMINDX);

    return MORPHO_NIL;
}

/** Print an array */
value Array_print(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    if (!MORPHO_ISARRAY(self)) return Object_print(v, nargs, args);
    
    array_print(v, MORPHO_GETARRAY(self));

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
    objectlist *new=object_newlist(a->ndim, a->data);

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
            out=slf->values[n];
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

    /* String */
    builtin_addfunction(STRING_CLASSNAME, string_constructor, BUILTIN_FLAGSEMPTY);
    value stringclass=builtin_addclass(STRING_CLASSNAME, MORPHO_GETCLASSDEFINITION(String), objclass);
    object_setveneerclass(OBJECT_STRING, stringclass);

    /* Array */
    builtin_addfunction(ARRAY_CLASSNAME, array_constructor, BUILTIN_FLAGSEMPTY);
    value arrayclass=builtin_addclass(ARRAY_CLASSNAME, MORPHO_GETCLASSDEFINITION(Array), objclass);
    object_setveneerclass(OBJECT_ARRAY, arrayclass);

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

    morpho_defineerror(ARRAY_ARGS, ERROR_HALT, ARRAY_ARGS_MSG);
    morpho_defineerror(ARRAY_INIT, ERROR_HALT, ARRAY_INIT_MSG);
    morpho_defineerror(ARRAY_CMPT, ERROR_HALT, ARRAY_CMPT_MSG);
    morpho_defineerror(STRING_IMMTBL, ERROR_HALT, STRING_IMMTBL_MSG);
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
