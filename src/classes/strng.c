/** @file strng.c
 *  @author T J Atherton
 *
 *  @brief Defines string object type and String class
 */

#include <stdio.h>

#include "morpho.h"
#include "classes.h"
#include "lex.h"
#include "common.h"

/* **********************************************************************
 * String objects
 * ********************************************************************** */

/** String object definitions */
void objectstring_printfn(object *obj, void *v) {
    morpho_printf(v, "%s", ((objectstring *) obj)->string);
}

size_t objectstring_sizefn(object *obj) {
    return sizeof(objectstring)+((objectstring *) obj)->length+1;
}

hash objectstring_hashfn(object *obj) {
    objectstring *str = (objectstring *) obj;
    return dictionary_hashcstring(str->string, str->length);
}

int objectstring_cmpfn(object *a, object *b) {
    objectstring *astring = (objectstring *) a;
    objectstring *bstring = (objectstring *) b;
    size_t len = (astring->length > bstring->length ? astring->length : bstring->length);

    return -strncmp(astring->string, bstring->string, len);
}

objecttypedefn objectstringdefn = {
    .printfn = objectstring_printfn,
    .markfn = NULL,
    .freefn = NULL,
    .sizefn = objectstring_sizefn,
    .hashfn = objectstring_hashfn,
    .cmpfn = objectstring_cmpfn
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
        if (in) {
            memcpy(new->string, in, length);
        } else {
            memset(new->string, 0, length);
        }
        new->string[length] = '\0'; /* Zero terminate the string to be compatible with C */
        new->length=strlen(new->string);
        out = MORPHO_OBJECT(new);
    }
    return out;
}

/** @brief Creates a string with given length
 *  @param length length of string to allocate
 *  @returns the object (as a value) which will be MORPHO_NIL on failure */
objectstring *object_stringwithsize(size_t length) {
    objectstring *new = (objectstring *) object_new(sizeof(objectstring) + sizeof(char) * (length + 1), OBJECT_STRING);

    if (new) {
        new->string=new->stringdata;
        new->string[length] = '\0'; // Ensure pre-null terminated
        memset(new->string, 0, length);
        new->length=length;
        return new;
    }
    return NULL;
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
 * String utility functions
 * ********************************************************************** */

/** Convert a string to a number */
bool string_tonumber(objectstring *string, value *out) {
    bool minus=false;
    lexer l;
    token tok;
    error err;
    error_init(&err);
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
    lex_clear(&l);

    return false;
}

/** Count number of characters in a string */
int string_countchars(objectstring *s) {
    int n=0;
    for (char *c = s->string; *c!='\0'; ) {
        c+=morpho_utf8numberofbytes(c);
        n++;
    }
    return n;
}

/** Get a pointer to the i'th character of a string */
char *string_index(objectstring *s, int i) {
    int n=0;
    for (char *c = s->string; *c!='\0'; ) {
        if (i==n) return (char *) c;
        c+=morpho_utf8numberofbytes(c);
        n++;
    }
    return NULL;
}

/* **********************************************************************
 * String class
 * ********************************************************************** */

/** Constructor function for strings */
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
    morpho_printvalue(v, MORPHO_SELF(args));

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
                out=object_stringfromcstring(c, morpho_utf8numberofbytes(c));
                morpho_bindobjects(v, 1, &out);
            } else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        }
    } else MORPHO_RAISE(v, ENUMERATE_ARGS);

    return out;
}

/** Tests if a string encodes a number */
value String_isnumber(vm *v, int nargs, value *args) {
    objectstring *slf = MORPHO_GETSTRING(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (string_tonumber(slf, &out)) return MORPHO_TRUE;

    return MORPHO_FALSE;
}

/** Splits a string */
value String_split(vm *v, int nargs, value *args) {
    objectstring *slf = MORPHO_GETSTRING(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (nargs==1 && MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        objectstring *split = MORPHO_GETSTRING(MORPHO_GETARG(args, 0));
        objectlist *new = object_newlist(0, NULL);

        if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return MORPHO_NIL; }

        char *last = slf->string;
        for (char *c = slf->string; *c!='\0'; c+=morpho_utf8numberofbytes(c)) { // Loop over string
            for (char *s = split->string; *s!='\0';) { // Loop over split chars
                int nbytes = morpho_utf8numberofbytes(s);
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
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, String_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(STRING_ISNUMBER_METHOD, String_isnumber, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(STRING_SPLIT_METHOD, String_split, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

objecttype objectstringtype;

void string_initialize(void) {
    // Create string object type
    //objectstringtype=object_addtype(&objectstringdefn);
    
    // Locate the Object class to use as the parent class of Range
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // Create String veneer class
    value stringclass=builtin_addclass(STRING_CLASSNAME, MORPHO_GETCLASSDEFINITION(String), objclass);
    object_setveneerclass(OBJECT_STRING, stringclass);
    
    // String constructor function
    morpho_addfunction(STRING_CLASSNAME, STRING_CLASSNAME " (...)", string_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
}
