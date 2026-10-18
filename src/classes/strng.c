/** @file strng.c
 *  @author T J Atherton
 *
 *  @brief Defines string object type and String class
 */

#include <stdio.h>
#include <ctype.h>

#include "morpho.h"
#include "classes.h"
#include "lex.h"
#include "common.h"

/* **********************************************************************
 * String object
 * ********************************************************************** */

/** Print the string  */
void objectstring_printfn(object *obj, void *v) {
    morpho_printf(v, "%s", ((objectstring *) obj)->string);
}

/** Allocation size. length is the content, as strlen would report; the extra byte is the stored terminator. */
size_t objectstring_sizefn(object *obj) {
    return sizeof(objectstring) + ((objectstring *) obj)->length + 1;
}

hash objectstring_hashfn(object *obj) {
    objectstring *str = (objectstring *) obj;
    return dictionary_hashcstring(str->string, str->length);
}

int objectstring_cmpfn(object *a, object *b) {
    objectstring *aa=(objectstring *) a;
    objectstring *bb=(objectstring *) b;

    size_t n=(aa->length<bb->length ? aa->length : bb->length);
    int cmp=memcmp(aa->string, bb->string, n);

    if (cmp) return -cmp;
    if (aa->length==bb->length) return MORPHO_EQUAL;
    return (bb->length>aa->length ? MORPHO_BIGGER : MORPHO_SMALLER);
}

objecttypedefn objectstringdefn = {
    .printfn = objectstring_printfn,
    .markfn = NULL,
    .freefn = NULL,
    .sizefn = objectstring_sizefn,
    .hashfn = objectstring_hashfn,
    .cmpfn = objectstring_cmpfn
};

/* **********************************************************************
 * objectstring constructors
 * ********************************************************************** */

/** internal helper to get character count */
static size_t _getcharcount(objectstring *str) {
    return atomic_load_explicit(&str->nchars, memory_order_relaxed);
}

/** Internal helper to set character count */
static void _setcharcount(objectstring *str, size_t n) {
    atomic_store_explicit(&str->nchars, n, memory_order_relaxed);
}

/** @brief Creates a string from an existing character array of specified length
 *  @param in     the string to copy
 *  @param length length of string to copy (e.g. strlen(in))
 *  @returns the object or NULL on failure
 *  @warning A trailing \0 character is added to the string to ensure it is zero-terminated, 
 *           however any interior \0 characters are part of the string content; these can
 *           cause issues where string contents are used with C APIs. */
objectstring *string_new(const char *in, size_t length) {
    objectstring *new = (objectstring *) object_new(sizeof(objectstring) + sizeof(char) * (length + 1), OBJECT_STRING);

    if (new) {
        new->string=new->stringdata;
        if (in) {
            memcpy(new->string, in, length);
        } else {
            memset(new->string, 0, length);
        }
        atomic_init(&new->nchars, (length==0 ? 0 : STRING_NCHARS_UNKNOWN)); // Initialize nchars cache
        new->string[length] = '\0'; // Ensure string is zero-terminated
        new->length=length;
    }
    
    return new;
}

/** @brief Converts a varraychar into a string.
 *  @param in  the varray to convert
 *  @returns the string
 *  @warning It is not necessary to add a nul terminator to the varray; the string constructor will ensure the string is null terminated. */
objectstring *string_fromvarraychar(varray_char *in) {
    return string_new(in->data, in->count);
}

/** Creates a string from a single utf8 character */
static objectstring *string_newcharacter(const char *c) {
    int nbytes=morpho_utf8numberofbytes(c);
    if (nbytes<=0) return NULL;

    objectstring *new=string_new(c, nbytes);
    if (new) _setcharcount(new, 1);

    return new;
}

/* **********************************************************************
 * String utility functions
 * ********************************************************************** */

/** Clones a string */
objectstring *string_clone(objectstring *s) {
    objectstring *new = string_new(s->string, s->length);
    if (new) _setcharcount(new, _getcharcount(s));
    return new;
}

/** @brief Concatenates strings together
 *  @param a      first string
 *  @param b      second string
 *  @returns a string object or NULL on failure  */
objectstring *string_concatenate(objectstring *a, objectstring *b) {
    size_t length = a->length + b->length;
    objectstring *new = (objectstring *) object_new(sizeof(objectstring) + sizeof(char) * (length + 1), OBJECT_STRING);

    if (new) {
        new->string=new->stringdata;
        new->length=length;
        // Copy across old strings
        if (a) memcpy(new->string, a->string, a->length);
        if (b) memcpy(new->string+(a ? a->length : 0), b->string, b->length);
        new->string[length]='\0';
        
        // Set up character count
        size_t na=_getcharcount(a), nb=_getcharcount(b);
        size_t newnchars = (na!=STRING_NCHARS_UNKNOWN && nb!=STRING_NCHARS_UNKNOWN ? na+nb : STRING_NCHARS_UNKNOWN);
        atomic_init(&new->nchars, newnchars);
    }
    return new;
}

/** @brief Join values into a string
 *  @param v      the virtual machine
 *  @param prefix if a String, prepended with no separator
 *  @param nparts number of values
 *  @param parts  values to stringify
 *  @param sep    separator inserted between elements, or NULL
 *  @returns a string object */
static objectstring *string_join(vm *v, value prefix, unsigned int nparts, value *parts, const char *sep) {
    varray_char buffer;
    varray_charinit(&buffer);

    if (MORPHO_ISSTRING(prefix)) {
        objectstring *s = MORPHO_GETSTRING(prefix);
        varray_charadd(&buffer, s->string, (int) s->length);
    }

    size_t seplen = sep ? strlen(sep) : 0;
    for (unsigned int i=0; i<nparts; i++) {
        if (i>0 && seplen>0) varray_charadd(&buffer, (char *) sep, (int) seplen);
        morpho_printtobuffer(v, parts[i], &buffer);
    }

    objectstring *new = string_fromvarraychar(&buffer);
    varray_charclear(&buffer);
    return new;
}

/** Count number of characters in a string */
size_t string_countchars(objectstring *s) {
    size_t n=_getcharcount(s);
    if (n!=STRING_NCHARS_UNKNOWN) return n;

    n=0;
    for (char *c=s->string, *end=s->string+s->length; c<end; n++) {
        int nbytes=morpho_utf8numberofbytes(c);
        if (nbytes<=0 || c+nbytes>end) return 0; // Protect against malformed strings

        c+=nbytes;
    }

    _setcharcount(s, n);
    return n;
}

/** Get a pointer to the i'th character of a string */
char *string_index(objectstring *s, int i) {
    if (i<0) return NULL;
    size_t n=_getcharcount(s);

    // ASCII fast path
    if (n==s->length) return ((size_t) i<s->length ? s->string+i : NULL);

    n=0;
    char *end=s->string+s->length;

    for (char *c=s->string; c<end; n++) {
        if ((size_t) i==n) return c;

        int nbytes=morpho_utf8numberofbytes(c);
        if (nbytes<=0 || c+nbytes>end) return NULL; // Protect against malformed strings

        c+=nbytes;
    }
    return NULL;
}

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

/** @brief Writes a trailing '\\0' so the buffer can be used as a C string.
 *  The terminator is stored at data[count] and is not included in count. */
void string_zeroterminatevarraychar(varray_char *string) {
    if (string && varray_charresize(string, 1)) string->data[string->count] = '\0';
}

/* ----------------------------
 * Legacy interface
 * ---------------------------- */

objectstring *object_newstringwithlength(const char *in, size_t length) {
    return string_new(in, length);
}

value object_stringfromvarraychar(varray_char *in) {
    objectstring *new=string_new(in->data, in->count);
    return (new ? MORPHO_OBJECT(new) : MORPHO_NIL);
}

/** @brief Concatenate two strings  */
value object_concatenatestring(value a, value b) {
    if (!MORPHO_ISSTRING(a) || !MORPHO_ISSTRING(b)) return MORPHO_NIL;
    objectstring *new = string_concatenate(MORPHO_GETSTRING(a), MORPHO_GETSTRING(b));
    return (new ? MORPHO_OBJECT(new) : MORPHO_NIL);
}

/** @brief Clone a string  */
value object_clonestring(value val) {
    objectstring *new = NULL;
    if (MORPHO_ISSTRING(val)) {
        new=string_clone(MORPHO_GETSTRING(val));
    }
    return (new ? MORPHO_OBJECT(new): MORPHO_NIL);
}

/** @brief Creates a string from an existing null terminated C string */
objectstring *object_newstring(const char *in) {
    return object_newstringwithlength(in, strlen(in));
}

/** @brief Creates a string from an existing character array of specified length  */
value object_stringfromcstring(const char *in, size_t length) {
    objectstring *str = object_newstringwithlength(in, length);
    return (str ? MORPHO_OBJECT(str): MORPHO_NIL);
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
    objectstring *new = string_clone(MORPHO_GETSTRING(MORPHO_SELF(args)));
    return morpho_wrapandbind(v, (object *) new);
}

/** Gets a specified character from a string */
value String_getindex(vm *v, int nargs, value *args) {
    objectstring *slf = MORPHO_GETSTRING(MORPHO_SELF(args));
    int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

    char *c = string_index(slf, n);
    if (!c) {
        morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        return MORPHO_NIL;
    }
    
    return morpho_wrapandbind(v, (object *) string_newcharacter(c));
}

/** Enumerate members of a string */
value String_enumerate(vm *v, int nargs, value *args) {
    objectstring *slf = MORPHO_GETSTRING(MORPHO_SELF(args));
    int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    if (n<0) return MORPHO_INTEGER(string_countchars(slf));
    return String_getindex(v, nargs, args);
}

/** Iterate protocol */
value String_iterate(vm *v, int nargs, value *args) {
    objectstring *slf = MORPHO_GETSTRING(MORPHO_SELF(args));
    int i=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    if (i<0) { morpho_runtimeerror(v, VM_INVALIDARGS); return MORPHO_NIL; }

    if (i<slf->length) {
        char *c = slf->string+i;
        int nbytes = morpho_utf8numberofbytes(c);
        objectstring *new = string_newcharacter(c);
        
        value ret[3]={ MORPHO_INTEGER(i+nbytes), morpho_wrapandbind(v, (object *) new), MORPHO_NIL };
        return morpho_return(v, 3, ret);
    }

    value ret[3]={ MORPHO_NIL, MORPHO_NIL, MORPHO_NIL };
    return morpho_return(v, 3, ret);
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

    return morpho_wrapandbindrecursive(v, (object *) new);
}

/** Gets a substring of the string */
value String_substring(vm *v, int nargs, value *args) {
    objectstring *slf=MORPHO_GETSTRING(MORPHO_SELF(args));
    int begin=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)), end=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    if (begin<0) begin=0;
    objectstring *new=NULL;

    size_t nchars=string_countchars(slf);
    if (end<0 || end<begin || (size_t) begin>=nchars) {
        new=object_newstringwithlength(NULL, 0);
    } else {
        if ((size_t) end>nchars) end=(int) nchars;

        if (nchars==slf->length) { // ASCII fast path using byte offsets
            new=object_newstringwithlength(slf->string+begin, end-begin);
        } else { // UTF-8 path
            char *first=string_index(slf, begin), *last=first;
            char *limit=slf->string+slf->length;

            for (int i=begin; i<end; i++) {
                int nbytes=morpho_utf8numberofbytes(last);
                if (nbytes<=0 || last+nbytes>limit) break;
                last+=nbytes;
            }

            new=object_newstringwithlength(first, last-first);
        }

        if (new) _setcharcount(new, (size_t) (end-begin));
    }
    
    return morpho_wrapandbind(v, (object *) new);
}

/** Join a list of values into a string */
value String_join__list(vm *v, int nargs, value *args) {
    objectlist *parts = MORPHO_GETLIST(MORPHO_GETARG(args, 0));
    return morpho_wrapandbind(v, (object *) string_join(v, MORPHO_SELF(args), parts->val.count, parts->val.data, NULL));
}

value String_join__list_string(vm *v, int nargs, value *args) {
    objectlist *parts = MORPHO_GETLIST(MORPHO_GETARG(args, 0));
    return morpho_wrapandbind(v, (object *) string_join(v, MORPHO_SELF(args), parts->val.count, parts->val.data, MORPHO_GETCSTRING(MORPHO_GETARG(args, 1))));
}

/** Removes leading and trailing whitespace */
value String_trim(vm *v, int nargs, value *args) {
    objectstring *slf = MORPHO_GETSTRING(MORPHO_SELF(args));
    size_t begin = 0, end = slf->length;

    while (begin<end && isspace((unsigned char) slf->string[begin])) begin++;
    while (end>begin && isspace((unsigned char) slf->string[end-1])) end--;

    if (begin==0 && end==slf->length) return MORPHO_SELF(args); // No need to trim

    objectstring *new = string_new(slf->string+begin, end-begin);

    // Recalculate character count if known
    if (new) {
        size_t nchars = _getcharcount(slf);
        if (nchars!=STRING_NCHARS_UNKNOWN) _setcharcount(new, nchars - begin - (slf->length-end));
    }

    return morpho_wrapandbind(v, (object *) new);
}

MORPHO_BEGINCLASS(String)
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int ()", String_count, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "String ()", String_print, MORPHO_FN_IO),
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, "String ()", String_clone, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "String (Int)", String_getindex, MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ENUMERATE_METHOD, "(Int)", String_enumerate, MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ITERATE_METHOD, "(Int)", String_iterate, MORPHO_FN_ALLOCATES|MORPHO_FN_PUREFN|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(STRING_ISNUMBER_METHOD, "Bool ()", String_isnumber, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(STRING_SPLIT_METHOD, "List (String)", String_split, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES),
MORPHO_METHOD_SIGNATURE(STRING_SUBSTRING_METHOD, "String (Int, Int)", String_substring, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES),
MORPHO_METHOD_SIGNATURE(MORPHO_JOIN_METHOD, "String (List)", String_join__list, MORPHO_FN_REENTRANT|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_JOIN_METHOD, "String (List, String)", String_join__list_string, MORPHO_FN_REENTRANT|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(STRING_TRIM_METHOD, "String ()", String_trim, MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

objecttype objectstringtype;

void string_initialize(void) {
    // Note the objectstringtype is created by builtin_initialize as it is needed before object definitions
    
    // Locate the Object class to use as the parent class of String
    value objclass = builtin_findclassfromcstring(OBJECT_CLASSNAME);
    
    // Create String veneer class
    value stringclass=builtin_addclass(STRING_CLASSNAME, MORPHO_GETCLASSDEFINITION(String), objclass);
    object_setveneerclass(OBJECT_STRING, stringclass);
    
    // String constructor function
    morpho_addfunction(STRING_CLASSNAME, STRING_CLASSNAME " (...)", string_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
}
