/** @file shortstring.c
 *  @author J Overholt
 *
 *  @brief Defines ShortString veneer class
 */
 
#include "shortstring.h"

#include "morpho.h"
#include "classes.h"
#include "common.h"

#ifdef MORPHO_NAN_BOXING
/** @brief Creates a short string from an existing character array with given length
 *  @param in     the null terminated c-string to copy
 *  @returns the object (as a value) which will be MORPHO_NIL on failure */
value value_shortstringfromnulcstring(const char *in) {
    char buf[SSTRING_CAPACITY];
    memset(buf, 0, SSTRING_CAPACITY);
    if (in) {
        int i = 0;
        while (i<SSTRING_LENGTH) {
            if (in[i]=='\0') break;
            buf[i] = in[i];
            ++i;
        }
        // Store length in empty space if small enough
        if (i < SSTRING_CAPACITY-2) buf[SSTRING_CAPACITY-1] = i;
    }
    return MORPHO_SHORTSTRING(buf);
}

/** @brief Creates a short string from an existing character array with given length
 *  @param in     the string to copy
 *  @param length the length of the string
 *  @returns the object (as a value) which will be MORPHO_NIL on failure */
value value_shortstringfromcstring(const char *in, size_t length) {
    char buf[SSTRING_CAPACITY];
    memset(buf, 0, SSTRING_CAPACITY);
    // printf("%s", in);
    if (in) {
        // printf("%s", in);
        if (length > SSTRING_LENGTH) return MORPHO_NIL;
        memcpy(buf, in, length);
        // Store length in empty space if small enough
        if (length < SSTRING_CAPACITY-2) buf[SSTRING_CAPACITY-1] = length;
    }
    return MORPHO_SHORTSTRING(buf);
}

/** @brief Gets the number of bytes used in the short string
    @note the number of bytes used is stored in the last byte 
          if the string does not use that byte for data or terminator */
int shortstring_length(value s) {
    char *str = MORPHO_GETSHORTSTRING(s);
    if (str[SSTRING_CAPACITY-2] == '\0') return (int) str[SSTRING_CAPACITY-1];
    else if (str[SSTRING_CAPACITY-1] == '\0') return SSTRING_CAPACITY-1;
    else return SSTRING_CAPACITY;
}

/** Count number of characters in a string */
int shortstring_countchars(value s) {
    char *str = MORPHO_GETSHORTSTRING(s);
    int n=0;
    for (char *c = str; *c!='\0' && (c-str)<SSTRING_LENGTH; n++) {
        c+=morpho_utf8numberofbytes(c);
    }
    return n;
}

/** Get a pointer to the i'th character of a short string */
char *shortstring_index(value *s, int i) {
    if (i<0) return NULL;
    char *str = MORPHO_GETSHORTSTRING(*s);
    int n=0;
    for (char *c = str; *c!='\0' && (c-str)<SSTRING_LENGTH; n++) {
        if (i==n) return (char *) c;
        c+=morpho_utf8numberofbytes(c);
    }
    return NULL;
}
#endif

/* -------------------------------------------------------
 * ShortString class
 * ------------------------------------------------------- */

/** Simple constructor function for short strings */
value shortstring_constructor(vm *v, int nargs, value *args) {
#ifndef MORPHO_NAN_BOXING
    morpho_runtimeerror(v, SSTRING_NONANBOX);
    return MORPHO_NIL;
#else
    objectstring *str = MORPHO_GETSTRING(MORPHO_GETARG(args, 0));

    value out=value_shortstringfromcstring(str->string, str->length);
    if (!MORPHO_ISSHORTSTRING(out)) morpho_runtimeerror(v, SSTRING_TOOLONG);

    return out;
#endif
}

/** Constructor function for short strings */
value shortstring_multconstructor(vm *v, int nargs, value *args) {
#ifndef MORPHO_NAN_BOXING
    morpho_runtimeerror(v, SSTRING_NONANBOX);
    return MORPHO_NIL;
#else
    varray_char buffer;
    varray_charinit(&buffer);

    for (unsigned int i=1; i<=nargs; i++) {
        morpho_printtobuffer(v, args[i], &buffer);
    }

    value out=value_shortstringfromcstring(buffer.data, buffer.count);
    if (!MORPHO_ISSHORTSTRING(out)) morpho_runtimeerror(v, SSTRING_TOOLONG);

    varray_charclear(&buffer);

    return out;
#endif
}

value ShortString_count(vm *v, int nargs, value *args) {
#ifndef MORPHO_NAN_BOXING
    morpho_runtimeerror(v, SSTRING_NONANBOX);
    return MORPHO_NIL;
#else
    return MORPHO_INTEGER(shortstring_countchars(MORPHO_SELF(args)));
#endif
}

value ShortString_print(vm *v, int nargs, value *args) {
#ifndef MORPHO_NAN_BOXING
    morpho_runtimeerror(v, SSTRING_NONANBOX);
    return MORPHO_NIL;
#else
    morpho_printvalue(v, MORPHO_SELF(args));

    return MORPHO_SELF(args);
#endif
}

value ShortString_getindex(vm *v, int nargs, value *args) {
#ifndef MORPHO_NAN_BOXING
    morpho_runtimeerror(v, SSTRING_NONANBOX);
    return MORPHO_NIL;
#else
    value out = MORPHO_NIL;
    value slf = MORPHO_SELF(args);
    int n = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

    char *c = shortstring_index(&slf, n);
    if (c) {
        out = value_shortstringfromcstring(c, morpho_utf8numberofbytes(c));
    } else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
    return out;
#endif
}

value ShortString_enumerate(vm *v, int nargs, value *args) {
#ifndef MORPHO_NAN_BOXING
    morpho_runtimeerror(v, SSTRING_NONANBOX);
    return MORPHO_NIL;
#else
    value out=MORPHO_NIL;
    value slf = MORPHO_SELF(args);
    int n = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

    if (n<0) {
        out = MORPHO_INTEGER(shortstring_countchars(slf));
    } else {
        char *c = shortstring_index(&slf, n);
        if (c) {
            out = value_shortstringfromcstring(c, morpho_utf8numberofbytes(c));
        } else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
    }

    return out;
#endif
}


MORPHO_BEGINCLASS(ShortString)
MORPHO_METHOD_SIGNATURE(MORPHO_CLASS_METHOD, "()", Object_class, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int ()", ShortString_count, MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "ShortString ()", ShortString_print, MORPHO_FN_IO|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "ShortString (Int)", ShortString_getindex, MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ENUMERATE_METHOD, "(Int)", ShortString_enumerate, MORPHO_FN_THROWS)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void shortstring_initialize(void) {
    
    // Create String veneer class
    value sstringclass=builtin_addclass(SSTRING_CLASSNAME, MORPHO_GETCLASSDEFINITION(ShortString), MORPHO_NIL);
    value_setveneerclass(MORPHO_SHORTSTRING(""), sstringclass);
    
    // String constructor function
    morpho_addfunction(SSTRING_CLASSNAME, SSTRING_CLASSNAME " (String)", shortstring_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
    morpho_addfunction(SSTRING_CLASSNAME, SSTRING_CLASSNAME " (...)", shortstring_multconstructor, MORPHO_FN_CONSTRUCTOR, NULL);

    // Define errors
    morpho_defineerror(SSTRING_TOOLONG, ERROR_HALT, SSTRING_TOOLONG_MSG);
    morpho_defineerror(SSTRING_NONANBOX, ERROR_HALT, SSTRING_NONANBOX_MSG);
}