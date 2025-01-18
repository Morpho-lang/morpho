/** @file strng.h
 *  @author T J Atherton
 *
 *  @brief Defines string object type and String class
 */

#ifndef strng_h
#define strng_h

#include <string.h>
#include "object.h"

/* -------------------------------------------------------
 * String object type
 * ------------------------------------------------------- */

extern objecttype objectstringtype;
#define OBJECT_STRING objectstringtype

/** A string object */
typedef struct {
    object obj;
    size_t length;
    char *string;
    char stringdata[];
} objectstring;

/** Tests whether an object is a string */
#define MORPHO_ISSTRING(val) object_istype(val, OBJECT_STRING)

/** Extracts the objectstring from a value */
#define MORPHO_GETSTRING(val)             ((objectstring *) MORPHO_GETOBJECT(val))

/** Extracts a C string from a value */
#define MORPHO_GETCSTRING(val)            (((objectstring *) MORPHO_GETOBJECT(val))->string)

/** Extracts the string length from a value */
#define MORPHO_GETSTRINGLENGTH(val)       (((objectstring *) MORPHO_GETOBJECT(val))->length)

/** Use to create static strings on the C stack */
#define MORPHO_STATICSTRING(cstring)      { .obj.type=OBJECT_STRING, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .string=cstring, .length=strlen(cstring) }

/** Use to create static strings on the C stack */
#define MORPHO_STATICSTRINGWITHLENGTH(cstring, len)      { .obj.type=OBJECT_STRING, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .string=cstring, .length=len }


#define OBJECT_STRINGLABEL "string" // These are only used by the parser... Should be moved?
#define OBJECT_SYMBOLLABEL "symbol"

/** Create a string object from a C string */
value object_stringfromcstring(const char *in, size_t length);

/** Create an empty string of specified size */
objectstring *object_stringwithsize(size_t length);

/** Create a string object from a character varray */
value object_stringfromvarraychar(varray_char *in);

/** Clone a string */
value object_clonestring(value val);

/** Concatenate two strings */
value object_concatenatestring(value a, value b);

/* -------------------------------------------------------
 * String veneer class
 * ------------------------------------------------------- */

#define STRING_CLASSNAME                  "String"

#define STRING_SPLIT_METHOD               "split"
#define STRING_ISNUMBER_METHOD            "isnumber"

/* -------------------------------------------------------
 * String error messages
 * ------------------------------------------------------- */

/* -------------------------------------------------------
 * String interface
 * ------------------------------------------------------- */

bool string_tonumber(objectstring *string, value *out);
int string_countchars(objectstring *s);
char *string_index(objectstring *s, int i);

void string_initialize(void);

#endif
