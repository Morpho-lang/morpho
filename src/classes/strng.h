/** @file strng.h
 *  @author T J Atherton
 *
 *  @brief Defines string object type and String class
 */

#ifndef strng_h
#define strng_h

#include <string.h>
#include <stdatomic.h>
#include "object.h"

/* -------------------------------------------------------
 * String object type
 * ------------------------------------------------------- */

extern MORPHO_API objecttype objectstringtype;
#define OBJECT_STRING objectstringtype

/** A string object */
typedef struct {
    object obj;
    size_t length; // Length of string in bytes
    _Atomic(size_t) nchars; // Number of unicode code points (cached)
    char *string; // Content
    char stringdata[]; // Flexible array member to enable contiguous allocation
} objectstring;

/** Tests whether an object is a string */
#define MORPHO_ISSTRING(val) object_istype(val, OBJECT_STRING)

/** Extracts the objectstring from a value */
#define MORPHO_GETSTRING(val)             ((objectstring *) MORPHO_GETOBJECT(val))

/** Extracts a C string from a value */
#define MORPHO_GETCSTRING(val)            (((objectstring *) MORPHO_GETOBJECT(val))->string)

/** Extracts the string length from a value */
#define MORPHO_GETSTRINGLENGTH(val)       (((objectstring *) MORPHO_GETOBJECT(val))->length)

/** Uninitialized nchars */
#define STRING_NCHARS_UNKNOWN SIZE_MAX

/** Use to create static strings on the C stack (cstring may be const char *). */
#define MORPHO_STATICSTRING(cstring)      { .obj.type=OBJECT_STRING, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .string=(char *)(cstring), .length=strlen(cstring), .nchars=STRING_NCHARS_UNKNOWN,  }

/** Use to create static strings on the C stack (cstring may be const char *). */
#define MORPHO_STATICSTRINGWITHLENGTH(cstring, len)      { .obj.type=OBJECT_STRING, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .string=(char *)(cstring), .length=len, .nchars=STRING_NCHARS_UNKNOWN }

#define OBJECT_STRINGLABEL "string" // These are only used by the parser... Should be moved?
#define OBJECT_SYMBOLLABEL "symbol"

/* -------------------------------------------------------
 * String veneer class
 * ------------------------------------------------------- */

#define STRING_CLASSNAME                  "String"

#define STRING_SPLIT_METHOD               "split"
#define STRING_ISNUMBER_METHOD            "isnumber"
#define STRING_SUBSTRING_METHOD           "substring"

/* -------------------------------------------------------
 * String error messages
 * ------------------------------------------------------- */

/* -------------------------------------------------------
 * Legacy interface
 * ------------------------------------------------------- */

objectstring *object_newstringwithlength(const char *in, size_t length);
objectstring *object_newstring(const char *in);
value object_stringfromcstring(const char *in, size_t length);
value object_stringfromvarraychar(varray_char *in);
value object_clonestring(value val);
value object_concatenatestring(value a, value b);

/* -------------------------------------------------------
 * String interface
 * ------------------------------------------------------- */

objectstring *string_new(const char *in, size_t length);
objectstring *string_fromvarraychar(varray_char *in);

objectstring *string_clone(objectstring *s);
objectstring *string_concatenate(objectstring *a, objectstring *b);

size_t string_countchars(objectstring *s);
char *string_index(objectstring *s, int i);

bool string_tonumber(objectstring *string, value *out);
void string_zeroterminatevarraychar(varray_char *string);

void string_initialize(void);

#endif
