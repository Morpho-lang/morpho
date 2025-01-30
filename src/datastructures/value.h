/** @file value.h
 *  @author T J Atherton
 *
 *  @brief Fundamental data type for morpho
*/

#ifndef value_h
#define value_h

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "build.h"
#include "varray.h"

/* Forward declarations of object structures */
typedef struct sobject object;

/* -------------------------------------------------------
 * Fundamental value type
 * ------------------------------------------------------- */

/** Values are the basic data type in morpho: each variable declared with 'var' corresponds to one value.
    Values can contain the following types:
        VALUE_NIL           - nil
        VALUE_INTEGER - 32 bit integer
        VALUE_DOUBLE  -
        VALUE_BOOL       - boolean type
        VALUE_OBJECT   - pointer to an object
    The implementation of a value is intentionally opaque and can be NAN boxed into a 64-bit double or left as a struct.
    This file therefore defines several kinds of macro to:
        * create values of a given type, e.g. MORPHO_INTEGER.
        * Test the type of a value, e.g. MORPHO_ISINTEGER
        * Extract a given type from a value and cast to the relevant C type, e.g. MORPHO_GETINTEGERVALUE */

/** NAN Boxing represents a value as a double, using the values that correspond to NAN to contain the remaining types. */
#ifdef MORPHO_NAN_BOXING

/** In this representation, we can extract non-double types from a 64 bit integer */
typedef uint64_t value;

/** Define macros that enable us to refer to various bits */
#define SIGN_BIT    ((uint64_t) 0x8000000000000000)
#define QNAN        ((uint64_t) 0x7ffc000000000000)
#define LOWER_WORD  ((uint64_t) 0x00000000ffffffff)

/** Store the type in bits 47-49 */
#define TAG_NIL     (1ull<<47) // 001
#define TAG_BOOL    (2ull<<47) // 010
#define TAG_INT     (3ull<<47) // 011
#define TAG_OBJ     SIGN_BIT

/** Bool values are stored in the lowest bit */
#define TAG_TRUE    1
#define TAG_FALSE   0

/** Bit mask used to select type bits */
#define TYPE_BITS (TAG_OBJ | TAG_NIL | TAG_BOOL | TAG_INT)

/** Map VALUE_XXX macros to type bits  */
#define VALUE_NIL       (TAG_NIL)
#define VALUE_INTEGER   (TAG_INT)
#define VALUE_DOUBLE    ()
#define VALUE_BOOL      (TAG_BOOL)
#define VALUE_OBJECT    (TAG_OBJ)

/** Get the type from a value */
#define MORPHO_GETTYPE(x)  ((x) & TYPE_BITS)

/** Union to enable conversion of a double to a 64 bit integer */
typedef union {
    uint64_t bits;
    double num;
} doubleunion;

/** Converts a double to a value by type punning */
static inline value doubletovalue(double num) {
  doubleunion data;
  data.num = num;
  return data.bits;
}

/** Converts a value to a double by type punning */
static inline double valuetodouble(value v) {
  doubleunion data;
  data.bits = v;
  return data.num;
}

/** Create a literal */
#define MORPHO_NIL                  ((value) (uint64_t) (QNAN | TAG_NIL))
#define MORPHO_TRUE                 ((value) (uint64_t) (QNAN | TAG_BOOL | TAG_TRUE))
#define MORPHO_FALSE                ((value) (uint64_t) (QNAN | TAG_BOOL | TAG_FALSE))

#define MORPHO_INTEGER(x)           ((((uint64_t) (x)) & LOWER_WORD) | QNAN | TAG_INT)
#define MORPHO_FLOAT(x)             doubletovalue(x)
#define MORPHO_BOOL(x)              ((x) ? MORPHO_TRUE : MORPHO_FALSE)
#define MORPHO_OBJECT(x)            ((value) (TAG_OBJ | QNAN | (uint64_t)(uintptr_t)(x)))

/** Test for the type of a value */
#define MORPHO_ISNIL(v)             ((v) == MORPHO_NIL)
#define MORPHO_ISINTEGER(v)         (((v) & (QNAN | TYPE_BITS)) == (QNAN | TAG_INT))
#define MORPHO_ISFLOAT(v)           (((v) & QNAN) != QNAN)
#define MORPHO_ISBOOL(v)            (((v) & (QNAN | TYPE_BITS)) == (QNAN | TAG_BOOL))
#define MORPHO_ISOBJECT(v) \
        (((v) & (QNAN | TYPE_BITS))== (QNAN | TAG_OBJ))

/** Get a value */
#define MORPHO_GETINTEGERVALUE(v)   ((int) ((uint32_t) (v & LOWER_WORD)))
#define MORPHO_GETFLOATVALUE(v)     valuetodouble(v)
#define MORPHO_GETBOOLVALUE(v)      ((v) == MORPHO_TRUE)
#define MORPHO_GETOBJECT(v)         ((object *) (uintptr_t) ((v) & ~(TAG_OBJ | QNAN)))

static inline bool morpho_ofsametype(value a, value b) {
    if (MORPHO_ISFLOAT(a) || MORPHO_ISFLOAT(b)) {
        return MORPHO_ISFLOAT(a) && MORPHO_ISFLOAT(b);
    } else {
        if ((a & TYPE_BITS)==(b & TYPE_BITS)) {
            return true;
        }
    }

    return false;
}

/** Get a non-object's type field as an integer */
static inline int _getorderedtype(value x) {
    return (MORPHO_ISFLOAT(x) ? 0 : (((x) & TYPE_BITS)>>47) & 0x7);
}
#define MORPHO_GETORDEREDTYPE(x) _getorderedtype(x)

/** Alternatively, we represent a value through a struct. */
#else

/** @brief A enumerated type defining the different types available in Morpho. */
enum {
    VALUE_DOUBLE, // Note that the order of these must match the boxed version above
    VALUE_NIL,
    VALUE_BOOL,
    VALUE_INTEGER,
    VALUE_OBJECT
};

typedef int valuetype;

/** @brief The unboxed value type. */
typedef struct {
    valuetype type;
    union {
        int integer;
        double real;
        bool boolean;
        struct sobject *obj;
    } as;
} value;

/** This macro gets the type of the value.
    @warning Not intended for broad use. */
#define MORPHO_GETTYPE(v) ((v).type)

/** Gets the ordered type of the value
    @warning Not intended for broad use. */
#define MORPHO_GETORDEREDTYPE(v) ((v).type)

/** Test for the type of a value */
#define MORPHO_ISNIL(v) ((v).type==VALUE_NIL)
#define MORPHO_ISINTEGER(v) ((v).type==VALUE_INTEGER)
#define MORPHO_ISFLOAT(v) ((v).type==VALUE_DOUBLE)
#define MORPHO_ISBOOL(v) ((v).type==VALUE_BOOL)
#define MORPHO_ISOBJECT(v) ((v).type==VALUE_OBJECT)

/** Create a literal */
#define MORPHO_NIL ((value) { VALUE_NIL, .as.integer = (int) 0 })
#define MORPHO_INTEGER(x) ((value) { VALUE_INTEGER, .as.integer = (int) (x) })
#define MORPHO_FLOAT(x) ((value) { VALUE_DOUBLE, .as.real = (double) x })
#define MORPHO_BOOL(x) ((value) { VALUE_BOOL, .as.boolean = (bool) x })
#define MORPHO_OBJECT(x) ((value) { VALUE_OBJECT, .as.obj = (object *) x })

#define MORPHO_TRUE MORPHO_BOOL(true)
#define MORPHO_FALSE MORPHO_BOOL(false)

/** Get a value */
#define MORPHO_GETINTEGERVALUE(v) ((v).as.integer)
#define MORPHO_GETFLOATVALUE(v) ((v).as.real)
#define MORPHO_GETBOOLVALUE(v) ((v).as.boolean)
#define MORPHO_GETOBJECT(v) ((v).as.obj)

static inline bool morpho_ofsametype(value a, value b) {
    return (a.type == b.type);
}

#endif

/* -------------------------------------------------------
 * Comparing values
 * ------------------------------------------------------- */

/** Check if two values are the same, i.e. identical or refer to the same object */
bool morpho_issame(value a, value b);

/** Test if two values are identical, i.e. identical or refer to the same object */
#define MORPHO_ISSAME(a,b) (morpho_issame(a,b))

/** Compare two values, checking contents of objects where supported */
int morpho_comparevalue(value a, value b);

/** Compare two values, even if they have inequivalent types e.g. int and float */
int morpho_extendedcomparevalue(value a, value b);

/** Macro to test if two values are equal, checking contents of objects where supported */
#define MORPHO_ISEQUAL(a,b) (!morpho_comparevalue(a,b))

/* -------------------------------------------------------
 * Type checking and conversion
 * ------------------------------------------------------- */

/** Detect if a value is a number */
bool morpho_isnumber(value a);

/** Define a unified notion of falsity/truthyness */
bool morpho_isfalse(value a);

/** Convert a value to an integer */
bool morpho_valuetoint(value v, int *out);

/** Convert a value to a float */
bool morpho_valuetofloat(value v, double *out);

/** Macro to detect if a value is a number */
#define MORPHO_ISNUMBER(v) (morpho_isnumber(v))

/** Conversion of integer to a float */
#define MORPHO_INTEGERTOFLOAT(x) (MORPHO_FLOAT((double) MORPHO_GETINTEGERVALUE((x))))

/** Conversion of a float to an integer with rounding */
#define MORPHO_FLOATTOINTEGER(x) (MORPHO_INTEGER((int) round(MORPHO_GETFLOATVALUE((x)))))

/** Macros to determine if a value is true or false */
#define MORPHO_ISFALSE(x) (morpho_isfalse(x))
#define MORPHO_ISTRUE(x) (!morpho_isfalse(x))

/* -------------------------------------------------------
 * Varrays of values
 * ------------------------------------------------------- */

DECLARE_VARRAY(value, value);

bool varray_valuefind(varray_value *varray, value v, unsigned int *out);
bool varray_valuefindsame(varray_value *varray, value v, unsigned int *out);

/* -------------------------------------------------------
 * Other utility functions
 * ------------------------------------------------------- */

bool value_promotenumberlist(unsigned int nv, value *v);
bool value_minmax(unsigned int nval, value *list, value *min, value *max);

void value_initialize(void);

#endif /* value_h */
