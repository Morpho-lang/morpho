/** @file value.h
 *  @author T J Atherton
 *
 *  @brief Fundamental data type for morpho
*/

#ifndef value_h
#define value_h

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
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
        VALUE_NIL      - nil
        VALUE_INTEGER  - 32 bit integer
        VALUE_DOUBLE   -
        VALUE_BOOL     - boolean type
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
#define QNAN         ((uint64_t) 0x7ff8000000000000ull)
#define LOWER_WORD   ((uint64_t) 0x00000000ffffffffull)

/** Store the type in bits 48-50 */
#define TAG_SHIFT    48
#define TAG_MASK     ((uint64_t) (0x7ull << TAG_SHIFT))  // bits 48..50
#define PAYLOAD_MASK ((uint64_t) 0x0000ffffffffffffull)  // bits 0..47
#define EXP_MASK     ((uint64_t) 0x7ff0000000000000ull)  // Exponent bits

#define TAG_NIL      ((uint64_t) 1ull << TAG_SHIFT)
#define TAG_BOOL     ((uint64_t) 2ull << TAG_SHIFT)
#define TAG_INT      ((uint64_t) 3ull << TAG_SHIFT)
#define TAG_OBJ      ((uint64_t) 4ull << TAG_SHIFT)

/** Manipulations */
#define MORPHO_EXPALLONES(v) ((((uint64_t)(v)) & EXP_MASK) == EXP_MASK)
#define MORPHO_TAGBITS(v)    (((uint64_t)(v)) & TAG_MASK)

/** Bool values are stored in the lowest bit */
#define TAG_TRUE    1
#define TAG_FALSE   0

/** Map VALUE_XXX macros to type bits  */
#define VALUE_NIL       (TAG_NIL)
#define VALUE_INTEGER   (TAG_INT)
#define VALUE_DOUBLE    ((uint64_t) 0ull)
#define VALUE_BOOL      (TAG_BOOL)
#define VALUE_OBJECT    (TAG_OBJ)

/** Get the type from a value */
#define MORPHO_GETTYPE(x) (MORPHO_ISBOXED(x) ? MORPHO_TAGBITS(x) : VALUE_DOUBLE)

/** Converts a double to a value */
static inline value doubletovalue(double num) {
    value bits;
    memcpy(&bits, &num, sizeof(bits));
    // If this is NaN or Inf (exp all ones), force tag bits to 0 so it is a genuine float NaN/Inf
    if ((bits & EXP_MASK) == EXP_MASK) {
        bits &= ~TAG_MASK; 
    }
    return bits;
}

/** Converts a value to a double */
static inline double valuetodouble(value v) {
    double num;
    memcpy(&num, &v, sizeof(num));
    return num;
}

/** Create a literal */
#define MORPHO_NIL        ((value) (QNAN | TAG_NIL))
#define MORPHO_TRUE       ((value) (QNAN | TAG_BOOL | TAG_TRUE))
#define MORPHO_FALSE      ((value) (QNAN | TAG_BOOL | TAG_FALSE))

#define MORPHO_BOOL(x)    ((x) ? MORPHO_TRUE : MORPHO_FALSE)
#define MORPHO_INTEGER(x) ((value) (QNAN | TAG_INT | (((uint64_t)(x)) & LOWER_WORD)))
#define MORPHO_FLOAT(x)             doubletovalue(x)
#define MORPHO_OBJECT(x)  ((value) (QNAN | TAG_OBJ | (((uint64_t)(uintptr_t)(x)) & PAYLOAD_MASK)))

/** Test for the type of a value */
#define MORPHO_ISNIL(v)      ((v) == MORPHO_NIL) 
#define MORPHO_ISBOXED(v)    (MORPHO_EXPALLONES(v) && (MORPHO_TAGBITS(v) != 0))
#define MORPHO_ISINTEGER(v)  (MORPHO_ISBOXED(v) && (MORPHO_TAGBITS(v) == TAG_INT))
#define MORPHO_ISBOOL(v)     (MORPHO_ISBOXED(v) && (MORPHO_TAGBITS(v) == TAG_BOOL))
#define MORPHO_ISOBJECT(v)   (MORPHO_ISBOXED(v) && (MORPHO_TAGBITS(v) == TAG_OBJ))
#define MORPHO_ISFLOAT(v)    (!MORPHO_ISBOXED(v))

/** Get a value */
#define MORPHO_GETPAYLOAD(v)        (((uint64_t)(v)) & PAYLOAD_MASK)
#define MORPHO_GETINTEGERVALUE(v)   ((int32_t) ((uint32_t)((uint64_t)(v) & LOWER_WORD)))
#define MORPHO_GETFLOATVALUE(v)     valuetodouble(v)
#define MORPHO_GETBOOLVALUE(v)      ((v) == MORPHO_TRUE)
#define MORPHO_GETOBJECT(v)         ((object *)(uintptr_t)(((uint64_t)(v)) & PAYLOAD_MASK))

static inline bool morpho_ofsametype(value a, value b) {
    bool af = MORPHO_ISFLOAT(a);
    bool bf = MORPHO_ISFLOAT(b);

    if (af || bf) return (af && bf);

    /* both are boxed: compare tag field only */
    return ((a & TAG_MASK) == (b & TAG_MASK));
}

/** Get a non-object's type field as an integer */
static inline int _getorderedtype(value x) {
    return MORPHO_ISFLOAT(x) ? 0 : (int)(((uint64_t)x & TAG_MASK) >> TAG_SHIFT);
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
 * Type checking
 * ------------------------------------------------------- */

/** Get the type associated with a value */
bool value_type(value v, value *type);

/** Check whether an actual type matches a required type */
bool value_typematch(value type, value match);

/** Check whether a value matches a required type */
bool value_istype(value v, value type);

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
