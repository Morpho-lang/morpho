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

/** @brief A Morpho value */

#ifdef MORPHO_NAN_BOXING

typedef uint64_t value;

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

#define TYPE_BITS (TAG_OBJ | TAG_NIL | TAG_BOOL | TAG_INT)

#define VALUE_NIL       (TAG_NIL)
#define VALUE_INTEGER   (TAG_INT)
#define VALUE_DOUBLE    ()
#define VALUE_BOOL      (TAG_BOOL)
#define VALUE_OBJECT    (TAG_OBJ)

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

/* Ordered type labels */

/* Computes an ordered type for a value  */
static inline unsigned int morpho_getorderedtype(value v) {
    if (MORPHO_ISFLOAT(v)) return 0;
    return ((v & (TAG_NIL | TAG_BOOL | TAG_INT))>>47) + ((v & TAG_OBJ) >> (63-2));
}

#else

/** @brief A enumerated type defining the different types available in Morpho. */
typedef enum {
    VALUE_NIL,
    VALUE_INTEGER,
    VALUE_DOUBLE,
    VALUE_BOOL,
    VALUE_OBJECT
} valuetype;

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

static inline bool morpho_isnumber(value a) {
    return (MORPHO_ISINTEGER(a) || MORPHO_ISFLOAT(a));
}

#define MORPHO_ISNUMBER(v) (morpho_isnumber(v))

/** Conversion of integer to a float */
#define MORPHO_INTEGERTOFLOAT(x) (MORPHO_FLOAT((double) MORPHO_GETINTEGERVALUE((x))))
/** Conversion of a float to an integer with rounding */
#define MORPHO_FLOATTOINTEGER(x) (MORPHO_INTEGER((int) round(MORPHO_GETFLOATVALUE((x)))))

/** Define notion of falsity/truthyness */
static inline bool morpho_isfalse(value a) {
    return (MORPHO_ISNIL(a) || (MORPHO_ISBOOL(a) && (MORPHO_GETBOOLVALUE(a)==false)));
}

#define MORPHO_ISFALSE(x) (morpho_isfalse(x))
#define MORPHO_ISTRUE(x) (!morpho_isfalse(x))

/* Conversion between types */

/* Convert a value to an integer */
static inline bool morpho_valuetoint(value v, int *out) {
    if (MORPHO_ISINTEGER(v)) { *out = MORPHO_GETINTEGERVALUE(v); return true; }
    if (MORPHO_ISFLOAT(v)) { *out = (int) MORPHO_GETFLOATVALUE(v); return true; }
    return false;
}

/* Convert a value to a float */
static inline bool morpho_valuetofloat(value v, double *out) {
    if (MORPHO_ISINTEGER(v)) { *out = (double) MORPHO_GETINTEGERVALUE(v); return true; }
    if (MORPHO_ISFLOAT(v)) { *out = MORPHO_GETFLOATVALUE(v); return true; }
    return false;
}

DECLARE_VARRAY(value, value);

bool varray_valuefind(varray_value *varray, value v, unsigned int *out);
bool varray_valuefindsame(varray_value *varray, value v, unsigned int *out);

bool value_promotenumberlist(unsigned int nv, value *v);

#endif /* value_h */
