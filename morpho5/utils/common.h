/** @file common.h
 *  @author T J Atherton
 *
 *  @brief Morpho virtual machine
 */

#ifndef common_h
#define common_h

#include <stddef.h>
#include <stdarg.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "value.h"
#include "object.h"
#include "builtin.h"
#include "error.h"

#define COMMON_NILSTRING   "nil"
#define COMMON_TRUESTRING  "true"
#define COMMON_FALSESTRING "false"

/* -----------------------------------------
 * Functions and macros for comparing values
 * ----------------------------------------- */

/** @brief Compares two values
 * @param a value to compare
 * @param b value to compare
 * @returns 0 if a and b are equal, a positive number if b\>a and a negative number if a\<b */
#define EQUAL 0
#define NOTEQUAL 1
#define BIGGER 1
#define SMALLER -1
static inline int morpho_comparevalue (value a, value b) {
    if (!morpho_ofsametype(a, b)) return NOTEQUAL;
    
    if (MORPHO_ISFLOAT(a)) {
        double x = MORPHO_GETFLOATVALUE(b) - MORPHO_GETFLOATVALUE(a);
        if (x>DBL_EPSILON) return BIGGER; /* Fast way out for clear cut cases */
        if (x<-DBL_EPSILON) return SMALLER;
        /* Assumes absolute tolerance is the same as relative tolerance. */
        if (fabs(x)<=DBL_EPSILON*fmax(1.0, fmax(MORPHO_GETFLOATVALUE(a), MORPHO_GETFLOATVALUE(b)))) return EQUAL;
        return (x>0 ? BIGGER : SMALLER);
    } else {
        switch (MORPHO_GETTYPE(a)) {
            case VALUE_NIL:
                return EQUAL; /** Nones are always the same */
            case VALUE_INTEGER:
                return (MORPHO_GETINTEGERVALUE(b) - MORPHO_GETINTEGERVALUE(a));
            case VALUE_BOOL:
                return (MORPHO_GETBOOLVALUE(b) != MORPHO_GETBOOLVALUE(a));
            case VALUE_OBJECT:
                {
                    if (MORPHO_GETOBJECTTYPE(a)!=MORPHO_GETOBJECTTYPE(b)) {
                        return 1; /* Objects of different type are always different */
                    } else if (MORPHO_ISSTRING(a)) {
                        objectstring *astring = MORPHO_GETSTRING(a);
                        objectstring *bstring = MORPHO_GETSTRING(b);
                        size_t len = (astring->length > bstring->length ? astring->length : bstring->length);
                        
                        return -strncmp(astring->string, bstring->string, len);
                    } else if (MORPHO_ISDOKKEY(a) && MORPHO_ISDOKKEY(b)) {
                        objectdokkey *akey = MORPHO_GETDOKKEY(a);
                        objectdokkey *bkey = MORPHO_GETDOKKEY(b);
                        
                        return ((MORPHO_GETDOKKEYCOL(akey)==MORPHO_GETDOKKEYCOL(bkey) &&
                                 MORPHO_GETDOKKEYROW(akey)==MORPHO_GETDOKKEYROW(bkey)) ? EQUAL : NOTEQUAL);
                    } else {
                        return (MORPHO_GETOBJECT(a) == MORPHO_GETOBJECT(b)? EQUAL: NOTEQUAL);
                    }
                }
            default:
                UNREACHABLE("unhandled value type for comparison [Check morpho_comparevalue]");
        }
    }
    return NOTEQUAL;
}
#undef EQUAL
#undef NOTEQUAL
#undef BIGGER
#undef SMALLER

/** @brief Compares two values, checking if two values are identical
 * @details Faster than morpho_comparevalue
 * @param a value to compare
 * @param b value to compare
 * @returns true if a and b are identical, false otherwise */
static inline bool morpho_comparevaluesame (value a, value b) {
#ifdef MORPHO_NAN_BOXING
    return (a==b);
#else
    if (a.type!=b.type) return false;
    
    switch (a.type) {
        case VALUE_NIL:
            return true; /** Nils are always the same */
        case VALUE_INTEGER:
            return (b.as.integer == a.as.integer);
        case VALUE_DOUBLE:
            /* The sign bit comparison is required to distinguish between -0 and 0. */
            return ((b.as.real == a.as.real) && (signbit(b.as.real)==signbit(a.as.real)));
        case VALUE_BOOL:
            return (b.as.boolean == a.as.boolean);
        case VALUE_OBJECT:
            return MORPHO_GETOBJECT(a) == MORPHO_GETOBJECT(b);
        default:
            UNREACHABLE("unhandled value type for comparison [Check morpho_comparevaluesame]");
    }
    
    return false;
#endif
}

/** Macros to compare values  */

/** Use this one to carefully compare the values in each object */
#define MORPHO_ISEQUAL(a,b) (!morpho_comparevalue(a,b))
/** Use this one where we want to check the values refer to the same object */
#define MORPHO_ISSAME(a,b) (morpho_comparevaluesame(a,b))

/** Check if a value is callable */
static inline bool morpho_iscallable(value a) {
    return (MORPHO_ISFUNCTION(a) ||
            MORPHO_ISBUILTINFUNCTION(a) ||
            MORPHO_ISINVOCATION(a) ||
            MORPHO_ISCLOSURE(a));
}

#define MORPHO_ISCALLABLE(x) (morpho_iscallable(x))

value morpho_concatenatestringvalues(int nval, value *v);

char *morpho_strdup(char *string);

int morpho_utf8numberofbytes(uint8_t *string);
unsigned int morpho_powerof2ceiling(unsigned int n);

bool morpho_isdirectory(const char *path);

#ifdef MORPHO_DEBUG
void morpho_unreachable(const char *explanation);
#endif

#endif /* common_h */
