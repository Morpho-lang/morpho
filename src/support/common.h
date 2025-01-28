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
#include "value.h"
#include "object.h"
#include "classes.h"

#define MORPHO_NILSTRING   "nil"
#define MORPHO_TRUESTRING  "true"
#define MORPHO_FALSESTRING "false"

/* -----------------------------------------
 * VM Callback functions
 * ----------------------------------------- */

typedef enum {
    MORPHO_INPUT_KEYPRESS,
    MORPHO_INPUT_LINE
} morphoinputmode;

/* Callback function used to obtain input from stdin */
typedef void (*morphoinputfn) (vm *v, void *ref, morphoinputmode mode, varray_char *str);

/* Callback function used to print text to stdout */
typedef void (*morphoprintfn) (vm *v, void *ref, char *str);

/* Callback function used to output a warning */
typedef void (*morphowarningfn) (vm *v, void *ref, error *warning);

/* Callback function used to enter debugger */
typedef void (*morphodebuggerfn) (vm *v, void *ref);

void morpho_setwarningfn(vm *v, morphowarningfn warningfn, void *ref);
void morpho_setprintfn(vm *v, morphoprintfn printfn, void *ref);
void morpho_setinputfn(vm *v, morphoinputfn inputfn, void *ref);
void morpho_setdebuggerfn(vm *v, morphodebuggerfn debuggerfn, void *ref);

/* -----------------------------------------
 * Functions and macros for comparing values
 * ----------------------------------------- */

/** @brief Promotes l and r to types that can be compared
 * @param l value to compare
 * @param r value to compare */
/*#define MORPHO_CMPPROMOTETYPE(l, r) \
    if (!morpho_ofsametype(l, r)) { \
        if (MORPHO_ISINTEGER(l) && MORPHO_ISFLOAT(r)) { \
            l = MORPHO_INTEGERTOFLOAT(l); \
        } else if (MORPHO_ISFLOAT(l) && MORPHO_ISINTEGER(r)) { \
            r = MORPHO_INTEGERTOFLOAT(r); \
        } \
    }*/

/** Check if a value is callable */
static inline bool morpho_iscallable(value a) {
    return (MORPHO_ISFUNCTION(a) ||
            MORPHO_ISBUILTINFUNCTION(a) ||
            MORPHO_ISMETAFUNCTION(a) ||
            MORPHO_ISINVOCATION(a) ||
            MORPHO_ISCLOSURE(a) ||
            MORPHO_ISCLASS(a));
}

#define MORPHO_ISCALLABLE(x) (morpho_iscallable(x))

bool morpho_printtobuffer(vm *v, value val, varray_char *buffer);
value morpho_concatenate(vm *v, int nval, value *val);

char *morpho_strdup(char *string);

int morpho_utf8numberofbytes(const char *string);
int morpho_utf8toint(const char *c);
int morpho_encodeutf8(int c, char *out);

unsigned int morpho_powerof2ceiling(unsigned int n);

#ifdef MORPHO_DEBUG
void morpho_unreachable(const char *explanation);
#endif

typedef enum {
    MORPHO_TUPLEMODE, // Generates tuples (all combinations of n elements)
    MORPHO_SETMODE // Generates sets (unique elements and indep of order)
} tuplemode;

void morpho_tuplesinit(unsigned int nval, unsigned int n, unsigned int *c, tuplemode mode);
bool morpho_tuples(unsigned int nval, value *list, unsigned int n, unsigned int *c, tuplemode mode, value *tuple);

#endif /* common_h */
