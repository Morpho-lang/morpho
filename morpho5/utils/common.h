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
#include <pthread.h>
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

/** @brief Promotes l and r to types that can be compared
 * @param l value to compare
 * @param r value to compare */
#define MORPHO_CMPPROMOTETYPE(l, r) \
    if (!morpho_ofsametype(l, r)) { \
        if (MORPHO_ISINTEGER(l) && MORPHO_ISFLOAT(r)) { \
            l = MORPHO_INTEGERTOFLOAT(l); \
        } else if (MORPHO_ISFLOAT(l) && MORPHO_ISINTEGER(r)) { \
            r = MORPHO_INTEGERTOFLOAT(r); \
        } \
    }

/** @brief Compares two values
 * @param l value to compare
 * @param r value to compare */
#define MORPHO_CHECKCMPTYPE(l, r) \
    if (!morpho_ofsametype(l, r)) { \
        if (MORPHO_ISINTEGER(l) && MORPHO_ISFLOAT(r)) { \
            l = MORPHO_INTEGERTOFLOAT(l); \
        } else if (MORPHO_ISFLOAT(l) && MORPHO_ISINTEGER(r)) { \
            r = MORPHO_INTEGERTOFLOAT(r); \
        } \
    }

int morpho_comparevalue (value a, value b);

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

void morpho_printtobuffer(vm *v, value val, varray_char *buffer);
value morpho_concatenate(vm *v, int nval, value *val);

char *morpho_strdup(char *string);

int morpho_utf8numberofbytes(uint8_t *string);
unsigned int morpho_powerof2ceiling(unsigned int n);

bool morpho_isdirectory(const char *path);
bool white_space_remainder(const char *s, int start);

#ifdef MORPHO_DEBUG
void morpho_unreachable(const char *explanation);
#endif

typedef enum {
    MORPHO_TUPLEMODE, // Generates tuples (all combinations of n elements)
    MORPHO_SETMODE // Generates sets (unique elements and indep of order)
} tuplemode;

void morpho_tuplesinit(unsigned int nval, unsigned int n, unsigned int *c, tuplemode mode);
bool morpho_tuples(unsigned int nval, value *list, unsigned int n, unsigned int *c, tuplemode mode, value *tuple);

/* -----------------------------------------
 * Thread pools
 * ----------------------------------------- */

/** A work function will be called by the threadpool once a thread is available.
    You must supply all relevant information for both input and output in a single structure passed as an opaque reference. */
typedef bool (* workfn) (void *arg);

typedef struct {
    workfn func;
    void *arg;
} task;

DECLARE_VARRAY(task, task);

typedef struct {
    pthread_mutex_t lock_mutex; /* Lock for access to threadpool structure. */
    pthread_cond_t work_available_cond; /* Signals that work is available. */
    pthread_cond_t work_halted_cond; /* Signals when no threads are processing. */
    int nprocessing; /* Number of threads actively processing work */
    int nthreads; /* Number of active threads. */
    bool stop; /* Indicates threads should terminate */
    
    varray_task queue; /* Queue of tasks lined up */
} threadpool;

bool threadpool_init(threadpool *pool, int nworkers);
void threadpool_clear(threadpool *pool);
bool threadpool_add_task(threadpool *pool, workfn func, void *arg);
void threadpool_fence(threadpool *pool);
void threadpool_wait(threadpool *pool);

void threadpool_test(void);

#endif /* common_h */
