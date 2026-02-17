/** @file varray.h
 *  @author T J Atherton
 *
 *  @brief Dynamically resizing array (varray) data structure
*/

#ifndef varray_h
#define varray_h

#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>

#include "memory.h"

/* -------------------------------------------------------
 * Variable array macros
 * ------------------------------------------------------- */

/** @brief Creates a generic varray containing a specified type.
 *
 * @details Varrays only differ by their contents, and so we use macros to
 *  conveniently define types and functions.
 * To use these:
 *  First, call DECLARE_VARRAY(NAME,TYPE) in your .h file with a selected
 *   name for your varray and the type of thing you want to store in it.
 *  This will define:
 *  1. A type called varray_NAME (where NAME is the name you gave).
 *  2. Functions
 *     varray_NAMEinit(v)               - Initializes the varray
 *     varray_NAMEadd(v, data[], count) - Adds elements to the varray
 *     varray_NAMEwrite(v, data)        - Writes a single element to the varray, returning the index
 *     varray_NAMEclear(v)              - Clears the varray, freeing memory
 *  Then, call DEFINE_VARRAY(NAME,TYPE) in your .c file to define the appropriate functions
 */
#define DECLARE_VARRAY(name, type) \
    typedef struct { \
        unsigned int count; \
        unsigned int capacity; \
        type *data; \
    } varray_##name; \
    \
    void varray_##name##init(varray_##name *v); \
    bool varray_##name##add(varray_##name *v, type *data, int count); \
    bool varray_##name##resize(varray_##name *v, int count); \
    int varray_##name##write(varray_##name *v, type data); \
    void varray_##name##clear(varray_##name *v);

#define DEFINE_VARRAY(name, type) \
void varray_##name##init(varray_##name *v) { \
    v->count = 0; \
    v->capacity=0; \
    v->data=NULL; \
} \
\
bool varray_##name##add(varray_##name *v, type *data, int count) { \
    if (v->capacity<v->count + count) { \
        unsigned int capacity = varray_powerof2ceiling(v->count + count); \
        v->data = (type *) morpho_allocate(v->data, v->capacity * sizeof(type), \
                            capacity * sizeof(type)); \
        v->capacity = capacity; \
    }; \
    \
    if (v->data && data) for (unsigned int i = 0; i < count; i++) { \
        v->data[v->count++] = data[i]; \
    } \
    return (v->data!=NULL); \
} \
\
bool varray_##name##resize(varray_##name *v, int count) { \
    if (v->capacity<v->count + count) { \
        unsigned int capacity = varray_powerof2ceiling(v->count + count); \
        v->data = (type *) morpho_allocate(v->data, v->capacity * sizeof(type), \
                            capacity * sizeof(type)); \
        v->capacity = capacity; \
    }; \
    return (v->data!=NULL); \
} \
\
int varray_##name##write(varray_##name *v, type data) { \
    varray_##name##add(v, &data, 1); \
    return v->count-1; \
} \
\
void varray_##name##clear(varray_##name *v) { \
    morpho_allocate(v->data, 0, 0); \
    varray_##name##init(v); \
} \
bool varray_##name##pop(varray_##name *v, type *dest) { \
    if (v->count>0) { \
        v->count-=1; \
        *dest = v->data[v->count]; \
        return true; \
    } \
    return false; \
} \

/* -------------------------------------------------------
 * Common varray types
 * ------------------------------------------------------- */

DECLARE_VARRAY(char, char);
DECLARE_VARRAY(int, int);
DECLARE_VARRAY(double, double);
DECLARE_VARRAY(ptrdiff, ptrdiff_t);

unsigned int varray_powerof2ceiling(unsigned int n);

#endif /* varray_h */
