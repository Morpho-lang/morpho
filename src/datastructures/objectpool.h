/** @file objectpool.h
 *  @author T J Atherton
 *
 *  @brief Pool of Morpho objects
 */

#ifndef objectpool_h
#define objectpool_h

#include <stddef.h>
#include <stdbool.h>
#include "object.h"

/* -------------------------------------------------------
 * Object pool
 * ------------------------------------------------------- */

/** An object pool is a contiguous chunk of memory holding nelements objects,
 *  of fixed size and specified objecttype. 
 *  Set a parent if the objectpool is part of an object bound to a VM.
 *  Access individual elements through objectpool_get(). */
typedef struct {
    size_t size;       /** Bytes per element (must be at least sizeof(object) to allocate) */
    size_t nelements;  /** Number of elements */
    void *pool;        /** Allocation, or NULL if not yet created */
    objecttype type;   /** Object type stamped on every slot */
    object *parent;    /** Optional parent for CHILD objects */
    bool ready;        /** True only after allocation and slot init have finished */
} objectpool;

void objectpool_init(objectpool *p, size_t size, objecttype type, object *parent);
void objectpool_clear(objectpool *p);
bool objectpool_alloc(objectpool *p, size_t nelements);
bool objectpool_isalloc(const objectpool *p);
void *objectpool_get(const objectpool *p, size_t indx);

#endif /* objectpool_h */
