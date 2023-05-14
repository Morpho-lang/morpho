/** @file error.c
 *  @author T J Atherton
 *
 *  @brief Morpho memory management
*/

#include <stdio.h>
#include "memory.h"

#ifdef MORPHO_DEBUG_STRESSGARBAGECOLLECTOR
#include "morpho.h"
void vm_collectgarbage(vm *v);
#endif

/** @brief Generic allocator function
 *  @param old      A previously allocated pointer, or NULL to allocate new memory
 *  @param oldsize  The previously allocated size
 *  @param newsize  New size to allocate
 *  @returns A pointer to allocated memory, or NULL on failure.
 */
void *morpho_allocate(void *old, size_t oldsize, size_t newsize) {
// [2/27/21] TJA: Disabled because Garbage collection should be triggered on binding, not on allocation.
//#ifdef MORPHO_DEBUG_STRESSGARBAGECOLLECTOR
//    if (newsize>oldsize) {
//        vm_collectgarbage(NULL);
//    }
//#endif
    
    if (newsize == 0) {
        free(old);
        return NULL;
    }

    return realloc(old, newsize);
}

