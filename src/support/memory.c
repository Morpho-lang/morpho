/** @file memory.c
 *  @author T J Atherton
 *
 *  @brief Morpho memory allocator
*/

#include "memory.h"

/** @brief Generic allocator function
 *  @param old      A previously allocated pointer, or NULL to allocate new memory
 *  @param oldsize  The previously allocated size
 *  @param newsize  New size to allocate
 *  @returns A pointer to allocated memory, or NULL on failure.
 */
void *morpho_allocate(void *old, size_t oldsize, size_t newsize) {
    if (newsize == 0) {
        free(old);
        return NULL;
    }

    return realloc(old, newsize);
}
