/** @file memory.h
 *  @author T J Atherton
 *
 *  @brief Morpho memory management
*/

#ifndef memory_h
#define memory_h

#include <stdlib.h>

/** Macro to redirect malloc through our memory management */
#define MORPHO_MALLOC(size) morpho_allocate(NULL, 0, size)

/** Macro to redirect free through our memory management */
#define MORPHO_FREE(x) morpho_allocate(x, 0, 0)

/** Macro to redirect realloc through our memory management */
#define MORPHO_REALLOC(x, size) morpho_allocate(x, 0, size)

void *morpho_allocate(void *old, size_t oldsize, size_t newsize);

#endif /* memory_h */
