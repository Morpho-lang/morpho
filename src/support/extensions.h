/** @file extensions.h
 *  @author T J Atherton
 *
 *  @brief Morpho extensions
 */

#ifndef extensions_h
#define extensions_h

#include <stdbool.h>

/** Extensions are libraries written in C, or a language that links with C, that are loaded dynamically at runtime using dlopen() */
bool morpho_loadextension(char *name);

void extensions_initialize(void);
void extensions_finalize(void);

#endif /* extensions_h */
