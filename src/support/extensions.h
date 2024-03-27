/** @file extensions.h
 *  @author T J Atherton
 *
 *  @brief Morpho extensions
 */

/** Extensions are libraries written in C, or a langauge that links with C, that are loaded dynamically at runtime using dlopen().
 * As these are loaded, they may defined functions and classes that can then be used by morpho programs */

#ifndef extensions_h
#define extensions_h

#include <stdbool.h>

#define MORPHO_EXTENSIONINITIALIZE "initialize" // Function to call upon initialization
#define MORPHO_EXTENSIONFINALIZE "finalize"     // Function to call upon finalization

bool extension_load(char *name, dictionary **functiontable, dictionary **classtable);

void extensions_initialize(void);
void extensions_finalize(void);

#endif /* extensions_h */
