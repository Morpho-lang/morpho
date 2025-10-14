/** @file help.h
 *  @author T J Atherton and others (see below)
 *
 *  @brief Morpho help
*/

#ifndef help_h
#define help_h

bool morpho_help(char *query, varray_char *result);

void help_initialize(void);
void help_finalize(void);

#endif /* help_h */
