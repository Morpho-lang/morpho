/** @file random.c
 *  @author T J Atherton and others (see below)
 *
 *  @brief Random number generation
*/

#ifndef random_h
#define random_h

#include <stdio.h>
#include <stdint.h>
#include <time.h>

double random_double(void);
unsigned int random_int(void);
void random_initialize(void);

#endif /* random_h */
