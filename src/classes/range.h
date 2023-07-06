/** @file range.h
 *  @author T J Atherton
 *
 *  @brief Defines range object type and Range class
 */

#ifndef range_h
#define range_h

#include "object.h"

#define RANGE_CLASSNAME "Range"

#define RANGE_ARGS                        "RngArgs"
#define RANGE_ARGS_MSG                    "Range expects numerical arguments: a start, an end and an optional stepsize."

int range_count(objectrange *range);
value range_iterate(objectrange *range, unsigned int i);

void range_initialize(void);
void range_finalize(void);

#endif
