/** @file range.h
 *  @author T J Atherton
 *
 *  @brief Defines range object type and Range class
 */

#ifndef range_h
#define range_h

#include "object.h"

/* -------------------------------------------------------
 * Range objects
 * ------------------------------------------------------- */

extern objecttype objectrangetype;
#define OBJECT_RANGE objectrangetype

typedef struct {
    object obj;
    unsigned int nsteps;
    value start;
    value end;
    value step;
    bool inclusive; 
} objectrange;

/** Tests whether an object is a range */
#define MORPHO_ISRANGE(val) object_istype(val, OBJECT_RANGE)

/** Gets the object as a range */
#define MORPHO_GETRANGE(val)   ((objectrange *) MORPHO_GETOBJECT(val))

/** Creates a new range object */
objectrange *object_newrange(value start, value end, value step, bool inclusive, errorid *errid);

/* -------------------------------------------------------
 * Range veneer class
 * ------------------------------------------------------- */

#define RANGE_CLASSNAME                   "Range"

#define RANGE_INCLUSIVE_CONSTRUCTOR       "InclusiveRange"

/* -------------------------------------------------------
 * Range error messages
 * ------------------------------------------------------- */

#define RANGE_ARGS                        "RngArgs"
#define RANGE_ARGS_MSG                    "Range expects numerical arguments: a start, an end and an optional stepsize."

#define RANGE_STPSZ                       "RngStpSz"
#define RANGE_STPSZ_MSG                   "Range stepsize too small."

/* -------------------------------------------------------
 * Range interface
 * ------------------------------------------------------- */

int range_count(objectrange *range);
value range_iterate(objectrange *range, unsigned int i);

void range_initialize(void);

#endif
