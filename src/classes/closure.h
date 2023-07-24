/** @file closure.h
 *  @author T J Atherton
 *
 *  @brief Defines closure object type and Closure class
 */

#ifndef closure_h
#define closure_h

#include "object.h"

/* -------------------------------------------------------
 * Closure objects
 * ------------------------------------------------------- */

extern objecttype objectclosuretype;
#define OBJECT_CLOSURE objectclosuretype

typedef struct {
    object obj;
    objectfunction *func;
    int nupvalues;
    objectupvalue *upvalues[];
} objectclosure;

objectclosure *object_newclosure(objectfunction *sf, objectfunction *func, indx np);

/** Tests whether an object is a closure */
#define MORPHO_ISCLOSURE(val) object_istype(val, OBJECT_CLOSURE)

/** Gets the object as a closure */
#define MORPHO_GETCLOSURE(val)   ((objectclosure *) MORPHO_GETOBJECT(val))

/** Retrieve the function object from a closure */
#define MORPHO_GETCLOSUREFUNCTION(val)  (((objectclosure *) MORPHO_GETOBJECT(val))->func)

/* -------------------------------------------------------
 * Closure veneer class
 * ------------------------------------------------------- */

#define CLOSURE_CLASSNAME "Closure"

/* -------------------------------------------------------
 * Closure error messages
 * ------------------------------------------------------- */

/* -------------------------------------------------------
 * Closure interface
 * ------------------------------------------------------- */

/* Initialization/finalization */
void closure_initialize(void);
void closure_finalize(void);

#endif
