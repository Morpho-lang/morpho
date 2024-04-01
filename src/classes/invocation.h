/** @file invocation.h
 *  @author T J Atherton
 *
 *  @brief Defines invocation object type and Invocation class
 */

/** Invocations (or bound methods) are created by the runtime when a method is accessed from an object without a call, e.g.
    
        var inv = Object.clone
 
    An invocation can then be called at a later point, and behaves as other callable objects like functions
 
        var a = inv()
*/

#ifndef invocation_h
#define invocation_h

#include "object.h"

/* -------------------------------------------------------
 * Invocation objects
 * ------------------------------------------------------- */

extern objecttype objectinvocationtype;
#define OBJECT_INVOCATION objectinvocationtype

typedef struct {
    object obj;
    value receiver;
    value method;
} objectinvocation;

/** Tests whether an object is an invocation */
#define MORPHO_ISINVOCATION(val) object_istype(val, OBJECT_INVOCATION)

/** Gets the object as an invocation */
#define MORPHO_GETINVOCATION(val)   ((objectinvocation *) MORPHO_GETOBJECT(val))

objectinvocation *object_newinvocation(value receiver, value method);

/* -------------------------------------------------------
 * Invocation veneer class
 * ------------------------------------------------------- */

#define INVOCATION_CLASSNAME              "Invocation"

/* -------------------------------------------------------
 * Invocation error messages
 * ------------------------------------------------------- */

#define INVOCATION_ARGS                   "InvocationArgs"
#define INVOCATION_ARGS_MSG               "Invocation must be called with an object and a method name as arguments."

#define INVOCATION_METHOD                 "InvocationMethod"
#define INVOCATION_METHOD_MSG             "Method not found."

/* -------------------------------------------------------
 * Invocation interface
 * ------------------------------------------------------- */

void invocation_initialize(void);

#endif
