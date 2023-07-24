/** @file upvalue.h
 *  @author T J Atherton
 *
 *  @brief Defines upvalue object type
 */

#ifndef upvalue_h
#define upvalue_h

#include "object.h"

/** Upvalues are used by the virtual machine to implement closures; they are not visible to the user */

/* -------------------------------------------------------
 * Upvalue structure
 * ------------------------------------------------------- */

/** An upvalue descriptor */
typedef struct {
    bool islocal; /** Set if the upvalue is local to this function */
    indx reg; /** An index that either:
                  if islocal - refers to the register
               OR otherwise  - refers to the upvalue array in the current closure */
} upvalue;

DECLARE_VARRAY(upvalue, upvalue)
DECLARE_VARRAY(varray_upvalue, varray_upvalue)

/* ---------------------------
 * Upvalue objects
 * --------------------------- */

extern objecttype objectupvaluetype;
#define OBJECT_UPVALUE objectupvaluetype

typedef struct sobjectupvalue {
    object obj;
    value* location; /** Pointer to the location of the upvalue */
    value  closed; /** Closed value of the upvalue */
    struct sobjectupvalue *next;
} objectupvalue;

void object_upvalueinit(objectupvalue *c);
objectupvalue *object_newupvalue(value *reg);

/** Gets an upvalue from a value */
#define MORPHO_GETUPVALUE(val)   ((objectupvalue *) MORPHO_GETOBJECT(val))

/** Tests whether an object is an upvalue */
#define MORPHO_ISUPVALUE(val) object_istype(val, object_upvaluetype)

/* -------------------------------------------------------
 * Upvalue error messages
 * ------------------------------------------------------- */

/* -------------------------------------------------------
 * Upvalue interface
 * ------------------------------------------------------- */

/* Initialization/finalization */
void upvalue_initialize(void);
void upvalue_finalize(void);

#endif
