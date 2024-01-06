/** @file tuple.h
 *  @author T J Atherton
 *
 *  @brief Defines tuple object type and Tuple class
 */

#ifndef tuple_h
#define tuple_h

#include "object.h"

/* -------------------------------------------------------
 * Tuple object type
 * ------------------------------------------------------- */

extern objecttype objecttupletype;
#define OBJECT_TUPLE objecttupletype

/** A string object */
typedef struct {
    object obj;
    unsigned int length;
    value *tuple;
    value tupledata[];
} objecttuple;

/** Tests whether an object is a tuple */
#define MORPHO_ISTUPLE(val) object_istype(val, OBJECT_TUPLE)

/** Extracts the objecttuple from a value */
#define MORPHO_GETTUPLE(val)             ((objecttuple *) MORPHO_GETOBJECT(val))

/** Extracts the length from a value */
#define MORPHO_GETTUPLELENGTH(val)             ((MORPHO_GETTUPLE(val))->length)

/** Extracts the value list from a value */
#define MORPHO_GETTUPLEVALUES(val)             ((MORPHO_GETTUPLE(val))->tuple)

/** Use to create static tuples on the C stack */
#define MORPHO_STATICTUPLE(list, len)      { .obj.type=OBJECT_TUPLE, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .tuple=list, .length=len }

/* -------------------------------------------------------
 * Tuple veneer class
 * ------------------------------------------------------- */

#define TUPLE_CLASSNAME                   "Tuple"

/* -------------------------------------------------------
 * Tuple error messages
 * ------------------------------------------------------- */

#define TUPLE_IMMTBL                     "StrngImmtbl"
#define TUPLE_IMMTBL_MSG                 "Strings are immutable."

/* -------------------------------------------------------
 * Tuple interface
 * ------------------------------------------------------- */

/** Create a tuple with an (optional) list of values */
objecttuple *object_newtuple(unsigned int length, value *v);

unsigned int tuple_length(objecttuple *tuple);
bool tuple_getelement(objecttuple *tuple, int i, value *out);

void tuple_initialize(void);

#endif
