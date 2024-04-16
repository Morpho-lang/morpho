/** @file metafunction.h
 *  @author T J Atherton
 *
 *  @brief Defines metafunction object type and Metafunction veneer class
 */

#ifndef metafunction_h
#define metafunction_h

#include "object.h"

/* -------------------------------------------------------
 * Metafunction objects
 * ------------------------------------------------------- */

extern objecttype objectmetafunctiontype;
#define OBJECT_METAFUNCTION objectmetafunctiontype

/** A metafunction object */
typedef struct sobjectmetafunction {
    object obj;
} objectmetafunction;

/** Gets an objectmetafunction from a value */
#define MORPHO_GETMETAFUNCTION(val)   ((objectmetafunction *) MORPHO_GETOBJECT(val))

/** Tests whether an object is a metafunction */
#define MORPHO_ISMETAFUNCTION(val) object_istype(val, OBJECT_METAFUNCTION)

/* -------------------------------------------------------
 * Metafunction veneer class
 * ------------------------------------------------------- */

#define METAFUNCTION_CLASSNAME "Metafunction"

/* -------------------------------------------------------
 * Metafunction error messages
 * ------------------------------------------------------- */

/* -------------------------------------------------------
 * Metafunction interface
 * ------------------------------------------------------- */

void metafunction_initialize(void);

#endif
