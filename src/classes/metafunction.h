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
    value name;
    varray_value fns; 
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

objectmetafunction *object_newmetafunction(value name);
bool metafunction_wrap(value name, value fn, value *out);

bool metafunction_add(objectmetafunction *f, value fn);
bool metafunction_typefromvalue(value v, value *out);
bool metafunction_resolve(objectmetafunction *f, int nargs, value *args, value *fn);

void metafunction_initialize(void);

#endif
