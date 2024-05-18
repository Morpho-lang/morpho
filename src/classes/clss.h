/** @file clss.h
 *  @author T J Atherton
 *
 *  @brief Defines class object type
 */

#ifndef clss_h
#define clss_h

#include "object.h"

/* -------------------------------------------------------
 * Class objects
 * ------------------------------------------------------- */

extern objecttype objectclasstype;
#define OBJECT_CLASS objectclasstype

typedef struct sobjectclass {
    object obj;
    struct sobjectclass *superclass; /** The class's superclass */
    value name; /** Class name */
    dictionary methods; /** Method dictionary */
    varray_value parents; /** Classes this class inherits from */
    varray_value children; /** Classes that inherit from this class */
    varray_value linearization; /** Classes that inherit from this class */
    int uid;
} objectclass;

/** Tests whether an object is a class */
#define MORPHO_ISCLASS(val) object_istype(val, OBJECT_CLASS)

/** Gets the object as a class */
#define MORPHO_GETCLASS(val)   ((objectclass *) MORPHO_GETOBJECT(val))

/** Gets the superclass */
#define MORPHO_GETSUPERCLASS(val)   (MORPHO_GETCLASS(val)->superclass)

/* -------------------------------------------------------
 * Class veneer class
 * ------------------------------------------------------- */

// #define CLASS_CLASSNAME                   "Class"
// #define CLASS_LINEARIZATION               "linearization"

/* -------------------------------------------------------
 * Class error messages
 * ------------------------------------------------------- */

#define CLASS_INVK                        "ClssInvk"
#define CLASS_INVK_MSG                    "Cannot invoke method '%s' on a class."

/* -------------------------------------------------------
 * Class interface
 * ------------------------------------------------------- */

objectclass *object_newclass(value name);
objectclass *morpho_lookupclass(value obj);

bool class_linearize(objectclass *klass);

void class_initialize(void);

#endif
