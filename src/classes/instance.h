/** @file instance.h
 *  @author T J Atherton
 *
 *  @brief Defines instance object type and Object base class
 */

#ifndef instance_h
#define instance_h

#include "object.h"

/* -------------------------------------------------------
 * Instance objects
 * ------------------------------------------------------- */

extern objecttype objectinstancetype;
#define OBJECT_INSTANCE objectinstancetype

typedef struct {
    object obj;
    objectclass *klass;
    dictionary fields;
} objectinstance;

/** Tests whether an object is a class */
#define MORPHO_ISINSTANCE(val) object_istype(val, OBJECT_INSTANCE)

/** Gets the object as a class */
#define MORPHO_GETINSTANCE(val)   ((objectinstance *) MORPHO_GETOBJECT(val))

objectinstance *object_newinstance(objectclass *klass);

/* -------------------------------------------------------
 * Object veneer class
 * ------------------------------------------------------- */

#define OBJECT_CLASSNAME "Object"

/* -------------------------------------------------------
 * Instance error messages
 * ------------------------------------------------------- */

#define OBJECT_CANTCLONE                  "ObjCantClone"
#define OBJECT_CANTCLONE_MSG              "Cannot clone this object."

#define OBJECT_IMMUTABLE                  "ObjImmutable"
#define OBJECT_IMMUTABLE_MSG              "Cannot modify this object."

#define SETINDEX_ARGS                     "SetIndxArgs"
#define SETINDEX_ARGS_MSG                 "Setindex method expects an index and a value as arguments."

#define ENUMERATE_ARGS                    "EnmrtArgs"
#define ENUMERATE_ARGS_MSG                "Enumerate method expects a single integer argument."

#define RESPONDSTO_ARG                    "RspndsToArg"
#define RESPONDSTO_ARG_MSG                "Method respondsto expects a single string argument or no argrument."

#define HAS_ARG                           "HasArg"
#define HAS_ARG_MSG                       "Method has expects a single string argument or no argument."

#define ISMEMBER_ARG                      "IsMmbrArg"
#define ISMEMBER_ARG_MSG                  "Method ismember expects a single argument."

/* -------------------------------------------------------
 * Instance interface
 * ------------------------------------------------------- */

/* Expose Object_print */
value Object_print(vm *v, int nargs, value *args);

bool objectinstance_setproperty(objectinstance *obj, value key, value val);
bool objectinstance_getproperty(objectinstance *obj, value key, value *val);
bool objectinstance_getpropertyinterned(objectinstance *obj, value key, value *val);

/* Initialization/finalization */
void instance_initialize(void);
void instance_finalize(void);

#endif
