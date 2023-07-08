/** @file veneer.h
 *  @author T J Atherton
 *
 *  @brief Veneer classes over built in objects
 */

#ifndef veneer_h
#define veneer_h

#include "builtin.h"
#include "function.h"
#include "closure.h"
#include "invocation.h"
#include "list.h"
#include "array.h"
#include "range.h"
#include "strng.h"
#include "dict.h"

#include "matrix.h"

/* ---------------------------
 * Veneer classes
 * --------------------------- */

#define OBJECT_CLASSNAME "Object"
#define ERROR_CLASSNAME "Error"

#define ERROR_TAG_PROPERTY "tag"
#define ERROR_MESSAGE_PROPERTY "message"

#define SETINDEX_ARGS                     "SetIndxArgs"
#define SETINDEX_ARGS_MSG                 "Setindex method expects an index and a value as arguments."

#define ENUMERATE_ARGS                    "EnmrtArgs"
#define ENUMERATE_ARGS_MSG                "Enumerate method expects a single integer argument."

#define RESPONDSTO_ARG                    "RspndsToArg"
#define RESPONDSTO_ARG_MSG                "Method respondsto expects a single string argument or no argrument."

#define HAS_ARG                    		  "HasArg"
#define HAS_ARG_MSG                		  "Method has expects a single string argument or no argument."

#define ISMEMBER_ARG                      "IsMmbrArg"
#define ISMEMBER_ARG_MSG                  "Method ismember expects a single argument."

#define CLASS_INVK                        "ClssInvk"
#define CLASS_INVK_MSG                    "Cannot invoke method '%s' on a class."

#define ERROR_ARGS                        "ErrorArgs"
#define ERROR_ARGS_MSG                    "Error much be called with a tag and a default message as arguments."

#define OBJECT_CANTCLONE                  "ObjCantClone"
#define OBJECT_CANTCLONE_MSG              "Cannot clone this object."

#define OBJECT_IMMUTABLE                  "ObjImmutable"
#define OBJECT_IMMUTABLE_MSG              "Cannot modify this object."

/* Object methods */
value Object_print(vm *v, int nargs, value *args);

objectclass *object_getclass(value v);

void veneer_initialize(void);

#endif /* veneer_h */
