/** @file veneer.h
 *  @author T J Atherton
 *
 *  @brief Veneer classes over built in objects
 */

#ifndef veneer_h
#define veneer_h

#include "builtin.h"
#include "list.h"
#include "range.h"
#include "strng.h"
#include "array.h"
#include "closure.h"
#include "matrix.h"

/* ---------------------------
 * Veneer classes
 * --------------------------- */

#define OBJECT_CLASSNAME "Object"
#define DICTIONARY_CLASSNAME "Dictionary"
#define FUNCTION_CLASSNAME "Function"
#define INVOCATION_CLASSNAME "Invocation"
#define ERROR_CLASSNAME "Error"

#define ERROR_TAG_PROPERTY "tag"
#define ERROR_MESSAGE_PROPERTY "message"

#define DICTIONARY_KEYS_METHOD "keys"
#define DICTIONARY_CONTAINS_METHOD "contains"
#define DICTIONARY_REMOVE_METHOD "remove"
#define DICTIONARY_CLEAR_METHOD "clear"

#define SETINDEX_ARGS                     "SetIndxArgs"
#define SETINDEX_ARGS_MSG                 "Setindex method expects an index and a value as arguments."

#define ENUMERATE_ARGS                    "EnmrtArgs"
#define ENUMERATE_ARGS_MSG                "Enumerate method expects a single integer argument."

#define DICT_DCTKYNTFND                   "DctKyNtFnd"
#define DICT_DCTKYNTFND_MSG               "Key not found in dictionary."

#define RESPONDSTO_ARG                    "RspndsToArg"
#define RESPONDSTO_ARG_MSG                "Method respondsto expects a single string argument or no argrument."

#define HAS_ARG                    		  "HasArg"
#define HAS_ARG_MSG                		  "Method has expects a single string argument or no argument."

#define ISMEMBER_ARG                      "IsMmbrArg"
#define ISMEMBER_ARG_MSG                  "Method ismember expects a single argument."

#define DICT_DCTSTARG                     "DctStArg"
#define DICT_DCTSTARG_MSG                 "Dictionary set methods (union, intersection, difference) expect a dictionary as the argument."

#define CLASS_INVK                        "ClssInvk"
#define CLASS_INVK_MSG                    "Cannot invoke method '%s' on a class."

#define INVOCATION_ARGS                   "InvocationArgs"
#define INVOCATION_ARGS_MSG               "Invocation must be called with an object and a method name as arguments."

#define INVOCATION_METHOD                 "InvocationMethod"
#define INVOCATION_METHOD_MSG             "Method not found."

#define ERROR_ARGS                        "ErrorArgs"
#define ERROR_ARGS_MSG                    "Error much be called with a tag and a default message as arguments."

#define OBJECT_CANTCLONE                  "ObjCantClone"
#define OBJECT_CANTCLONE_MSG              "Cannot clone this object."

#define OBJECT_IMMUTABLE                  "ObjImmutable"
#define OBJECT_IMMUTABLE_MSG              "Cannot modify this object."

/* Object methods */
value Object_print(vm *v, int nargs, value *args);

void veneer_initialize(void);

#endif /* veneer_h */
