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
#include "instance.h"
#include "list.h"
#include "array.h"
#include "range.h"
#include "strng.h"
#include "dict.h"

#include "matrix.h"

/* ---------------------------
 * Veneer classes
 * --------------------------- */

#define ERROR_CLASSNAME "Error"

#define ERROR_TAG_PROPERTY "tag"
#define ERROR_MESSAGE_PROPERTY "message"

#define ISMEMBER_ARG                      "IsMmbrArg"
#define ISMEMBER_ARG_MSG                  "Method ismember expects a single argument."

#define CLASS_INVK                        "ClssInvk"
#define CLASS_INVK_MSG                    "Cannot invoke method '%s' on a class."

#define ERROR_ARGS                        "ErrorArgs"
#define ERROR_ARGS_MSG                    "Error much be called with a tag and a default message as arguments."

void veneer_initialize(void);

#endif /* veneer_h */
