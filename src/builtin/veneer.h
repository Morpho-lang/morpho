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
#include "err.h"

#include "matrix.h"

/* ---------------------------
 * Veneer classes
 * --------------------------- */

#define CLASS_INVK                        "ClssInvk"
#define CLASS_INVK_MSG                    "Cannot invoke method '%s' on a class."
 
void veneer_initialize(void);

#endif /* veneer_h */
