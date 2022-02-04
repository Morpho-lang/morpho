/** @file functions.h
 *  @author T J Atherton
 *
 *  @brief Built in functions
 */

#ifndef functions_h
#define functions_h

#include <stdio.h>

void functions_initialize(void);

#define FUNCTION_RANDOM        "random"
#define FUNCTION_RANDOMINT     "randomint"
#define FUNCTION_RANDOMNORMAL  "randomnormal"
#define FUNCTION_CLOCK         "clock"
#define FUNCTION_SYSTEM        "system"

#define FUNCTION_INT           "int"
#define FUNCTION_FLOAT         "float"
#define FUNCTION_BOOL          "bool"
#define FUNCTION_MOD           "mod"
#define FUNCTION_ABS           "abs"
#define FUNCTION_ISCALLABLE    "iscallable"
#define FUNCTION_MIN           "min"
#define FUNCTION_MAX           "max"
#define FUNCTION_BOUNDS        "bounds"

#define FUNCTION_SIGN           "sign"

#define FUNCTION_APPLY         "apply"

#define FUNCTION_ARCTAN        "arctan"

#define MATH_ARGS                    "ExpctNmArgs"
#define MATH_ARGS_MSG                "Function '%s' expects numerical arguments."

#define MATH_NUMARGS                 "ExpctArgNm"
#define MATH_NUMARGS_MSG             "Function '%s' expects 1 numerical argument."

#define MATH_ATANARGS                "AtanArgNm"
#define MATH_ATANARGS_MSG            "Function 'arctan' expects either 1 or 2 numerical arguments."

#define TYPE_NUMARGS                 "TypArgNm"
#define TYPE_NUMARGS_MSG             "Function '%s' expects one argument."

#define MAX_ARGS                     "MnMxArgs"
#define MAX_ARGS_MSG                 "Function '%s' expects at least one numerical argument, list or matrix."

#endif /* functions_h */
