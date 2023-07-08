/** @file veneer.c
 *  @author T J Atherton
 *
 *  @brief Veneer classes over built in objects
 */

#include "morpho.h"
#include "veneer.h"

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void veneer_initialize(void) {
    morpho_defineerror(CLASS_INVK, ERROR_HALT, CLASS_INVK_MSG);
}
