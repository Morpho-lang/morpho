/** @file newlinalg.c
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#include "newlinalg.h"

/* -------------------------------------------------------
 * Initialization and finalization
 * ------------------------------------------------------- */

void newlinalg_initialize(void) { 
    xmatrix_initialize();
}

void newlinalg_finalize(void) { 
}
