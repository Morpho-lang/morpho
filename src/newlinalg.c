/** @file newlinalg.c
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#define ACCELERATE_NEW_LAPACK
#define MORPHO_INCLUDE_LINALG

#include "newlinalg.h"

/* -------------------------------------------------------
 * Initialization and finalization
 * ------------------------------------------------------- */

void newlinalg_initialize(void) { 
    xmatrix_initialize();
}

void newlinalg_finalize(void) { 
}
