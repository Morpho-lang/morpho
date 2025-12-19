/** @file newlinalg.c
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#define ACCELERATE_NEW_LAPACK
#define MORPHO_INCLUDE_LINALG

#include "newlinalg.h"

/* -------------------------------------------------------
 * Errors
 * ------------------------------------------------------- */

void linalg_raiseerror(vm *v, linalgError_t err) {
    switch (err) {
        case LINALGERR_OK: break;
        case LINALGERR_INCOMPATIBLE_DIM: morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES); break;
        case LINALGERR_INDX_OUT_OF_BNDS: morpho_runtimeerror(v, LINALG_INDICESOUTSIDEBOUNDS); break;
        case LINALGERR_MATRIX_SINGULAR: morpho_runtimeerror(v, LINALG_SINGULAR); break;
        case LINALGERR_NOT_SQUARE: morpho_runtimeerror(v, LINALG_NOTSQ); break;
        case LINALGERR_LAPACK_INVLD_ARGS: morpho_runtimeerror(v, LINALG_LAPACK_ARGS); break;
        case LINALGERR_ALLOC: morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); break;
    }
}

/* -------------------------------------------------------
 * Initialization and finalization
 * ------------------------------------------------------- */

void newlinalg_initialize(void) { 
    xmatrix_initialize();
    
    morpho_defineerror(LINALG_INCOMPATIBLEMATRICES, ERROR_HALT, LINALG_INCOMPATIBLEMATRICES_MSG);
    morpho_defineerror(LINALG_INDICESOUTSIDEBOUNDS, ERROR_HALT, LINALG_INDICESOUTSIDEBOUNDS_MSG);
    morpho_defineerror(LINALG_SINGULAR,             ERROR_HALT, LINALG_SINGULAR_MSG);
    morpho_defineerror(LINALG_NOTSQ,                ERROR_HALT, LINALG_NOTSQ_MSG);
    morpho_defineerror(LINALG_LAPACK_ARGS,          ERROR_HALT, LINALG_LAPACK_ARGS_MSG);
}

void newlinalg_finalize(void) { 
}
