/** @file linalg.c
 *  @author T J Atherton
 *
 *  @brief Improved linear algebra library
*/

#include "linalg.h"

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
        case LINALGERR_OP_FAILED: morpho_runtimeerror(v, LINALG_OPFAILED); break;
        case LINALGERR_NOT_SUPPORTED: morpho_runtimeerror(v, LINALG_NOTSUPPORTED); break;
        case LINALGERR_NON_NUMERICAL: morpho_runtimeerror(v, LINALG_NNNMRCL_ARG); break;
        case LINALGERR_INVLD_ARG: morpho_runtimeerror(v, LINALG_INVLDARGS); break;
        case LINALGERR_ALLOC: morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); break;
    }
}

/* -------------------------------------------------------
 * Initialization and finalization
 * ------------------------------------------------------- */

void linalg_initialize(void) {
    matrix_initialize();
    
    morpho_defineerror(LINALG_INCOMPATIBLEMATRICES, ERROR_HALT, LINALG_INCOMPATIBLEMATRICES_MSG);
    morpho_defineerror(LINALG_INDICESOUTSIDEBOUNDS, ERROR_HALT, LINALG_INDICESOUTSIDEBOUNDS_MSG);
    morpho_defineerror(LINALG_SINGULAR,             ERROR_HALT, LINALG_SINGULAR_MSG);
    morpho_defineerror(LINALG_NOTSQ,                ERROR_HALT, LINALG_NOTSQ_MSG);
    morpho_defineerror(LINALG_LAPACK_ARGS,          ERROR_HALT, LINALG_LAPACK_ARGS_MSG);
    morpho_defineerror(LINALG_OPFAILED,             ERROR_HALT, LINALG_OPFAILED_MSG);
    morpho_defineerror(LINALG_NOTSUPPORTED,         ERROR_HALT, LINALG_NOTSUPPORTED_MSG);
    morpho_defineerror(LINALG_INVLDARGS,            ERROR_HALT, LINALG_INVLDARGS_MSG);
    morpho_defineerror(LINALG_NNNMRCL_ARG,          ERROR_HALT, LINALG_NNNMRCL_ARG_MSG);
    morpho_defineerror(LINALG_NORMARGS,             ERROR_HALT, LINALG_NORMARGS_MSG);
    morpho_defineerror(LINALG_ARITHARGS,            ERROR_HALT, LINALG_ARITHARGS_MSG);
}

