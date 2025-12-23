
/** @file newlinalg.h
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#ifndef newlinalg_h
#define newlinalg_h

#include <morpho.h>
#include <classes.h>

/* -------------------------------------------------------
 * objectmatrixerror type
 * ------------------------------------------------------- */

typedef enum {
    LINALGERR_OK,                // Operation performed correctly
    LINALGERR_INCOMPATIBLE_DIM,  // Matrices have incompatible dimensions, e.g. for multiplication
    LINALGERR_INDX_OUT_OF_BNDS,  // Index out of bounds, e.g. for access.
    LINALGERR_MATRIX_SINGULAR,   // Matrix is singular
    LINALGERR_NOT_SQUARE,        // Matrix is required to be square for this algorithm
    LINALGERR_LAPACK_INVLD_ARGS, // Invalid arguments to LAPACK routine
    LINALGERR_OP_FAILED,         // Matrix operation failed
    LINALGERR_NOT_SUPPORTED,     // Operation not supported for this matrix type
    LINALGERR_ALLOC              // Memory allocation failed
} linalgError_t;

/* -------------------------------------------------------
 * Errors
 * ------------------------------------------------------- */

#define LINALG_INCOMPATIBLEMATRICES       "LnAlgMtrxIncmptbl"
#define LINALG_INCOMPATIBLEMATRICES_MSG   "Matrices have incompatible shape."

#define LINALG_INDICESOUTSIDEBOUNDS       "LnAlgMtrxIndxBnds"
#define LINALG_INDICESOUTSIDEBOUNDS_MSG   "Matrix index out of bounds."

#define LINALG_SINGULAR                   "LnAlgMtrxSnglr"
#define LINALG_SINGULAR_MSG               "Matrix is singular."

#define LINALG_NOTSQ                      "LnAlgMtrxNtSq"
#define LINALG_NOTSQ_MSG                  "Matrix is not square."

#define LINALG_LAPACK_ARGS                "LnAlgLapackArgs"
#define LINALG_LAPACK_ARGS_MSG            "Lapack function called with invalid arguments."

#define LINALG_OPFAILED                   "LnAlgMtrxOpFld"
#define LINALG_OPFAILED_MSG               "Matrix operation failed."

#define LINALG_NOTSUPPORTED               "LnAlgMtrxNtSpprtd"
#define LINALG_NOTSUPPORTED_MSG           "Operation not supported for this matrix type."

/* -------------------------------------------------------
 * Interface
 * ------------------------------------------------------- */

void linalg_raiseerror(vm *v, linalgError_t err);

/** Macro to simplify error checking:
    - evaluates expression f that returns linalgError_t;
    - if an error occurred, raises the corresponding error in a vm called v */
#define LINALG_ERRCHECKVM(f) { linalgError_t err = f; if (err!=LINALGERR_OK) linalg_raiseerror(v, err); }

/* -------------------------------------------------------
 * Include the rest of the library
 * ------------------------------------------------------- */

#include "xmatrix.h"
#include "xcomplexmatrix.h"

#endif
