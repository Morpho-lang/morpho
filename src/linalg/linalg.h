
/** @file linalg.h
 *  @author T J Atherton
 *
 *  @brief Improved linear algebra library
*/

#ifndef linalg_h
#define linalg_h

#include "morpho.h"

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
    LINALGERR_NON_NUMERICAL,     // Non numerical args supplied
    LINALGERR_INVLD_ARG,         // Invalid argument supplied
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

#define LINALG_INVLDARGS                  "LnAlgMtrxInvldArg"
#define LINALG_INVLDARGS_MSG              "Invalid arguments to matrix method."

#define LINALG_NNNMRCL_ARG                "LnAlgMtrxNnNmrclArg"
#define LINALG_NNNMRCL_ARG_MSG            "Matrix method requires numerical arguments."

#define LINALG_NORMARGS                   "LnAlgMtrxNrmArgs"
#define LINALG_NORMARGS_MSG               "Method 'norm' requires a supported argument: 1 or inf."

#define LINALG_ARITHARGS                  "LnAlgInvldArg"
#define LINALG_ARITHARGS_MSG              "Matrix arithmetic methods expect a matrix or number as their argument."

#define LINALG_INVLDINDICES               "LnAlgInvldIndx"
#define LINALG_INVLDINDICES_MSG           "Matrices require two arguments for indexing."

/* -------------------------------------------------------
 * Interface
 * ------------------------------------------------------- */

void linalg_raiseerror(vm *v, linalgError_t err);

/** Macros to simplify error checking:
    - evaluates expression f that returns linalgError_t;
    - if an error occurred, raises the corresponding error in a vm called v */
#define LINALG_ERRCHECKVM(f) { linalgError_t err = f; if (err!=LINALGERR_OK) linalg_raiseerror(v, err); }

/** As for LINALG_ERRCHECKVM but additionally jumps to a given label */
#define LINALG_ERRCHECKVMGOTO(f, label) { linalgError_t err = f; if (err!=LINALGERR_OK) { linalg_raiseerror(v, err); goto label; }}

/** As for LINALG_ERRCHECKVM but additionally returnsl */
#define LINALG_ERRCHECKVMRETURN(f, ret) { linalgError_t err = f; if (err!=LINALGERR_OK) { linalg_raiseerror(v, err); return ret; }}

/** Similar to the above, except returns the error rather than raising it */
#define LINALG_ERRCHECKRETURN(f) { linalgError_t err = f; if (err!=LINALGERR_OK) return err; }

/* -------------------------------------------------------
 * Include the rest of the library
 * ------------------------------------------------------- */

#ifdef MORPHO_INCLUDE_LINALG
#include "matrix.h"
#include "complexmatrix.h"
#endif

/* -------------------------------------------------------
 * Initialization and finalization
 * ------------------------------------------------------- */

void linalg_initialize(void);

#endif /* linalg_h */
