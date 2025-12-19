
/** @file newlinalg.h
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#ifndef newlinalg_h
#define newlinalg_h

#include <morpho.h>
#include <classes.h>

#include "xmatrix.h"

/* -------------------------------------------------------
 * objectmatrixerror type
 * ------------------------------------------------------- */

typedef enum {
    LINALGERR_OK,                // Operation performed correctly
    LINALGERR_INCOMPATIBLE_DIM,  // Matrices have incompatible dimensions, e.g. for multiplication
    LINALGERR_INDX_OUT_OF_BNDS,  // Index out of bounds, e.g. for access.
    LINALGERR_MATRIX_SINGULAR,   // Matrix is singular
    LINALGERR_NOT_SQUARE,        // Matrix is required to be square for this algorithm
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

#endif
