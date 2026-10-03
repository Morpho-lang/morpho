/** @file linalg_includes.h
 *  @author T J Atherton
 * 
 *  @brief Include BLAS/LAPACK for use by Morpho. Not a public header. 
 */

#ifndef linalg_includes_h
#define linalg_includes_h

/** Use Apple's Accelerate library for LAPACK and BLAS */
#ifdef __APPLE__
#ifdef MORPHO_LINALG_USE_ACCELERATE
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define MATRIX_LAPACK_PRESENT
#endif
#endif

/** Otherwise, use LAPACKE */
#ifndef MATRIX_LAPACK_PRESENT
#include <cblas.h>
#include <lapacke.h>
#define MORPHO_LINALG_USE_LAPACKE
#define MATRIX_LAPACK_PRESENT
#endif

#ifdef MORPHO_LINALG_USE_LAPACKE
typedef lapack_complex_double linalg_complexdouble_t;
typedef lapack_int            linalg_int_t;
#else
typedef __LAPACK_double_complex linalg_complexdouble_t;
typedef __LAPACK_int            linalg_int_t;
#endif

#endif /* linalg_includes_h */
