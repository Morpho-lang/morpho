/** @file complexmatrix.h
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#ifndef complexmatrix_h
#define complexmatrix_h

#include "matrix.h"

/* -------------------------------------------------------
 * ComplexMatrix veneer class
 * ------------------------------------------------------- */

#define COMPLEXMATRIX_CLASSNAME                   "ComplexMatrix"

#define COMPLEXMATRIX_CONJTRANSPOSE_METHOD        "conjTranspose"

extern objecttype objectcomplexmatrixtype;
#define OBJECT_COMPLEXMATRIX objectcomplexmatrixtype

typedef objectmatrix objectcomplexmatrix;

/** Tests whether an object is a complex matrix */
#define MORPHO_ISCOMPLEXMATRIX(val) object_istype(val, OBJECT_COMPLEXMATRIX)

/** Gets the object as a complex matrix */
#define MORPHO_GETCOMPLEXMATRIX(val) ((objectcomplexmatrix *) MORPHO_GETOBJECT(val))

void complexmatrix_initialize(void);

#endif
