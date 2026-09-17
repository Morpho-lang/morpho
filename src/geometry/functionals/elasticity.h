/** @file elasticity.h
 *  @author T J Atherton
 *
 *  @brief LinearElasticity functional
 */

#ifndef morpho_functionals_elasticity_h
#define morpho_functionals_elasticity_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "functional.h"

/* -------------------------------------------------------
 * LinearElasticity veneer class
 * ------------------------------------------------------- */

#define LINEARELASTICITY_CLASSNAME            "LinearElasticity"

#define LINEARELASTICITY_REFERENCE_PROPERTY   "reference"
#define LINEARELASTICITY_WTBYREF_PROPERTY     "weightByReference"
#define LINEARELASTICITY_POISSON_PROPERTY     "poissonratio"
#define LINEARELASTICITY_CACHE_PROPERTY       "_refcache"

void linearelasticity_calculategram(objectmatrix *vert, int dim, int nv, int *vid, objectmatrix *gram);

void elasticity_initialize(void);

#endif

#endif /* morpho_functionals_elasticity_h */
