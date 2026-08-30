/** @file curvature.h
 *  @author T J Atherton
 *
 *  @brief Line and surface curvature functionals
 */

#ifndef morpho_functionals_curvature_h
#define morpho_functionals_curvature_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

/* -------------------------------------------------------
 * Curvature veneer classes
 * ------------------------------------------------------- */

#define LINECURVATURESQ_CLASSNAME      "LineCurvatureSq"
#define LINETORSIONSQ_CLASSNAME        "LineTorsionSq"
#define MEANCURVATURESQ_CLASSNAME      "MeanCurvatureSq"
#define GAUSSCURVATURE_CLASSNAME       "GaussCurvature"

#define CURVATURE_INTEGRANDONLY_PROPERTY "integrandonly"
#define CURVATURE_GEODESIC_PROPERTY      "geodesic"

void curvature_initialize(void);

#endif

#endif /* morpho_functionals_curvature_h */
