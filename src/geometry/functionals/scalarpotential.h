/** @file scalarpotential.h
 *  @author T J Atherton
 *
 *  @brief ScalarPotential functional
 */

#ifndef morpho_functionals_scalarpotential_h
#define morpho_functionals_scalarpotential_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

/* -------------------------------------------------------
 * ScalarPotential veneer class
 * ------------------------------------------------------- */

#define SCALARPOTENTIAL_CLASSNAME             "ScalarPotential"

#define SCALARPOTENTIAL_FUNCTION_PROPERTY     "function"
#define SCALARPOTENTIAL_GRADFUNCTION_PROPERTY "gradfunction"

/* -------------------------------------------------------
 * ScalarPotential error messages
 * ------------------------------------------------------- */

#define SCALARPOTENTIAL_FNCLLBL               "SclrPtFnCllbl"
#define SCALARPOTENTIAL_FNCLLBL_MSG           "ScalarPotential function is not callable."

void scalarpotential_initialize(void);

#endif

#endif /* morpho_functionals_scalarpotential_h */
