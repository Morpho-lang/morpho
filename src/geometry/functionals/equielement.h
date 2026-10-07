/** @file equielement.h
 *  @author T J Atherton
 *
 *  @brief EquiElement functional
 */

#ifndef morpho_functionals_equielement_h
#define morpho_functionals_equielement_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

/* -------------------------------------------------------
 * EquiElement veneer class
 * ------------------------------------------------------- */

#define EQUIELEMENT_CLASSNAME          "EquiElement"

#define EQUIELEMENT_WEIGHT_PROPERTY    "weight"

void equielement_initialize(void);

#endif

#endif /* morpho_functionals_equielement_h */
