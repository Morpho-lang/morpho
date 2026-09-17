/** @file nematic.h
 *  @author T J Atherton
 *
 *  @brief Nematic and NematicElectric functionals
 */

#ifndef morpho_functionals_nematic_h
#define morpho_functionals_nematic_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

/* -------------------------------------------------------
 * Nematic veneer classes
 * ------------------------------------------------------- */

#define NEMATIC_CLASSNAME              "Nematic"
#define NEMATICELECTRIC_CLASSNAME      "NematicElectric"

#define NEMATIC_KSPLAY_PROPERTY        "ksplay"
#define NEMATIC_KTWIST_PROPERTY        "ktwist"
#define NEMATIC_KBEND_PROPERTY         "kbend"
#define NEMATIC_PITCH_PROPERTY         "pitch"
#define NEMATIC_DIRECTOR_PROPERTY      "director"

void nematic_initialize(void);

#endif

#endif /* morpho_functionals_nematic_h */
