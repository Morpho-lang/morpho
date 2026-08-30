/** @file size.h
 *  @author T J Atherton
 *
 *  @brief Length, Area, Volume and enclosed-size functionals
 */

#ifndef morpho_functionals_size_h
#define morpho_functionals_size_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

/* -------------------------------------------------------
 * Size veneer classes
 * ------------------------------------------------------- */

#define LENGTH_CLASSNAME               "Length"
#define AREA_CLASSNAME                 "Area"
#define AREAENCLOSED_CLASSNAME         "AreaEnclosed"
#define VOLUME_CLASSNAME               "Volume"
#define VOLUMEENCLOSED_CLASSNAME       "VolumeEnclosed"

/* -------------------------------------------------------
 * Size error messages
 * ------------------------------------------------------- */

#define VOLUMEENCLOSED_ZERO            "VolEnclZero"
#define VOLUMEENCLOSED_ZERO_MSG        "VolumeEnclosed detected an element of zero size. Check that a mesh point is not coincident with the origin."

void size_initialize(void);

#endif

#endif /* morpho_functionals_size_h */
