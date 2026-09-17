/** @file geometry.c
 *  @author T J Atherton
 *
 *  @brief Geometry wrapper
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "geometry.h"

void geometry_initialize(void) {
    mesh_initialize();
    integrate_initialize();
    field_initialize();
    functional_initialize();
    fespace_initialize();
    selection_initialize();
}

#endif