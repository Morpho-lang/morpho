/** @file geometry.c
 *  @author T J Atherton
 *
 *  @brief Geometry wrapper
 */

#include "geometry.h"

void geometry_initialize(void) {
    mesh_initialize();
    integrate_initialize();
    field_initialize();
    functional_initialize();
    discretization_initialize();
    selection_initialize();
}
