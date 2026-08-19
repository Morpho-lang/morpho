/** @file geometry.h
 *  @author T J Atherton
 *
 *  @brief Geometry wrapper
 */

#ifndef geometry_h
#define geometry_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "mesh.h"
#include "field.h"
#include "selection.h"
#include "functional.h"
#include "fespace.h"
#include "integrate.h"

void geometry_initialize(void);

#endif

#endif /* geometry_h */
