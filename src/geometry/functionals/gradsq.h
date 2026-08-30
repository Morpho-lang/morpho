/** @file gradsq.h
 *  @author T J Atherton
 *
 *  @brief GradSq functional and P1 gradient kernels
 */

#ifndef morpho_functionals_gradsq_h
#define morpho_functionals_gradsq_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "functional.h"

/* -------------------------------------------------------
 * GradSq veneer class
 * ------------------------------------------------------- */

#define GRADSQ_CLASSNAME               "GradSq"

bool gradsq_evaluategradient(objectmesh *mesh, objectfield *field, int nv, int *vid, double *out);
bool gradsq_evaluategradient3d(objectmesh *mesh, objectfield *field, int nv, int *vid, double *out);
bool gradsq_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, fieldref *ref);
void _gradsq_initfield(objectinstance *self, value fieldval);

void gradsq_initialize(void);

#endif

#endif /* morpho_functionals_gradsq_h */
