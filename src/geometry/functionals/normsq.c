/** @file normsq.c
 *  @author T J Atherton
 *
 *  @brief NormSq functional
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "functional.h"
#include "morpho.h"
#include "classes.h"
#include "common.h"
#include "linalg.h"
#include "sparse.h"
#include "geometry.h"
#include "gradsq.h"
#include "normsq.h"

/* ----------------------------------------------
 * NormSq
 * ---------------------------------------------- */

/** Calculate the norm squared of a field quantity */
bool normsq_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    fieldref *eref = ref;
    unsigned int nentries;
    double *entries;

    if (field_getelementaslist(eref->field, MESH_GRADE_VERTEX, id, 0, &nentries, &entries)) {
        *out = functional_vecdot(nentries, entries, entries);
        return true;
    }

    return false;
}

/** Analytic fieldgradient: d/dq |q|^2 = 2q. */
bool normsq_fieldgradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectfield *grad) {
    fieldref *eref = ref;
    unsigned int nentries;
    double *entries, *gentries;

    if (!field_getelementaslist(eref->field, MESH_GRADE_VERTEX, id, 0, &nentries, &entries)) return false;
    if (!field_getelementaslist(grad, MESH_GRADE_VERTEX, id, 0, &nentries, &gentries)) return false;

    for (unsigned int i=0; i<nentries; i++) functional_accum(&gentries[i], 2.0*entries[i]);
    return true;
}

value NormSq_init__field(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    _gradsq_initfield(self, MORPHO_GETARG(args, 0));
    functional_setgrade(self, MESH_GRADE_VERTEX);
    return MORPHO_NIL;
}

FUNCTIONAL_MD_REF_BIND_FORCEGRADE_START(NormSq, fieldref, gradsq_prepareref, normsq_integrand, FUNCTIONAL_ARGS, MESH_GRADE_VERTEX, fieldref_startfn)
FUNCTIONAL_MD_REF_INTEGRAND_COST(NormSq, fieldref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_TOTAL_COST(NormSq, fieldref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT_COST(NormSq, fieldref, MESH_GRADE_VERTEX, NULL, SYMMETRY_NONE, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_ANALYTICALFIELDGRADIENT_COST(NormSq, fieldref, MESH_GRADE_VERTEX, normsq_fieldgradient, FUNCTIONAL_COST_CHEAPEST)

MORPHO_BEGINCLASS(NormSq)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Field)", NormSq_init__field, MORPHO_FN_MUTATES),

FUNCTIONAL_MD_INTEGRAND_METHODS(NormSq),
FUNCTIONAL_MD_TOTAL_METHODS(NormSq),
FUNCTIONAL_MD_GRADIENT_METHODS(NormSq),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS(NormSq)
MORPHO_ENDCLASS

void normsq_initialize(void) {
    objectstring objclassname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objclassname));

    builtin_addclass(NORMSQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(NormSq), objclass);
}

#endif
