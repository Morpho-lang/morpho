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

typedef bool (normsq_nodefn) (objectfield *q, grade g, elementid el, int indx, void *ref);

/** Visit every node of `q`, optionally restricted by `sel`. */
static bool normsq_foreachnode(objectfield *q, objectselection *sel, normsq_nodefn *fn, void *ref) {
    for (grade g=0; g<(grade) q->ngrades; g++) {
        unsigned int dof=field_dofforgrade(q, g);
        if (dof==0) continue;
        elementid nel=mesh_nelementsforgrade(q->mesh, g);
        for (elementid el=0; el<nel; el++) {
            if (sel && !selection_isselected(sel, g, el)) continue;
            for (unsigned int indx=0; indx<dof; indx++) {
                if (!fn(q, g, el, (int) indx, ref)) return false;
            }
        }
    }
    return true;
}

/** Compute |q|^2 at one node. */
static bool normsq_nodedot(objectfield *q, grade g, elementid el, int indx, double *out) {
    unsigned int nent;
    double *p;
    if (!field_getelementaslist(q, g, el, indx, &nent, &p)) return false;
    *out=functional_vecdot(nent, p, p);
    return true;
}

/** Write |q|^2 into a scalar Field with the same node layout. */
static bool normsq_writeintegrand(objectfield *q, grade g, elementid el, int indx, void *ref) {
    double nsq;
    if (!normsq_nodedot(q, g, el, indx, &nsq)) return false;
    return field_setelement((objectfield *) ref, g, el, indx, MORPHO_FLOAT(nsq));
}

/** Add |q|^2 at one node into a running total. */
static bool normsq_addintegrand(objectfield *q, grade g, elementid el, int indx, void *ref) {
    double nsq;
    if (!normsq_nodedot(q, g, el, indx, &nsq)) return false;
    *((double *) ref)+=nsq;
    return true;
}

/** Compute d/dq |q|^2 = 2q at one node. */
static bool normsq_writefieldgrad(objectfield *q, grade g, elementid el, int indx, void *ref) {
    unsigned int nent, ng;
    double *p, *gp;
    if (!field_getelementaslist(q, g, el, indx, &nent, &p)) return false;
    if (!field_getelementaslist((objectfield *) ref, g, el, indx, &ng, &gp) || ng!=nent) return false;
    for (unsigned int i=0; i<nent; i++) functional_accum(&gp[i], 2.0*p[i]);
    return true;
}

/** Single-element integrand: vertex |q|^2 (existing dispatch). */
bool normsq_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    fieldref *eref = ref;
    return normsq_nodedot(eref->field, MESH_GRADE_VERTEX, id, 0, out);
}

/** Full-map integrand: scalar Field with the input's node layout. */
static bool normsq_mapintegrand(vm *v, functional_mapinfo *info, value *out) {
    fieldref *eref=info->ref;
    objectfield *q=eref->field;
    objectfield *new=object_newfield(q->mesh, MORPHO_NIL, q->fnspc, q->dof);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }

    if (!normsq_foreachnode(q, info->sel, normsq_writeintegrand, new)) {
        object_free((object *) new);
        return false;
    }

    *out=MORPHO_OBJECT(new);
    return true;
}

/** Sum |q|^2 at every node. */
static bool normsq_maptotal(vm *v, functional_mapinfo *info, value *out) {
    fieldref *eref=info->ref;
    double sum=0.0;
    if (!normsq_foreachnode(eref->field, info->sel, normsq_addintegrand, &sum)) return false;
    *out=MORPHO_FLOAT(sum);
    return true;
}

/** Analytic fieldgradient over every node; other Fields yield zero. */
static bool normsq_mapfieldgradient(vm *v, functional_mapinfo *info, value *out) {
    fieldref *eref=info->ref;
    if (!info->field) MORPHO_FAIL(v, FUNCTIONAL_ARGS);

    objectfield *new=object_newfield(info->mesh, info->field->prototype, info->field->fnspc, info->field->dof);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }
    field_zero(new);

    if (info->field==eref->field &&
        !normsq_foreachnode(eref->field, info->sel, normsq_writefieldgrad, new)) {
        object_free((object *) new);
        return false;
    }

    *out=MORPHO_OBJECT(new);
    return true;
}

value NormSq_init__field(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    _gradsq_initfield(self, MORPHO_GETARG(args, 0));
    functional_setgrade(self, MESH_GRADE_VERTEX);
    return MORPHO_NIL;
}

FUNCTIONAL_MD_REF_BIND_FORCEGRADE_START(NormSq, fieldref, gradsq_prepareref, normsq_integrand, FUNCTIONAL_ARGS, MESH_GRADE_VERTEX, fieldref_startfn)
FUNCTIONAL_MD_REF_HELPER(NormSq, integrand, fieldref, MESH_GRADE_VERTEX, normsq_mapintegrand, true, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_HELPER(NormSq, integrand_elem, fieldref, MESH_GRADE_VERTEX, functional_mapintegrandforelement, false, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_OVERLOADS(NormSq, integrand, _NormSq_integrand)
FUNCTIONAL_MD_REF__MESH_INT(NormSq, integrand, _NormSq_integrand_elem)
FUNCTIONAL_MD_REF__MESH_INT_INT(NormSq, integrand, _NormSq_integrand_elem)
FUNCTIONAL_MD_REF_MAP_COST(NormSq, total, fieldref, MESH_GRADE_VERTEX, normsq_maptotal, false, NULL, NULL, SYMMETRY_NONE, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT_COST(NormSq, fieldref, MESH_GRADE_VERTEX, NULL, SYMMETRY_NONE, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_FIELDGRADIENT_MAP_COST(NormSq, fieldref, MESH_GRADE_VERTEX, normsq_mapfieldgradient, NULL, NULL, FUNCTIONAL_COST_CHEAPEST)

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
