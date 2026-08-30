/** @file scalarpotential.c
 *  @author T J Atherton
 *
 *  @brief ScalarPotential functional
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
#include "scalarpotential.h"

/* ----------------------------------------------
 * Scalar potential
 * ---------------------------------------------- */

static value scalarpotential_functionproperty;
static value scalarpotential_gradfunctionproperty;

typedef struct {
    value fn;
} scalarpotentialref;

bool scalarpotential_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, scalarpotentialref *ref) {
    ref->fn=MORPHO_NIL;
    return (objectinstance_getpropertyinterned(self, scalarpotential_functionproperty, &ref->fn) &&
            MORPHO_ISCALLABLE(ref->fn));
}

/** Evaluate the scalar potential */
bool scalarpotential_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x;
    value fn = ((scalarpotentialref *) ref)->fn;
    value args[mesh->dim];
    value ret;

    if (matrix_getcolumnptr(mesh->vert, id, &x)!=LINALGERR_OK) return false;
    for (int i=0; i<mesh->dim; i++) args[i]=MORPHO_FLOAT(x[i]);

    if (morpho_call(v, fn, mesh->dim, args, &ret)) {
        return morpho_valuetofloat(ret, out);
    }

    return false;
}

/** Evaluate the gradient of the scalar potential */
bool scalarpotential_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    double *x;
    value fn = ((scalarpotentialref *) ref)->fn;
    value args[mesh->dim];
    value ret;

    if (matrix_getcolumnptr(mesh->vert, id, &x)!=LINALGERR_OK) return false;
    for (int i=0; i<mesh->dim; i++) args[i]=MORPHO_FLOAT(x[i]);

    if (morpho_call(v, fn, mesh->dim, args, &ret)) {
        if (MORPHO_ISMATRIX(ret)) {
            objectmatrix *vf=MORPHO_GETMATRIX(ret);

            if (vf->nrows*vf->ncols==frc->nrows) {
                return functional_addtocolumn(frc, id, 1.0, vf->elements);
            }
        }
    }

    return false;
}

value ScalarPotential_init(vm *v, int nargs, value *args) {
    functional_setgrade(MORPHO_GETINSTANCE(MORPHO_SELF(args)), MESH_GRADE_VERTEX);
    return MORPHO_NIL;
}

value ScalarPotential_init__fn(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    functional_setgrade(self, MESH_GRADE_VERTEX);
    objectinstance_setproperty(self, scalarpotential_functionproperty, MORPHO_GETARG(args, 0));
    return MORPHO_NIL;
}

value ScalarPotential_init__fn_fn(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    functional_setgrade(self, MESH_GRADE_VERTEX);
    objectinstance_setproperty(self, scalarpotential_functionproperty, MORPHO_GETARG(args, 0));
    objectinstance_setproperty(self, scalarpotential_gradfunctionproperty, MORPHO_GETARG(args, 1));
    return MORPHO_NIL;
}

FUNCTIONAL_MD_REF_BIND(ScalarPotential, scalarpotentialref, scalarpotential_prepareref, scalarpotential_integrand, SCALARPOTENTIAL_FNCLLBL)
FUNCTIONAL_MD_REF_INTEGRAND_COST(ScalarPotential, scalarpotentialref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_TOTAL_COST(ScalarPotential, scalarpotentialref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAPEST)

static functional_mapcallback *ScalarPotential_choosegradient(vm *v, objectinstance *self, functional_mapinfo *info) {
    scalarpotentialref *ref = info->ref;
    value fn;

    if (objectinstance_getpropertyinterned(self, scalarpotential_gradfunctionproperty, &fn)) {
        if (!MORPHO_ISCALLABLE(fn)) {
            morpho_runtimeerror(v, SCALARPOTENTIAL_FNCLLBL);
            return NULL;
        }
        ref->fn = fn;
        info->grad = scalarpotential_gradient;
        return functional_mapgradient;
    }

    if (!_ScalarPotential_bindref(v, self, info, ref)) return NULL;
    return functional_mapnumericalgradient;
}

FUNCTIONAL_MD_REF_CHOOSEGRADIENT(ScalarPotential, scalarpotentialref, MESH_GRADE_VERTEX, ScalarPotential_choosegradient)
FUNCTIONAL_MD_REF_HESSIAN(ScalarPotential, scalarpotentialref, MESH_GRADE_VERTEX, NULL, SYMMETRY_NONE)

#define SP_MAPFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_MAPFLAGS)
#define SP_TOTALFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_TOTALFLAGS)
#define SP_ELEMFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_ELEMFLAGS)

MORPHO_BEGINCLASS(ScalarPotential)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", ScalarPotential_init, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Callable)", ScalarPotential_init__fn, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Callable, Callable)", ScalarPotential_init__fn_fn, MORPHO_FN_MUTATES),

FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(ScalarPotential, SP_MAPFLAGS, SP_ELEMFLAGS),
FUNCTIONAL_MD_TOTAL_METHODS_FLAGS(ScalarPotential, SP_TOTALFLAGS),
FUNCTIONAL_MD_GRADIENT_METHODS_FLAGS(ScalarPotential, SP_MAPFLAGS),
FUNCTIONAL_MD_HESSIAN_METHODS_FLAGS(ScalarPotential, SP_MAPFLAGS)
MORPHO_ENDCLASS

void scalarpotential_initialize(void) {
    scalarpotential_functionproperty=builtin_internsymbolascstring(SCALARPOTENTIAL_FUNCTION_PROPERTY);
    scalarpotential_gradfunctionproperty=builtin_internsymbolascstring(SCALARPOTENTIAL_GRADFUNCTION_PROPERTY);

    objectstring objclassname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objclassname));

    builtin_addclass(SCALARPOTENTIAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(ScalarPotential), objclass);

    morpho_defineerror(SCALARPOTENTIAL_FNCLLBL, ERROR_HALT, SCALARPOTENTIAL_FNCLLBL_MSG);
}

#endif
