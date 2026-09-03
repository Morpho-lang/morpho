/** @file hydrogel.c
 *  @author T J Atherton
 *
 *  @brief Hydrogel functional
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
#include <math.h>
#include "hydrogel.h"

/* ----------------------------------------------
* Hydrogel
* ---------------------------------------------- */

static value hydrogel_aproperty;
static value hydrogel_bproperty;
static value hydrogel_cproperty;
static value hydrogel_dproperty;
static value hydrogel_phirefproperty;
static value hydrogel_phi0property;
static value hydrogel_referenceproperty;

typedef struct {
    objectmesh *refmesh;
    grade grade;
    double a, b, c, d, phiref; // Hydrogel coefficients
    value phi0; // Can be a number or a field. (Ensuring flexibility for supplying a phi0 field in the future)
} hydrogelref;

/** Prepares the reference structure from the object's properties */
bool hydrogel_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, hydrogelref *ref) {
    bool success=false;
    value refmesh=MORPHO_NIL, grade=MORPHO_NIL, phi0=MORPHO_NIL;
    value a=MORPHO_NIL, b=MORPHO_NIL, c=MORPHO_NIL, d=MORPHO_NIL, phiref=MORPHO_NIL;

    if (objectinstance_getpropertyinterned(self, hydrogel_referenceproperty, &refmesh) &&
        MORPHO_ISMESH(refmesh) &&
        objectinstance_getpropertyinterned(self, functional_gradeproperty, &grade) &&
        MORPHO_ISINTEGER(grade) &&
        objectinstance_getpropertyinterned(self, hydrogel_aproperty, &a) &&
        MORPHO_ISNUMBER(a) &&
        objectinstance_getpropertyinterned(self, hydrogel_bproperty, &b) &&
        MORPHO_ISNUMBER(b) &&
        objectinstance_getpropertyinterned(self, hydrogel_cproperty, &c) &&
        MORPHO_ISNUMBER(c) &&
        objectinstance_getpropertyinterned(self, hydrogel_dproperty, &d) &&
        MORPHO_ISNUMBER(d) &&
        objectinstance_getpropertyinterned(self, hydrogel_phirefproperty, &phiref) &&
        MORPHO_ISNUMBER(phiref) &&
        objectinstance_getpropertyinterned(self, hydrogel_phi0property, &phi0) &&
        (MORPHO_ISNUMBER(phi0) || MORPHO_ISFIELD(phi0))) {
        ref->refmesh=MORPHO_GETMESH(refmesh);
        ref->grade=MORPHO_GETINTEGERVALUE(grade);

        if (ref->grade<0) ref->grade=mesh_maxgrade(mesh);

        if (morpho_valuetofloat(a, &ref->a) &&
            morpho_valuetofloat(b, &ref->b) &&
            morpho_valuetofloat(c, &ref->c) &&
            morpho_valuetofloat(d, &ref->d) &&
            morpho_valuetofloat(phiref, &ref->phiref)) {
            ref->phi0 = phi0;
            success=true;
        }
    }
    return success;
}

/** Calculate the Hydrogel energy */
bool hydrogel_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    hydrogelref *info = (hydrogelref *) ref;
    value vphi0 = info->phi0;
    double V=0.0, V0=0.0, phi0=0.0;

    if (!functional_elementsize(v, info->refmesh, info->grade, id, nv, vid, &V0)) return false;
    if (!functional_elementsize(v, mesh, info->grade, id, nv, vid, &V)) return false;

    if (V0<1e-8) {
        morpho_runtimewarning(v, HYDROGEL_ZEEROREFELEMENT, id, V, V0);
    }

    if (fabs(V)<MORPHO_EPS) {
        *out = 0;
        return true;
    }

    // Determine phi0 either as a number or by looking up something in a field
    if (MORPHO_ISFIELD(info->phi0)) {
        objectfield *p = MORPHO_GETFIELD(info->phi0);
        if (!field_getelement(p, info->grade, id, 0, &vphi0)) MORPHO_FAILVARGS(v, HYDROGEL_FLDGRD, (unsigned int) info->grade);
    }
    if (MORPHO_ISNUMBER(vphi0)) {
        if (!morpho_valuetofloat(vphi0, &phi0)) return false;
    }

    double phi = phi0/(V/V0);
    double pr = info->phiref;
    if (phi<0 || 1-phi<0) {
        morpho_runtimewarning(v, HYDROGEL_BNDS, id, V, V0, phi, 1-phi);
    }

    if (phi>1-MORPHO_EPS) phi = 1-MORPHO_EPS;
    if (phi<MORPHO_EPS) phi = MORPHO_EPS;

    *out = (info->a * phi*log(phi) +
            info->b * (1-phi)*log(1-phi) +
            info->c * phi*(1-phi))*V +
            info->d * (log(pr/phi)/3.0 - pow((pr/phi), (2.0/3)) + 1.0)*V0;

    return true;
}

/** Calculate gradient */
bool hydrogel_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {

    hydrogelref *info = (hydrogelref *) ref;
    value vphi0 = info->phi0;
    double V=0.0, V0=0.0, phi0=0.0;

    if (!functional_elementsize(v, info->refmesh, info->grade, id, nv, vid, &V0)) return false;
    if (!functional_elementsize(v, mesh, info->grade, id, nv, vid, &V)) return false;

    if (V0<1e-8) {
        morpho_runtimewarning(v, HYDROGEL_ZEEROREFELEMENT, id, V, V0);
    }

    if (fabs(V)<MORPHO_EPS) return true;

    // Determine phi0 either as a number or by looking up something in a field
    if (MORPHO_ISFIELD(info->phi0)) {
        objectfield *p = MORPHO_GETFIELD(info->phi0);
        if (!field_getelement(p, info->grade, id, 0, &vphi0)) MORPHO_FAILVARGS(v, HYDROGEL_FLDGRD, (unsigned int) info->grade);
    }
    if (MORPHO_ISNUMBER(vphi0)) {
        if (!morpho_valuetofloat(vphi0, &phi0)) return false;
    }

    double phi = phi0/(V/V0);
    double pr = info->phiref;
    if (phi<0 || 1-phi<0) {
        morpho_runtimewarning(v, HYDROGEL_BNDS, id, V, V0, phi, 1-phi);
    }

    if (phi>1-MORPHO_EPS) phi = 1-MORPHO_EPS;
    if (phi<MORPHO_EPS) phi = MORPHO_EPS;

    double grad = (-info->a * phi +
            info->b * ( phi + log(1-phi) ) +
            info->c * phi*phi +
            info->d * (pr/phi0) * ((phi/pr)/3.0 - (2.0/3) * pow((phi/pr), (1.0/3)) ) );

    // Compute grad * element gradient
    if (!functional_elementgradient_scale(v, mesh, info->grade, id, nv, vid, frc, grad)) return false;

    return true;
}

value Hydrogel_init__mesh(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    objectmesh *mesh = MORPHO_GETMESH(MORPHO_GETARG(args, 0));
    value grade=MORPHO_INTEGER(-1);
    value a=MORPHO_NIL, b=MORPHO_NIL, c=MORPHO_NIL, d=MORPHO_NIL, phiref=MORPHO_NIL, phi0=MORPHO_NIL;

    builtin_options(v, nargs, args, NULL, 7,
                    hydrogel_aproperty, &a,
                    hydrogel_bproperty, &b,
                    hydrogel_cproperty, &c,
                    hydrogel_dproperty, &d,
                    hydrogel_phirefproperty, &phiref,
                    hydrogel_phi0property, &phi0,
                    functional_gradeproperty, &grade);

    objectinstance_setproperty(self, hydrogel_aproperty, a);
    objectinstance_setproperty(self, hydrogel_bproperty, b);
    objectinstance_setproperty(self, hydrogel_cproperty, c);
    objectinstance_setproperty(self, hydrogel_dproperty, d);
    objectinstance_setproperty(self, hydrogel_phirefproperty, phiref);
    objectinstance_setproperty(self, hydrogel_phi0property, phi0);
    objectinstance_setproperty(self, hydrogel_referenceproperty, MORPHO_GETARG(args, 0));

    if (MORPHO_ISINTEGER(grade) && MORPHO_GETINTEGERVALUE(grade)>=0) {
        functional_setgrade(self, MORPHO_GETINTEGERVALUE(grade));
    } else {
        functional_setgrade(self, mesh_maxgrade(mesh));
    }
    return MORPHO_NIL;
}

FUNCTIONAL_MD_REF_BIND(Hydrogel, hydrogelref, hydrogel_prepareref, hydrogel_integrand, FUNCTIONAL_ARGS)
FUNCTIONAL_MD_REF_INTEGRAND(Hydrogel, hydrogelref, ref.grade)
FUNCTIONAL_MD_REF_TOTAL(Hydrogel, hydrogelref, ref.grade)
FUNCTIONAL_MD_REF_GRADIENT(Hydrogel, hydrogelref, ref.grade, hydrogel_gradient, SYMMETRY_ADD)

MORPHO_BEGINCLASS(Hydrogel)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Mesh)", Hydrogel_init__mesh, MORPHO_FN_MUTATES|MORPHO_FN_OPTARGS),

FUNCTIONAL_MD_INTEGRAND_METHODS(Hydrogel),
FUNCTIONAL_MD_TOTAL_METHODS(Hydrogel),
FUNCTIONAL_MD_GRADIENT_METHODS(Hydrogel)
MORPHO_ENDCLASS

void hydrogel_initialize(void) {
    hydrogel_aproperty=builtin_internsymbolascstring(HYDROGEL_A_PROPERTY);
    hydrogel_bproperty=builtin_internsymbolascstring(HYDROGEL_B_PROPERTY);
    hydrogel_cproperty=builtin_internsymbolascstring(HYDROGEL_C_PROPERTY);
    hydrogel_dproperty=builtin_internsymbolascstring(HYDROGEL_D_PROPERTY);
    hydrogel_phirefproperty=builtin_internsymbolascstring(HYDROGEL_PHIREF_PROPERTY);
    hydrogel_phi0property=builtin_internsymbolascstring(HYDROGEL_PHI0_PROPERTY);
    hydrogel_referenceproperty=builtin_internsymbolascstring(HYDROGEL_REFERENCE_PROPERTY);

    value objclass = builtin_findclassfromcstring(OBJECT_CLASSNAME);

    builtin_addclass(HYDROGEL_CLASSNAME, MORPHO_GETCLASSDEFINITION(Hydrogel), objclass);

    morpho_defineerror(HYDROGEL_FLDGRD, ERROR_HALT, HYDROGEL_FLDGRD_MSG);
    morpho_defineerror(HYDROGEL_ZEEROREFELEMENT, ERROR_WARNING, HYDROGEL_ZEEROREFELEMENT_MSG);
    morpho_defineerror(HYDROGEL_BNDS, ERROR_WARNING, HYDROGEL_BNDS_MSG);
}

#endif
