/** @file gradsq.c
 *  @author T J Atherton
 *
 *  @brief GradSq functional and P1 gradient kernels
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
#include "gradsq.h"

/* ----------------------------------------------
 * GradSq
 * ---------------------------------------------- */

/* TODO: Support other FEspaces. Today this is vertex P1 (CG1), same kernel as
 * Nematic / NematicElectric; NormSq is vertex |q|^2. High-order |grad q|^2 is
 * Integral + grad(). */

bool gradsq_computeperpendicular(unsigned int n, double *s1, double *s2, double *out) {
    double s1s2, s2s2, sout;

    /* Compute s1 - (s1.s2) s2 / (s2.2) */
    s1s2 = functional_vecdot(n, s1, s2);
    s2s2 = functional_vecdot(n, s2, s2);
    if (fabs(s2s2)<MORPHO_EPS) return false; // Check for side of zero weight

    double temp[n];
    functional_vecscale(n, s1s2/s2s2, s2, temp);
    functional_vecsub(n, s1, temp, out);

    /* Scale by 1/|t|^2 */
    sout = functional_vecnorm(n, out);
    if (fabs(sout)<MORPHO_EPS) return false; // Check for side of zero weight

    functional_vecscale(n, 1/(sout*sout), out, out);
    return true;
}

/** Evaluates the gradient of a field quantity
 @param[in] mesh - object to use
 @param[in] field - field to compute gradient of
 @param[in] nv - number of vertices
 @param[in] vid - vertex ids
 @param[out] out - should be field->psize * mesh->dim units of storage */
bool gradsq_evaluategradient(objectmesh *mesh, objectfield *field, int nv, int *vid, double *out) {    double *f[nv]; // Field value lists
    double *x[nv]; // Vertex coordinates
    unsigned int nentries=0;

    // Get field values and vertex coordinates
    for (unsigned int i=0; i<nv; i++) {
        if (!mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i])) return false;
        if (!field_getelementaslist(field, MESH_GRADE_VERTEX, vid[i], 0, &nentries, &f[i])) return false;
    }

    double s[3][mesh->dim], t[3][mesh->dim];

    /* Vector sides */
    functional_vecsub(mesh->dim, x[1], x[0], s[0]);
    functional_vecsub(mesh->dim, x[2], x[1], s[1]);
    functional_vecsub(mesh->dim, x[0], x[2], s[2]);

    /* Perpendicular vectors. Collapsed sides contribute zero gradient. */
    if (!gradsq_computeperpendicular(mesh->dim, s[2], s[1], t[0]) ||
        !gradsq_computeperpendicular(mesh->dim, s[0], s[2], t[1]) ||
        !gradsq_computeperpendicular(mesh->dim, s[1], s[0], t[2])) {
        for (unsigned int i=0; i<mesh->dim*nentries; i++) out[i]=0;
        return true;
    }

    /* Compute the gradient */
    for (unsigned int i=0; i<mesh->dim*nentries; i++) out[i]=0;
    for (unsigned int j=0; j<nv; j++) {
        for (unsigned int i=0; i<nentries; i++) {
            functional_vecaddscale(mesh->dim, &out[i*mesh->dim], f[j][i], t[j], &out[i*mesh->dim]);
        }
    }

    return true;
}

/** Evaluates the gradient of a field quantity in 3D
 @param[in] mesh - object to use
 @param[in] field - field to compute gradient of
 @param[in] nv - number of vertices
 @param[in] vid - vertex ids
 @param[out] out - should be field->psize * mesh->dim units of storage */
bool gradsq_evaluategradient3d(objectmesh *mesh, objectfield *field, int nv, int *vid, double *out) {
    double *f[nv]; // Field value lists
    double *x[nv]; // Vertex coordinates
    double xarray[nv*mesh->dim]; // Vertex coordinates
    double xtarray[nv*mesh->dim]; // Vertex coordinates
    unsigned int nentries=0;

    // Get field values and vertex coordinates
    for (unsigned int i=0; i<nv; i++) {
        if (!mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i])) return false;
        if (!field_getelementaslist(field, MESH_GRADE_VERTEX, vid[i], 0, &nentries, &f[i])) return false;
    }

    // Build a matrix such that the columns are x_i - x_0
    for (unsigned int i=1; i<nv; i++) {
        functional_vecsub(mesh->dim, x[i], x[0], &xarray[(i-1)*mesh->dim]);
    }

    for (unsigned int i=0; i<mesh->dim*nentries; i++) out[i]=0;

    objectmatrix M = MORPHO_STATICMATRIX(xarray, mesh->dim, mesh->dim);
    objectmatrix Mt = MORPHO_STATICMATRIX(xtarray, mesh->dim, mesh->dim);
    matrix_transpose(&M, &Mt);

    objectmatrix grad = MORPHO_STATICMATRIX(out, mesh->dim, nentries);

    // Loop over elements of the field
    for (unsigned int i=0; i<nentries; i++) {
        // Copy across the field values to form the rhs
        for (unsigned int j=0; j<mesh->dim; j++) out[i*mesh->dim+j] = f[j+1][i]-f[0][i];
    }

    // Solve to obtain the gradient of each element
    matrix_solvesmall(&Mt, &grad);

    return true;
}

/** Prepares the gradsq reference */
bool gradsq_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, fieldref *ref) {
    bool success=false, grdset=false;
    value field=MORPHO_NIL, grd=MORPHO_NIL;

    if (objectinstance_getpropertyinterned(self, functional_fieldproperty, &field) &&
        MORPHO_ISFIELD(field)) {
        ref->field=MORPHO_GETFIELD(field);
        success=true;
    }

    if (objectinstance_getpropertyinterned(self, functional_gradeproperty, &grd) &&
        MORPHO_ISINTEGER(grd)) {
        ref->grade=MORPHO_GETINTEGERVALUE(grd);
        if (ref->grade>0) grdset=true;
    }
    if (!grdset) ref->grade=mesh_maxgrade(mesh);

    return success;
}

/** Clones the nematic reference with a given substitute field */
void *gradsq_cloneref(void *ref, objectfield *field, objectfield *sub) {
    fieldref *nref = (fieldref *) ref;
    fieldref *clone = MORPHO_MALLOC(sizeof(fieldref));
    
    if (clone) {
        *clone = *nref;
        if (clone->field==field) clone->field=sub;
    }
    
    return clone;
}

/** Calculate the |grad q|^2 energy */
bool gradsq_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    fieldref *eref = ref;
    double size=0; // Length area or volume of the element
    double grad[eref->field->psize*mesh->dim];

    if (!functional_elementsize(v, mesh, eref->grade, id, nv, vid, &size)) return false;

    if (eref->grade==2) {
        if (!gradsq_evaluategradient(mesh, eref->field, nv, vid, grad)) return false;
    } else if (eref->grade==3) {
        if (!gradsq_evaluategradient3d(mesh, eref->field, nv, vid, grad)) return false;
    } else {
        MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) eref->grade);
    }

    double gradnrm=functional_vecnorm(eref->field->psize*mesh->dim, grad);
    *out = gradnrm*gradnrm*size;

    return true;
}

void _gradsq_initfield(objectinstance *self, value fieldval) {
    objectinstance_setproperty(self, functional_fieldproperty, fieldval);
    functional_setgrade(self, mesh_maxgrade(MORPHO_GETFIELD(fieldval)->mesh));
}

value GradSq_init__field(vm *v, int nargs, value *args) {
    _gradsq_initfield(MORPHO_GETINSTANCE(MORPHO_SELF(args)), MORPHO_GETARG(args, 0));
    return MORPHO_NIL;
}

value GradSq_init__field_int(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    _gradsq_initfield(self, MORPHO_GETARG(args, 0));
    functional_setgrade(self, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1)));
    return MORPHO_NIL;
}

FUNCTIONAL_MD_REF_BIND_START(GradSq, fieldref, gradsq_prepareref, gradsq_integrand, FUNCTIONAL_ARGS, fieldref_startfn)
FUNCTIONAL_MD_REF_INTEGRAND(GradSq, fieldref, ref.grade)
FUNCTIONAL_MD_REF_TOTAL(GradSq, fieldref, ref.grade)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(GradSq, fieldref, ref.grade, NULL, SYMMETRY_ADD)
FUNCTIONAL_MD_REF_FIELDGRADIENT(GradSq, fieldref, ref.grade, gradsq_cloneref, NULL)

MORPHO_BEGINCLASS(GradSq)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Field)", GradSq_init__field, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Field, Int)", GradSq_init__field_int, MORPHO_FN_MUTATES),

FUNCTIONAL_MD_INTEGRAND_METHODS(GradSq),
FUNCTIONAL_MD_TOTAL_METHODS(GradSq),
FUNCTIONAL_MD_GRADIENT_METHODS(GradSq),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS(GradSq)
MORPHO_ENDCLASS

void gradsq_initialize(void) {
    objectstring objclassname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objclassname));

    builtin_addclass(GRADSQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(GradSq), objclass);
}

#endif
