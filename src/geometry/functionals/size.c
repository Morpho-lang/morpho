/** @file size.c
 *  @author T J Atherton
 *
 *  @brief Length, Area, Volume and enclosed-size functionals
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
#include <float.h>
#include <math.h>
#include "size.h"

bool length_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out);
bool area_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out);
bool volume_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out);

/** Calculate element size */
bool functional_elementsize(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, double *out) {
    switch (g) {
        case 1: return length_integrand(v, mesh, id, nv, vid, NULL, out);
        case 2: return area_integrand(v, mesh, id, nv, vid, NULL, out);
        case 3: return volume_integrand(v, mesh, id, nv, vid, NULL, out);
    }
    return false;
}

bool length_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale);
bool area_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale);
bool volume_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale);

bool length_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc);
bool area_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc);
bool volume_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc);

/** Calculate a scaled element gradient */
bool functional_elementgradient_scale(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, objectmatrix *frc, double scale) {
    switch (g) {
        case 1: return length_gradient_scale(v, mesh, id, nv, vid, NULL, frc, scale);
        case 2: return area_gradient_scale(v, mesh, id, nv, vid, NULL, frc, scale);
        case 3: return volume_gradient_scale(v, mesh, id, nv, vid, NULL, frc, scale);
    }
    return false;
}

/** Calculate element gradient */
bool functional_elementgradient(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, objectmatrix *frc) {
    switch (g) {
        case 1: return length_gradient(v, mesh, id, nv, vid, NULL, frc);
        case 2: return area_gradient(v, mesh, id, nv, vid, NULL, frc);
        case 3: return volume_gradient(v, mesh, id, nv, vid, NULL, frc);
    }
    return false;
}

/* ----------------------------------------------
 * Length
 * ---------------------------------------------- */

/** Calculate area */
bool length_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    if (nv!=2) { *out=0; return true; }
    double *x[nv], s0[mesh->dim];
    for (int j=0; j<nv; j++) matrix_getcolumnptr(mesh->vert, vid[j], &x[j]);

    functional_vecsub(mesh->dim, x[1], x[0], s0);

    *out=functional_vecnorm(mesh->dim, s0);
    return true;
}

/** Calculate scaled gradient */
bool length_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale) {
    double *x[nv], s0[mesh->dim], norm;
    for (int j=0; j<nv; j++) matrix_getcolumnptr(mesh->vert, vid[j], &x[j]);

    functional_vecsub(mesh->dim, x[1], x[0], s0);
    norm=functional_vecnorm(mesh->dim, s0);
    if (norm<MORPHO_EPS) return true;
    
    if (!functional_addtocolumn(frc, vid[0], -1.0/norm*scale, s0)) return false;
    if (!functional_addtocolumn(frc, vid[1], 1./norm*scale, s0)) return false;

    return true;
}

/** Calculate gradient */
bool length_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    return length_gradient_scale(v, mesh, id, nv, vid, NULL, frc, 1.0);
}

FUNCTIONAL_INIT(Length, MESH_GRADE_LINE)
FUNCTIONAL_MD_INTEGRAND_COST(Length, MESH_GRADE_LINE, length_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_TOTAL_COST(Length, MESH_GRADE_LINE, length_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_GRADIENT_COST(Length, MESH_GRADE_LINE, length_gradient, SYMMETRY_ADD, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_HESSIAN(Length, MESH_GRADE_LINE, length_integrand)

MORPHO_BEGINCLASS(Length)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", Length_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(Length),
FUNCTIONAL_MD_TOTAL_METHODS(Length),
FUNCTIONAL_MD_GRADIENT_METHODS(Length),
FUNCTIONAL_MD_HESSIAN_METHODS(Length)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Enclosed area
 * ---------------------------------------------- */

/** Calculate area enclosed */
bool areaenclosed_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], cx[mesh->dim], normcx;
    for (int j=0; j<nv; j++) matrix_getcolumnptr(mesh->vert, vid[j], &x[j]);

    if (mesh->dim==2) {
        functional_veccross2d(x[0], x[1], cx);
        normcx=fabs(cx[0]);
    } else {
        functional_veccross(x[0], x[1], cx);
        normcx=functional_vecnorm(mesh->dim, cx);
    }

    *out=0.5*normcx;

    return true;
}

/** Analytic gradient of 1/2 |x0 × x1|. */
bool areaenclosed_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    if (nv!=2) return true;
    double *x[2], x0[3]={0,0,0}, x1[3]={0,0,0}, c[3], f0[3], f1[3], nrm;
    for (int j=0; j<nv; j++) if (matrix_getcolumnptr(mesh->vert, vid[j], &x[j])!=LINALGERR_OK) return false;
    for (int k=0; k<mesh->dim; k++) { x0[k]=x[0][k]; x1[k]=x[1][k]; }

    functional_veccross(x0, x1, c);
    nrm=functional_vecnorm(3, c);
    if (nrm<MORPHO_EPS) return true;

    functional_veccross(x1, c, f0); /* ∂/∂x0 ~ x1 × c */
    functional_veccross(c, x0, f1); /* ∂/∂x1 ~ c × x0 */

    if (!functional_addtocolumn(frc, vid[0], 0.5/nrm, f0)) return false;
    if (!functional_addtocolumn(frc, vid[1], 0.5/nrm, f1)) return false;

    return true;
}

FUNCTIONAL_INIT(AreaEnclosed, MESH_GRADE_LINE)
FUNCTIONAL_MD_INTEGRAND_COST(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_TOTAL_COST(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_GRADIENT_COST(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_gradient, SYMMETRY_ADD, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_HESSIAN(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand)

MORPHO_BEGINCLASS(AreaEnclosed)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", AreaEnclosed_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(AreaEnclosed),
FUNCTIONAL_MD_TOTAL_METHODS(AreaEnclosed),
FUNCTIONAL_MD_GRADIENT_METHODS(AreaEnclosed),
FUNCTIONAL_MD_HESSIAN_METHODS(AreaEnclosed)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Area
 * ---------------------------------------------- */

/** Calculate area */
bool area_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    if (nv!=3) { *out=0; return true; }
    double *x[nv], s0[3], s1[3], cx[3];
    for (int j=0; j<3; j++) { s0[j]=0; s1[j]=0; cx[j]=0; }
    for (int j=0; j<nv; j++) matrix_getcolumnptr(mesh->vert, vid[j], &x[j]);

    functional_vecsub(mesh->dim, x[1], x[0], s0);
    functional_vecsub(mesh->dim, x[2], x[1], s1);

    functional_veccross(s0, s1, cx);
    *out=0.5*functional_vecnorm(3, cx);
    
    return true;
}

/** Calculate scaled gradient */
bool area_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale) {
    double *x[nv], s0[3], s1[3], s01[3], s010[3], s011[3];
    double norm;
    for (int j=0; j<3; j++) { s0[j]=0; s1[j]=0; s01[j]=0; s010[j]=0; s011[j]=0; }
    for (int j=0; j<nv; j++) if (matrix_getcolumnptr(mesh->vert, vid[j], &x[j])!=LINALGERR_OK) return false;

    functional_vecsub(mesh->dim, x[1], x[0], s0);
    functional_vecsub(mesh->dim, x[2], x[1], s1);

    functional_veccross(s0, s1, s01);
    norm=functional_vecnorm(3, s01);
    if (norm<MORPHO_EPS) return true;

    functional_veccross(s01, s0, s010);
    functional_veccross(s01, s1, s011);

    if (!functional_addtocolumn(frc, vid[0], 0.5/norm*scale, s011)) return false;
    if (!functional_addtocolumn(frc, vid[2], 0.5/norm*scale, s010)) return false;

    functional_vecadd(mesh->dim, s010, s011, s0);

    if (!functional_addtocolumn(frc, vid[1], -0.5/norm*scale, s0)) return false;

    return true;
}

/** Calculate gradient */
bool area_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    return area_gradient_scale(v, mesh, id, nv, vid, NULL, frc, 1.0);
}

FUNCTIONAL_INIT(Area, MESH_GRADE_AREA)
FUNCTIONAL_MD_INTEGRAND_COST(Area, MESH_GRADE_AREA, area_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_TOTAL_COST(Area, MESH_GRADE_AREA, area_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_GRADIENT(Area, MESH_GRADE_AREA, area_gradient, SYMMETRY_ADD)
FUNCTIONAL_MD_HESSIAN(Area, MESH_GRADE_AREA, area_integrand)

MORPHO_BEGINCLASS(Area)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", Area_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(Area),
FUNCTIONAL_MD_TOTAL_METHODS(Area),
FUNCTIONAL_MD_GRADIENT_METHODS(Area),
FUNCTIONAL_MD_HESSIAN_METHODS(Area)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Enclosed volume
 * ---------------------------------------------- */

/** Calculate enclosed volume */
bool volumeenclosed_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], cx[mesh->dim];
    for (int j=0; j<nv; j++) if (matrix_getcolumnptr(mesh->vert, vid[j], &x[j])!=LINALGERR_OK) return false;

    functional_veccross(x[0], x[1], cx);

    *out=fabs(functional_vecdot(mesh->dim, cx, x[2]))/6.0;
    return true;
}

/** Calculate gradient */
bool volumeenclosed_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    double *x[nv], cx[mesh->dim], dot;
    for (int j=0; j<nv; j++) if (matrix_getcolumnptr(mesh->vert, vid[j], &x[j])!=LINALGERR_OK) return false;

    functional_veccross(x[0], x[1], cx);
    dot=functional_vecdot(mesh->dim, cx, x[2]);
    if (fabs(dot)<=DBL_MIN) MORPHO_FAIL(v, VOLUMEENCLOSED_ZERO);
    
    dot/=fabs(dot);

    if (!functional_addtocolumn(frc, vid[2], dot/6.0, cx)) return false;

    functional_veccross(x[1], x[2], cx);
    if (!functional_addtocolumn(frc, vid[0], dot/6.0, cx)) return false;

    functional_veccross(x[2], x[0], cx);
    if (!functional_addtocolumn(frc, vid[1], dot/6.0, cx)) return false;

    return true;
}

FUNCTIONAL_INIT(VolumeEnclosed, MESH_GRADE_AREA)
FUNCTIONAL_MD_INTEGRAND_COST(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_TOTAL_COST(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_GRADIENT_COST(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_gradient, SYMMETRY_ADD, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_HESSIAN(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_integrand)

MORPHO_BEGINCLASS(VolumeEnclosed)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", VolumeEnclosed_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(VolumeEnclosed),
FUNCTIONAL_MD_TOTAL_METHODS(VolumeEnclosed),
FUNCTIONAL_MD_GRADIENT_METHODS(VolumeEnclosed),
FUNCTIONAL_MD_HESSIAN_METHODS(VolumeEnclosed)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Volume
 * ---------------------------------------------- */

/** Calculate enclosed volume */
bool volume_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], s10[mesh->dim], s20[mesh->dim], s30[mesh->dim], cx[mesh->dim];
    for (int j=0; j<nv; j++) matrix_getcolumnptr(mesh->vert, vid[j], &x[j]);

    functional_vecsub(mesh->dim, x[1], x[0], s10);
    functional_vecsub(mesh->dim, x[2], x[0], s20);
    functional_vecsub(mesh->dim, x[3], x[0], s30);

    functional_veccross(s20, s30, cx);

    *out=fabs(functional_vecdot(mesh->dim, s10, cx))/6.0;
    return true;
}

/** Calculate scaled gradient */
bool volume_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale) {
    double *x[nv], s10[mesh->dim], s20[mesh->dim], s30[mesh->dim];
    double s31[mesh->dim], s21[mesh->dim], cx[mesh->dim], uu;
    for (int j=0; j<nv; j++) matrix_getcolumnptr(mesh->vert, vid[j], &x[j]);

    functional_vecsub(mesh->dim, x[1], x[0], s10);
    functional_vecsub(mesh->dim, x[2], x[0], s20);
    functional_vecsub(mesh->dim, x[3], x[0], s30);
    functional_vecsub(mesh->dim, x[3], x[1], s31);
    functional_vecsub(mesh->dim, x[2], x[1], s21);

    functional_veccross(s20, s30, cx);
    uu=functional_vecdot(mesh->dim, s10, cx);
    uu=(uu>0 ? 1.0 : -1.0);

    if (!functional_addtocolumn(frc, vid[1], uu/6.0*scale, cx)) return false;

    functional_veccross(s31, s21, cx);
    if (!functional_addtocolumn(frc, vid[0], uu/6.0*scale, cx)) return false;

    functional_veccross(s30, s10, cx);
    if (!functional_addtocolumn(frc, vid[2], uu/6.0*scale, cx)) return false;

    functional_veccross(s10, s20, cx);
    if (!functional_addtocolumn(frc, vid[3], uu/6.0*scale, cx)) return false;

    return true;
}

/** Calculate gradient */
bool volume_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    return volume_gradient_scale(v, mesh, id, nv, vid, NULL, frc, 1.0);
}

FUNCTIONAL_INIT(Volume, MESH_GRADE_VOLUME)
FUNCTIONAL_MD_INTEGRAND_COST(Volume, MESH_GRADE_VOLUME, volume_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_TOTAL_COST(Volume, MESH_GRADE_VOLUME, volume_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_GRADIENT_COST(Volume, MESH_GRADE_VOLUME, volume_gradient, SYMMETRY_ADD, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_HESSIAN(Volume, MESH_GRADE_VOLUME, volume_integrand)

MORPHO_BEGINCLASS(Volume)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", Volume_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(Volume),
FUNCTIONAL_MD_TOTAL_METHODS(Volume),
FUNCTIONAL_MD_GRADIENT_METHODS(Volume),
FUNCTIONAL_MD_HESSIAN_METHODS(Volume)
MORPHO_ENDCLASS

void size_initialize(void) {
    objectstring objclassname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objclassname));

    builtin_addclass(LENGTH_CLASSNAME, MORPHO_GETCLASSDEFINITION(Length), objclass);
    builtin_addclass(AREA_CLASSNAME, MORPHO_GETCLASSDEFINITION(Area), objclass);
    builtin_addclass(AREAENCLOSED_CLASSNAME, MORPHO_GETCLASSDEFINITION(AreaEnclosed), objclass);
    builtin_addclass(VOLUMEENCLOSED_CLASSNAME, MORPHO_GETCLASSDEFINITION(VolumeEnclosed), objclass);
    builtin_addclass(VOLUME_CLASSNAME, MORPHO_GETCLASSDEFINITION(Volume), objclass);

    morpho_defineerror(VOLUMEENCLOSED_ZERO, ERROR_HALT, VOLUMEENCLOSED_ZERO_MSG);
}

#endif
