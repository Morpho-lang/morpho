/** @file curvature.c
 *  @author T J Atherton
 *
 *  @brief Line and surface curvature functionals
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
#include "curvature.h"

/* **********************************************************************
 * Curvatures
 * ********************************************************************** */

/* ----------------------------------------------
 * LineCurvatureSq
 * ---------------------------------------------- */

static value curvature_integrandonlyproperty;

typedef struct {
    objectsparse *lineel; // Lines
    objectselection *selection; // Selection
    bool integrandonly; // Output integrated curvature or 'bare' curvature.
} curvatureref;

bool curvature_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, curvatureref *ref) {
    bool success = true;

    ref->selection=sel;

    ref->lineel = mesh_getconnectivityelement(mesh, MESH_GRADE_VERTEX, MESH_GRADE_LINE);
    if (ref->lineel) success=sparse_checkformat(ref->lineel, SPARSE_CCS, true, false);

    if (success) {
        objectsparse *s = mesh_getconnectivityelement(mesh, MESH_GRADE_LINE, MESH_GRADE_VERTEX);
        if (!s) s=mesh_addconnectivityelement(mesh, MESH_GRADE_LINE, MESH_GRADE_VERTEX);
        success=s;
    }

    if (success) {
        value integrandonly=MORPHO_FALSE;
        objectinstance_getpropertyinterned(self, curvature_integrandonlyproperty, &integrandonly);
        ref->integrandonly=MORPHO_ISTRUE(integrandonly);
    }

    return success;
}

/** Finds the points that a point depends on  */
bool linecurvsq_dependencies(functional_mapinfo *info, elementid id, varray_elementid *out) {
    objectmesh *mesh = info->mesh;
    curvatureref *cref = info->ref;
    bool success=false;
    varray_elementid nbrs;
    varray_elementidinit(&nbrs);

    varray_elementidwrite(out, id); // LinecurvSq is a vertex element, and hence depends on itself
    
    if (mesh_findneighbors(mesh, MESH_GRADE_VERTEX, id, MESH_GRADE_LINE, &nbrs)>0) {
        for (unsigned int i=0; i<nbrs.count; i++) {
            int nentries, *entries; // Get the vertices for this edge
            if (!sparseccs_getrowindices(&cref->lineel->ccs, nbrs.data[i], &nentries, &entries)) goto linecurvsq_dependencies_cleanup;
            for (unsigned int j=0; j<nentries; j++) {
                if (entries[j]==id) continue;
                varray_elementidwrite(out, entries[j]);
            }
        }
    }
    success=true;

linecurvsq_dependencies_cleanup:
    varray_elementidclear(&nbrs);

    return success;
}

/** Calculate the integral of the curvature squared  */
bool linecurvsq_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    curvatureref *cref = (curvatureref *) ref;
    double result = 0.0;
    varray_elementid nbrs;
    varray_elementid synid;
    varray_elementidinit(&nbrs);
    varray_elementidinit(&synid);

    double s0[mesh->dim], s1[mesh->dim], *s[2] = { s0, s1}, sgn=-1.0;

    if (mesh_findneighbors(mesh, MESH_GRADE_VERTEX, id, MESH_GRADE_LINE, &nbrs)>0 &&
        mesh_getsynonyms(mesh, MESH_GRADE_VERTEX, id, &synid)) {
        if (nbrs.count!=2) goto linecurvsq_integrand_cleanup;

        for (unsigned int i=0; i<2; i++) {
            int nentries, *entries; // Get the vertices for this edge
            if (!sparseccs_getrowindices(&cref->lineel->ccs, nbrs.data[i], &nentries, &entries)) break;

            double *x0, *x1;
            if (mesh_getvertexcoordinatesaslist(mesh, entries[0], &x0) &&
                mesh_getvertexcoordinatesaslist(mesh, entries[1], &x1)) {
                functional_vecsub(mesh->dim, x0, x1, s[i]);
            }
            if (!(entries[0]==id || functional_inlist(&synid, entries[0]))) sgn*=-1;
        }

        double s0s0=functional_vecdot(mesh->dim, s0, s0),
               s0s1=functional_vecdot(mesh->dim, s0, s1),
               s1s1=functional_vecdot(mesh->dim, s1, s1);

        s0s0=sqrt(s0s0); s1s1=sqrt(s1s1);

        if (s0s0<MORPHO_EPS || s1s1<MORPHO_EPS) goto linecurvsq_integrand_cleanup;

        double u=sgn*s0s1/s0s0/s1s1,
               len=0.5*(s0s0+s1s1);

        if (u<1) u=acos(u); else u=0;

        result = u*u/len;
        if (cref->integrandonly) result /= len; // Get the bare curvature.
    }

linecurvsq_integrand_cleanup:

    *out = result;
    varray_elementidclear(&nbrs);
    varray_elementidclear(&synid);

    return true;
}

FUNCTIONAL_INIT(LineCurvatureSq, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_BIND_FORCEGRADE(LineCurvatureSq, curvatureref, curvature_prepareref, linecurvsq_integrand, FUNCTIONAL_ARGS, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_INTEGRAND_COST(LineCurvatureSq, curvatureref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_REF_TOTAL_COST(LineCurvatureSq, curvatureref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(LineCurvatureSq, curvatureref, MESH_GRADE_VERTEX, linecurvsq_dependencies, SYMMETRY_ADD)
FUNCTIONAL_MD_REF_HESSIAN(LineCurvatureSq, curvatureref, MESH_GRADE_VERTEX, linecurvsq_dependencies, SYMMETRY_ADD)

MORPHO_BEGINCLASS(LineCurvatureSq)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", LineCurvatureSq_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(LineCurvatureSq),
FUNCTIONAL_MD_TOTAL_METHODS(LineCurvatureSq),
FUNCTIONAL_MD_GRADIENT_METHODS(LineCurvatureSq),
FUNCTIONAL_MD_HESSIAN_METHODS(LineCurvatureSq)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * LineTorsionSq
 * ---------------------------------------------- */

/** Return a list of vertices that an element depends on  */
bool linetorsionsq_dependencies(functional_mapinfo *info, elementid id, varray_elementid *out) {
    objectmesh *mesh = info->mesh;
    curvatureref *cref = info->ref;
    bool success=false;
    varray_elementid nbrs;
    varray_elementid synid;

    varray_elementidinit(&nbrs);
    varray_elementidinit(&synid);

    if (mesh_findneighbors(mesh, MESH_GRADE_LINE, id, MESH_GRADE_LINE, &nbrs)>0) {
        for (unsigned int i=0; i<nbrs.count; i++) {
            int nentries, *entries; // Get the vertices for this edge
            if (!sparseccs_getrowindices(&cref->lineel->ccs, nbrs.data[i], &nentries, &entries)) goto linetorsionsq_dependencies_cleanup;
            for (unsigned int j=0; j<nentries; j++) {
                varray_elementidwriteunique(out, entries[j]);
            }
        }
    }
    success=true;

linetorsionsq_dependencies_cleanup:
    varray_elementidclear(&nbrs);
    varray_elementidclear(&synid);

    return success;
}


/** Calculate the integral of the torsion squared  */
bool linetorsionsq_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    curvatureref *cref = (curvatureref *) ref;
    int tmpi; elementid tmpid;
    bool success=false;

    //double result = 0.0;
    varray_elementid nbrs;
    varray_elementid synid;
    varray_elementidinit(&nbrs);
    varray_elementidinit(&synid);
    elementid vlist[6]; // List of vertices in order  n
    int type[6];
    for (unsigned int i=0; i<6; i++) type[i]=-1;

    /* We want an ordered list of vertex indices:
     *               v the element
     *    0 --- 1/2 --- 3/4 --- 5
     * Where 1/2 and 3/4 are the same vertex, but could have different indices due to symmetries */
     vlist[2] = vid[0]; vlist[3] = vid[1]; // Copy the current element into place

    /* First identify neighbors and get the vertex ids for each element */
    if (mesh_findneighbors(mesh, MESH_GRADE_LINE, id, MESH_GRADE_LINE, &nbrs)>0) {
        if (nbrs.count<2) {
            *out = 0; success=true;
            goto linecurvsq_torsion_cleanup;
        }

        for (unsigned int i=0; i<nbrs.count; i++) {
            int nentries, *entries; // Get the vertices for this edge
            if (!sparseccs_getrowindices(&cref->lineel->ccs, nbrs.data[i], &nentries, &entries)) goto linecurvsq_torsion_cleanup;
            for (unsigned int j=0; j<nentries; j++) { // Copy the vertexids
                vlist[4*i+j] = entries[j];
            }
        }
    }

    /* The vertex ids are not yet in the right order. Let's identify which vertex is which */
    for (int i=0; i<2; i++) {
        if (mesh_getsynonyms(mesh, 0, vid[i], &synid)) {
            for (int j=0; j<6; j++) if (vlist[j]==vid[i] || functional_inlist(&synid, vlist[j])) type[j]=i;
        }
    }
    /* The type array now contains either 0,1 depending on which vertex we have, or -1 if the vertex is not a synonym for the element's vertices */
#define SWAP(var, i, j, tmp) { tmp=var[i]; var[i]=var[j]; var[j]=tmp; }
    if (type[0]==1 || type[1]==1) { // Make sure the first segment corresponds to the first vertex
        SWAP(vlist, 0, 4, tmpid); SWAP(vlist, 1, 5, tmpid);
        SWAP(type, 0, 4, tmpi); SWAP(type, 1, 5, tmpi);
    }

    if (type[1]==-1) { // Check order of first segment
        SWAP(vlist, 0, 1, tmpid);
        SWAP(type, 0, 1, tmpi);
    }

    if (type[4]==-1) { // Check order of first segment
        SWAP(vlist, 4, 5, tmpid);
        SWAP(type, 4, 5, tmpi);
    }
#undef SWAP

    /* We now have an ordered list of vertices.
       Get the vertex positions */
    double *x[6];
    for (int i=0; i<6; i++) matrix_getcolumnptr(mesh->vert, vlist[i], &x[i]);

    double A[3], B[3], C[3], crossAB[3], crossBC[3];
    functional_vecsub(3, x[1], x[0], A);
    functional_vecsub(3, x[3], x[2], B);
    functional_vecsub(3, x[5], x[4], C);

    functional_veccross(A, B, crossAB);
    functional_veccross(B, C, crossBC);

    double normB=functional_vecnorm(3, B),
           normAB=functional_vecnorm(3, crossAB),
           normBC=functional_vecnorm(3, crossBC);

    double S = functional_vecdot(3, A, crossBC)*normB;
    if (normAB>MORPHO_EPS) S/=normAB;
    if (normBC>MORPHO_EPS) S/=normBC;

    S=asin(S);
    *out=S*S/normB;
    success=true;

linecurvsq_torsion_cleanup:
    varray_elementidclear(&nbrs);
    varray_elementidclear(&synid);

    return success;
}

FUNCTIONAL_INIT(LineTorsionSq, MESH_GRADE_LINE)
FUNCTIONAL_MD_REF_BIND_FORCEGRADE(LineTorsionSq, curvatureref, curvature_prepareref, linetorsionsq_integrand, FUNCTIONAL_ARGS, MESH_GRADE_LINE)
FUNCTIONAL_MD_REF_INTEGRAND_COST(LineTorsionSq, curvatureref, MESH_GRADE_LINE, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_REF_TOTAL_COST(LineTorsionSq, curvatureref, MESH_GRADE_LINE, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(LineTorsionSq, curvatureref, MESH_GRADE_LINE, linetorsionsq_dependencies, SYMMETRY_ADD)
FUNCTIONAL_MD_REF_HESSIAN(LineTorsionSq, curvatureref, MESH_GRADE_LINE, linetorsionsq_dependencies, SYMMETRY_ADD)

MORPHO_BEGINCLASS(LineTorsionSq)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", LineTorsionSq_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(LineTorsionSq),
FUNCTIONAL_MD_TOTAL_METHODS(LineTorsionSq),
FUNCTIONAL_MD_GRADIENT_METHODS(LineTorsionSq),
FUNCTIONAL_MD_HESSIAN_METHODS(LineTorsionSq)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * MeanCurvatureSq
 * ---------------------------------------------- */

static value curvature_geodesicproperty;

typedef struct {
    objectsparse *areael; // Areas
    objectselection *selection; // Selection
    bool integrandonly; // Output integrated curvature or 'bare' curvature.
    bool geodesic; // Compute the geodesic curvature instead of the Gauss curvature (see https://cuhkmath.wordpress.com/2016/06/21/the-discrete-gauss-bonnet-theorem/)
} areacurvatureref;

bool areacurvature_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, areacurvatureref *ref) {
    bool success = true;

    ref->selection=sel;

    ref->areael = mesh_getconnectivityelement(mesh, MESH_GRADE_VERTEX, MESH_GRADE_AREA);
    if (ref->areael) success=sparse_checkformat(ref->areael, SPARSE_CCS, true, false);

    if (success) {
        objectsparse *s = mesh_getconnectivityelement(mesh, MESH_GRADE_AREA, MESH_GRADE_VERTEX);
        if (!s) s=mesh_addconnectivityelement(mesh, MESH_GRADE_AREA, MESH_GRADE_VERTEX);
        success=s;
    }

    if (success) {
        value integrandonly=MORPHO_FALSE;
        objectinstance_getpropertyinterned(self, curvature_integrandonlyproperty, &integrandonly);
        ref->integrandonly=MORPHO_ISTRUE(integrandonly);

        value geodesic=MORPHO_FALSE;
        objectinstance_getpropertyinterned(self, curvature_geodesicproperty, &geodesic);
        ref->geodesic=MORPHO_ISTRUE(geodesic);
    }

    return success;
}

/** Return a list of vertices that an element depends on  */
bool meancurvaturesq_dependencies(functional_mapinfo *info, elementid id, varray_elementid *out) {
    objectmesh *mesh = info->mesh;
    areacurvatureref *cref = info->ref;
    bool success=false;
    varray_elementid nbrs;
    varray_elementid synid;

    varray_elementidinit(&nbrs);
    varray_elementidinit(&synid);

    mesh_getsynonyms(mesh, MESH_GRADE_VERTEX, id, &synid);
    varray_elementidwriteunique(&synid, id);

    /* Loop over synonyms of the element id */
    mesh_findneighbors(mesh, MESH_GRADE_VERTEX, id, MESH_GRADE_AREA, &nbrs);

    for (unsigned int i=0; i<nbrs.count; i++) { /* Loop over adjacent triangles */
        int nvert, *vids;
        if (!sparseccs_getrowindices(&cref->areael->ccs, nbrs.data[i], &nvert, &vids)) goto meancurvsq_dependencies_cleanup;

        for (unsigned int j=0; j<nvert; j++) {
            if (vids[j]==id) continue;
            varray_elementidwriteunique(out, vids[j]);
        }
    }
    success=true;

meancurvsq_dependencies_cleanup:
    varray_elementidclear(&nbrs);
    varray_elementidclear(&synid);

    return success;
}

/** Orders the vertices in the list vids so that the vertex in synid is first */
bool curvature_ordervertices(varray_elementid *synid, int nv, int *vids) {
    int posn=-1;
    for (unsigned int i=0; i<nv && posn<0; i++) {
        for (unsigned int k=0; k<synid->count; k++) if (synid->data[k]==vids[i]) { posn = i; break; }
    }

    if (posn>0) { // If the desired vertex isn't in first position, move it there.
        int tmp=vids[posn];
        vids[posn]=vids[0]; vids[0]=tmp;
    }

    return (posn>=0);
}

/** Calculate the integral of the mean curvature squared  */
bool meancurvaturesq_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    areacurvatureref *cref = (areacurvatureref *) ref;
    double areasum = 0;
    bool success=false;

    varray_elementid nbrs;
    varray_elementid synid;
    varray_elementidinit(&nbrs);
    varray_elementidinit(&synid);

    mesh_getsynonyms(mesh, MESH_GRADE_VERTEX, id, &synid);
    varray_elementidwriteunique(&synid, id);

    double frc[mesh->dim]; // This will hold the total force due to the triangles present
    for (unsigned int i=0; i<mesh->dim; i++) frc[i]=0.0;

    mesh_findneighbors(mesh, MESH_GRADE_VERTEX, id, MESH_GRADE_AREA, &nbrs);

    for (unsigned int i=0; i<nbrs.count; i++) { /* Loop over adjacent triangles */
        int nvert, *ovids;
        if (!sparseccs_getrowindices(&cref->areael->ccs, nbrs.data[i], &nvert, &ovids)) goto meancurvsq_cleanup;

        int vids[nvert]; // Copy so we can reorder
        for (int j=0; j<nvert; j++) vids[j]=ovids[j];
        
        /* Order the vertices */
        if (!curvature_ordervertices(&synid, nvert, vids)) goto meancurvsq_cleanup;

        double *x[3], s0[3], s1[3], s01[3], s101[3];
        double norm;
        for (int j=0; j<3; j++) matrix_getcolumnptr(mesh->vert, vids[j], &x[j]);

        /* s0 = x1-x0; s1 = x2-x1 */
        functional_vecsub(mesh->dim, x[1], x[0], s0);
        functional_vecsub(mesh->dim, x[2], x[1], s1);

        /* F(v0) = (s1 x s0 x s1)/|s0 x x1|/2 */
        functional_veccross(s0, s1, s01);
        norm=functional_vecnorm(mesh->dim, s01);
        if (norm<MORPHO_EPS) goto meancurvsq_cleanup;

        areasum+=norm/2;
        functional_veccross(s1, s01, s101);

        functional_vecaddscale(mesh->dim, frc, 0.5/norm, s101, frc);
    }

    *out = functional_vecdot(mesh->dim, frc, frc)/(areasum/3.0)/4.0;
    if (cref->integrandonly) *out /= (areasum/3.0);
    success=true;

meancurvsq_cleanup:
    varray_elementidclear(&nbrs);
    varray_elementidclear(&synid);

    return success;
}

FUNCTIONAL_INIT(MeanCurvatureSq, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_BIND_FORCEGRADE(MeanCurvatureSq, areacurvatureref, areacurvature_prepareref, meancurvaturesq_integrand, FUNCTIONAL_ARGS, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_INTEGRAND_COST(MeanCurvatureSq, areacurvatureref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_REF_TOTAL_COST(MeanCurvatureSq, areacurvatureref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(MeanCurvatureSq, areacurvatureref, MESH_GRADE_VERTEX, meancurvaturesq_dependencies, SYMMETRY_ADD)

MORPHO_BEGINCLASS(MeanCurvatureSq)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", MeanCurvatureSq_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(MeanCurvatureSq),
FUNCTIONAL_MD_TOTAL_METHODS(MeanCurvatureSq),
FUNCTIONAL_MD_GRADIENT_METHODS(MeanCurvatureSq)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * GaussCurvature
 * ---------------------------------------------- */

/** Calculate the integral of the gaussian curvature  */
bool gausscurvature_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    areacurvatureref *cref = (areacurvatureref *) ref;
    double anglesum = 0, areasum = 0;
    bool success=false;

    varray_elementid nbrs;
    varray_elementid synid;
    varray_elementidinit(&nbrs);
    varray_elementidinit(&synid);

    mesh_getsynonyms(mesh, MESH_GRADE_VERTEX, id, &synid);
    varray_elementidwriteunique(&synid, id);

    double frc[mesh->dim]; // This will hold the total force due to the triangles present
    for (unsigned int i=0; i<mesh->dim; i++) frc[i]=0.0;

    mesh_findneighbors(mesh, MESH_GRADE_VERTEX, id, MESH_GRADE_AREA, &nbrs);

    for (unsigned int i=0; i<nbrs.count; i++) { /* Loop over adjacent triangles */
        int nvert, *ovids;
        if (!sparseccs_getrowindices(&cref->areael->ccs, nbrs.data[i], &nvert, &ovids)) goto gausscurv_cleanup;
        
        int vids[nvert]; // Copy so we can reorder
        for (int j=0; j<nvert; j++) vids[j]=ovids[j];

        /* Order the vertices */
        if (!curvature_ordervertices(&synid, nvert, vids)) goto gausscurv_cleanup;

        double *x[3], s0[3], s1[3], s01[3];
        for (int j=0; j<3; j++) matrix_getcolumnptr(mesh->vert, vids[j], &x[j]);

        /* s0 = x1-x0; s1 = x2-x0 */
        functional_vecsub(mesh->dim, x[1], x[0], s0);
        functional_vecsub(mesh->dim, x[2], x[0], s1);

        functional_veccross(s0, s1, s01);
        double area = functional_vecnorm(mesh->dim, s01);
        anglesum+=atan2(area, functional_vecdot(mesh->dim, s0, s1));

        areasum+=area/2;
    }

    *out = 2*M_PI-anglesum;
    if (cref->geodesic) *out = M_PI-anglesum;
    if (cref->integrandonly) *out /= (areasum/3.0);
    success=true;

gausscurv_cleanup:
    varray_elementidclear(&nbrs);
    varray_elementidclear(&synid);

    return success;
}

FUNCTIONAL_INIT(GaussCurvature, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_BIND_FORCEGRADE(GaussCurvature, areacurvatureref, areacurvature_prepareref, gausscurvature_integrand, FUNCTIONAL_ARGS, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_INTEGRAND_COST(GaussCurvature, areacurvatureref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_REF_TOTAL_COST(GaussCurvature, areacurvatureref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(GaussCurvature, areacurvatureref, MESH_GRADE_VERTEX, meancurvaturesq_dependencies, SYMMETRY_ADD)

MORPHO_BEGINCLASS(GaussCurvature)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", GaussCurvature_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(GaussCurvature),
FUNCTIONAL_MD_TOTAL_METHODS(GaussCurvature),
FUNCTIONAL_MD_GRADIENT_METHODS(GaussCurvature)
MORPHO_ENDCLASS

void curvature_initialize(void) {
    curvature_integrandonlyproperty=builtin_internsymbolascstring(CURVATURE_INTEGRANDONLY_PROPERTY);
    curvature_geodesicproperty=builtin_internsymbolascstring(CURVATURE_GEODESIC_PROPERTY);

    value objclass = builtin_findclassfromcstring(OBJECT_CLASSNAME);

    builtin_addclass(LINECURVATURESQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(LineCurvatureSq), objclass);
    builtin_addclass(LINETORSIONSQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(LineTorsionSq), objclass);
    builtin_addclass(MEANCURVATURESQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(MeanCurvatureSq), objclass);
    builtin_addclass(GAUSSCURVATURE_CLASSNAME, MORPHO_GETCLASSDEFINITION(GaussCurvature), objclass);
}

#endif
