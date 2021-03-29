/** @file functional.c
 *  @author T J Atherton
 *
 *  @brief Functionals
 */

#include "functional.h"
#include "builtin.h"
#include "common.h"
#include "error.h"
#include "value.h"
#include "object.h"
#include "morpho.h"
#include "matrix.h"
#include "sparse.h"
#include "mesh.h"

static value functional_gradeproperty;

/* **********************************************************************
 * Common functional routines
 * ********************************************************************** */

/** Validates the arguments provided to a functional
 * @param[in] v - vm
 * @param[in] nargs - number of arguments
 * @param[in] args - the arguments
 * @param[out] mesh - mesh to use
 * @param[out] sel - selection to use */
bool functional_validateargs(vm *v, int nargs, value *args, objectmesh **mesh, objectselection **sel) {
    if (nargs>0 && MORPHO_ISMESH(MORPHO_GETARG(args,0))) {
        *mesh = MORPHO_GETMESH(MORPHO_GETARG(args,0));
    }
    
    if (nargs>1 && MORPHO_ISSELECTION(MORPHO_GETARG(args,1))) {
        *sel = MORPHO_GETSELECTION(MORPHO_GETARG(args,1));
    }
    
    if (*mesh) return true;
    morpho_runtimeerror(v, FUNC_INTEGRAND_MESH);
    return false;
}

/** Integrand function */
typedef bool (functional_integrand) (vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out);

/** Gradient function */
typedef bool (functional_gradient) (vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc);


/** Internal function to count the number of elements */
static bool functional_countelements(vm *v, objectmesh *mesh, grade g, int *n, objectsparse **s) {
    /* How many elements? */
    if (g==MESH_GRADE_VERTEX) {
        *n=mesh->vert->ncols;
    } else {
        *s=mesh_getconnectivityelement(mesh, 0, g);
        if (*s) {
            *n=(*s)->ccs.ncols; // Number of elements
        } else {
            morpho_runtimeerror(v, FUNC_ELNTFND, g);
            return false;
        }
    }
    return true;
}

/** Sums an integrand
 * @param[in] v - virtual machine in use
 * @param[in] mesh - mesh to calculate integrand over
 * @param[in] sel - selection
 * @param[in] g - grade of the functional
 * @param[in] integrand - function to compute integrand for the element
 * @param[in] ref - a reference
 * @param[out] out - a matrix of integrand values
 * @returns true on success, false otherwise. Error reporting through VM. */
bool functional_sumintegrand(vm *v, objectmesh *mesh, objectselection *sel, grade g, functional_integrand *integrand, void *ref, value *out) {
    objectsparse *s=mesh_getconnectivityelement(mesh, 0, g);
    int n=0;
    
    if (!functional_countelements(v, mesh, g, &n, &s)) return false;
    
    if (n>0) {
        int vertexid; // Use this if looping over grade 0
        int *vid=(g==0 ? &vertexid : NULL),
            nv=(g==0 ? 1 : 0); // The vertex indices
        double sum=0.0, c=0.0, y, t, result;
        
        if (sel) { // Loop over selection
            if (sel->selected[g].count>0) for (unsigned int k=0; k<sel->selected[g].capacity; k++) {
                if (!MORPHO_ISINTEGER(sel->selected[g].contents[k].key)) continue;
                elementid i = MORPHO_GETINTEGERVALUE(sel->selected[g].contents[k].key);
            
                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;
            
                if (vid && nv>0) {
                    if ((*integrand) (v, mesh, i, nv, vid, ref, &result)) {
                        y=result-c; t=sum+y; c=(t-sum)-y; sum=t; // Kahan summation
                    } else return false;
                }
            }
        } else { // Loop over elements
            for (elementid i=0; i<n; i++) {
                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;

                if (vid && nv>0) {
                    if ((*integrand) (v, mesh, i, nv, vid, ref, &result)) {
                        y=result-c; t=sum+y; c=(t-sum)-y; sum=t; // Kahan summation
                    } else return false;
                }
            }
        }
        
        *out=MORPHO_FLOAT(sum);
    }
    
    return false;
}

/** Calculate an integrand
 * @param[in] v - virtual machine in use
 * @param[in] mesh - mesh to calculate integrand over
 * @param[in] sel - selection
 * @param[in] g - grade of the functional
 * @param[in] integrand - function to compute integrand for the element
 * @param[in] ref - a reference
 * @param[out] out - a matrix of integrand values
 * @returns true on success, false otherwise. Error reporting through VM. */
bool functional_mapintegrand(vm *v, objectmesh *mesh, objectselection *sel, grade g, functional_integrand *integrand, void *ref, value *out) {
    objectsparse *s=NULL;
    objectmatrix *new=NULL;
    bool ret=false;
    int n=0;
        
    /* How many elements? */
    if (!functional_countelements(v, mesh, g, &n, &s)) return false;
    
    /* Create the output matrix */
    if (n>0) {
        new=object_newmatrix(1, n, true);
        if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }
    }
    
    if (new) {
        int vertexid; // Use this if looping over grade 0
        int *vid=(g==0 ? &vertexid : NULL),
            nv=(g==0 ? 1 : 0); // The vertex indices
        double result;
        
        if (sel) { // Loop over selection
            if (sel->selected[g].count>0) for (unsigned int k=0; k<sel->selected[g].capacity; k++) {
                if (!MORPHO_ISINTEGER(sel->selected[g].contents[k].key)) continue;
                elementid i = MORPHO_GETINTEGERVALUE(sel->selected[g].contents[k].key);
                
                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;
            
                if (vid && nv>0) {
                    if ((*integrand) (v, mesh, i, nv, vid, ref, &result)) {
                        matrix_setelement(new, 0, i, result);
                    } else return false;
                }
            }
        } else { // Loop over elements
            for (elementid i=0; i<n; i++) {
                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;

                if (vid && nv>0) {
                    if ((*integrand) (v, mesh, i, nv, vid, ref, &result)) {
                        matrix_setelement(new, 0, i, result);
                    } else return false;
                }
            }
        }
        *out = MORPHO_OBJECT(new);
        ret=true;
    }
    
    return ret;
}

/** Calculate gradient
 * @param[in] v - virtual machine in use
 * @param[in] mesh - mesh to calculate integrand over
 * @param[in] sel - selection
 * @param[in] g - grade of the functional
 * @param[in] grad - function to compute gradient for the element
 * @param[in] ref - a reference
 * @param[out] out - a matrix of integrand values
 * @returns true on success, false otherwise. Error reporting through VM. */
bool functional_mapgradient(vm *v, objectmesh *mesh, objectselection *sel, grade g, functional_gradient *grad, void *ref, value *out) {
    objectsparse *s=NULL;
    objectmatrix *frc=NULL;
    bool ret=false;
    int n=0;
    
    /* How many elements? */
    if (!functional_countelements(v, mesh, g, &n, &s)) return false;
    
    /* Create the output matrix */
    if (n>0) {
        frc=object_newmatrix(mesh->vert->nrows, mesh->vert->ncols, true);
        if (!frc)  { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }
    }
    
    if (frc) {
        int vertexid; // Use this if looping over grade 0
        int *vid=(g==0 ? &vertexid : NULL),
            nv=(g==0 ? 1 : 0); // The vertex indices
            
        
        if (sel) { // Loop over selection
            if (sel->selected[g].count>0) for (unsigned int k=0; k<sel->selected[g].capacity; k++) {
                if (!MORPHO_ISINTEGER(sel->selected[g].contents[k].key)) continue;
                
                elementid i = MORPHO_GETINTEGERVALUE(sel->selected[g].contents[k].key);
                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;
            
                if (vid && nv>0) {
                    if (!(*grad) (v, mesh, i, nv, vid, ref, frc)) goto functional_mapgradient_cleanup;
                }
            }
        } else { // Loop over elements
            for (elementid i=0; i<n; i++) {
                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;

                if (vid && nv>0) {
                    if (!(*grad) (v, mesh, i, nv, vid, ref, frc)) goto functional_mapgradient_cleanup;
                }
            }
        }
        
        *out = MORPHO_OBJECT(frc);
        ret=true;
    }
    
functional_mapgradient_cleanup:
    if (!ret) object_free((object *) frc);
    
    return ret;
}

/* Calculates a numerical gradient */
static bool functional_numericalgradient(vm *v, objectmesh *mesh, elementid i, int nv, int *vid, functional_integrand *integrand, void *ref, objectmatrix *frc) {
    double f0,fp,fm,x0,eps=1e-10; // Should use sqrt(machineeps)*(1+|x|) here
    
    // Loop over vertices in element
    for (unsigned int j=0; j<nv; j++) {
        // Loop over coordinates
        for (unsigned int k=0; k<mesh->dim; k++) {
            matrix_getelement(frc, k, vid[j], &f0);
            
            matrix_getelement(mesh->vert, k, vid[j], &x0);
            matrix_setelement(mesh->vert, k, vid[j], x0+eps);
            if (!(*integrand) (v, mesh, i, nv, vid, ref, &fp)) return false;
            matrix_setelement(mesh->vert, k, vid[j], x0-eps);
            if (!(*integrand) (v, mesh, i, nv, vid, ref, &fm)) return false;
            matrix_setelement(mesh->vert, k, vid[j], x0);
            
            matrix_setelement(frc, k, vid[j], f0+(fp-fm)/(2*eps));
        }
    }
    return true;
}

/** Map numerical gradient over the elements
 * @param[in] v - virtual machine in use
 * @param[in] mesh - mesh to calculate integrand over
 * @param[in] sel - selection
 * @param[in] g - grade of the functional
 * @param[in] integrand - function to compute integrand for the element
 * @param[in] ref - a reference
 * @param[out] out - a matrix of integrand values
 * @returns true on success, false otherwise. Error reporting through VM. */
bool functional_mapnumericalgradient(vm *v, objectmesh *mesh, objectselection *sel, grade g, functional_integrand *integrand, void *ref, value *out) {
    objectsparse *s=NULL;
    objectmatrix *frc=NULL;
    bool ret=false;
    int n=0;
    
    /* How many elements? */
    if (!functional_countelements(v, mesh, g, &n, &s)) return false;
    
    /* Create the output matrix */
    if (n>0) {
        frc=object_newmatrix(mesh->vert->nrows, mesh->vert->ncols, true);
        if (!frc)  { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }
    }
    
    if (frc) {
        int vertexid; // Use this if looping over grade 0
        int *vid=(g==0 ? &vertexid : NULL),
            nv=(g==0 ? 1 : 0); // The vertex indices
        
        if (sel) { // Loop over selection
            if (sel->selected[g].count>0) for (unsigned int k=0; k<sel->selected[g].capacity; k++) {
                if (!MORPHO_ISINTEGER(sel->selected[g].contents[k].key)) continue;
                
                elementid i = MORPHO_GETINTEGERVALUE(sel->selected[g].contents[k].key);
                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;
            
                if (vid && nv>0) {
                    if (!functional_numericalgradient(v, mesh, i, nv, vid, integrand, ref, frc)) goto functional_numericalgradient_cleanup;
                }
            }
        } else { // Loop over elements
            for (elementid i=0; i<n; i++) {
                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;

                if (vid && nv>0) {
                    if (!functional_numericalgradient(v, mesh, i, nv, vid, integrand, ref, frc)) goto functional_numericalgradient_cleanup;
                }
            }
        }
        
        *out = MORPHO_OBJECT(frc);
        ret=true;
    }

functional_numericalgradient_cleanup:
    if (!ret) object_free((object *) frc);
    
    return ret;
}


/* **********************************************************************
 * Common library functions
 * ********************************************************************** */

/** Calculate the difference of two vectors */
void functional_vecadd(unsigned int n, double *a, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]+b[i];
}

/** Calculate the difference of two vectors */
void functional_vecsub(unsigned int n, double *a, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]-b[i];
}

/** Calculate the norm of a vector */
double functional_vecnorm(unsigned int n, double *a) {
    return cblas_dnrm2(n, a, 1);
}

/** Dot product of two vectors */
double functional_vecdot(unsigned int n, double *a, double *b) {
    return cblas_ddot(n, a, 1, b, 1);
}

/** 3D cross product  */
void functional_veccross(double *a, double *b, double *out) {
    out[0]=a[1]*b[2]-a[2]*b[1];
    out[1]=a[2]*b[0]-a[0]*b[2];
    out[2]=a[0]*b[1]-a[1]*b[0];
}

/* **********************************************************************
 * Functionals
 * ********************************************************************** */

/** Initialize a functional */
#define FUNCTIONAL_INIT(name, grade) value name##_init(vm *v, int nargs, value *args) { \
    objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), functional_gradeproperty, MORPHO_INTEGER(grade)); \
    return MORPHO_NIL; \
}

/** Evaluate an integrand */
#define FUNCTIONAL_INTEGRAND(name, grade, integrandfn) value name##_integrand(vm *v, int nargs, value *args) { \
    objectmesh *mesh=NULL; \
    objectselection *sel=NULL; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &mesh, &sel)) { \
        functional_mapintegrand(v, mesh, sel, grade, integrandfn, NULL, &out); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    return out; \
}

/** Evaluate a gradient */
#define FUNCTIONAL_GRADIENT(name, grade, gradientfn) \
value name##_gradient(vm *v, int nargs, value *args) { \
    objectmesh *mesh=NULL; \
    objectselection *sel=NULL; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &mesh, &sel)) { \
        functional_mapgradient(v, mesh, sel, grade, gradientfn, NULL, &out); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    \
    return out; \
}

/** Total an integrand */
#define FUNCTIONAL_TOTAL(name, grade, totalfn) \
value name##_total(vm *v, int nargs, value *args) { \
    objectmesh *mesh=NULL; \
    objectselection *sel=NULL; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &mesh, &sel)) { \
        functional_sumintegrand(v, mesh, sel, grade, totalfn, NULL, &out); \
    } \
    \
    return out; \
}

/* ----------------------------------------------
 * Length
 * ---------------------------------------------- */

/** Calculate area */
bool length_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], s0[mesh->dim];
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);
    
    functional_vecsub(mesh->dim, x[1], x[0], s0);
    
    *out=functional_vecnorm(mesh->dim, s0);
    return true;
}

/** Calculate gradient */
bool length_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    double *x[nv], s0[mesh->dim], norm;
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);
    
    functional_vecsub(mesh->dim, x[1], x[0], s0);
    norm=functional_vecnorm(mesh->dim, s0);
    if (norm<MORPHO_EPS) return false;
    
    matrix_addtocolumn(frc, vid[0], -1.0/norm, s0);
    matrix_addtocolumn(frc, vid[1], 1./norm, s0);
    
    return true;
}

FUNCTIONAL_INIT(Length, MESH_GRADE_LINE)
FUNCTIONAL_INTEGRAND(Length, MESH_GRADE_LINE, length_integrand)
FUNCTIONAL_GRADIENT(Length, MESH_GRADE_LINE, length_gradient)
FUNCTIONAL_TOTAL(Length, MESH_GRADE_LINE, length_integrand)

MORPHO_BEGINCLASS(Length)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Length_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Length_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Length_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Length_total, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Enclosed area
 * ---------------------------------------------- */

/** Calculate area enclosed */
bool areaenclosed_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], cx[mesh->dim];
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);
    
    functional_veccross(x[0], x[1], cx);
    
    *out=0.5*functional_vecnorm(mesh->dim, cx);
    return true;
}

/** Calculate gradient */
bool areaenclosed_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    double *x[nv], cx[mesh->dim], s[mesh->dim];
    double norm;
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);
    
    functional_veccross(x[0], x[1], cx);
    norm=functional_vecnorm(mesh->dim, cx);
    if (norm<MORPHO_EPS) return false;
    
    functional_veccross(x[1], cx, s);
    matrix_addtocolumn(frc, vid[0], 0.5/norm, s);
    
    functional_veccross(cx, x[0], s);
    matrix_addtocolumn(frc, vid[1], 0.5/norm, s);
    
    return true;
}

FUNCTIONAL_INIT(AreaEnclosed, MESH_GRADE_LINE)
FUNCTIONAL_INTEGRAND(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand)
FUNCTIONAL_GRADIENT(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_gradient)
FUNCTIONAL_TOTAL(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand)

MORPHO_BEGINCLASS(AreaEnclosed)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, AreaEnclosed_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, AreaEnclosed_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, AreaEnclosed_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, AreaEnclosed_total, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Area
 * ---------------------------------------------- */

/** Calculate area */
bool area_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], s0[mesh->dim], s1[mesh->dim], cx[mesh->dim];
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);
    
    functional_vecsub(mesh->dim, x[1], x[0], s0);
    functional_vecsub(mesh->dim, x[2], x[1], s1);
    
    functional_veccross(s0, s1, cx);
    
    *out=0.5*functional_vecnorm(mesh->dim, cx);
    return true;
}

/** Calculate gradient */
bool area_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    double *x[nv], s0[3], s1[3], s01[3], s010[3], s011[3];
    double norm;
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);
    
    functional_vecsub(mesh->dim, x[1], x[0], s0);
    functional_vecsub(mesh->dim, x[2], x[1], s1);
    
    functional_veccross(s0, s1, s01);
    norm=functional_vecnorm(mesh->dim, s01);
    if (norm<MORPHO_EPS) return false;
    
    functional_veccross(s01, s0, s010);
    functional_veccross(s01, s1, s011);
    
    matrix_addtocolumn(frc, vid[0], 0.5/norm, s011);
    matrix_addtocolumn(frc, vid[2], 0.5/norm, s010);
    
    functional_vecadd(mesh->dim, s010, s011, s0);
    
    matrix_addtocolumn(frc, vid[1], -0.5/norm, s0);
    
    return true;
}

FUNCTIONAL_INIT(Area, MESH_GRADE_AREA)
FUNCTIONAL_INTEGRAND(Area, MESH_GRADE_AREA, area_integrand)
FUNCTIONAL_GRADIENT(Area, MESH_GRADE_AREA, area_gradient)
FUNCTIONAL_TOTAL(Area, MESH_GRADE_AREA, area_integrand)

MORPHO_BEGINCLASS(Area)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Area_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Area_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Area_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Area_total, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Enclosed volume
 * ---------------------------------------------- */

/** Calculate enclosed volume */
bool volumeenclosed_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], cx[mesh->dim];
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);
    
    functional_veccross(x[0], x[1], cx);
    
    *out=fabs(functional_vecdot(mesh->dim, cx, x[2]))/6.0;
    return true;
}

/** Calculate gradient */
bool volumeenclosed_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    double *x[nv], cx[mesh->dim], dot;
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);
    
    functional_veccross(x[0], x[1], cx);
    dot=functional_vecdot(mesh->dim, cx, x[2]);
    dot/=fabs(dot);
    
    matrix_addtocolumn(frc, vid[2], dot/6.0, cx);
    
    functional_veccross(x[1], x[2], cx);
    matrix_addtocolumn(frc, vid[0], dot/6.0, cx);
    
    functional_veccross(x[2], x[0], cx);
    matrix_addtocolumn(frc, vid[1], dot/6.0, cx);
    
    return true;
}

FUNCTIONAL_INIT(VolumeEnclosed, MESH_GRADE_AREA)
FUNCTIONAL_INTEGRAND(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_integrand)
FUNCTIONAL_GRADIENT(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_gradient)
FUNCTIONAL_TOTAL(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_integrand)

MORPHO_BEGINCLASS(VolumeEnclosed)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, VolumeEnclosed_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, VolumeEnclosed_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, VolumeEnclosed_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, VolumeEnclosed_total, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Volume
 * ---------------------------------------------- */

/** Calculate enclosed volume */
bool volume_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], s10[mesh->dim], s20[mesh->dim], s30[mesh->dim], cx[mesh->dim];
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);
    
    functional_vecsub(mesh->dim, x[1], x[0], s10);
    functional_vecsub(mesh->dim, x[2], x[0], s20);
    functional_vecsub(mesh->dim, x[3], x[0], s30);
    
    functional_veccross(s20, s30, cx);
    
    *out=fabs(functional_vecdot(mesh->dim, s10, cx))/6.0;
    return true;
}

/** Calculate gradient */
bool volume_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    double *x[nv], s10[mesh->dim], s20[mesh->dim], s30[mesh->dim];
    double s31[mesh->dim], s21[mesh->dim], cx[mesh->dim], uu;
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);
    
    functional_vecsub(mesh->dim, x[1], x[0], s10);
    functional_vecsub(mesh->dim, x[2], x[0], s20);
    functional_vecsub(mesh->dim, x[3], x[0], s30);
    functional_vecsub(mesh->dim, x[3], x[1], s31);
    functional_vecsub(mesh->dim, x[2], x[1], s21);
    
    functional_veccross(s20, s30, cx);
    uu=functional_vecdot(mesh->dim, s10, cx);
    uu=(uu>0 ? 1.0 : -1.0);
    
    matrix_addtocolumn(frc, vid[1], uu/6.0, cx);
    
    functional_veccross(s31, s21, cx);
    matrix_addtocolumn(frc, vid[0], uu/6.0, cx);
    
    functional_veccross(s30, s10, cx);
    matrix_addtocolumn(frc, vid[2], uu/6.0, cx);
    
    functional_veccross(s10, s20, cx);
    matrix_addtocolumn(frc, vid[3], uu/6.0, cx);
    
    return true;
}

FUNCTIONAL_INIT(Volume, MESH_GRADE_VOLUME)
FUNCTIONAL_INTEGRAND(Volume, MESH_GRADE_VOLUME, volume_integrand)
FUNCTIONAL_GRADIENT(Volume, MESH_GRADE_VOLUME, volume_gradient)
FUNCTIONAL_TOTAL(Volume, MESH_GRADE_VOLUME, volume_integrand)

MORPHO_BEGINCLASS(Volume)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Volume_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Volume_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Volume_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Volume_total, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Scalar potential
 * ---------------------------------------------- */

static value scalarpotential_functionproperty;
static value scalarpotential_gradfunctionproperty;

/** Evaluate the scalar potential */
bool scalarpotential_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x;
    value fn = *(value *) ref;
    value args[mesh->dim];
    value ret;
    
    matrix_getcolumn(mesh->vert, id, &x);
    for (int i=0; i<mesh->dim; i++) args[i]=MORPHO_FLOAT(x[i]);
    
    if (morpho_call(v, fn, mesh->dim, args, &ret)) {
        return morpho_valuetofloat(ret, out);
    }
    
    return false;
}

/** Evaluate the gradient of the scalar potential */
bool scalarpotential_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    double *x;
    value fn = *(value *) ref;
    value args[mesh->dim];
    value ret;
    
    matrix_getcolumn(mesh->vert, id, &x);
    for (int i=0; i<mesh->dim; i++) args[i]=MORPHO_FLOAT(x[i]);
    
    if (morpho_call(v, fn, mesh->dim, args, &ret)) {
        if (MORPHO_ISMATRIX(ret)) {
            objectmatrix *vf=MORPHO_GETMATRIX(ret);
            
            if (vf->nrows*vf->ncols==frc->nrows) {
                return matrix_addtocolumn(frc, id, 1.0, vf->elements);
            }
        }
    }
    
    return false;
}


/** Initialize a scalar potential */
value ScalarPotential_init(vm *v, int nargs, value *args) {
    objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), functional_gradeproperty, MORPHO_INTEGER(MESH_GRADE_VERTEX));
    
    /* First argument is the potential function */
    if (nargs>0) {
        if (MORPHO_ISCALLABLE(MORPHO_GETARG(args, 0))) {
            objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), scalarpotential_functionproperty, MORPHO_GETARG(args, 0));
        } else morpho_runtimeerror(v, SCALARPOTENTIAL_FNCLLBL);
    }
    /* Second argument is the gradient of the potential function */
    if (nargs>1) {
        if (MORPHO_ISCALLABLE(MORPHO_GETARG(args, 1))) {
            objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), scalarpotential_gradfunctionproperty, MORPHO_GETARG(args, 1));
        } else morpho_runtimeerror(v, SCALARPOTENTIAL_FNCLLBL);
    }
    
    return MORPHO_NIL;
}

/** Integrand function */
value ScalarPotential_integrand(vm *v, int nargs, value *args) {
    objectmesh *mesh=NULL;
    objectselection *sel=NULL;
    value out=MORPHO_NIL;
    
    if (functional_validateargs(v, nargs, args, &mesh, &sel)) {
        value fn;
        if (objectinstance_getproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), scalarpotential_functionproperty, &fn)) {
            if (MORPHO_ISCALLABLE(fn)) {
                functional_mapintegrand(v, mesh, sel, MESH_GRADE_VERTEX, scalarpotential_integrand, &fn, &out);
            } else morpho_runtimeerror(v, SCALARPOTENTIAL_FNCLLBL);
        } else morpho_runtimeerror(v, VM_OBJECTLACKSPROPERTY, SCALARPOTENTIAL_FUNCTION_PROPERTY);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

/** Evaluate a gradient */
value ScalarPotential_gradient(vm *v, int nargs, value *args) {
    objectmesh *mesh=NULL;
    objectselection *sel=NULL;
    value out=MORPHO_NIL;
    
    if (functional_validateargs(v, nargs, args, &mesh, &sel)) {
        value fn;
        // Check if a gradient function is available
        if (objectinstance_getproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), scalarpotential_gradfunctionproperty, &fn)) {
            if (MORPHO_ISCALLABLE(fn)) {
                functional_mapgradient(v, mesh, sel, MESH_GRADE_VERTEX, scalarpotential_gradient, &fn, &out);
            } else morpho_runtimeerror(v, SCALARPOTENTIAL_FNCLLBL);
        } else if (objectinstance_getproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), scalarpotential_functionproperty, &fn)) {
            // Otherwise try to use the regular scalar function
            UNREACHABLE("Numerical derivative not implemented");
        } else morpho_runtimeerror(v, VM_OBJECTLACKSPROPERTY, SCALARPOTENTIAL_FUNCTION_PROPERTY);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    
    return out;
}

/** Total function */
value ScalarPotential_total(vm *v, int nargs, value *args) {
    objectmesh *mesh=NULL;
    objectselection *sel=NULL;
    value out=MORPHO_NIL;
    
    if (functional_validateargs(v, nargs, args, &mesh, &sel)) {
        value fn;
        if (objectinstance_getproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), scalarpotential_functionproperty, &fn)) {
            if (MORPHO_ISCALLABLE(fn)) {
                functional_sumintegrand(v, mesh, sel, MESH_GRADE_VERTEX, scalarpotential_integrand, &fn, &out);
            } else morpho_runtimeerror(v, SCALARPOTENTIAL_FNCLLBL);
        } else morpho_runtimeerror(v, VM_OBJECTLACKSPROPERTY, SCALARPOTENTIAL_FUNCTION_PROPERTY);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(ScalarPotential)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, ScalarPotential_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, ScalarPotential_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, ScalarPotential_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, ScalarPotential_total, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Linear Elasticity
 * ---------------------------------------------- */

static value linearelasticity_referenceproperty;
static value linearelasticity_poissonproperty;

typedef struct {
    objectmesh *refmesh;
    grade grade;
    double lambda; // Lamé coefficients
    double mu;     //
} linearelasticityref;

/** Calculates the Gram matrix */
void linearelasticity_calculategram(objectmatrix *vert, int dim, int nv, int *vid, objectmatrix *gram) {
    int gdim=nv-1; // Dimension of Gram matrix
    double *x[nv], // Positions of vertices
            s[gdim][nv]; // Side vectors
    
    for (int j=0; j<nv; j++) matrix_getcolumn(vert, vid[j], &x[j]); // Get vertices
    for (int j=1; j<nv; j++) functional_vecsub(dim, x[j], x[0], s[j-1]); // u_i = X_i - X_0
    // <u_i, u_j>
    for (int i=0; i<nv-1; i++) for (int j=0; j<nv-1; j++) gram->elements[i+j*gdim]=functional_vecdot(dim, s[i], s[j]);
}

/** Calculate the linear elastic energy */
bool linearelasticity_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double weight=0.0;
    linearelasticityref *info = (linearelasticityref *) ref;
    int gdim=nv-1; // Dimension of Gram matrix
    
    /* Construct static matrices */
    double gramrefel[gdim*gdim], gramdefel[gdim*gdim], qel[gdim*gdim], rel[gdim*gdim], cgel[gdim*gdim];
    objectmatrix gramref = MORPHO_STATICMATRIX(gramrefel, gdim, gdim); // Gram matrices
    objectmatrix gramdef = MORPHO_STATICMATRIX(gramdefel, gdim, gdim); //
    objectmatrix q = MORPHO_STATICMATRIX(qel, gdim, gdim); // Inverse of Gram in source domain
    objectmatrix r = MORPHO_STATICMATRIX(rel, gdim, gdim); // Intermediate calculations
    objectmatrix cg = MORPHO_STATICMATRIX(cgel, gdim, gdim); // Cauchy-Green strain tensor
    
    linearelasticity_calculategram(info->refmesh->vert, mesh->dim, nv, vid, &gramref);
    linearelasticity_calculategram(mesh->vert, mesh->dim, nv, vid, &gramdef);
    
    if (matrix_inverse(&gramref, &q)!=MATRIX_OK) return false;
    if (matrix_mul(&gramdef, &q, &r)!=MATRIX_OK) return false;
    
    matrix_identity(&cg);
    matrix_scale(&cg, -0.5);
    matrix_accumulate(&cg, 0.5, &r);
    
    double trcg=0.0, trcgcg=0.0;
    matrix_trace(&cg, &trcg);
    
    matrix_mul(&cg, &cg, &r);
    matrix_trace(&r, &trcgcg);
    
    switch (info->grade) {
        case 1: length_integrand(v, info->refmesh, id, nv, vid, NULL, &weight); break;
        case 2: area_integrand(v, info->refmesh, id, nv, vid, NULL, &weight); break;
        case 3: volume_integrand(v, info->refmesh, id, nv, vid, NULL, &weight); break;
    }
    
    *out=weight*(info->mu*trcgcg + 0.5*info->lambda*trcg*trcg);
    
    return true;
}

/** Prepares the reference structure from the LinearElasticity object's properties */
bool linearelasticity_prepareref(objectinstance *self, linearelasticityref *ref) {
    bool success=false;
    value refmesh=MORPHO_NIL;
    value grade=MORPHO_NIL;
    value poisson=MORPHO_NIL;
    
    if (objectinstance_getproperty(self, linearelasticity_referenceproperty, &refmesh) &&
        objectinstance_getproperty(self, functional_gradeproperty, &grade) &&
        MORPHO_ISINTEGER(grade) &&
        objectinstance_getproperty(self, linearelasticity_poissonproperty, &poisson) &&
        MORPHO_ISNUMBER(poisson)) {
        ref->refmesh=MORPHO_GETMESH(refmesh);
        ref->grade=MORPHO_GETINTEGERVALUE(grade);
        
        double nu = MORPHO_GETFLOATVALUE(poisson);
        
        ref->mu=0.5/(1+nu);
        ref->lambda=nu/(1+nu)/(1-2*nu);
        success=true;
    }
    return success;
}

value LinearElasticity_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    /* First argument is the reference mesh */
    if (nargs>0) {
        if (MORPHO_ISMESH(MORPHO_GETARG(args, 0))) {
            objectinstance_setproperty(self, linearelasticity_referenceproperty, MORPHO_GETARG(args, 0));
            objectmesh *mesh = MORPHO_GETMESH(MORPHO_GETARG(args, 0));
            
            objectinstance_setproperty(self, functional_gradeproperty, MORPHO_INTEGER(mesh_maxgrade(mesh)));
            objectinstance_setproperty(self, linearelasticity_poissonproperty, MORPHO_FLOAT(0.3));
        } else morpho_runtimeerror(v, LINEARELASTICITY_REF);
    } else morpho_runtimeerror(v, LINEARELASTICITY_REF);
    
    /* Second (optional) argument is the grade to act on */
    if (nargs>1) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 1))) {
            objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), functional_gradeproperty, MORPHO_GETARG(args, 1));
        }
    }
    
    return MORPHO_NIL;
}

/** Integrand function */
value LinearElasticity_integrand(vm *v, int nargs, value *args) {
    objectmesh *mesh=NULL;
    objectselection *sel=NULL;
    linearelasticityref ref;
    value out=MORPHO_NIL;
    
    if (functional_validateargs(v, nargs, args, &mesh, &sel)) {
        if (linearelasticity_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), &ref)) {
            functional_mapintegrand(v, mesh, sel, ref.grade, linearelasticity_integrand, &ref, &out);
        } else morpho_runtimeerror(v, LINEARELASTICITY_PRP);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

/** Integrand function */
value LinearElasticity_total(vm *v, int nargs, value *args) {
    objectmesh *mesh=NULL;
    objectselection *sel=NULL;
    linearelasticityref ref;
    value out=MORPHO_NIL;
    
    if (functional_validateargs(v, nargs, args, &mesh, &sel)) {
        if (linearelasticity_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), &ref)) {
            functional_sumintegrand(v, mesh, sel, ref.grade, linearelasticity_integrand, &ref, &out);
        } else morpho_runtimeerror(v, LINEARELASTICITY_PRP);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

/** Integrand function */
value LinearElasticity_gradient(vm *v, int nargs, value *args) {
    objectmesh *mesh=NULL;
    objectselection *sel=NULL;
    linearelasticityref ref;
    value out=MORPHO_NIL;
    
    if (functional_validateargs(v, nargs, args, &mesh, &sel)) {
        if (linearelasticity_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), &ref)) {
            functional_mapnumericalgradient(v, mesh, sel, ref.grade, linearelasticity_integrand, &ref, &out);
        } else morpho_runtimeerror(v, LINEARELASTICITY_PRP);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(LinearElasticity)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, LinearElasticity_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, LinearElasticity_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, LinearElasticity_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, LinearElasticity_gradient, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void functional_initialize(void) {
    functional_gradeproperty=builtin_internsymbolascstring(FUNCTIONAL_GRADE_PROPERTY);
    scalarpotential_functionproperty=builtin_internsymbolascstring(SCALARPOTENTIAL_FUNCTION_PROPERTY);
    scalarpotential_gradfunctionproperty=builtin_internsymbolascstring(SCALARPOTENTIAL_GRADFUNCTION_PROPERTY);
    linearelasticity_referenceproperty=builtin_internsymbolascstring(LINEARELASTICITY_REFERENCE_PROPERTY);
    linearelasticity_poissonproperty=builtin_internsymbolascstring(LINEARELASTICITY_POISSON_PROPERTY);
    
    builtin_addclass(LENGTH_CLASSNAME, MORPHO_GETCLASSDEFINITION(Length), MORPHO_NIL);
    builtin_addclass(AREA_CLASSNAME, MORPHO_GETCLASSDEFINITION(Area), MORPHO_NIL);
    builtin_addclass(AREAENCLOSED_CLASSNAME, MORPHO_GETCLASSDEFINITION(AreaEnclosed), MORPHO_NIL);
    builtin_addclass(VOLUMEENCLOSED_CLASSNAME, MORPHO_GETCLASSDEFINITION(VolumeEnclosed), MORPHO_NIL);
    builtin_addclass(VOLUME_CLASSNAME, MORPHO_GETCLASSDEFINITION(Volume), MORPHO_NIL);
    
    builtin_addclass(SCALARPOTENTIAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(ScalarPotential), MORPHO_NIL);
    
    builtin_addclass(LINEARELASTICITY_CLASSNAME, MORPHO_GETCLASSDEFINITION(LinearElasticity), MORPHO_NIL);
    
    morpho_defineerror(FUNC_INTEGRAND_MESH, ERROR_HALT, FUNC_INTEGRAND_MESH_MSG);
    morpho_defineerror(FUNC_ELNTFND, ERROR_HALT, FUNC_ELNTFND_MSG);
    morpho_defineerror(SCALARPOTENTIAL_FNCLLBL, ERROR_HALT, SCALARPOTENTIAL_FNCLLBL_MSG);
    morpho_defineerror(LINEARELASTICITY_REF, ERROR_HALT, LINEARELASTICITY_REF_MSG);
    morpho_defineerror(LINEARELASTICITY_PRP, ERROR_HALT, LINEARELASTICITY_PRP_MSG);
}
