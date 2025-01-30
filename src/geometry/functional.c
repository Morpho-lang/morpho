/** @file functional.c
 *  @author T J Atherton
 *
 *  @brief Functionals
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include <float.h>

#include "functional.h"
#include "morpho.h"
#include "classes.h"
#include "common.h"

#include "threadpool.h"

#include "matrix.h"
#include "sparse.h"
#include "integrate.h"
#include <math.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

value functional_gradeproperty;
value functional_fieldproperty;
//static value functional_functionproperty;

/* **********************************************************************
 * Utility functions
 * ********************************************************************** */

double fddelta1, // = pow(MORPHO_EPS, 1.0/3.0),
       fddelta2; // = pow(MORPHO_EPS, 1.0/4.0);

// Estimates the correct stepsize for cell centered finite differences
double functional_fdstepsize(double x, int order) {
    double h = fddelta1;
    if (order==2) h = fddelta2;
    
    // h should be multiplied by an estimate of the lengthscale over which f changes,
    //      | f / f''' | ^ (1/3)
    double absx = fabs(x);
    if (absx>1) h*=absx; // In the absence of other information, and unless we're near 0, use x as the best estimate.
    
    // Ensure stepsize results in a representable number
    volatile double temp = x+h; // Prevent compiler optimizing this away
    return temp-x;
}

static void functional_clearmapinfo(functional_mapinfo *info) {
    info->mesh=NULL;
    info->field=NULL;
    info->sel=NULL;
    info->g=0;
    info->id=0;
    info->integrand=NULL;
    info->grad=NULL;
    info->dependencies=NULL;
    info->cloneref=NULL;
    info->freeref=NULL;
    info->ref=NULL;
    info->sym=SYMMETRY_NONE;
}

/** Validates the arguments provided to a functional
 * @param[in] v - vm
 * @param[in] nargs - number of arguments
 * @param[in] args - the arguments
 * @param[out] info - mapinfo block  */
bool functional_validateargs(vm *v, int nargs, value *args, functional_mapinfo *info) {
    functional_clearmapinfo(info);

    for (unsigned int i=0; i<nargs; i++) {
        if (MORPHO_ISMESH(MORPHO_GETARG(args,i))) {
            info->mesh = MORPHO_GETMESH(MORPHO_GETARG(args,i));
        } else if (MORPHO_ISSELECTION(MORPHO_GETARG(args,i))) {
            info->sel = MORPHO_GETSELECTION(MORPHO_GETARG(args,i));
        } else if (MORPHO_ISFIELD(MORPHO_GETARG(args,i))) {
            info->field = MORPHO_GETFIELD(MORPHO_GETARG(args,i));
            if (info->field) info->mesh = (info->field->mesh); // Retrieve the mesh from the field
        } else if (MORPHO_ISINTEGER(MORPHO_GETARG(args,i))) {
            info->id = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args,i));
        }
    }


    if (info->mesh) return true;
    morpho_runtimeerror(v, FUNC_INTEGRAND_MESH);
    return false;
}


/* **********************************************************************
 * Common routines
 * ********************************************************************** */

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

static int functional_symmetryimagelistfn(const void *a, const void *b) {
    elementid i=*(elementid *) a; elementid j=*(elementid *) b;
    return (int) i-j;
}

/** Gets a list of all image elements (those that map onto a target element)
 * @param[in] mesh - the mesh
 * @param[in] g - grade to look up
 * @param[in] sort - whether to sort othe results
 * @param[out] ids - varray is filled with image element ids */
void functional_symmetryimagelist(objectmesh *mesh, grade g, bool sort, varray_elementid *ids) {
    objectsparse *conn=mesh_getconnectivityelement(mesh, g, g);

    ids->count=0; // Initialize the varray

    if (conn) {
        int i,j;
        void *ctr=sparsedok_loopstart(&conn->dok);

        while (sparsedok_loop(&conn->dok, &ctr, &i, &j)) {
            varray_elementidwrite(ids, j);
        }

        if (sort) qsort(ids->data, ids->count, sizeof(elementid), functional_symmetryimagelistfn);
    }
}

/** Sums forces on symmetry vertices
 * @param[in] mesh - mesh object
 * @param frc - force object; updated if symmetries are present. */
bool functional_symmetrysumforces(objectmesh *mesh, objectmatrix *frc) {
    objectsparse *s=mesh_getconnectivityelement(mesh, 0, 0); // Checking for vertex symmetries

    if (s) {
        int i,j;
        void *ctr = sparsedok_loopstart(&s->dok);
        double *fi, *fj, fsum[mesh->dim];

        while (sparsedok_loop(&s->dok, &ctr, &i, &j)) {
            if (matrix_getcolumn(frc, i, &fi) &&
                matrix_getcolumn(frc, j, &fj)) {

                for (unsigned int k=0; k<mesh->dim; k++) fsum[k]=fi[k]+fj[k];
                matrix_setcolumn(frc, i, fsum);
                matrix_setcolumn(frc, j, fsum);
            }
        }
    }

    return s;
}

bool functional_inlist(varray_elementid *list, elementid id) {
    for (unsigned int i=0; i<list->count; i++) if (list->data[i]==id) return true;
    return false;
}

bool functional_containsvertex(int nv, int *vid, elementid id) {
    for (unsigned int i=0; i<nv; i++) if (vid[i]==id) return true;
    return false;
}

/* **********************************************************************
 * Map functions
 * ********************************************************************** */

/** Sums an integrand
 * @param[in] v - virtual machine in use
 * @param[in] info - map info
 * @param[out] out - a matrix of integrand values
 * @returns true on success, false otherwise. Error reporting through VM. */
bool functional_sumintegrandX(vm *v, functional_mapinfo *info, value *out) {
    bool success=false;
    objectmesh *mesh = info->mesh;
    objectselection *sel = info->sel;
    grade g = info->g;
    functional_integrand *integrand = info->integrand;
    void *ref = info->ref;
    objectsparse *s=NULL;
    int n=0;

    if (!functional_countelements(v, mesh, g, &n, &s)) return false;

    /* Find any image elements so we can skip over them */
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    functional_symmetryimagelist(mesh, g, true, &imageids);

    if (n>0) {
        int vertexid; // Use this if looping over grade 0
        int *vid=(g==0 ? &vertexid : NULL),
            nv=(g==0 ? 1 : 0); // The vertex indices
        int sindx=0; // Index into imageids array
        double sum=0.0, c=0.0, y, t, result;

        if (sel) { // Loop over selection
            if (sel->selected[g].count>0) for (unsigned int k=0; k<sel->selected[g].capacity; k++) {
                if (!MORPHO_ISINTEGER(sel->selected[g].contents[k].key)) continue;
                elementid i = MORPHO_GETINTEGERVALUE(sel->selected[g].contents[k].key);

                // Skip this element if it's an image element:
                if ((imageids.count>0) && (sindx<imageids.count) && imageids.data[sindx]==i) { sindx++; continue; }

                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;

                if (vid && nv>0) {
                    if ((*integrand) (v, mesh, i, nv, vid, ref, &result)) {
                        y=result-c; t=sum+y; c=(t-sum)-y; sum=t; // Kahan summation
                    } else goto functional_sumintegrand_cleanup;
                }
            }
        } else { // Loop over elements
            for (elementid i=0; i<n; i++) {
                // Skip this element if it's an image element
                if ((imageids.count>0) && (sindx<imageids.count) && imageids.data[sindx]==i) { sindx++; continue; }

                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;

                if (vid && nv>0) {
                    if ((*integrand) (v, mesh, i, nv, vid, ref, &result)) {
                        y=result-c; t=sum+y; c=(t-sum)-y; sum=t; // Kahan summation
                    } else goto functional_sumintegrand_cleanup;
                }
            }
        }

        *out=MORPHO_FLOAT(sum);
    }

    success=true;

functional_sumintegrand_cleanup:
    varray_elementidclear(&imageids);
    return success;
}

/** Calculate an integrand
 * @param[in] v - virtual machine in use
 * @param[in] info - map info
 * @param[out] out - a matrix of integrand values
 * @returns true on success, false otherwise. Error reporting through VM. */
bool functional_mapintegrandX(vm *v, functional_mapinfo *info, value *out) {
    objectmesh *mesh = info->mesh;
    objectselection *sel = info->sel;
    grade g = info->g;
    functional_integrand *integrand = info->integrand;
    void *ref = info->ref;
    objectsparse *s=NULL;
    objectmatrix *new=NULL;
    bool ret=false;
    int n=0;

    /* How many elements? */
    if (!functional_countelements(v, mesh, g, &n, &s)) return false;

    /* Find any image elements so we can skip over them */
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    functional_symmetryimagelist(mesh, g, true, &imageids);

    /* Create the output matrix */
    if (n>0) {
        new=object_newmatrix(1, n, true);
        if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }
    }

    if (new) {
        int vertexid; // Use this if looping over grade 0
        int *vid=(g==0 ? &vertexid : NULL),
            nv=(g==0 ? 1 : 0); // The vertex indices
        int sindx=0; // Index into imageids array
        double result;

        if (sel) { // Loop over selection
            if (sel->selected[g].count>0) for (unsigned int k=0; k<sel->selected[g].capacity; k++) {
                if (!MORPHO_ISINTEGER(sel->selected[g].contents[k].key)) continue;
                elementid i = MORPHO_GETINTEGERVALUE(sel->selected[g].contents[k].key);

                // Skip this element if it's an image element
                if ((imageids.count>0) && (sindx<imageids.count) && imageids.data[sindx]==i) { sindx++; continue; }

                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;

                if (vid && nv>0) {
                    if ((*integrand) (v, mesh, i, nv, vid, ref, &result)) {
                        matrix_setelement(new, 0, i, result);
                    } else goto functional_mapintegrand_cleanup;
                }
            }
        } else { // Loop over elements
            for (elementid i=0; i<n; i++) {
                // Skip this element if it's an image element
                if ((imageids.count>0) && (sindx<imageids.count) && imageids.data[sindx]==i) { sindx++; continue; }

                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;

                if (vid && nv>0) {
                    if ((*integrand) (v, mesh, i, nv, vid, ref, &result)) {
                        matrix_setelement(new, 0, i, result);
                    } else goto functional_mapintegrand_cleanup;
                }
            }
        }
        *out = MORPHO_OBJECT(new);
        ret=true;
    }

    varray_elementidclear(&imageids);
    return ret;

functional_mapintegrand_cleanup:
    object_free((object *) new);
    varray_elementidclear(&imageids);
    return false;
}

/** Calculate gradient
 * @param[in] v - virtual machine in use
 * @param[in] info - map info structure
 * @param[out] out - a matrix of integrand values
 * @returns true on success, false otherwise. Error reporting through VM. */
bool functional_mapgradientX(vm *v, functional_mapinfo *info, value *out) {
    objectmesh *mesh = info->mesh;
    objectselection *sel = info->sel;
    grade g = info->g;
    functional_gradient *grad = info->grad;
    void *ref = info->ref;
    symmetrybhvr sym = info->sym;
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

        if (sym==SYMMETRY_ADD) functional_symmetrysumforces(mesh, frc);

        *out = MORPHO_OBJECT(frc);
        ret=true;
    }

functional_mapgradient_cleanup:
    if (!ret) object_free((object *) frc);

    return ret;
}

/** Calculate field gradient
 * @param[in] v - virtual machine in use
 * @param[in] info - map info structure
 * @param[out] out - a field of integrand values
 * @returns true on success, false otherwise. Error reporting through VM. */
bool functional_mapfieldgradientX(vm *v, functional_mapinfo *info, value *out) {
    objectmesh *mesh = info->mesh;
    objectfield *field = info->field;
    objectselection *sel = info->sel;
    objectfield *grad=NULL;
    grade g = info->g;
    functional_fieldgradient *fgrad = info->fieldgrad;
    void *ref = info->ref;
    objectsparse *s=NULL;
    bool ret=false;
    int n=0;

    /* How many elements? */
    if (!functional_countelements(v, mesh, g, &n, &s)) return false;

    /* Create the output field */
    if (n>0) {
        grad=object_newfield(mesh, field->prototype, field->dof);
        if (!grad) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }
        field_zero(grad);
    }

    if (grad) {
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
                    if (!(*fgrad) (v, mesh, i, nv, vid, ref, grad)) goto functional_mapfieldgradient_cleanup;
                }
            }
        } else { // Loop over elements
            for (elementid i=0; i<n; i++) {
                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;

                if (vid && nv>0) {
                    if (!(*fgrad) (v, mesh, i, nv, vid, ref, grad)) goto functional_mapfieldgradient_cleanup;
                }
            }
        }

        *out = MORPHO_OBJECT(grad);
        ret=true;
    }

functional_mapfieldgradient_cleanup:
    if (!ret) object_free((object *) grad);

    return ret;
}

static bool functional_numericalremotegradient(vm *v, functional_mapinfo *info, objectsparse *conn, elementid remoteid, elementid i, int nv, int *vid, objectmatrix *frc);

/* Calculates a numerical gradient */
bool functional_numericalgradient(vm *v, objectmesh *mesh, elementid i, int nv, int *vid, functional_integrand *integrand, void *ref, objectmatrix *frc) {
    double f0,fp,fm,x0,eps=1e-6;

    // Loop over vertices in element
    for (unsigned int j=0; j<nv; j++) {
        // Loop over coordinates
        for (unsigned int k=0; k<mesh->dim; k++) {
            matrix_getelement(frc, k, vid[j], &f0);

            matrix_getelement(mesh->vert, k, vid[j], &x0);
            
            eps=functional_fdstepsize(x0, 1);
            
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

static bool functional_numericalremotegradient(vm *v, functional_mapinfo *info, objectsparse *conn, elementid remoteid, elementid i, int nv, int *vid, objectmatrix *frc) {
    objectmesh *mesh = info->mesh;
    double f0,fp,fm,x0,eps=1e-6;

    // Loop over coordinates
    for (unsigned int k=0; k<mesh->dim; k++) {
        matrix_getelement(frc, k, remoteid, &f0);

        matrix_getelement(mesh->vert, k, remoteid, &x0);
        eps=functional_fdstepsize(x0, 1);
        
        matrix_setelement(mesh->vert, k, remoteid, x0+eps);
        if (!(*info->integrand) (v, mesh, i, nv, vid, info->ref, &fp)) return false;
        matrix_setelement(mesh->vert, k, remoteid, x0-eps);
        if (!(*info->integrand) (v, mesh, i, nv, vid, info->ref, &fm)) return false;
        matrix_setelement(mesh->vert, k, remoteid, x0);

        matrix_setelement(frc, k, remoteid, f0+(fp-fm)/(2*eps));
    }

    return true;
}

double functional_sumlist(double *list, unsigned int nel) {
    double sum=0.0, c=0.0, y,t;

    for (unsigned int i=0; i<nel; i++) {
        y=list[i]-c;
        t=sum+y;
        c=(t-sum)-y;
        sum=t;
    }

    return sum;
}

/* *************************
 * Map functions
 * ************************* */

/** Map numerical gradient over the elements
 * @param[in] v - virtual machine in use
 * @param[in] info - map info
 * @param[out] out - a matrix of integrand values
 * @returns true on success, false otherwise. Error reporting through VM. */
bool functional_mapnumericalgradientX(vm *v, functional_mapinfo *info, value *out) {
    objectmesh *mesh = info->mesh;
    objectselection *sel = info->sel;
    grade g = info->g;
    functional_integrand *integrand = info->integrand;
    void *ref = info->ref;
    symmetrybhvr sym = info->sym;
    objectsparse *s=NULL;
    objectmatrix *frc=NULL;
    bool ret=false;
    int n=0;

    varray_elementid dependencies;
    if (info->dependencies) varray_elementidinit(&dependencies);

    /* Find any image elements so we can skip over them */
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    functional_symmetryimagelist(mesh, g, true, &imageids);

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
        int sindx=0; // Index into imageids array

        if (sel) { // Loop over selection
            if (sel->selected[g].count>0) for (unsigned int k=0; k<sel->selected[g].capacity; k++) {
                if (!MORPHO_ISINTEGER(sel->selected[g].contents[k].key)) continue;

                elementid i = MORPHO_GETINTEGERVALUE(sel->selected[g].contents[k].key);
                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;

                // Skip this element if it's an image element
                if ((imageids.count>0) && (sindx<imageids.count) && imageids.data[sindx]==i) { sindx++; continue; }

                if (vid && nv>0) {
                    if (!functional_numericalgradient(v, mesh, i, nv, vid, integrand, ref, frc)) goto functional_numericalgradient_cleanup;

                    if (info->dependencies && // Loop over dependencies if there are any
                        (info->dependencies) (info, i, &dependencies)) {
                        for (int j=0; j<dependencies.count; j++) {
                            if (!functional_numericalremotegradient(v, info, s, dependencies.data[j], i, nv, vid, frc)) goto functional_numericalgradient_cleanup;
                        }
                        dependencies.count=0;
                    }
                }
            }
        } else { // Loop over elements
            for (elementid i=0; i<n; i++) {
                // Skip this element if it's an image element
                if ((imageids.count>0) && (sindx<imageids.count) && imageids.data[sindx]==i) { sindx++; continue; }

                if (s) sparseccs_getrowindices(&s->ccs, i, &nv, &vid);
                else vertexid=i;

                if (vid && nv>0) {

                    if (!functional_numericalgradient(v, mesh, i, nv, vid, integrand, ref, frc)) goto functional_numericalgradient_cleanup;

                    if (info->dependencies && // Loop over dependencies if there are any
                        (info->dependencies) (info, i, &dependencies)) {
                        for (int j=0; j<dependencies.count; j++) {
                            if (functional_containsvertex(nv, vid, dependencies.data[j])) continue;
                            if (!functional_numericalremotegradient(v, info, s, dependencies.data[j], i, nv, vid, frc)) goto functional_numericalgradient_cleanup;
                        }
                        dependencies.count=0;
                    }
                }
            }
        }

        if (sym==SYMMETRY_ADD) functional_symmetrysumforces(mesh, frc);

        *out = MORPHO_OBJECT(frc);
        ret=true;
    }

functional_numericalgradient_cleanup:
    varray_elementidclear(&imageids);
    if (info->dependencies) varray_elementidclear(&dependencies);
    if (!ret) object_free((object *) frc);

    return ret;
}

bool functional_mapnumericalfieldgradientX(vm *v, functional_mapinfo *info, value *out) {
    objectmesh *mesh = info->mesh;
    objectselection *sel = info->sel;
    objectfield *field = info->field;
    grade grd = info->g;
    functional_integrand *integrand = info->integrand;
    void *ref = info->ref;
    //symmetrybhvr sym = info->sym;

    double eps=1e-6;
    bool ret=false;
    objectsparse *conn=mesh_getconnectivityelement(mesh, 0, grd); // Connectivity for the element

    /* Create the output field */
    objectfield *grad=object_newfield(mesh, field->prototype, field->dof);
    if (!grad) return false;

    field_zero(grad);

    /* Loop over elements in the field */
    for (grade g=0; g<field->ngrades; g++) {
        if (field->dof[g]==0) continue;
        int nentries=1, *entries, nv, *vid;
        double fr,fl;
        objectsparse *rconn=mesh_addconnectivityelement(mesh, grd, g); // Find dependencies for the grade

        for (elementid id=0; id<mesh_nelementsforgrade(mesh, g); id++) {
            entries = &id; // if there's no connectivity matrix, we'll just use the id itself

            if ((!rconn) || mesh_getconnectivity(rconn, id, &nentries, &entries)) {
                for (int i=0; i<nentries; i++) {
                    if (conn) {
                        if (sel) if (!selection_isselected(sel, grd, entries[i])) continue;
                        // Check selections here
                        sparseccs_getrowindices(&conn->ccs, entries[i], &nv, &vid);
                    } else {
                        if (sel) if (!selection_isselected(sel, grd, id)) continue;
                        nv=1; vid=&id;
                    }

                    /* Loop over dofs in field entry */
                    for (int j=0; j<field->psize*field->dof[g]; j++) {
                        int k=field->offset[g]+id*field->psize*field->dof[g]+j;
                        double fld=field->data.elements[k];
                        
                        eps=functional_fdstepsize(fld, 1);
                        
                        field->data.elements[k]+=eps;

                        if (!(*integrand) (v, mesh, id, nv, vid, ref, &fr)) goto functional_mapnumericalfieldgradient_cleanup;

                        field->data.elements[k]=fld-eps;

                        if (!(*integrand) (v, mesh, id, nv, vid, ref, &fl)) goto functional_mapnumericalfieldgradient_cleanup;

                        field->data.elements[k]=fld;

                        grad->data.elements[k]+=(fr-fl)/(2*eps);
                    }
                }
            }
        }

        *out = MORPHO_OBJECT(grad);
        ret=true;
    }

functional_mapnumericalfieldgradient_cleanup:
    if (!ret) object_free((object *) grad);

    return ret;
}

/* **********************************************************************
 * Multithreaded map functions
 * ********************************************************************** */

threadpool functional_pool;
bool functional_poolinitialized;

/** Gradient function */
typedef bool (functional_mapfn) (vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out);

/** Optionally process results from mapfn */
typedef bool (functional_processfn) (void *task);

/** Work to be done is divided into "tasks" which are then dispatched to the threadpool for execution. */
typedef struct {
    elementid start, end; /* Start and end indices for the task */
    elementid id; /* Current element id */
    elementid nel; /* Current element id */
    
    varray_elementid *skip; /* Sorted list of element ids to skip; set to NULL if not needed */
    unsigned int sindx;
    
    grade g; /* Grade of element */
    objectsparse *conn; /* Connectivity matrix */
    
    functional_mapfn *mapfn; /* Map function */
    functional_processfn *processfn; /* Post process results */
    
    vm *v; /* Virtual machine in use */
    objectmesh *mesh; /* Mesh in use */
    objectfield *field; /* Field in use */
    objectselection *selection; /* Selection to use if any */
    void *ref; /* Ref as an opaque pointer */
    
    void *result; /* Result of individual element as an opaque pointer */
    void *out; /* Overall output as an opaque pointer */
    _MORPHO_PADDING;
} functional_task;

/* Initialize a task structure */
void functionaltask_init(functional_task *task, elementid start, elementid end, functional_mapinfo *info)  {
    task->start=start;
    task->end=end;
    task->nel=0;
    task->id=0;
    task->g=(info ? info->g : 0);
    
    task->skip=NULL;
    task->sindx=0;
    
    task->conn=NULL;
    
    task->mapfn=NULL;
    task->processfn=NULL;
    
    task->mesh=(info ? info->mesh : NULL);
    task->field=(info ? info->field : NULL);
    task->selection=(info ? info->sel : NULL);
    
    task->v=NULL;
    task->ref=(info ? info->ref : NULL);
    task->out=NULL;
    task->result=NULL;
}

/** Check if we should skip element id */
bool functional_checkskip(functional_task *task) {
    if ((task->skip) &&
        (task->sindx<task->skip->count) &&
        task->skip->data[task->sindx]==task->id) {
        task->sindx++;
        return true;
    }
    return false;
}

/** Worker function to map a function over elements */
bool functional_mapfn_elements(void *arg) {
    functional_task *task = (functional_task *) arg;
    dictionary *selected=NULL;
    elementid *vid=&task->id; /* Will hold element definition */
    int nv=1; /* Number of vertices per element; default to 1  */
    
    if (task->selection) {
        selected=&task->selection->selected[task->g];
        if (selected->count==0) return true;
    }
    
    // Loop over required elements
    for (elementid i=task->start; i<task->end; i++) {
        if (selected) {
            // Skip empty dictionary entries
            if (!MORPHO_ISINTEGER(selected->contents[i].key)) continue;
            
            // Fetch the element id from the dictionary
            task->id = MORPHO_GETINTEGERVALUE(selected->contents[i].key);
        } else task->id = i;
        
        // Skip this element if it's an image element
        if (functional_checkskip(task)) continue;
        
        // Fetch element definition
        if (task->conn) {
            if (!sparseccs_getrowindices(&task->conn->ccs, task->id, &nv, &vid)) return false;
        }
        
        // Perform the map function
        if (!(*task->mapfn) (task->v, task->mesh, task->id, nv, vid, task->ref, task->result)) return false;
        
        // Perform post-processing if needed
        if (task->processfn) if (!(*task->processfn) (task)) return false;
        
        // Clean out temporary objects
        vm_cleansubkernel(task->v);
    }
    return true;
}

/** Dispatches tasks to threadpool */
bool functional_parallelmap(int ntasks, functional_task *tasks) {
    int nthreads = morpho_threadnumber();
    if (!nthreads) nthreads=1;
    
    if (!functional_poolinitialized) {
        functional_poolinitialized=threadpool_init(&functional_pool, nthreads);
        if (!functional_poolinitialized) return false;
    }
    
    for (int i=0; i<ntasks; i++) {
       threadpool_add_task(&functional_pool, functional_mapfn_elements, (void *) &tasks[i]);
    }
    threadpool_fence(&functional_pool);
    
    return true;
}

/** Calculate bin sizes */
void functional_binbounds(int nel, int nbins, int *binbounds) {
    int binsizes[nbins+1];
    
    int defsize = nel / nbins;
    for (int i=0; i<nbins; i++) binsizes[i]=defsize;
    
    int rem = nel % nbins;
    while (rem>0) {
        for (int i=0; i<nbins && rem>0; i++) { binsizes[i]++; rem--; }
    }
    
    int bindx=0;
    for (int i=0; i<=nbins; i++) {
        binbounds[i]=bindx;
        bindx+=binsizes[i];
    }
}

/** Prepare tasks for submitting
 * @param[in] v - Virtual machine to use
 * @param[in] info - Info structure with functional information
 * @param[in] ntask - Number of tasks
 * @param[out] task - Task structures updated
 * @param[out] imageids - Updated to include symmetry image ids */
int functional_preparetasks(vm *v, functional_mapinfo *info, int ntask, functional_task *task, varray_elementid *imageids) {
    int nel=0;
    objectsparse *conn=NULL; // The associated connectivity matrix if any
    
    /* Work out the number of elements */
    if (!functional_countelements(v, info->mesh, info->g, &nel, &conn)) return false;
    
    int cmax=nel;
    if (info->sel) {
        cmax=info->sel->selected[info->g].capacity;
    }
    
    int bins[ntask+1];
    functional_binbounds(cmax, ntask, bins);
    
    /* Ensure all mesh topology matrices have CCS */
    int maxgrade=mesh_maxgrade(info->mesh);
    for (int i=0; i<=maxgrade; i++) {
        for (int j=0; j<=maxgrade; j++) {
            objectsparse *s = mesh_getconnectivityelement(info->mesh, i, j);
            if (s) sparse_checkformat(s, SPARSE_CCS, true, false);
        }
    }
    
    /* Find any image elements so they can be skipped */
    functional_symmetryimagelist(info->mesh, info->g, true, imageids);
    if (info->field) field_addpool(info->field);
    
    vm *subkernels[ntask];
    if (!vm_subkernels(v, ntask, subkernels)) return false; 
    
    /** Initialize task structures */
    for (int i=0; i<ntask; i++) {
        functionaltask_init(task+i, bins[i], bins[i+1], info); // Setup the task
        
        task[i].v=subkernels[i];
        task[i].nel=nel;
        task[i].conn=conn;
        if (imageids->count>0) task[i].skip=imageids;
    }
    
    return true;
}

/** Cleans up task structures after executing them. */
void functional_cleanuptasks(vm *v, int ntask, functional_task *task) {
    for (int i=0; i<ntask; i++) {
        if (task[i].v!=v) vm_releasesubkernel(task[i].v);
    }
}

/* ----------------------------
 * Sum integrands
 * ---------------------------- */

/* Structure to store intermediate results for Kahan summation */
typedef struct {
    double result;
    double c;
    double sum;
    _MORPHO_PADDING;
} functional_sumintermediate;

/** Perform Kahan summation for total */
bool functional_sumintegrandprocessfn(void *arg) {
    functional_task *task = (functional_task *) arg;
    functional_sumintermediate *ks = (functional_sumintermediate *) task->out;
    double y=ks->result-ks->c;
    double t=ks->sum+y;
    ks->c=(t-ks->sum)-y;
    ks->sum=t; // Kahan summation
    return true;
}

/** Sum the integrand, mapping over integrand function */
bool functional_sumintegrand(vm *v, functional_mapinfo *info, value *out) {
    int ntask=morpho_threadnumber();
    if (!ntask) return functional_sumintegrandX(v, info, out);
    
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    functional_sumintermediate sums[ntask];
    
    for (int i=0; i<ntask; i++) {
        task[i].mapfn=(functional_mapfn *) info->integrand;
        task[i].processfn=functional_sumintegrandprocessfn;
        
        task[i].result=(void *) &sums[i].result;
        task[i].out=(void *) &sums[i];
        sums[i].c=0.0; sums[i].sum=0.0;
    }
    
    functional_parallelmap(ntask, task);
    
    // Sum up the results from each task...
    double sumlist[ntask];
    for (int i=0; i<ntask; i++) sumlist[i]=sums[i].sum;
    
    // ...and return the result
    *out = MORPHO_FLOAT(functional_sumlist(sumlist, ntask));
    
    functional_cleanuptasks(v, ntask, task);
    varray_elementidclear(&imageids);
    return true;
}

/* ----------------------------
 * Map integrands
 * ---------------------------- */

/** Calculate the integrand at a particular element
 * @param[in] v - virtual machine in use
 * @param[in] info - map info
 * @param[out] out - a matrix of integrand values
 * @returns true on success, false otherwise. Error reporting through VM. */
bool functional_mapintegrandforelement(vm *v, functional_mapinfo *info, value *out) {
    objectmesh *mesh = info->mesh;
    grade g = info->g;
    elementid id = info->id;
    functional_integrand *integrand = info->integrand;
    void *ref = info->ref;
    objectsparse *s=NULL;
    bool ret=false;
    int n=0;
    
    /* How many elements? */
    if (!functional_countelements(v, mesh, g, &n, &s)) return false;
    // Check if the requested element id is out of range
    if (id>=n) return false;
    
    int vertexid; // Use this if looping over grade 0
    int *vid=(g==0 ? &vertexid : NULL),
        nv=(g==0 ? 1 : 0); // The vertex indices
    if (s) sparseccs_getrowindices(&s->ccs, id, &nv, &vid);
    else vertexid=id;

    double result=0.0;
    if (vid && nv>0) {
        if (! (*integrand) (v, mesh, id, nv, vid, ref, &result)) {
            return false;
        }
    }
    *out = MORPHO_FLOAT(result);
    ret=true;
    
    return ret;
}

/** Set relevant matrix element to the result of the integrand */
bool functional_mapintegrandprocessfn(void *arg) {
    functional_task *task = (functional_task *) arg;
    objectmatrix *new = (objectmatrix *) task->out;
    matrix_setelement(new, 0, task->id, *(double *) task->result);
    return true;
}

/** Map integrand function, storing the results in a matrix */
bool functional_mapintegrand(vm *v, functional_mapinfo *info, value *out) {
    int ntask=morpho_threadnumber();
    if (!ntask) return functional_mapintegrandX(v, info, out);
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectmatrix *new = NULL;
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    /* Create output matrix */
    if (task[0].nel>0) {
        new=object_newmatrix(1, task[0].nel, true);
        if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }
    }
    
    functional_sumintermediate sums[ntask];
    
    for (int i=0; i<ntask; i++) {
        task[i].mapfn=(functional_mapfn *) info->integrand;
        task[i].processfn=functional_mapintegrandprocessfn;
    
        task[i].result=(void *) &sums[i].result;
        task[i].out=(void *) new;
    }
    
    functional_parallelmap(ntask, task);
    
    // ...and return the result
    *out = MORPHO_OBJECT(new);
    
    functional_cleanuptasks(v, ntask, task);
    varray_elementidclear(&imageids);
    return true;
}

/* ----------------------------
 * Map gradients
 * ---------------------------- */

/** Compute the gradient */
bool functional_mapgradient(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=morpho_threadnumber();
    if (!ntask) return functional_mapgradientX(v, info, out);
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectmatrix *new[ntask];
    for (int i=0; i<ntask; i++) new[i]=NULL;
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    /* Create output matrix */
    for (int i=0; i<ntask; i++) {
        // Create one per thread
        new[i]=object_newmatrix(info->mesh->vert->nrows, info->mesh->vert->ncols, true);
        if (!new[i]) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapgradient_cleanup; }
        
        task[i].mapfn=(functional_mapfn *) info->grad;
        task[i].result=(void *) new[i];
    }
    
    functional_parallelmap(ntask, task);
    
    /* Then add up all the matrices */
    for (int i=1; i<ntask; i++) matrix_add(new[0], new[i], new[0]);
    
    // Use symmetry actions
    if (info->sym==SYMMETRY_ADD) functional_symmetrysumforces(info->mesh, new[0]);
    
    success=true;
    
functional_mapgradient_cleanup:
    for (int i=1; i<ntask; i++) if (new[i]) object_free((object *) new[i]);
    
    // ...and return the result
    *out = MORPHO_OBJECT(new[0]);
    
    functional_cleanuptasks(v, ntask, task);
    varray_elementidclear(&imageids);
    return success;
}

/* ----------------------------
 * Map numerical gradients
 * ---------------------------- */

/** Computes the gradient of element eid with respect to vertex i */
bool functional_numericalgrad(vm *v, objectmesh *mesh, elementid eid, elementid i, int nv, int *vid, functional_integrand *integrand, void *ref, objectmatrix *frc) {
    double f0,fp,fm,x0,eps=1e-6;
    
    // Loop over coordinates
    for (unsigned int k=0; k<mesh->dim; k++) {
        matrix_getelement(frc, k, i, &f0);

        matrix_getelement(mesh->vert, k, i, &x0);
        
        eps=functional_fdstepsize(x0, 1);
        matrix_setelement(mesh->vert, k, i, x0+eps);
        if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fp)) return false;
        matrix_setelement(mesh->vert, k, i, x0-eps);
        if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fm)) return false;
        matrix_setelement(mesh->vert, k, i, x0);

        matrix_setelement(frc, k, i, f0+(fp-fm)/(2*eps));
    }
    
    return true;
}

/** Computes the gradient of element id with respect to its constituent vertices and any dependencies */
bool functional_numericalgradientmapfn(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out) {
    bool success=true;
    functional_mapinfo *info=(functional_mapinfo *) ref;
    
    for (int i=0; i<nv; i++) {
        if (!functional_numericalgrad(v, mesh, id, vid[i], nv, vid, info->integrand, info->ref, out)) return false;
    }
    
    // Now handle dependencies
    if (info->dependencies) {
        varray_elementid dependencies;
        varray_elementidinit(&dependencies);
        
        // Get list of vertices this element depends on
        if ((info->dependencies) (info, id, &dependencies)) {
            for (int j=0; j<dependencies.count; j++) {
                if (functional_containsvertex(nv, vid, dependencies.data[j])) continue;
                if (!functional_numericalgrad(v, mesh, id, dependencies.data[j], nv, vid, info->integrand, info->ref, out)) success=false;
            }
        }
        
        varray_elementidclear(&dependencies);
    }
    
    return success;
}

/** Compute the gradient numerically */
bool functional_mapnumericalgradient(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=morpho_threadnumber();
    if (!ntask) return functional_mapnumericalgradientX(v, info, out);
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectmatrix *new[ntask]; // Create an output matrix for each thread
    for (int i=0; i<ntask; i++) new[i]=NULL;
    
    objectmesh meshclones[ntask]; // Create shallow clones of the mesh with different vertex matrices
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    for (int i=0; i<ntask; i++) {
        // Create one output matrix per thread
        new[i]=object_newmatrix(info->mesh->vert->nrows, info->mesh->vert->ncols, true);
        if (!new[i]) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapgradient_cleanup; }
        
        // Clone the vertex matrix for each thread
        meshclones[i]=*info->mesh;
        meshclones[i].vert=object_clonematrix(info->mesh->vert);
        task[i].mesh=&meshclones[i];
        
        task[i].ref=(void *) info; // Use this to pass the info structure
        task[i].mapfn=functional_numericalgradientmapfn;
        task[i].result=(void *) new[i];
    }
    
    functional_parallelmap(ntask, task);
    
    /* Then add up all the matrices */
    for (int i=1; i<ntask; i++) matrix_add(new[0], new[i], new[0]);
    
    success=true;
    
    // Use symmetry actions
    if (info->sym==SYMMETRY_ADD) functional_symmetrysumforces(info->mesh, new[0]);
    
    // ...and return the result
    *out = MORPHO_OBJECT(new[0]);
    
functional_mapgradient_cleanup:
    // Free the temporary copies of the vertex matrices
    for (int i=0; i<ntask; i++) object_free((object *) meshclones[i].vert);
    // Free spare output matrices
    for (int i=1; i<ntask; i++) if (new[i]) object_free((object *) new[i]);
    
    functional_cleanuptasks(v, ntask, task);
    varray_elementidclear(&imageids);
    
    return success;
}

/* ----------------------------
 * Map field gradients
 * ---------------------------- */

/** Compute the field gradient */
bool functional_mapfieldgradient(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=morpho_threadnumber();
    if (!ntask) return functional_mapfieldgradientX(v, info, out);
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectfield *new[ntask];
    for (int i=0; i<ntask; i++) new[i]=NULL;
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    /* Create output fields */
    for (int i=0; i<ntask; i++) {
        // Create one per thread
        new[i]=object_newfield(info->mesh, info->field->prototype, info->field->dof);
        if (!new[i]) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapfieldgradient_cleanup; }
        field_zero(new[i]);
        
        task[i].mapfn=(functional_mapfn *) info->fieldgrad;
        task[i].result=(void *) new[i];
    }
    
    functional_parallelmap(ntask, task);
    
    /* Then add up all the fields using their underlying data stores */
    for (int i=1; i<ntask; i++) matrix_add(&new[0]->data, &new[1]->data, &new[0]->data);
    
    // TODO: Use symmetry actions
    //if (info->sym==SYMMETRY_ADD) functional_symmetrysumforces(info->mesh, new[0]);
    
    success=true;
    
functional_mapfieldgradient_cleanup:
    for (int i=1; i<ntask; i++) if (new[i]) object_free((object *) new[i]);
    
    // ...and return the result
    *out = MORPHO_OBJECT(new[0]);
    
    functional_cleanuptasks(v, ntask, task);
    varray_elementidclear(&imageids);
    return success;
}

/* ----------------------------
 * Map numerical field gradients
 * ---------------------------- */

/** Computes the field gradient of element eid with respect to field grade g element i */
bool functional_numericalfieldgrad(vm *v, objectmesh *mesh, elementid eid, objectfield *field, grade g, elementid i, int nv, int *vid, functional_integrand *integrand, void *ref, objectfield *grad) {
    double fr,fl,eps=1e-6;
    
    /* Loop over dofs in field entry */
    for (int j=0; j<field->psize*field->dof[g]; j++) {
        int k=field->offset[g]+i*field->psize*field->dof[g]+j;
        double f0=field->data.elements[k];
        
        eps=functional_fdstepsize(f0, 1);
        
        field->data.elements[k]+=eps;
        if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fr)) return false;

        field->data.elements[k]=f0-eps;
        if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fl)) return false;

        field->data.elements[k]=f0;

        grad->data.elements[k]+=(fr-fl)/(2*eps);
    }
    
    return true;
}

typedef struct {
    objectfield *field;
    functional_integrand *integrand;
    void *ref;
} functional_numericalfieldgradientref;

/** Computes the gradient of element id with respect to its constituent vertices and any dependencies */
bool functional_numericalfieldgradientmapfn(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out) {
    functional_numericalfieldgradientref *tref=(functional_numericalfieldgradientref *) ref;
    grade g=0;
    
    /* Temporary code: Should establish dependencies from the discretization
       For now, we simply loop over the vertices */
    for (elementid k=0; k<nv; k++) {
        if (!functional_numericalfieldgrad(v, mesh, id, tref->field, g, vid[k], nv, vid, tref->integrand, tref->ref, out)) return false;
    }
    
    return true;
}

/** Compute the field gradient numerically */
bool functional_mapnumericalfieldgradient(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=morpho_threadnumber();
    if (!ntask) return functional_mapnumericalfieldgradientX(v, info, out);
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectfield *new[ntask]; // Create an output field for each thread
    objectfield *fieldclones[ntask]; // Create clones of the field for each thread
    functional_numericalfieldgradientref tref[ntask];
    for (int i=0; i<ntask; i++) {
        new[i]=NULL; fieldclones[i]=NULL;
    }
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    for (int i=0; i<ntask; i++) {
        // Create one output field per thread
        new[i] = object_newfield(info->mesh, info->field->prototype, info->field->dof);
        if (!new[i]) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapfieldgradient_cleanup; }
        field_zero(new[i]);
        
        // Clone the vertex matrix for each thread
        fieldclones[i]=field_clone(info->field);
        tref[i].ref=info->ref;
        if (info->cloneref) {
            tref[i].ref=(info->cloneref) (info->ref, info->field, fieldclones[i]);
        } else UNREACHABLE("Functional calls numericalfieldgradient but doesn't provide cloneref");
        tref[i].integrand=info->integrand;
        tref[i].field=fieldclones[i];
        
        task[i].ref=(void *) &tref[i]; // Use this to pass the info structure
        task[i].mapfn=functional_numericalfieldgradientmapfn;
        task[i].result=(void *) new[i];
    }
    
    functional_parallelmap(ntask, task);
    
    /* Then add up all the fields */
    for (int i=1; i<ntask; i++) matrix_add(&new[0]->data, &new[i]->data, &new[0]->data);
    
    success=true;
    
    // ...and return the result
    *out = MORPHO_OBJECT(new[0]);
    
functional_mapfieldgradient_cleanup:
    for (int i=0; i<ntask; i++) {
        // Free any cloned references
        if (info->freeref) (info->freeref) (tref[i].ref);
        else if (info->cloneref) MORPHO_FREE(tref[i].ref);
        
        // Free the temporary copies of the fields
        object_free((object *) fieldclones[i]);
    }
    
    // Free spare output matrices
    for (int i=1; i<ntask; i++) if (new[i]) object_free((object *) new[i]);
    
    functional_cleanuptasks(v, ntask, task);
    varray_elementidclear(&imageids);
    
    return success;
}

/* ----------------------------
 * Map numerical hessians
 * ---------------------------- */

/** Adds a value to an element of a sparse matrix */
bool functional_sparseaccumulate(objectsparse *A, int i, int j, double val) {
    double f0 = 0.0;
    value h0;
    if (sparsedok_get(&A->dok, i, j, &h0)) {
        if (!morpho_valuetofloat(h0, &f0)) return false;
    }
    
    sparsedok_insert(&A->dok, i, j, MORPHO_FLOAT(f0+val));
    return true;
}

/** Computes the contribution to the hessian of element eid with respect to vertices i and j */
bool functional_numericalhess(vm *v, objectmesh *mesh, elementid eid, elementid i, elementid j, int nv, int *vid, functional_integrand *integrand, void *ref, objectsparse *hess) {
    double x0,y0,epsx=1e-4,epsy=1e-4;
    
    for (unsigned int k=0; k<mesh->dim; k++) { // Loop over coordinates in vertex i
        matrix_getelement(mesh->vert, k, i, &x0);
        epsx=functional_fdstepsize(x0, 2);
        
        if (i==j) { // Use a special formula for diagonal elements
            double fc, fr, fl;
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fc)) return false;
            
            matrix_setelement(mesh->vert, k, i, x0+epsx);
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fr)) return false;
            
            matrix_setelement(mesh->vert, k, i, x0-epsx);
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fl)) return false;
            
            matrix_setelement(mesh->vert, k, i, x0); // Restore vertex to original position
            
            functional_sparseaccumulate(hess, i*mesh->dim+k, i*mesh->dim+k, (fr + fl - 2*fc)/(epsx*epsx));
        }
        
        // Loop over coordinates in vertex j
        for (unsigned int l=0; //(i==j? k+1 : k); // Detect whether we're in an off diagonal block
             l<mesh->dim; l++) {
            if (i==j && k==l) continue;
            double fll,frr,flr,frl;
            
            matrix_getelement(mesh->vert, l, j, &y0);
            epsy=functional_fdstepsize(y0, 2);
            
            matrix_setelement(mesh->vert, k, i, x0+epsx);
            matrix_setelement(mesh->vert, l, j, y0+epsy);
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &frr)) return false;
            
            matrix_setelement(mesh->vert, l, j, y0-epsy);
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &frl)) return false;
            
            matrix_setelement(mesh->vert, k, i, x0-epsx);
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fll)) return false;
            
            matrix_setelement(mesh->vert, l, j, y0+epsy);
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &flr)) return false;
            
            matrix_setelement(mesh->vert, k, i, x0); // Restore vertices to original position
            matrix_setelement(mesh->vert, l, j, y0);
            
            functional_sparseaccumulate(hess, i*mesh->dim+k, j*mesh->dim+l, (frr + fll - flr - frl)/(4*epsx*epsy));
            //functional_sparseaccumulate(hess, j*mesh->dim+l, i*mesh->dim+k, (frr + fll - flr - frl)/(4*epsx*epsy));
        }
    }
    
    return true;
}

/** Computes the gradient of element id with respect to its constituent vertices and any dependencies */
bool functional_numericalhessianmapfn(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out) {
    bool success=true;
    functional_mapinfo *info=(functional_mapinfo *) ref;
    
    // TODO: Exploit symmetry of hessian to reduce work
    
    for (int i=0; i<nv; i++) {
        for (int j=0; j<nv; j++) {
            if (!functional_numericalhess(v, mesh, id, vid[i], vid[j], nv, vid, info->integrand, info->ref, out)) return false;
        }
    }
    
    // Now handle dependencies
    if (info->dependencies) {
        varray_elementid dependencies;
        varray_elementidinit(&dependencies);
        
        // Get list of vertices this element depends on
        if ((info->dependencies) (info, id, &dependencies)) {
            for (int i=0; i<dependencies.count; i++) {
                for (int j=0; j<dependencies.count; j++) {
                    if (functional_containsvertex(nv, vid, dependencies.data[i]) && functional_containsvertex(nv, vid, dependencies.data[j])) continue;
                    if (!functional_numericalhess(v, mesh, id, dependencies.data[i], dependencies.data[j], nv, vid, info->integrand, info->ref, out)) success=false;
                }
            }
        }
        
        varray_elementidclear(&dependencies);
    }
    
    return success;
}

static int _sparsecmp(const void *a, const void *b) {
    objectsparse *aa = *(objectsparse **) a;
    objectsparse *bb = *(objectsparse **) b;
    return bb->dok.dict.count - aa->dok.dict.count;
}

/** Compute the hessian numerically */
bool functional_mapnumericalhessian(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=morpho_threadnumber();
    if (ntask==0) ntask = 1;
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectsparse *new[ntask]; // Create an output matrix for each thread
    objectmesh meshclones[ntask]; // Create shallow clones of the mesh with different vertex matrices
    
    for (int i=0; i<ntask; i++) {
        new[i]=NULL;
        meshclones[i].vert=NULL;
    }
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    for (int i=0; i<ntask; i++) {
        int N = info->mesh->dim*mesh_nvertices(info->mesh);
        
        // Create one output matrix per thread
        new[i]=object_newsparse(&N, &N);
        if (!new[i]) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_maphessian_cleanup; }
        
        // Clone the vertex matrix for each thread
        meshclones[i]=*info->mesh;
        meshclones[i].vert=object_clonematrix(info->mesh->vert);
        if (!meshclones[i].vert) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_maphessian_cleanup; }
        task[i].mesh=&meshclones[i];
        
        task[i].ref=(void *) info; // Use this to pass the info structure
        task[i].mapfn=functional_numericalhessianmapfn;
        task[i].result=(void *) new[i];
    }
    
    functional_parallelmap(ntask, task);
    
    qsort(new, ntask, sizeof(objectsparse *), _sparsecmp);
    
    if (!sparse_checkformat(new[0], SPARSE_CCS, true, true)) {
        morpho_runtimeerror(v, SPARSE_OPFAILEDERR);
        goto functional_maphessian_cleanup;
    }
    
    /* Then add up all the matrices */
    for (int i=1; i<ntask; i++) {
        if (!new[i]->dok.dict.count) continue;
        objectsparse out = MORPHO_STATICSPARSE();
        sparsedok_init(&out.dok);
        sparseccs_init(&out.ccs);
        
        if (sparse_add(new[0], new[i], 1.0, 1.0, &out)==SPARSE_OK) {
            sparseccs_clear(&new[0]->ccs);
            new[0]->ccs = out.ccs;
        } else {
            morpho_runtimeerror(v, SPARSE_OPFAILEDERR);
            goto functional_maphessian_cleanup;
        }
    }
    success=true;
    
    // Use symmetry actions
    //if (info->sym==SYMMETRY_ADD) functional_symmetrysumforces(info->mesh, new[0]);
    
    sparsedok_clear(&new[0]->dok); // Remove dok info
    
    // ...and return the result
    *out = MORPHO_OBJECT(new[0]);
    
functional_maphessian_cleanup:
    // Free the temporary copies of the vertex matrices
    for (int i=0; i<ntask; i++) object_free((object *) meshclones[i].vert);
    // Free spare output matrices
    for (int i=1; i<ntask; i++) if (new[i]) object_free((object *) new[i]);
    
    functional_cleanuptasks(v, ntask, task);
    varray_elementidclear(&imageids);
    
    return success;
}


/* **********************************************************************
 * Common library functions
 * ********************************************************************** */

/** Calculate the difference of two vectors */
void functional_vecadd(unsigned int n, double *a, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]+b[i];
}

/** Add with scale */
void functional_vecaddscale(unsigned int n, double *a, double lambda, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]+lambda*b[i];
}

/** Calculate the difference of two vectors */
void functional_vecsub(unsigned int n, double *a, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]-b[i];
}

/** Scale a vector */
void functional_vecscale(unsigned int n, double lambda, double *a, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=lambda*a[i];
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

/** 2D cross product  */
void functional_veccross2d(double *a, double *b, double *out) {
    *out=a[0]*b[1]-a[1]*b[0];
}

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
    if (nv!=2) return false;
    double *x[nv], s0[mesh->dim];
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);

    functional_vecsub(mesh->dim, x[1], x[0], s0);

    *out=functional_vecnorm(mesh->dim, s0);
    return true;
}

/** Calculate scaled gradient */
bool length_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale) {
    double *x[nv], s0[mesh->dim], norm;
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);

    functional_vecsub(mesh->dim, x[1], x[0], s0);
    norm=functional_vecnorm(mesh->dim, s0);
    if (norm<MORPHO_EPS) return false;

    matrix_addtocolumn(frc, vid[0], -1.0/norm*scale, s0);
    matrix_addtocolumn(frc, vid[1], 1./norm*scale, s0);

    return true;
}

/** Calculate gradient */
bool length_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    return length_gradient_scale(v, mesh, id, nv, vid, NULL, frc, 1.0);
}

FUNCTIONAL_INIT(Length, MESH_GRADE_LINE)
FUNCTIONAL_INTEGRAND(Length, MESH_GRADE_LINE, length_integrand)
FUNCTIONAL_INTEGRANDFORELEMENT(Length, MESH_GRADE_LINE, length_integrand)
FUNCTIONAL_GRADIENT(Length, MESH_GRADE_LINE, length_gradient, SYMMETRY_ADD)
FUNCTIONAL_TOTAL(Length, MESH_GRADE_LINE, length_integrand)
FUNCTIONAL_HESSIAN(Length, MESH_GRADE_LINE, length_integrand)

MORPHO_BEGINCLASS(Length)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Length_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Length_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRANDFORELEMENT_METHOD, Length_integrandForElement, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Length_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Length_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, Length_hessian, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Enclosed area
 * ---------------------------------------------- */

/** Calculate area enclosed */
bool areaenclosed_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], cx[mesh->dim], normcx;
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);

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

/** Calculate gradient */
bool areaenclosed_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    double *x[nv], cx[3], s[3];
    double norm;
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);

    if (mesh->dim==3) {
        functional_veccross(x[0], x[1], cx);
        norm=functional_vecnorm(mesh->dim, cx);
        if (norm<MORPHO_EPS) return false;

        functional_veccross(x[1], cx, s);
        matrix_addtocolumn(frc, vid[0], 0.5/norm, s);

        functional_veccross(cx, x[0], s);
        matrix_addtocolumn(frc, vid[1], 0.5/norm, s);
    } else if (mesh->dim==2) {
        functional_veccross2d(x[0], x[1], cx);

    }

    return true;
}

FUNCTIONAL_INIT(AreaEnclosed, MESH_GRADE_LINE)
FUNCTIONAL_INTEGRAND(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand)
FUNCTIONAL_INTEGRANDFORELEMENT(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand)
FUNCTIONAL_NUMERICALGRADIENT(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand, SYMMETRY_ADD)
//FUNCTIONAL_GRADIENT(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_gradient, SYMMETRY_ADD)
FUNCTIONAL_TOTAL(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand)
FUNCTIONAL_HESSIAN(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand)

MORPHO_BEGINCLASS(AreaEnclosed)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, AreaEnclosed_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, AreaEnclosed_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRANDFORELEMENT_METHOD, AreaEnclosed_integrandForElement, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, AreaEnclosed_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, AreaEnclosed_hessian, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, AreaEnclosed_total, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Area
 * ---------------------------------------------- */

/** Calculate area */
bool area_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    if (nv!=3) return false;
    double *x[nv], s0[3], s1[3], cx[3];
    for (int j=0; j<3; j++) { s0[j]=0; s1[j]=0; cx[j]=0; }
    for (int j=0; j<nv; j++) matrix_getcolumn(mesh->vert, vid[j], &x[j]);

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
    for (int j=0; j<nv; j++) if (!matrix_getcolumn(mesh->vert, vid[j], &x[j])) return false;

    functional_vecsub(mesh->dim, x[1], x[0], s0);
    functional_vecsub(mesh->dim, x[2], x[1], s1);

    functional_veccross(s0, s1, s01);
    norm=functional_vecnorm(3, s01);
    if (norm<MORPHO_EPS) return false;

    functional_veccross(s01, s0, s010);
    functional_veccross(s01, s1, s011);

    matrix_addtocolumn(frc, vid[0], 0.5/norm*scale, s011);
    matrix_addtocolumn(frc, vid[2], 0.5/norm*scale, s010);

    functional_vecadd(mesh->dim, s010, s011, s0);

    matrix_addtocolumn(frc, vid[1], -0.5/norm*scale, s0);

    return true;
}

/** Calculate gradient */
bool area_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    return area_gradient_scale(v, mesh, id, nv, vid, NULL, frc, 1.0);
}

FUNCTIONAL_INIT(Area, MESH_GRADE_AREA)
FUNCTIONAL_INTEGRAND(Area, MESH_GRADE_AREA, area_integrand)
FUNCTIONAL_INTEGRANDFORELEMENT(Area, MESH_GRADE_AREA, area_integrand)
FUNCTIONAL_GRADIENT(Area, MESH_GRADE_AREA, area_gradient, SYMMETRY_ADD)
FUNCTIONAL_TOTAL(Area, MESH_GRADE_AREA, area_integrand)
FUNCTIONAL_HESSIAN(Area, MESH_GRADE_AREA, area_integrand)

MORPHO_BEGINCLASS(Area)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Area_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Area_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRANDFORELEMENT_METHOD, Area_integrandForElement, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Area_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Area_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, Area_hessian, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Enclosed volume
 * ---------------------------------------------- */

/** Calculate enclosed volume */
bool volumeenclosed_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], cx[mesh->dim];
    for (int j=0; j<nv; j++) if (!matrix_getcolumn(mesh->vert, vid[j], &x[j])) return false;

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
    if (fabs(dot)<=DBL_MIN) {
        morpho_runtimeerror(v, VOLUMEENCLOSED_ZERO);
        return false;
    }
    
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
FUNCTIONAL_GRADIENT(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_gradient, SYMMETRY_ADD)
FUNCTIONAL_TOTAL(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_integrand)
FUNCTIONAL_HESSIAN(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_integrand)

MORPHO_BEGINCLASS(VolumeEnclosed)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, VolumeEnclosed_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, VolumeEnclosed_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, VolumeEnclosed_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, VolumeEnclosed_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, VolumeEnclosed_hessian, BUILTIN_FLAGSEMPTY)
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

/** Calculate scaled gradient */
bool volume_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale) {
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

    matrix_addtocolumn(frc, vid[1], uu/6.0*scale, cx);

    functional_veccross(s31, s21, cx);
    matrix_addtocolumn(frc, vid[0], uu/6.0*scale, cx);

    functional_veccross(s30, s10, cx);
    matrix_addtocolumn(frc, vid[2], uu/6.0*scale, cx);

    functional_veccross(s10, s20, cx);
    matrix_addtocolumn(frc, vid[3], uu/6.0*scale, cx);

    return true;
}

/** Calculate gradient */
bool volume_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    return volume_gradient_scale(v, mesh, id, nv, vid, NULL, frc, 1.0);
}

FUNCTIONAL_INIT(Volume, MESH_GRADE_VOLUME)
FUNCTIONAL_INTEGRAND(Volume, MESH_GRADE_VOLUME, volume_integrand)
FUNCTIONAL_GRADIENT(Volume, MESH_GRADE_VOLUME, volume_gradient, SYMMETRY_ADD)
FUNCTIONAL_TOTAL(Volume, MESH_GRADE_VOLUME, volume_integrand)
FUNCTIONAL_HESSIAN(Volume, MESH_GRADE_VOLUME, volume_integrand)

MORPHO_BEGINCLASS(Volume)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Volume_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Volume_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Volume_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Volume_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, Volume_hessian, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

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

FUNCTIONAL_METHOD(ScalarPotential, integrand, MESH_GRADE_VERTEX, scalarpotentialref, scalarpotential_prepareref, functional_mapintegrand, scalarpotential_integrand, NULL, SCALARPOTENTIAL_FNCLLBL, SYMMETRY_NONE)

FUNCTIONAL_METHOD(ScalarPotential, total, MESH_GRADE_VERTEX, scalarpotentialref, scalarpotential_prepareref, functional_sumintegrand, scalarpotential_integrand, NULL, SCALARPOTENTIAL_FNCLLBL, SYMMETRY_NONE)

FUNCTIONAL_METHOD(ScalarPotential, hessian, MESH_GRADE_VERTEX, scalarpotentialref, scalarpotential_prepareref, functional_mapnumericalhessian, scalarpotential_integrand, NULL, SCALARPOTENTIAL_FNCLLBL, SYMMETRY_NONE)

/** Evaluate a gradient */
value ScalarPotential_gradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        value fn;
        // Check if a gradient function is available
        if (objectinstance_getpropertyinterned(MORPHO_GETINSTANCE(MORPHO_SELF(args)), scalarpotential_gradfunctionproperty, &fn)) {
            info.g = MESH_GRADE_VERTEX;
            info.grad = scalarpotential_gradient;
            info.ref = &fn;
            if (MORPHO_ISCALLABLE(fn)) {
                functional_mapgradient(v, &info, &out);
            } else morpho_runtimeerror(v, SCALARPOTENTIAL_FNCLLBL);
        } else if (objectinstance_getpropertyinterned(MORPHO_GETINSTANCE(MORPHO_SELF(args)), scalarpotential_functionproperty, &fn)) {
            // Otherwise try to use the regular scalar function

            value fn;
            if (objectinstance_getpropertyinterned(MORPHO_GETINSTANCE(MORPHO_SELF(args)), scalarpotential_functionproperty, &fn)) {
                info.g = MESH_GRADE_VERTEX;
                info.integrand = scalarpotential_integrand;
                info.ref = &fn;
                if (MORPHO_ISCALLABLE(fn)) {
                    functional_mapnumericalgradient(v, &info, &out);
                } else morpho_runtimeerror(v, SCALARPOTENTIAL_FNCLLBL);
            } else morpho_runtimeerror(v, VM_OBJECTLACKSPROPERTY, SCALARPOTENTIAL_FUNCTION_PROPERTY);

        } else morpho_runtimeerror(v, VM_OBJECTLACKSPROPERTY, SCALARPOTENTIAL_FUNCTION_PROPERTY);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);

    return out;
}

MORPHO_BEGINCLASS(ScalarPotential)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, ScalarPotential_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, ScalarPotential_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, ScalarPotential_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, ScalarPotential_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, ScalarPotential_hessian, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Linear Elasticity
 * ---------------------------------------------- */

static value linearelasticity_referenceproperty;
static value linearelasticity_weightbyreferenceproperty;
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

    if (!functional_elementsize(v, info->refmesh, info->grade, id, nv, vid, &weight)) return false;

    *out=weight*(info->mu*trcgcg + 0.5*info->lambda*trcg*trcg);

    return true;
}

/** Prepares the reference structure from the LinearElasticity object's properties */
bool linearelasticity_prepareref(objectinstance *self, linearelasticityref *ref) {
    bool success=false;
    value refmesh=MORPHO_NIL;
    value grade=MORPHO_NIL;
    value poisson=MORPHO_NIL;

    if (objectinstance_getpropertyinterned(self, linearelasticity_referenceproperty, &refmesh) &&
        objectinstance_getpropertyinterned(self, functional_gradeproperty, &grade) &&
        MORPHO_ISINTEGER(grade) &&
        objectinstance_getpropertyinterned(self, linearelasticity_poissonproperty, &poisson) &&
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
    functional_mapinfo info;
    linearelasticityref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (linearelasticity_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), &ref)) {
            info.g = ref.grade;
            info.integrand = linearelasticity_integrand;
            info.ref = &ref;
            functional_mapintegrand(v, &info, &out);
        } else morpho_runtimeerror(v, LINEARELASTICITY_PRP);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

/** Total function */
value LinearElasticity_total(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    linearelasticityref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (linearelasticity_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), &ref)) {
            info.g = ref.grade;
            info.integrand = linearelasticity_integrand;
            info.ref = &ref;
            functional_sumintegrand(v, &info, &out);
        } else morpho_runtimeerror(v, LINEARELASTICITY_PRP);
    }
    return out;
}

/** Integrand function */
value LinearElasticity_gradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    linearelasticityref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (linearelasticity_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), &ref)) {
            info.g = ref.grade;
            info.integrand = linearelasticity_integrand;
            info.ref = &ref;
            info.sym = SYMMETRY_ADD;
            functional_mapnumericalgradient(v, &info, &out);
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

/* ----------------------------------------------
* Hydrogel
* ---------------------------------------------- */

static value hydrogel_aproperty;
static value hydrogel_bproperty;
static value hydrogel_cproperty;
static value hydrogel_dproperty;
static value hydrogel_phirefproperty;
static value hydrogel_phi0property;

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

    if (objectinstance_getpropertyinterned(self, linearelasticity_referenceproperty, &refmesh) &&
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

    if (fabs(V)<MORPHO_EPS) return false;

    // Determine phi0 either as a number or by looking up something in a field
    if (MORPHO_ISFIELD(info->phi0)) {
        objectfield *p = MORPHO_GETFIELD(info->phi0);
        if (!field_getelement(p, info->grade, id, 0, &vphi0)) {
            morpho_runtimeerror(v, HYDROGEL_FLDGRD, (unsigned int) info->grade);
            return false;
        }
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

    if (phi<0 || 1-phi<0) return false;

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

    if (fabs(V)<MORPHO_EPS) return false;

    // Determine phi0 either as a number or by looking up something in a field
    if (MORPHO_ISFIELD(info->phi0)) {
        objectfield *p = MORPHO_GETFIELD(info->phi0);
        if (!field_getelement(p, info->grade, id, 0, &vphi0)) {
            morpho_runtimeerror(v, HYDROGEL_FLDGRD, (unsigned int) info->grade);
            return false;
        }
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

/** Evaluate a gradient */
value Hydrogel_gradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    value out=MORPHO_NIL;
    hydrogelref ref;
    if (functional_validateargs(v, nargs, args, &info)) {
        if (hydrogel_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, -1, info.sel, &ref)) {
            info.g = ref.grade;
            info.grad = hydrogel_gradient;
            info.ref = &ref;
            info.sym = SYMMETRY_ADD;
            functional_mapgradient(v, &info, &out);
        }
    }

    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);

    return out;
}


value Hydrogel_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    int nfixed;
    value grade=MORPHO_INTEGER(-1);
    value a=MORPHO_NIL, b=MORPHO_NIL, c=MORPHO_NIL, d=MORPHO_NIL, phiref=MORPHO_NIL, phi0=MORPHO_NIL;

    if (builtin_options(v, nargs, args, &nfixed, 6,
                        hydrogel_aproperty, &a,
                        hydrogel_bproperty, &b,
                        hydrogel_cproperty, &c,
                        hydrogel_dproperty, &d,
                        hydrogel_phirefproperty, &phiref,
                        hydrogel_phi0property, &phi0,
                        functional_gradeproperty, &grade)) {

        objectinstance_setproperty(self, hydrogel_aproperty, a);
        objectinstance_setproperty(self, hydrogel_bproperty, b);
        objectinstance_setproperty(self, hydrogel_cproperty, c);
        objectinstance_setproperty(self, hydrogel_dproperty, d);
        objectinstance_setproperty(self, hydrogel_phirefproperty, phiref);
        objectinstance_setproperty(self, hydrogel_phi0property, phi0);
        objectinstance_setproperty(self, functional_gradeproperty, grade);

        if (nfixed==1 && MORPHO_ISMESH(MORPHO_GETARG(args, 0))) {
            objectinstance_setproperty(self, linearelasticity_referenceproperty, MORPHO_GETARG(args, 0));
        } else morpho_runtimeerror(v, HYDROGEL_ARGS);
    } else morpho_runtimeerror(v, HYDROGEL_ARGS);

    return MORPHO_NIL;
}

FUNCTIONAL_METHOD(Hydrogel, integrand, (ref.grade), hydrogelref, hydrogel_prepareref, functional_mapintegrand, hydrogel_integrand, NULL, HYDROGEL_PRP, SYMMETRY_NONE)

FUNCTIONAL_METHOD(Hydrogel, total, (ref.grade), hydrogelref, hydrogel_prepareref, functional_sumintegrand, hydrogel_integrand, NULL, HYDROGEL_PRP, SYMMETRY_NONE)

MORPHO_BEGINCLASS(Hydrogel)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Hydrogel_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Hydrogel_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Hydrogel_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Hydrogel_gradient, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Equielement
 * ---------------------------------------------- */

static value equielement_weightproperty;

typedef struct {
    grade grade;
    objectsparse *vtoel; // Connect vertices to elements
    objectsparse *eltov; // Connect elements to vertices
    objectmatrix *weight; // Weight field
    double mean;
} equielementref;

/** Prepares the reference structure from the Equielement object's properties */
bool equielement_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, equielementref *ref) {
    bool success=false;
    value grade=MORPHO_NIL;
    value weight=MORPHO_NIL;

    if (objectinstance_getpropertyinterned(self, functional_gradeproperty, &grade) &&
        MORPHO_ISINTEGER(grade) ) {
        ref->grade=MORPHO_GETINTEGERVALUE(grade);
        ref->weight=NULL;

        int maxgrade=mesh_maxgrade(mesh);
        if (ref->grade<0 || ref->grade>maxgrade) ref->grade = maxgrade;

        ref->vtoel=mesh_addconnectivityelement(mesh, ref->grade, 0);
        ref->eltov=mesh_addconnectivityelement(mesh, 0, ref->grade);

        if (ref->vtoel && ref->eltov) success=true;
    }

    if (objectinstance_getpropertyinterned(self, equielement_weightproperty, &weight) &&
        MORPHO_ISMATRIX(weight) ) {
        ref->weight=MORPHO_GETMATRIX(weight);
        if (ref->weight) {
            ref->mean=matrix_sum(ref->weight);
            ref->mean/=ref->weight->ncols;
        }
    }

    return success;
}


bool equielement_contains(varray_elementid *nbrs, elementid id) {
    for (unsigned int i=0; i<nbrs->count; i++) {
        if (nbrs->data[i]==id) return true;
    }
    return false;
}

/** Finds the points that a point depends on  */
bool equielement_dependencies(functional_mapinfo *info, elementid id, varray_elementid *out) {
    objectmesh *mesh = info->mesh;
    equielementref *eref = info->ref;
    bool success=false;
    varray_elementid nbrs;
    varray_elementidinit(&nbrs);

    // varray_elementidwrite(out, id); // EquiElement is a vertex element, and hence depends on itself
    
    if (mesh_findneighbors(mesh, MESH_GRADE_VERTEX, id, eref->grade, &nbrs)>0) {
        for (unsigned int i=0; i<nbrs.count; i++) {
            int nentries, *entries; // Get the vertices for this element
            if (!sparseccs_getrowindices(&eref->eltov->ccs, nbrs.data[i], &nentries, &entries)) goto equieleement_dependencies_cleanup;

            for (unsigned int j=0; j<nentries; j++) {
                if (entries[j]==id) continue;
                if (equielement_contains(out, entries[j])) continue;
                varray_elementidwrite(out, entries[j]);
            }
        }
    }
    success=true;

equieleement_dependencies_cleanup:
    varray_elementidclear(&nbrs);

    return success;
}

/** Calculate the equielement energy */
bool equielement_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *r, double *out) {
    equielementref *ref = (equielementref *) r;
    int nconn, *conn;

    if (sparseccs_getrowindices(&ref->vtoel->ccs, id, &nconn, &conn)) {
        if (nconn==1) { *out = 0; return true; }

        double size[nconn], mean=0.0, total=0.0;

        for (int i=0; i<nconn; i++) {
            int nv, *vid;
            sparseccs_getrowindices(&ref->eltov->ccs, conn[i], &nv, &vid);
            functional_elementsize(v, mesh, ref->grade, conn[i], nv, vid, &size[i]);
            mean+=size[i];
        }

        mean /= ((double) nconn);

        if (fabs(mean)<MORPHO_EPS) return false;

        /* Now evaluate the functional at this vertex */
        if (!ref->weight || fabs(ref->mean)<MORPHO_EPS) {
            for (unsigned int i=0; i<nconn; i++) total+=(1.0-size[i]/mean)*(1.0-size[i]/mean);
        } else {
            double weight[nconn], wmean=0.0;

            for (int i=0; i<nconn; i++) {
                weight[i]=1.0;
                matrix_getelement(ref->weight, 0, conn[i], &weight[i]);
                wmean+=weight[i];
            }

            wmean /= ((double) nconn);
            if (fabs(wmean)<MORPHO_EPS) wmean = 1.0;

            for (unsigned int i=0; i<nconn; i++) {
                double term = (1.0-weight[i]*size[i]/mean/wmean);
                total+=term*term;
            }
        }

        *out = total;
    }

    return true;
}

value EquiElement_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    int nfixed;
    value grade=MORPHO_INTEGER(-1);
    value weight=MORPHO_NIL;

    if (builtin_options(v, nargs, args, &nfixed, 2, equielement_weightproperty, &weight, functional_gradeproperty, &grade)) {
        objectinstance_setproperty(self, equielement_weightproperty, weight);
        objectinstance_setproperty(self, functional_gradeproperty, grade);
    } else morpho_runtimeerror(v, EQUIELEMENT_ARGS);

    return MORPHO_NIL;
}

FUNCTIONAL_METHOD(EquiElement, integrand, MESH_GRADE_VERTEX, equielementref, equielement_prepareref, functional_mapintegrand, equielement_integrand, NULL, EQUIELEMENT_ARGS, SYMMETRY_NONE)

FUNCTIONAL_METHOD(EquiElement, total, MESH_GRADE_VERTEX, equielementref, equielement_prepareref, functional_sumintegrand, equielement_integrand, NULL, EQUIELEMENT_ARGS, SYMMETRY_NONE)

FUNCTIONAL_METHOD(EquiElement, gradient, MESH_GRADE_VERTEX, equielementref, equielement_prepareref, functional_mapnumericalgradient, equielement_integrand, equielement_dependencies, EQUIELEMENT_ARGS, SYMMETRY_ADD)

FUNCTIONAL_METHOD(EquiElement, hessian, MESH_GRADE_VERTEX, equielementref, equielement_prepareref, functional_mapnumericalhessian, equielement_integrand, equielement_dependencies, EQUIELEMENT_ARGS, SYMMETRY_ADD)

MORPHO_BEGINCLASS(EquiElement)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, EquiElement_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, EquiElement_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, EquiElement_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, EquiElement_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, EquiElement_hessian, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

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

        if (s0s0<MORPHO_EPS || s1s1<MORPHO_EPS) return false;

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
FUNCTIONAL_METHOD(LineCurvatureSq, integrand, MESH_GRADE_VERTEX, curvatureref, curvature_prepareref, functional_mapintegrand, linecurvsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(LineCurvatureSq, integrandForElement, MESH_GRADE_VERTEX, curvatureref, curvature_prepareref, functional_mapintegrandforelement, linecurvsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(LineCurvatureSq, total, MESH_GRADE_VERTEX, curvatureref, curvature_prepareref, functional_sumintegrand, linecurvsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(LineCurvatureSq, gradient, MESH_GRADE_VERTEX, curvatureref, curvature_prepareref, functional_mapnumericalgradient, linecurvsq_integrand, linecurvsq_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)
FUNCTIONAL_METHOD(LineCurvatureSq, hessian, MESH_GRADE_VERTEX, curvatureref, curvature_prepareref, functional_mapnumericalhessian, linecurvsq_integrand, linecurvsq_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)

MORPHO_BEGINCLASS(LineCurvatureSq)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, LineCurvatureSq_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, LineCurvatureSq_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRANDFORELEMENT_METHOD, LineCurvatureSq_integrandForElement, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, LineCurvatureSq_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, LineCurvatureSq_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, LineCurvatureSq_hessian, BUILTIN_FLAGSEMPTY)
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
    for (int i=0; i<6; i++) matrix_getcolumn(mesh->vert, vlist[i], &x[i]);

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
FUNCTIONAL_METHOD(LineTorsionSq, integrand, MESH_GRADE_LINE, curvatureref, curvature_prepareref, functional_mapintegrand, linetorsionsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(LineTorsionSq, total, MESH_GRADE_LINE, curvatureref, curvature_prepareref, functional_sumintegrand, linetorsionsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(LineTorsionSq, gradient, MESH_GRADE_LINE, curvatureref, curvature_prepareref, functional_mapnumericalgradient, linetorsionsq_integrand, linetorsionsq_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)
FUNCTIONAL_METHOD(LineTorsionSq, hessian, MESH_GRADE_LINE, curvatureref, curvature_prepareref, functional_mapnumericalhessian, linetorsionsq_integrand, linetorsionsq_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)

MORPHO_BEGINCLASS(LineTorsionSq)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, LineTorsionSq_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, LineTorsionSq_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, LineTorsionSq_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, LineTorsionSq_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, LineTorsionSq_hessian, BUILTIN_FLAGSEMPTY)
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
        for (int j=0; j<3; j++) matrix_getcolumn(mesh->vert, vids[j], &x[j]);

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
FUNCTIONAL_METHOD(MeanCurvatureSq, integrand, MESH_GRADE_VERTEX, areacurvatureref, areacurvature_prepareref, functional_mapintegrand, meancurvaturesq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(MeanCurvatureSq, total, MESH_GRADE_VERTEX, areacurvatureref, areacurvature_prepareref, functional_sumintegrand, meancurvaturesq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(MeanCurvatureSq, gradient, MESH_GRADE_VERTEX, areacurvatureref, areacurvature_prepareref, functional_mapnumericalgradient, meancurvaturesq_integrand, meancurvaturesq_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)

MORPHO_BEGINCLASS(MeanCurvatureSq)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, MeanCurvatureSq_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, MeanCurvatureSq_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, MeanCurvatureSq_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, MeanCurvatureSq_total, BUILTIN_FLAGSEMPTY)
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
        for (int j=0; j<3; j++) matrix_getcolumn(mesh->vert, vids[j], &x[j]);

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
FUNCTIONAL_METHOD(GaussCurvature, integrand, MESH_GRADE_VERTEX, areacurvatureref, areacurvature_prepareref, functional_mapintegrand, gausscurvature_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(GaussCurvature, total, MESH_GRADE_VERTEX, areacurvatureref, areacurvature_prepareref, functional_sumintegrand, gausscurvature_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(GaussCurvature, gradient, MESH_GRADE_VERTEX, areacurvatureref, areacurvature_prepareref, functional_mapnumericalgradient, gausscurvature_integrand, meancurvaturesq_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)

MORPHO_BEGINCLASS(GaussCurvature)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, GaussCurvature_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, GaussCurvature_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, GaussCurvature_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, GaussCurvature_total, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Fields
 * ********************************************************************** */

typedef struct {
    objectfield *field;
    grade grade;
} fieldref;

/* ----------------------------------------------
 * GradSq
 * ---------------------------------------------- */

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

    /* Perpendicular vectors */
    gradsq_computeperpendicular(mesh->dim, s[2], s[1], t[0]);
    gradsq_computeperpendicular(mesh->dim, s[0], s[2], t[1]);
    gradsq_computeperpendicular(mesh->dim, s[1], s[0], t[2]);

    /* Compute the gradient */
    for (unsigned int i=0; i<mesh->dim*nentries; i++) out[i]=0;
    for (unsigned int j=0; j<nv; j++) {
        for (unsigned int i=0; i<nentries; i++) {
            functional_vecaddscale(mesh->dim, &out[i*mesh->dim], f[j][i], t[j], &out[i*mesh->dim]);
        }
    }

    return true;
}

/** Evaluates the gradient of a field quantity in 1D
 @param[in] mesh - object to use
 @param[in] field - field to compute gradient of
 @param[in] nv - number of vertices
 @param[in] vid - vertex ids
 @param[out] out - should be field->psize * mesh->dim units of storage */
bool gradsq_evaluategradient1d(objectmesh *mesh, objectfield *field, int nv, int *vid, double *out) {
    UNREACHABLE("GradSq in 1D not implemented.");
    double *f[nv]; // Field value lists
    double *x[nv]; // Vertex coordinates
    unsigned int nentries=0;

    // Get field values and vertex coordinates
    for (unsigned int i=0; i<nv; i++) {
        if (!mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i])) return false;
        if (!field_getelementaslist(field, MESH_GRADE_VERTEX, vid[i], 0, &nentries, &f[i])) return false;
    }

    double s[mesh->dim];

    /* Vector sides */
    functional_vecsub(mesh->dim, x[1], x[0], s);

    /* Compute the gradient */
    for (unsigned int i=0; i<mesh->dim*nentries; i++) out[i]=0;
    for (unsigned int j=0; j<nv; j++) {
        for (unsigned int i=0; i<nentries; i++) {
//            functional_vecaddscale(mesh->dim, &out[i*mesh->dim], f[j][i], t[j], &out[i*mesh->dim]);
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

    double farray[nentries*mesh->dim]; // Field elements
    objectmatrix frhs = MORPHO_STATICMATRIX(farray, mesh->dim, nentries);
    objectmatrix grad = MORPHO_STATICMATRIX(out, mesh->dim, nentries);

    // Loop over elements of the field
    for (unsigned int i=0; i<nentries; i++) {
        // Copy across the field values to form the rhs
        for (unsigned int j=0; j<mesh->dim; j++) farray[i*mesh->dim+j] = f[j+1][i]-f[0][i];
    }

    // Solve to obtain the gradient of each element
    matrix_divs(&Mt, &frhs, &grad);

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
        return false;
    }

    double gradnrm=functional_vecnorm(eref->field->psize*mesh->dim, grad);
    *out = gradnrm*gradnrm*size;

    return true;
}

/** Initialize a GradSq object */
value GradSq_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));

    if (nargs>0 && MORPHO_ISFIELD(MORPHO_GETARG(args, 0))) {
        objectinstance_setproperty(self, functional_fieldproperty, MORPHO_GETARG(args, 0));
    } else {
        morpho_runtimeerror(v, VM_INVALIDARGS);
        return MORPHO_FALSE;
    }

    /* Second (optional) argument is the grade to act on */
    if (nargs>1) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 1))) {
            objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), functional_gradeproperty, MORPHO_GETARG(args, 1));
        }
    }

    return MORPHO_NIL;
}

FUNCTIONAL_METHOD(GradSq, integrand, (ref.grade), fieldref, gradsq_prepareref, functional_mapintegrand, gradsq_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(GradSq, total, (ref.grade), fieldref, gradsq_prepareref, functional_sumintegrand, gradsq_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(GradSq, gradient, (ref.grade), fieldref, gradsq_prepareref, functional_mapnumericalgradient, gradsq_integrand, NULL, GRADSQ_ARGS, SYMMETRY_ADD);

value GradSq_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    fieldref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (gradsq_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, MESH_GRADE_AREA, info.sel, &ref)) {
            info.g = ref.grade;
            info.field = ref.field;
            info.integrand = gradsq_integrand;
            info.cloneref = gradsq_cloneref;
            info.ref = &ref;
            functional_mapnumericalfieldgradient(v, &info, &out);
            //functional_mapfieldgradient(v, &info, &out);
        } else morpho_runtimeerror(v, GRADSQ_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(GradSq)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, GradSq_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, GradSq_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, GradSq_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, GradSq_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, GradSq_fieldgradient, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Nematic
 * ---------------------------------------------- */

static value nematic_ksplayproperty;
static value nematic_ktwistproperty;
static value nematic_kbendproperty;
static value nematic_pitchproperty;

typedef struct {
    double ksplay,ktwist,kbend,pitch;
    bool haspitch;
    objectfield *field;
    grade grade;
} nematicref;

/** Prepares the nematic reference */
bool nematic_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, nematicref *ref) {
    bool success=false, grdset=false;
    value field=MORPHO_NIL, grd=MORPHO_NIL;
    value val=MORPHO_NIL;
    ref->ksplay=1.0; ref->ktwist=1.0; ref->kbend=1.0; ref->pitch=0.0;
    ref->haspitch=false;

    if (objectinstance_getpropertyinterned(self, functional_fieldproperty, &field) &&
        MORPHO_ISFIELD(field)) {
        ref->field=MORPHO_GETFIELD(field);
        success=true;
    }
    if (objectinstance_getpropertyinterned(self, nematic_ksplayproperty, &val) && MORPHO_ISNUMBER(val)) {
        morpho_valuetofloat(val, &ref->ksplay);
    }
    if (objectinstance_getpropertyinterned(self, nematic_ktwistproperty, &val) && MORPHO_ISNUMBER(val)) {
        morpho_valuetofloat(val, &ref->ktwist);
    }
    if (objectinstance_getpropertyinterned(self, nematic_kbendproperty, &val) && MORPHO_ISNUMBER(val)) {
        morpho_valuetofloat(val, &ref->kbend);
    }
    if (objectinstance_getpropertyinterned(self, nematic_pitchproperty, &val) && MORPHO_ISNUMBER(val)) {
        morpho_valuetofloat(val, &ref->pitch);
        ref->haspitch=true;
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
void *nematic_cloneref(void *ref, objectfield *field, objectfield *sub) {
    nematicref *nref = (nematicref *) ref;
    nematicref *clone = MORPHO_MALLOC(sizeof(nematicref));
    
    if (clone) {
        *clone = *nref;
        if (clone->field==field) clone->field=sub;
    }
    
    return clone;
}

/* Integrates two linear functions with values at vertices f[0]...f[2] and g[0]...g[2] */
double nematic_bcint(double *f, double *g) {
    return (f[0]*(2*g[0]+g[1]+g[2]) + f[1]*(g[0]+2*g[1]+g[2]) + f[2]*(g[0]+g[1]+2*g[2]))/12;
}

/* Integrates a linear vector function with values at vertices f[0]...f[2] */
double nematic_bcint1(double *f) {
    return (f[0] + f[1] + f[2])/3;
}

/* Integrates a linear vector function with values at vertices f[0]...f[n]
   Works for dimensions 1-3 at least */
double nematic_bcintf(unsigned int n, double *f) {
    double sum = 0;
    for (unsigned int i=0; i<n; i++) sum+=f[i];
    return sum/n;
}

/* Integrates a product of two linear functions with values at vertices
   f[0]...f[n] and g[0]...g[n].
   Works for dimensions 1-3 at least */
double nematic_bcintfg(unsigned int n, double *f, double *g) {
    double sum = 0;
    for (unsigned int i=0; i<n; i++) {
        for (unsigned int j=0; j<n; j++) sum+=f[i]*g[j];
        sum+=f[i]*g[i];
    }
    return sum/(n*(n+1));
}

/** Calculate the nematic energy */
bool nematic_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    nematicref *eref = ref;
    double size=0; // Length area or volume of the element
    double gradnnraw[eref->field->psize*3];
    double gradnn[eref->field->psize*3];
    double divnn, curlnn[3] = { 0.0, 0.0, 0.0 };
    
    for (int i=0; i<eref->field->psize*3; i++) { gradnn[i]=0.0; gradnnraw[i]=0.0; }

    if (!functional_elementsize(v, mesh, eref->grade, id, nv, vid, &size)) return false;

    // Get nematic director components
    double *nn[nv]; // Field value lists
    unsigned int nentries=0;
    for (unsigned int i=0; i<nv; i++) {
        if (!field_getelementaslist(eref->field, MESH_GRADE_VERTEX, vid[i], 0, &nentries, &nn[i])) return false;
    }

    // Evaluate gradients of the director
    if (eref->grade==2) {
        if (!gradsq_evaluategradient(mesh, eref->field, nv, vid, gradnnraw)) return
            false;
    } else if (eref->grade==3) {
        if (!gradsq_evaluategradient3d(mesh, eref->field, nv, vid, gradnnraw)) return
            false;
    }
    
    // Copy into 3x3 matrix
    for (int j=0; j<3; j++) for (int i=0; i<mesh->dim; i++) gradnn[3*j+i] = gradnnraw[mesh->dim*j+i];
    
    // Output of this is the matrix:
    // [ nx,x ny,x nz,x ] [ 0 3 6 ] <- indices
    // [ nx,y ny,y nz,y ] [ 1 4 7 ]
    // [ nx,z ny,z nz,z ] [ 2 5 8 ]
    objectmatrix gradnnmat = MORPHO_STATICMATRIX(gradnn, 3, 3);

    matrix_trace(&gradnnmat, &divnn);
    curlnn[0]=gradnn[7]-gradnn[5]; // nz,y - ny,z
    curlnn[1]=gradnn[2]-gradnn[6]; // nx,z - nz,x
    curlnn[2]=gradnn[3]-gradnn[1]; // ny,x - nx,y

    /* From components of the curl, construct the coefficients that go in front of integrals of
           nx^2, ny^2, nz^2, nx*ny, ny*nz, and nz*nx over the element. */
    double ctwst[6] = { curlnn[0]*curlnn[0], curlnn[1]*curlnn[1], curlnn[2]*curlnn[2],
                        2*curlnn[0]*curlnn[1], 2*curlnn[1]*curlnn[2], 2*curlnn[2]*curlnn[0]};

    double cbnd[6] = { ctwst[1] + ctwst[2], ctwst[0] + ctwst[2], ctwst[0] + ctwst[1],
                       -ctwst[3], -ctwst[4], -ctwst[5] };

    /* Calculate integrals of nx^2, ny^2, nz^2, nx*ny, ny*nz, and nz*nx over the element */
    double nnt[3][nv]; // The transpose of nn
    for (unsigned int i=0; i<nv; i++)
        for (unsigned int j=0; j<3; j++) nnt[j][i]=nn[i][j];

    double integrals[] = {  nematic_bcintfg(nv, nnt[0], nnt[0]),
                            nematic_bcintfg(nv, nnt[1], nnt[1]),
                            nematic_bcintfg(nv, nnt[2], nnt[2]),
                            nematic_bcintfg(nv, nnt[0], nnt[1]),
                            nematic_bcintfg(nv, nnt[1], nnt[2]),
                            nematic_bcintfg(nv, nnt[2], nnt[0])
    };

    /* Now we can calculate the components of splay, twist and bend */
    double splay=0.0, twist=0.0, bend=0.0, chol=0.0;

    /* Evaluate the three contributions to the integral */
    splay = 0.5*eref->ksplay*size*divnn*divnn;
    for (unsigned int i=0; i<6; i++) {
        twist += ctwst[i]*integrals[i];
        bend += cbnd[i]*integrals[i];
    }
    twist *= 0.5*eref->ktwist*size;
    bend *= 0.5*eref->kbend*size;

    if (eref->haspitch) {
        /* Cholesteric terms: 0.5 * k22 * [- 2 q (cx <nx> + cy <ny> + cz <nz>) + q^2] */
        for (unsigned i=0; i<3; i++) {
            chol += -2*curlnn[i]*nematic_bcintf(nv, nnt[i])*eref->pitch;
        }
        chol += (eref->pitch*eref->pitch);
        chol *= 0.5*eref->ktwist*size;
    }

    *out = splay+twist+bend+chol;

    return true;
}

/** Initialize a Nematic object */
value Nematic_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));

    int nfixed=nargs;
    value ksplay=MORPHO_FLOAT(1.0),
          ktwist=MORPHO_FLOAT(1.0),
          kbend=MORPHO_FLOAT(1.0);
    value pitch=MORPHO_NIL;

    if (builtin_options(v, nargs, args, &nfixed, 4,
                        nematic_ksplayproperty, &ksplay,
                        nematic_ktwistproperty, &ktwist,
                        nematic_kbendproperty, &kbend,
                        nematic_pitchproperty, &pitch)) {
        objectinstance_setproperty(self, nematic_ksplayproperty, ksplay);
        objectinstance_setproperty(self, nematic_ktwistproperty, ktwist);
        objectinstance_setproperty(self, nematic_kbendproperty, kbend);
        objectinstance_setproperty(self, nematic_pitchproperty, pitch);
    } else morpho_runtimeerror(v, NEMATIC_ARGS);

    if (nfixed==1 && MORPHO_ISFIELD(MORPHO_GETARG(args, 0))) {
        objectinstance_setproperty(self, functional_fieldproperty, MORPHO_GETARG(args, 0));
    } else morpho_runtimeerror(v, NEMATIC_ARGS);

    return MORPHO_NIL;
}

FUNCTIONAL_METHOD(Nematic, integrand, (ref.grade), nematicref, nematic_prepareref, functional_mapintegrand, nematic_integrand, NULL, NEMATIC_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(Nematic, total, (ref.grade), nematicref, nematic_prepareref, functional_sumintegrand, nematic_integrand, NULL, NEMATIC_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(Nematic, gradient, (ref.grade), nematicref, nematic_prepareref, functional_mapnumericalgradient, nematic_integrand, NULL, NEMATIC_ARGS, SYMMETRY_NONE);

value Nematic_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    nematicref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (nematic_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, MESH_GRADE_AREA, info.sel, &ref)) {
            info.g=ref.grade;
            info.integrand=nematic_integrand;
            info.ref=&ref;
            info.cloneref=nematic_cloneref;
            functional_mapnumericalfieldgradient(v, &info, &out);
        } else morpho_runtimeerror(v, GRADSQ_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(Nematic)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Nematic_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Nematic_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Nematic_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Nematic_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, Nematic_fieldgradient, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * NematicElectric
 * ---------------------------------------------- */

typedef struct {
    objectfield *director;
    value field;
    grade grade;
} nematicelectricref;

/** Prepares the nematicelectric reference */
bool nematicelectric_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, nematicelectricref *ref) {
    bool success=false, grdset=false;
    ref->field=MORPHO_NIL;
    value fieldlist=MORPHO_NIL, grd=MORPHO_NIL;

    if (objectinstance_getpropertyinterned(self, functional_fieldproperty, &fieldlist) &&
        MORPHO_ISLIST(fieldlist)) {
        objectlist *lst = MORPHO_GETLIST(fieldlist);
        value director = MORPHO_NIL;
        list_getelement(lst, 0, &director);
        list_getelement(lst, 1, &ref->field);

        if (MORPHO_ISFIELD(director)) ref->director=MORPHO_GETFIELD(director);

        if (MORPHO_ISFIELD(ref->field) || MORPHO_ISMATRIX(ref->field)) success=true;
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
void *nematicelectric_cloneref(void *ref, objectfield *field, objectfield *sub) {
    nematicelectricref *nref = (nematicelectricref *) ref;
    nematicelectricref *clone = MORPHO_MALLOC(sizeof(nematicelectricref));
    
    if (clone) {
        *clone = *nref;
        if (clone->director==field) clone->director=sub;
        if (MORPHO_ISFIELD(clone->field) &&
            MORPHO_GETFIELD(clone->field)==field) {
            clone->field=MORPHO_OBJECT(sub);
        }
    }
    
    return clone;
}

/** Calculate the integral (n.E)^2 energy, where E is calculated from the electric potential */
bool nematicelectric_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    nematicelectricref *eref = ref;
    double size=0; // Length area or volume of the element

    if (!functional_elementsize(v, mesh, eref->grade, id, nv, vid, &size)) return false;

    // Get nematic director components
    double *nn[nv]; // Field value lists
    unsigned int nentries=0;
    for (unsigned int i=0; i<nv; i++) {
        if (!field_getelementaslist(eref->director, MESH_GRADE_VERTEX, vid[i], 0, &nentries, &nn[i])) return false;
    }

    // The electric field ends up being constant over the element
    double ee[mesh->dim];
    if (MORPHO_ISFIELD(eref->field)) {
        if (eref->grade==2) {
            if (!gradsq_evaluategradient(mesh, MORPHO_GETFIELD(eref->field), nv, vid, ee)) return false;
        } else if (eref->grade==3) {
            if (!gradsq_evaluategradient3d(mesh, MORPHO_GETFIELD(eref->field), nv, vid, ee)) return false;
        }
    }

    /* Calculate integrals of nx^2, ny^2, nz^2, nx*ny, ny*nz, and nz*nx over the element */
    double nnt[mesh->dim][nv]; // The transpose of nn
    for (unsigned int i=0; i<nv; i++)
        for (unsigned int j=0; j<mesh->dim; j++) nnt[j][i]=nn[i][j];

    /* Calculate integral (n.e)^2 using the above results */
    double total = ee[0]*ee[0]*nematic_bcintfg(nv, nnt[0], nnt[0])+
                   ee[1]*ee[1]*nematic_bcintfg(nv, nnt[1], nnt[1])+
                   ee[2]*ee[2]*nematic_bcintfg(nv, nnt[2], nnt[2])+
                   2*ee[0]*ee[1]*nematic_bcintfg(nv, nnt[0], nnt[1])+
                   2*ee[1]*ee[2]*nematic_bcintfg(nv, nnt[1], nnt[2])+
                   2*ee[2]*ee[0]*nematic_bcintfg(nv, nnt[2], nnt[0]);

    *out = size*total;

    return true;
}

/** Initialize a NematicElectric object */
value NematicElectric_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));

    if (nargs==2 && MORPHO_ISFIELD(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISFIELD(MORPHO_GETARG(args, 1))) {
        objectlist *new = object_newlist(2, &MORPHO_GETARG(args, 0));
        if (new) {
            value lst = MORPHO_OBJECT(new);
            objectinstance_setproperty(self, functional_fieldproperty, lst);
            morpho_bindobjects(v, 1, &lst);
        }
    } else morpho_runtimeerror(v, NEMATICELECTRIC_ARGS);

    return MORPHO_NIL;
}

FUNCTIONAL_METHOD(NematicElectric, integrand, (ref.grade), nematicelectricref, nematicelectric_prepareref, functional_mapintegrand, nematicelectric_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(NematicElectric, total, (ref.grade), nematicelectricref, nematicelectric_prepareref, functional_sumintegrand, nematicelectric_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(NematicElectric, gradient, (ref.grade), nematicelectricref, nematicelectric_prepareref, functional_mapnumericalgradient, nematicelectric_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

value NematicElectric_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    nematicelectricref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (nematicelectric_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, MESH_GRADE_AREA, info.sel, &ref)) {
            info.g=ref.grade;
            info.integrand=nematicelectric_integrand;
            info.cloneref=nematicelectric_cloneref;
            info.ref=&ref;
            functional_mapnumericalfieldgradient(v, &info, &out);
        } else morpho_runtimeerror(v, GRADSQ_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(NematicElectric)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, NematicElectric_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, NematicElectric_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, NematicElectric_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, NematicElectric_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, NematicElectric_fieldgradient, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

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

FUNCTIONAL_METHOD(NormSq, integrand, MESH_GRADE_VERTEX, fieldref, gradsq_prepareref, functional_mapintegrand, normsq_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(NormSq, total, MESH_GRADE_VERTEX, fieldref, gradsq_prepareref, functional_sumintegrand, normsq_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(NormSq, gradient, MESH_GRADE_VERTEX, fieldref, gradsq_prepareref, functional_mapnumericalgradient, normsq_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

value NormSq_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    fieldref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (gradsq_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, MESH_GRADE_VERTEX, info.sel, &ref)) {
            info.g=MESH_GRADE_VERTEX;
            info.ref=&ref;
            info.field=ref.field;
            info.integrand=normsq_integrand;
            info.cloneref=gradsq_cloneref;
            functional_mapnumericalfieldgradient(v, &info, &out);
        } else morpho_runtimeerror(v, GRADSQ_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(NormSq)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, GradSq_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, NormSq_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, NormSq_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, NormSq_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, NormSq_fieldgradient, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Integrals
 * ********************************************************************** */

/** Integral references
 @brief Used to pass through the functional element mapping system.
 A thread local copy is made with cloned fields */

typedef struct {
    value integrand;
    int nfields;
    value *fields;
    value *originalfields; // Original fields
    value method; // Method dictionary
    objectmesh *mref; // Reference mesh
    vm *v;
    bool weightbyref; // Use reference mesh for the element
} integralref;

/* ----------------------------------------------
 * Integrand functions
 * ---------------------------------------------- */

/** Integral element references
 @brief used to store information about the current element in thread-local storage. We wrap them in an object so that they can be safely stored in a value.
 Guaranteed to be thread local */

typedef struct {
    object obj;
    objectmesh *mesh;    // The current mesh object
    grade g;             // Current grade
    elementid id;        // Current element
    int nv;              // Number of vertices
    int *vid;            // Vertex ids
    double **vertexposn; // List of vertex positions
    double elementsize;  // Size of the element
    value *qinterpolated; // List of interpolated quantities (this allows us to identify operators on fields)
    value *qgrad;        // Gradients
    integralref *iref;
} objectintegralelementref;

size_t objectintegralelementref_sizefn(object *obj) {
    return sizeof(objectintegralelementref);
}

void objectintegralelementref_printfn(object *obj, void *v) {
    morpho_printf(v, "<Elementref>");
}

objecttypedefn objectintegralelementrefdefn = {
    .printfn=objectintegralelementref_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectintegralelementref_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

objecttype objectintegralelementreftype;
#define OBJECT_INTEGRALELEMENTREF objectintegralelementreftype

/** Tests whether an object is an element ref */
#define MORPHO_ISINTEGRALELEMENTREF(val) object_istype(val, OBJECT_INTEGRALELEMENTREF)

/** Gets the object as an element ref */
#define MORPHO_GETINTEGRALELEMENTREF(val) ((objectintegralelementref *) MORPHO_GETOBJECT(val))

/** Static element ref */
#define MORPHO_STATICINTEGRALELEMENTREF(mesh, grade, id, nv, vid)      { .obj.type=OBJECT_INTEGRALELEMENTREF, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .g=grade, .mesh=mesh, .id=id, .nv=nv, .vid=vid, .qinterpolated=NULL }

int elementhandle;

/** Get the current element ref from thread-local storage in the VM */
objectintegralelementref *integral_getelementref(vm *v) {
    value elref=MORPHO_NIL;
    vm_gettlvar(v, elementhandle, &elref);
    if (MORPHO_ISINTEGRALELEMENTREF(elref)) return MORPHO_GETINTEGRALELEMENTREF(elref);
    
    return NULL;
}

/* --------
 * Tangent
 * -------- */

int tangenthandle; // TL storage handle for tangent vectors

/** Evaluate the tangent vector */
void integral_evaluatetangent(vm *v, value *out) {
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) { morpho_runtimeerror(v, INTEGRAL_SPCLFN, TANGENT_FUNCTION); return; }
    
    int dim = elref->mesh->dim;
    
    objectmatrix *mtangent = object_newmatrix(dim, 1, false);
    if (!mtangent) {
        morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        return;
    }
    
    functional_vecsub(dim, elref->vertexposn[1], elref->vertexposn[0], mtangent->elements);

    double tnorm=functional_vecnorm(dim, mtangent->elements);
    if (fabs(tnorm)>MORPHO_EPS) functional_vecscale(dim, 1.0/tnorm, mtangent->elements, mtangent->elements);
    
    vm_settlvar(v, tangenthandle, MORPHO_OBJECT(mtangent));
    *out = MORPHO_OBJECT(mtangent);
}

static value integral_tangent(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    
    vm_gettlvar(v, tangenthandle, &out);
    if (MORPHO_ISNIL(out)) integral_evaluatetangent(v, &out);
    
    return out;
}

/* --------
 * Normal
 * -------- */

int normlhandle; // TL storage handle for normal vectors

/** Evaluates the normal vector */
void integral_evaluatenormal(vm *v, value *out) {
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) { morpho_runtimeerror(v, INTEGRAL_SPCLFN, NORMAL_FUNCTION); return; }
    
    int dim = elref->mesh->dim;
    double s0[dim], s1[dim];
    objectmatrix *mnormal = object_newmatrix(dim, 1, false);
    if (!mnormal) {
        morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        return;
    }
    
    functional_vecsub(dim, elref->vertexposn[1], elref->vertexposn[0], s0);
    functional_vecsub(dim, elref->vertexposn[2], elref->vertexposn[1], s1);
    functional_veccross(s0, s1, mnormal->elements);
    
    double nnorm=functional_vecnorm(dim, mnormal->elements);
    if (fabs(nnorm)>MORPHO_EPS) functional_vecscale(dim, 1.0/nnorm, mnormal->elements, mnormal->elements);
    
    vm_settlvar(v, normlhandle, MORPHO_OBJECT(mnormal));
    *out = MORPHO_OBJECT(mnormal);
}

static value integral_normal(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;

    vm_gettlvar(v, normlhandle, &out);
    if (MORPHO_ISNIL(out)) integral_evaluatenormal(v, &out);
    
    return out;
}

/* --------
 * Gradient
 * -------- */

void integral_evaluategradient(vm *v, value q, value *out) {
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) { morpho_runtimeerror(v, INTEGRAL_SPCLFN, GRAD_FUNCTION); return; }
    
    /* Identify the field being referred to */
    int ifld, xfld=-1;
    for (ifld=0; ifld<elref->iref->nfields; ifld++) {
        if (MORPHO_ISFIELD(q) && MORPHO_ISSAME(elref->iref->originalfields[ifld], q)) break;
        else if (MORPHO_ISSAME(elref->qinterpolated[ifld], q)) {
            if (xfld>=0) { morpho_runtimeerror(v, INTEGRAL_AMBGSFLD); return; }
            // @warning: This will fail if two fields happen to have the same value(!)
            xfld=ifld;
        }
    }
    if (xfld>=0) ifld = xfld;
    
    // Raise an error if we couldn't find it
    if (ifld>=elref->iref->nfields) {
        morpho_runtimeerror(v, INTEGRAL_FLD); return;
    }
    
    *out = elref->qgrad[ifld];
    if (!MORPHO_ISNIL(*out)) return;
    
    objectfield *fld = MORPHO_GETFIELD(elref->iref->fields[ifld]);
    
    // Evaluate the gradient
    int dim = elref->mesh->dim;
    int ndof = fld->psize; // Number of degrees of freedom per element
    double grad[ndof*dim]; // Storage for gradient
    value gradx[dim]; // Components of the gradient
    for (int i=0; i<dim; i++) gradx[i]=MORPHO_NIL;
    
    bool gsucc = false;
    
    // Evaluate correct gradient
    if (elref->g==2) gsucc=gradsq_evaluategradient(elref->mesh, fld, elref->nv, elref->vid, grad);
    else if (elref->g==3) gsucc=gradsq_evaluategradient3d(elref->mesh, fld, elref->nv, elref->vid, grad);
    
    if (!gsucc) {
        UNREACHABLE("Couldn't evaluate gradient");
        return;
    }

    // Copy into a list or matrix as appropriate
    if (ndof==1) {
        objectmatrix *mgrad=object_newmatrix(dim, 1, false);
        if (!mgrad) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return; }
        memcpy(mgrad->elements, grad, sizeof(grad));
        *out = MORPHO_OBJECT(mgrad);
        // Don't bind these; this will be freed by the integrand caller
    } else {
        if (!MORPHO_ISMATRIX(fld->prototype)) UNREACHABLE("Field type not supported in grad");
        objectmatrix *proto = MORPHO_GETMATRIX(fld->prototype);
        for (int i=0; i<dim; i++) {
            objectmatrix *mgrad=object_newmatrix(proto->nrows, proto->ncols, false); // Should copy prototype dimensions!
            if (!mgrad) goto integral_evaluategradient_cleanup;
            for (int k=0; k<ndof; k++) mgrad->elements[k]=grad[k*dim+i];
            gradx[i]=MORPHO_OBJECT(mgrad);
        }
        objectlist *glst = object_newlist(dim, gradx);
        if (!glst) goto integral_evaluategradient_cleanup;
        *out = MORPHO_OBJECT(glst);
        // Don't bind these; they will be freed by the integrand caller
    }
    
    // Store for further use
    elref->qgrad[ifld]=*out;
    
    return;

integral_evaluategradient_cleanup:
    
    for (int i=0; i<dim; i++) if (MORPHO_ISOBJECT(gradx[i]))
    morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return;
}

static value integral_gradfn(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    if (nargs==1) {
        integral_evaluategradient(v, MORPHO_GETARG(args, 0), &out);
    } else morpho_runtimeerror(v, INTEGRAL_FLD);
    
    return out;
}

/* -------------------
 * Cauchy green strain
 * ------------------- */

int cauchygreenhandle; // TL storage handle for CG tensor

/** Evaluates the cg strain tensor */
void integral_evaluatecg(vm *v, value *out) {
    objectintegralelementref *elref = integral_getelementref(v);
    
    if (!elref || !elref->iref->mref) {
        morpho_runtimeerror(v, INTEGRAL_SPCLFN, CGTENSOR_FUNCTION); return;
    }
    
    int gdim=elref->nv-1; // Dimension of Gram matrix
    
    objectmatrix *cg=object_newmatrix(gdim, gdim, true);
    if (!cg) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return; }
    
    double gramrefel[gdim*gdim], gramdefel[gdim*gdim], qel[gdim*gdim], rel[gdim*gdim];
    objectmatrix gramref = MORPHO_STATICMATRIX(gramrefel, gdim, gdim); // Gram matrices
    objectmatrix gramdef = MORPHO_STATICMATRIX(gramdefel, gdim, gdim); //
    objectmatrix q = MORPHO_STATICMATRIX(qel, gdim, gdim); // Inverse of Gram in source domain
    objectmatrix r = MORPHO_STATICMATRIX(rel, gdim, gdim); // Intermediate calculations
    
    linearelasticity_calculategram(elref->iref->mref->vert, elref->mesh->dim, elref->nv, elref->vid, &gramref);
    linearelasticity_calculategram(elref->mesh->vert, elref->mesh->dim, elref->nv, elref->vid, &gramdef);
    
    if (matrix_inverse(&gramref, &q)!=MATRIX_OK) return;
    if (matrix_mul(&gramdef, &q, &r)!=MATRIX_OK) return;

    matrix_identity(cg);
    matrix_scale(cg, -0.5);
    matrix_accumulate(cg, 0.5, &r);
    
    vm_settlvar(v, cauchygreenhandle, MORPHO_OBJECT(cg));
    *out = MORPHO_OBJECT(cg);
}

static value integral_cgfn(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;

    vm_gettlvar(v, cauchygreenhandle, &out);
    if (MORPHO_ISNIL(out)) integral_evaluatecg(v, &out);
    
    return out;
}

/* ----------------------
 * General initialization
 * ---------------------- */

/** Clears threadlocal storage */
void integral_cleartlvars(vm *v) {
    int handles[] = { elementhandle, normlhandle, tangenthandle, cauchygreenhandle, -1 };
    
    for (int i=0; handles[i]>=0; i++) {
        vm_settlvar(v, handles[i], MORPHO_NIL);
    }
}

void integral_freetlvars(vm *v) {
    int handles[] = { normlhandle, tangenthandle, cauchygreenhandle, -1 };
    
    for (int i=0; handles[i]>=0; i++) {
        value val;
        vm_gettlvar(v, handles[i], &val);
        if (MORPHO_ISOBJECT(val)) morpho_freeobject(val);
    }
    
    integral_cleartlvars(v);
}

/* ----------------------------------------------
 * LineIntegral
 * ---------------------------------------------- */

value functional_methodproperty;

/** Prepares an integral reference */
bool integral_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, integralref *ref) {
    bool success=false;
    value func=MORPHO_NIL;
    value mref=MORPHO_NIL;
    value wtbyref=MORPHO_NIL;
    value field=MORPHO_NIL;
    value method=MORPHO_NIL;
    ref->v=NULL;
    ref->nfields=0;
    ref->method=MORPHO_NIL;
    ref->mref=NULL;
    ref->weightbyref=false;

    if (objectinstance_getpropertyinterned(self, scalarpotential_functionproperty, &func) &&
        MORPHO_ISCALLABLE(func)) {
        ref->integrand=func;
        success=true;
    }
    if (objectinstance_getpropertyinterned(self, linearelasticity_referenceproperty, &mref) &&
        MORPHO_ISMESH(mref)) {
        ref->mref=MORPHO_GETMESH(mref);
    }
    if (objectinstance_getpropertyinterned(self, linearelasticity_weightbyreferenceproperty, &wtbyref)) {
        ref->weightbyref=!morpho_isfalse(wtbyref);
    }
    if (objectinstance_getpropertyinterned(self, functional_methodproperty, &method)) {
        ref->method=method;
    }
    if (objectinstance_getpropertyinterned(self, functional_fieldproperty, &field) &&
        MORPHO_ISLIST(field)) {
        objectlist *list = MORPHO_GETLIST(field);
        ref->nfields=list->val.count;
        ref->fields=list->val.data;
        ref->originalfields=list->val.data;
        
        for (int i=0; i<ref->nfields; i++) {
            if (MORPHO_ISFIELD(ref->fields[i])) {
                objectfield *fld = MORPHO_GETFIELD(ref->fields[i]);
                field_addpool(fld);
            }
        }
    }
    return success;
}

/** Clones the integral reference with a given substitute field */
void *integral_cloneref(void *ref, objectfield *field, objectfield *sub) {
    integralref *nref = (integralref *) ref;
    integralref *clone = MORPHO_MALLOC(sizeof(integralref));
    
    if (clone) {
        *clone = *nref;
        clone->originalfields=nref->originalfields;
        clone->fields=MORPHO_MALLOC(sizeof(value)*clone->nfields);
        if (!clone->fields) { MORPHO_FREE(clone); return NULL; }
        
        for (int i=0; i<clone->nfields; i++) {
            clone->fields[i]=nref->fields[i];
            if (MORPHO_ISFIELD(nref->fields[i]) &&
                MORPHO_GETFIELD(nref->fields[i])==field) {
                clone->fields[i]=MORPHO_OBJECT(sub);
            }
        }
    }
    
    return clone;
}

/** Frees a reference */
void integral_freeref(void *ref) {
    integralref *nref = (integralref *) ref;
    MORPHO_FREE(nref->fields);
    MORPHO_FREE(ref);
}

bool integral_integrandfn(unsigned int dim, double *t, double *x, unsigned int nquantity, value *quantity, void *ref, double *fout) {
    integralref *iref = ref;
    objectmatrix posn = MORPHO_STATICMATRIX(x, dim, 1);
    value args[nquantity+1], out;

    args[0]=MORPHO_OBJECT(&posn);
    for (unsigned int i=0; i<nquantity; i++) args[i+1]=quantity[i];
    
    objectintegralelementref *elref = integral_getelementref(iref->v);
    if (elref) elref->qinterpolated=quantity;
    
    if (morpho_call(iref->v, iref->integrand, nquantity+1, args, &out)) {
        morpho_valuetofloat(out,fout);
        return true;
    }

    return false;
}

/** Integrate a function over a line */
bool lineintegral_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    integralref iref = *(integralref *) ref;
    double *x[nv];
    bool success;
    value qgrad[iref.nfields+1];
    for (int i=0; i<iref.nfields; i++) qgrad[i] = MORPHO_NIL;
    
    objectintegralelementref elref = MORPHO_STATICINTEGRALELEMENTREF(mesh, MESH_GRADE_LINE, id, nv, vid);
    elref.iref = &iref;
    elref.vertexposn = x;
    elref.qgrad=qgrad;
    
    if (!functional_elementsize(v, mesh, MESH_GRADE_LINE, id, nv, vid, &elref.elementsize)) return false;

    iref.v=v;
    for (unsigned int i=0; i<nv; i++) {
        mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i]);
    }

    /* Set up quantities */
    integral_cleartlvars(v);
    vm_settlvar(v, elementhandle, MORPHO_OBJECT(&elref));

    value q0[iref.nfields+1], q1[iref.nfields+1];
    value *q[2] = { q0, q1 };
    for (unsigned int k=0; k<iref.nfields; k++) {
        for (unsigned int i=0; i<nv; i++) {
            field_getelement(MORPHO_GETFIELD(iref.fields[k]), MESH_GRADE_VERTEX, vid[i], 0, &q[i][k]);
        }
    }

    if (MORPHO_ISDICTIONARY(iref.method)) {
        double err;
        success=integrate(integral_integrandfn, MORPHO_GETDICTIONARY(iref.method), mesh->dim, MESH_GRADE_LINE, x, iref.nfields, q, &iref, out, &err);
    } else {
        success=integrate_integrate(integral_integrandfn, mesh->dim, MESH_GRADE_LINE, x, iref.nfields, q, &iref, out);
    }
    
    if (success) *out *=elref.elementsize;

    integral_freetlvars(v);
    
    return success;
}

FUNCTIONAL_METHOD(LineIntegral, integrand, MESH_GRADE_LINE, integralref, integral_prepareref, functional_mapintegrand, lineintegral_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(LineIntegral, total, MESH_GRADE_LINE, integralref, integral_prepareref, functional_sumintegrand, lineintegral_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(LineIntegral, gradient, MESH_GRADE_LINE, integralref, integral_prepareref, functional_mapnumericalgradient, lineintegral_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(LineIntegral, hessian, MESH_GRADE_LINE, integralref, integral_prepareref, functional_mapnumericalhessian, lineintegral_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE)

/** Initialize a LineIntegral object */
value LineIntegral_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    int nparams = -1;
    int nfixed;
    value method=MORPHO_NIL;
    value mref=MORPHO_NIL;
    value wtbyref=MORPHO_NIL;

    if (builtin_options(v, nargs, args, &nfixed, 3,
                        functional_methodproperty, &method,
                        linearelasticity_referenceproperty, &mref,
                        linearelasticity_weightbyreferenceproperty, &wtbyref)) {
        if (MORPHO_ISDICTIONARY(method)) objectinstance_setproperty(self, functional_methodproperty, method);
        if (MORPHO_ISMESH(mref)) objectinstance_setproperty(self, linearelasticity_referenceproperty, mref);
        if (MORPHO_ISBOOL(wtbyref)) objectinstance_setproperty(self, linearelasticity_weightbyreferenceproperty, wtbyref);
    } else {
        morpho_runtimeerror(v, LINEINTEGRAL_ARGS);
        return MORPHO_NIL;
    }
    
    if (nfixed>0) {
        value f = MORPHO_GETARG(args, 0);

        if (morpho_countparameters(f, &nparams)) {
            objectinstance_setproperty(self, scalarpotential_functionproperty, MORPHO_GETARG(args, 0));
        } else {
            morpho_runtimeerror(v, LINEINTEGRAL_ARGS);
            return MORPHO_NIL;
        }
    }

    if (nparams!=nfixed) {
        morpho_runtimeerror(v, LINEINTEGRAL_NFLDS);
        return MORPHO_NIL;
    }

    if (nfixed>1) {
        /* Remaining arguments should be fields */
        objectlist *list = object_newlist(nfixed-1, & MORPHO_GETARG(args, 1));
        if (!list) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return MORPHO_NIL; }

        for (unsigned int i=1; i<nfixed; i++) {
            if (!MORPHO_ISFIELD(MORPHO_GETARG(args, i))) {
                morpho_runtimeerror(v, LINEINTEGRAL_ARGS);
                object_free((object *) list);
                return MORPHO_NIL;
            }
        }

        value field = MORPHO_OBJECT(list);
        objectinstance_setproperty(self, functional_fieldproperty, field);
        morpho_bindobjects(v, 1, &field);
    }

    return MORPHO_NIL;
}

/** Field gradients for Line Integrals */
value LineIntegral_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    integralref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        // Should check whether the field is known about here...
        if (integral_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, MESH_GRADE_LINE, info.sel, &ref)) {
            info.g=MESH_GRADE_LINE;
            info.integrand=lineintegral_integrand;
            info.cloneref=integral_cloneref;
            info.freeref=integral_freeref;
            info.ref=&ref;
            functional_mapnumericalfieldgradient(v, &info, &out);
        } else morpho_runtimeerror(v, GRADSQ_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(LineIntegral)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, LineIntegral_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, LineIntegral_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, LineIntegral_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, LineIntegral_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, LineIntegral_fieldgradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, LineIntegral_hessian, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * AreaIntegral
 * ---------------------------------------------- */

/** Integrate a function over an area */
bool areaintegral_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    integralref iref = *(integralref *) ref;
    double *x[nv];
    bool success;
    
    value qgrad[iref.nfields+1];
    for (int i=0; i<iref.nfields; i++) qgrad[i] = MORPHO_NIL;
    
    objectintegralelementref elref = MORPHO_STATICINTEGRALELEMENTREF(mesh, MESH_GRADE_AREA, id, nv, vid);
    elref.iref = &iref;
    elref.vertexposn = x;
    elref.qgrad=qgrad;
    
    if (iref.weightbyref) {
        if (!functional_elementsize(v, iref.mref, MESH_GRADE_AREA, id, nv, vid, &elref.elementsize)) return false;
    } else {
        if (!functional_elementsize(v, mesh, MESH_GRADE_AREA, id, nv, vid, &elref.elementsize)) return false;
    }

    iref.v=v;
    for (unsigned int i=0; i<nv; i++) {
        mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i]);
    }

    /* Set up quantities */
    integral_cleartlvars(v);
    vm_settlvar(v, elementhandle, MORPHO_OBJECT(&elref));

    value q0[iref.nfields+1], q1[iref.nfields+1], q2[iref.nfields+1];
    value *q[3] = { q0, q1, q2 };
    for (unsigned int k=0; k<iref.nfields; k++) {
        for (unsigned int i=0; i<nv; i++) {
            field_getelement(MORPHO_GETFIELD(iref.fields[k]), MESH_GRADE_VERTEX, vid[i], 0, &q[i][k]);
        }
    }

    if (MORPHO_ISDICTIONARY(iref.method)) {
        double err;
        success=integrate(integral_integrandfn, MORPHO_GETDICTIONARY(iref.method), mesh->dim, MESH_GRADE_AREA, x, iref.nfields, q, &iref, out, &err);
    } else {
        success=integrate_integrate(integral_integrandfn, mesh->dim, MESH_GRADE_AREA, x, iref.nfields, q, &iref, out);
    }
    
    if (success) *out *= elref.elementsize;

    integral_freetlvars(v);
    for (int i=0; i<iref.nfields; i++) {
        if (MORPHO_ISLIST(qgrad[i])) {
            objectlist *l = MORPHO_GETLIST(qgrad[i]);
            for (int j=0; j<l->val.count; j++) morpho_freeobject(l->val.data[j]);
        }
        morpho_freeobject(qgrad[i]);
    }
    
    return success;
}

FUNCTIONAL_METHOD(AreaIntegral, integrand, MESH_GRADE_AREA, integralref, integral_prepareref, functional_mapintegrand, areaintegral_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(AreaIntegral, total, MESH_GRADE_AREA, integralref, integral_prepareref, functional_sumintegrand, areaintegral_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(AreaIntegral, gradient, MESH_GRADE_AREA, integralref, integral_prepareref, functional_mapnumericalgradient, areaintegral_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

/** Field gradients for Area Integrals */
value AreaIntegral_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    integralref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        // Should check whether the field is known about here...
        if (integral_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, MESH_GRADE_AREA, info.sel, &ref)) {
            info.g=MESH_GRADE_AREA;
            info.integrand=areaintegral_integrand;
            info.cloneref=integral_cloneref;
            info.freeref=integral_freeref;
            info.ref=&ref;
            functional_mapnumericalfieldgradient(v, &info, &out);
        } else morpho_runtimeerror(v, GRADSQ_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(AreaIntegral)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, LineIntegral_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, AreaIntegral_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, AreaIntegral_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, AreaIntegral_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, AreaIntegral_fieldgradient, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * VolumeIntegral
 * ---------------------------------------------- */

/** Integrate a function over a volume */
bool volumeintegral_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    integralref iref = *(integralref *) ref;
    double *x[nv];
    bool success;
    
    value qgrad[iref.nfields+1];
    for (int i=0; i<iref.nfields; i++) qgrad[i] = MORPHO_NIL;
    
    objectintegralelementref elref = MORPHO_STATICINTEGRALELEMENTREF(mesh, MESH_GRADE_VOLUME, id, nv, vid);
    elref.iref = &iref;
    elref.vertexposn = x;
    elref.qgrad=qgrad;

    if (!functional_elementsize(v, mesh, MESH_GRADE_VOLUME, id, nv, vid, &elref.elementsize)) return false;

    iref.v=v;
    for (unsigned int i=0; i<nv; i++) {
        mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i]);
    }
    
    /* Set up quantities */
    integral_cleartlvars(v);
    vm_settlvar(v, elementhandle, MORPHO_OBJECT(&elref));

    value q0[iref.nfields+1], q1[iref.nfields+1], q2[iref.nfields+1], q3[iref.nfields+1];
    value *q[4] = { q0, q1, q2, q3 };
    for (unsigned int k=0; k<iref.nfields; k++) {
        for (unsigned int i=0; i<nv; i++) {
            field_getelement(MORPHO_GETFIELD(iref.fields[k]), MESH_GRADE_VERTEX, vid[i], 0, &q[i][k]);
        }
    }

    if (MORPHO_ISDICTIONARY(iref.method)) {
        double err;
        success=integrate(integral_integrandfn, MORPHO_GETDICTIONARY(iref.method), mesh->dim, MESH_GRADE_VOLUME, x, iref.nfields, q, &iref, out, &err);
    } else {
        success=integrate_integrate(integral_integrandfn, mesh->dim, MESH_GRADE_VOLUME, x, iref.nfields, q, &iref, out);
    }
    
    if (success) *out *=elref.elementsize;

    integral_freetlvars(v);
    for (int i=0; i<iref.nfields; i++) {
        if (MORPHO_ISLIST(qgrad[i])) {
            objectlist *l = MORPHO_GETLIST(qgrad[i]);
            for (int j=0; j<l->val.count; j++) morpho_freeobject(l->val.data[j]);
        }
        morpho_freeobject(qgrad[i]);
    }
    
    return success;
}

FUNCTIONAL_METHOD(VolumeIntegral, integrand, MESH_GRADE_VOLUME, integralref, integral_prepareref, functional_mapintegrand, volumeintegral_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(VolumeIntegral, total, MESH_GRADE_VOLUME, integralref, integral_prepareref, functional_sumintegrand, volumeintegral_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD(VolumeIntegral, gradient, MESH_GRADE_VOLUME, integralref, integral_prepareref, functional_mapnumericalgradient, volumeintegral_integrand, NULL, GRADSQ_ARGS, SYMMETRY_NONE);

/** Field gradients for Volume Integrals */
value VolumeIntegral_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    integralref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        // Should check whether the field is known about here...
        if (integral_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, MESH_GRADE_VOLUME, info.sel, &ref)) {
            info.g=MESH_GRADE_VOLUME;
            info.integrand=volumeintegral_integrand;
            info.cloneref=integral_cloneref;
            info.freeref=integral_freeref;
            info.ref=&ref;
            functional_mapnumericalfieldgradient(v, &info, &out);
        } else morpho_runtimeerror(v, GRADSQ_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(VolumeIntegral)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, LineIntegral_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, VolumeIntegral_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, VolumeIntegral_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, VolumeIntegral_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, VolumeIntegral_fieldgradient, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS


/* **********************************************************************
 * Initialization
 * ********************************************************************** */

/*double ff(double x) {
    return exp(x);
}

double dff(double x) {
    return exp(x);
}

void functional_fdtest(void) {
    double h1 = 1e-8;
    
    //double xi[] = { -100, -10, -1.0, 0.0, 1e-7, 1e-5, 1e-2, 0.1, 1, 10, 100, 1e100 };
    
    for (int i=-6; i<3; i++) {
        double x = pow(10.0, (double) i);
        double fex = dff(x);
    
        double f1=(ff(x+h1)-ff(x-h1))/(2*h1);
        
        double h2=functional_fdstepsize(x, 1);
        double f2=(ff(x+h2)-ff(x-h2))/(2*h2);
        
        printf("%g: %g %g %g\n", x, fex, fabs((f1-fex)/fex), fabs((f2-fex)/fex));
    }
    
}*/

void functional_initialize(void) {
    fddelta1 = pow(MORPHO_EPS, 1.0/3.0);
    fddelta2 = pow(MORPHO_EPS, 1.0/4.0);
    
    functional_gradeproperty=builtin_internsymbolascstring(FUNCTIONAL_GRADE_PROPERTY);
    functional_fieldproperty=builtin_internsymbolascstring(FUNCTIONAL_FIELD_PROPERTY);
    scalarpotential_functionproperty=builtin_internsymbolascstring(SCALARPOTENTIAL_FUNCTION_PROPERTY);
    scalarpotential_gradfunctionproperty=builtin_internsymbolascstring(SCALARPOTENTIAL_GRADFUNCTION_PROPERTY);
    linearelasticity_referenceproperty=builtin_internsymbolascstring(LINEARELASTICITY_REFERENCE_PROPERTY);
    linearelasticity_weightbyreferenceproperty=builtin_internsymbolascstring(LINEARELASTICITY_WTBYREF_PROPERTY);
    linearelasticity_poissonproperty=builtin_internsymbolascstring(LINEARELASTICITY_POISSON_PROPERTY);
    hydrogel_aproperty=builtin_internsymbolascstring(HYDROGEL_A_PROPERTY);
    hydrogel_bproperty=builtin_internsymbolascstring(HYDROGEL_B_PROPERTY);
    hydrogel_cproperty=builtin_internsymbolascstring(HYDROGEL_C_PROPERTY);
    hydrogel_dproperty=builtin_internsymbolascstring(HYDROGEL_D_PROPERTY);
    hydrogel_phirefproperty=builtin_internsymbolascstring(HYDROGEL_PHIREF_PROPERTY);
    hydrogel_phi0property=builtin_internsymbolascstring(HYDROGEL_PHI0_PROPERTY);
    equielement_weightproperty=builtin_internsymbolascstring(EQUIELEMENT_WEIGHT_PROPERTY);
    nematic_ksplayproperty=builtin_internsymbolascstring(NEMATIC_KSPLAY_PROPERTY);
    nematic_ktwistproperty=builtin_internsymbolascstring(NEMATIC_KTWIST_PROPERTY);
    nematic_kbendproperty=builtin_internsymbolascstring(NEMATIC_KBEND_PROPERTY);
    nematic_pitchproperty=builtin_internsymbolascstring(NEMATIC_PITCH_PROPERTY);
    
    functional_methodproperty=builtin_internsymbolascstring(INTEGRAL_METHOD_PROPERTY);

    curvature_integrandonlyproperty=builtin_internsymbolascstring(CURVATURE_INTEGRANDONLY_PROPERTY);
    curvature_geodesicproperty=builtin_internsymbolascstring(CURVATURE_GEODESIC_PROPERTY);

    objectstring objclassname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objclassname));

    builtin_addclass(LENGTH_CLASSNAME, MORPHO_GETCLASSDEFINITION(Length), objclass);
    builtin_addclass(AREA_CLASSNAME, MORPHO_GETCLASSDEFINITION(Area), objclass);
    builtin_addclass(AREAENCLOSED_CLASSNAME, MORPHO_GETCLASSDEFINITION(AreaEnclosed), objclass);
    builtin_addclass(VOLUMEENCLOSED_CLASSNAME, MORPHO_GETCLASSDEFINITION(VolumeEnclosed), objclass);
    builtin_addclass(VOLUME_CLASSNAME, MORPHO_GETCLASSDEFINITION(Volume), objclass);
    builtin_addclass(SCALARPOTENTIAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(ScalarPotential), objclass);
    builtin_addclass(LINEARELASTICITY_CLASSNAME, MORPHO_GETCLASSDEFINITION(LinearElasticity), objclass);
    builtin_addclass(HYDROGEL_CLASSNAME, MORPHO_GETCLASSDEFINITION(Hydrogel), objclass);
    builtin_addclass(EQUIELEMENT_CLASSNAME, MORPHO_GETCLASSDEFINITION(EquiElement), objclass);
    builtin_addclass(LINECURVATURESQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(LineCurvatureSq), objclass);
    builtin_addclass(LINETORSIONSQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(LineTorsionSq), objclass);
    builtin_addclass(MEANCURVATURESQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(MeanCurvatureSq), objclass);
    builtin_addclass(GAUSSCURVATURE_CLASSNAME, MORPHO_GETCLASSDEFINITION(GaussCurvature), objclass);
    builtin_addclass(GRADSQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(GradSq), objclass);
    builtin_addclass(NORMSQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(NormSq), objclass);
    builtin_addclass(LINEINTEGRAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(LineIntegral), objclass);
    builtin_addclass(AREAINTEGRAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(AreaIntegral), objclass);
    builtin_addclass(VOLUMEINTEGRAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(VolumeIntegral), objclass);
    builtin_addclass(NEMATIC_CLASSNAME, MORPHO_GETCLASSDEFINITION(Nematic), objclass);
    builtin_addclass(NEMATICELECTRIC_CLASSNAME, MORPHO_GETCLASSDEFINITION(NematicElectric), objclass);

    builtin_addfunction(TANGENT_FUNCTION, integral_tangent, BUILTIN_FLAGSEMPTY);
    builtin_addfunction(NORMAL_FUNCTION, integral_normal, BUILTIN_FLAGSEMPTY);
    builtin_addfunction(GRAD_FUNCTION, integral_gradfn, BUILTIN_FLAGSEMPTY);
    builtin_addfunction(CGTENSOR_FUNCTION, integral_cgfn, BUILTIN_FLAGSEMPTY);

    morpho_defineerror(VOLUMEENCLOSED_ZERO, ERROR_HALT, VOLUMEENCLOSED_ZERO_MSG);
    morpho_defineerror(FUNC_INTEGRAND_MESH, ERROR_HALT, FUNC_INTEGRAND_MESH_MSG);
    morpho_defineerror(FUNC_ELNTFND, ERROR_HALT, FUNC_ELNTFND_MSG);

    morpho_defineerror(SCALARPOTENTIAL_FNCLLBL, ERROR_HALT, SCALARPOTENTIAL_FNCLLBL_MSG);

    morpho_defineerror(LINEARELASTICITY_REF, ERROR_HALT, LINEARELASTICITY_REF_MSG);
    morpho_defineerror(LINEARELASTICITY_PRP, ERROR_HALT, LINEARELASTICITY_PRP_MSG);

    morpho_defineerror(HYDROGEL_ARGS, ERROR_HALT, HYDROGEL_ARGS_MSG);
    morpho_defineerror(HYDROGEL_PRP, ERROR_HALT, HYDROGEL_PRP_MSG);
    morpho_defineerror(HYDROGEL_FLDGRD, ERROR_HALT, HYDROGEL_FLDGRD_MSG);
    morpho_defineerror(HYDROGEL_ZEEROREFELEMENT, ERROR_WARNING, HYDROGEL_ZEEROREFELEMENT_MSG);
    morpho_defineerror(HYDROGEL_BNDS, ERROR_WARNING, HYDROGEL_BNDS_MSG);

    morpho_defineerror(EQUIELEMENT_ARGS, ERROR_HALT, EQUIELEMENT_ARGS_MSG);
    morpho_defineerror(GRADSQ_ARGS, ERROR_HALT, GRADSQ_ARGS_MSG);
    morpho_defineerror(NEMATIC_ARGS, ERROR_HALT, NEMATIC_ARGS_MSG);
    morpho_defineerror(NEMATICELECTRIC_ARGS, ERROR_HALT, NEMATICELECTRIC_ARGS_MSG);

    morpho_defineerror(FUNCTIONAL_ARGS, ERROR_HALT, FUNCTIONAL_ARGS_MSG);

    morpho_defineerror(LINEINTEGRAL_ARGS, ERROR_HALT, LINEINTEGRAL_ARGS_MSG);
    morpho_defineerror(LINEINTEGRAL_NFLDS, ERROR_HALT, LINEINTEGRAL_NFLDS_MSG);
    morpho_defineerror(INTEGRAL_FLD, ERROR_HALT, INTEGRAL_FLD_MSG);
    morpho_defineerror(INTEGRAL_AMBGSFLD, ERROR_HALT, INTEGRAL_AMBGSFLD_MSG);
    morpho_defineerror(INTEGRAL_SPCLFN, ERROR_HALT, INTEGRAL_SPCLFN_MSG);
    
    functional_poolinitialized = false;
    
    objectintegralelementreftype=object_addtype(&objectintegralelementrefdefn);
    elementhandle=vm_addtlvar();
    tangenthandle=vm_addtlvar();
    normlhandle=vm_addtlvar();
    cauchygreenhandle=vm_addtlvar();
    
    morpho_addfinalizefn(functional_finalize);
}

void functional_finalize(void) {
    if (functional_poolinitialized) threadpool_clear(&functional_pool);
}

#endif
