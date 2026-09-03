/** @file equielement.c
 *  @author T J Atherton
 *
 *  @brief EquiElement functional
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
#include "equielement.h"

/* ----------------------------------------------
 * Equielement
 * ---------------------------------------------- */

static value equielement_weightproperty;

typedef struct {
    grade grade;
    objectsparse *vtoel; // Connect vertices to elements
    objectsparse *eltov; // Connect elements to vertices
    objectmatrix *weight; // Weight matrix (one row), possibly a view of a Field
    double mean;
    objectmatrix wview; // Field grade slice viewed as Matrix(1, nel)
} equielementref;

/** Bind optional weight: nil, Matrix row, or scalar Field on the functional grade. */
static bool equielement_bindweight(objectinstance *self, objectmesh *mesh, equielementref *ref) {
    value weight=MORPHO_NIL;
    if (!objectinstance_getpropertyinterned(self, equielement_weightproperty, &weight) ||
        MORPHO_ISNIL(weight)) return true;

    if (MORPHO_ISMATRIX(weight)) {
        ref->weight=MORPHO_GETMATRIX(weight);
    } else if (MORPHO_ISFIELD(weight)) { // Construct a view onto the field's data
        objectfield *f=MORPHO_GETFIELD(weight);
        if (f->mesh!=mesh) return false;

        double *p;
        unsigned int nent;
        if (!field_getelementaslist(f, ref->grade, 0, 0, &nent, &p) || nent!=1) return false; // Find first element
        if (field_dofforgrade(f, ref->grade)!=1) return false; // Check 1 dof

        elementid n=mesh_nelementsforgrade(mesh, ref->grade);
        if (n==0) return false;

        ref->wview=(objectmatrix) MORPHO_STATICMATRIX(p, 1, (MatrixIdx_t) n);
        ref->weight=&ref->wview;
    } else {
        return false;
    }

    if (ref->weight) {
        double sum[ref->weight->nvals];
        matrix_sum(ref->weight, sum);
        ref->mean=sum[0];
        ref->mean/=ref->weight->ncols;
    }

    return true;
}

/** Prepares the reference structure from the Equielement object's properties */
bool equielement_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, equielementref *ref) {
    bool success=false;
    value grade=MORPHO_NIL;

    ref->weight=NULL;
    ref->mean=0.0;

    if (objectinstance_getpropertyinterned(self, functional_gradeproperty, &grade) &&
        MORPHO_ISINTEGER(grade) ) {
        ref->grade=MORPHO_GETINTEGERVALUE(grade);

        int maxgrade=mesh_maxgrade(mesh);
        if (ref->grade<0 || ref->grade>maxgrade) ref->grade = maxgrade;

        ref->vtoel=mesh_addconnectivityelement(mesh, ref->grade, 0);
        ref->eltov=mesh_addconnectivityelement(mesh, 0, ref->grade);

        if (ref->vtoel && ref->eltov && equielement_bindweight(self, mesh, ref)) success=true;
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

        if (fabs(mean)<MORPHO_EPS) { *out = 0; return true; }

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
    value grade=MORPHO_INTEGER(-1);
    value weight=MORPHO_NIL;

    builtin_options(v, nargs, args, NULL, 2,
                    equielement_weightproperty, &weight,
                    functional_gradeproperty, &grade);

    objectinstance_setproperty(self, equielement_weightproperty, weight);
    objectinstance_setproperty(self, functional_gradeproperty, grade);
    return MORPHO_NIL;
}

FUNCTIONAL_MD_REF_BIND_FORCEGRADE(EquiElement, equielementref, equielement_prepareref, equielement_integrand, FUNCTIONAL_ARGS, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_INTEGRAND_COST(EquiElement, equielementref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_TOTAL_COST(EquiElement, equielementref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(EquiElement, equielementref, MESH_GRADE_VERTEX, equielement_dependencies, SYMMETRY_ADD)
FUNCTIONAL_MD_REF_HESSIAN(EquiElement, equielementref, MESH_GRADE_VERTEX, equielement_dependencies, SYMMETRY_ADD)

MORPHO_BEGINCLASS(EquiElement)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", EquiElement_init, MORPHO_FN_MUTATES|MORPHO_FN_OPTARGS),

FUNCTIONAL_MD_INTEGRAND_METHODS(EquiElement),
FUNCTIONAL_MD_TOTAL_METHODS(EquiElement),
FUNCTIONAL_MD_GRADIENT_METHODS(EquiElement),
FUNCTIONAL_MD_HESSIAN_METHODS(EquiElement)
MORPHO_ENDCLASS

void equielement_initialize(void) {
    equielement_weightproperty=builtin_internsymbolascstring(EQUIELEMENT_WEIGHT_PROPERTY);

    value objclass = builtin_findclassfromcstring(OBJECT_CLASSNAME);

    builtin_addclass(EQUIELEMENT_CLASSNAME, MORPHO_GETCLASSDEFINITION(EquiElement), objclass);
}

#endif
