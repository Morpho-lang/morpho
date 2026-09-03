/** @file elasticity.c
 *  @author T J Atherton
 *
 *  @brief LinearElasticity functional
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
#include "elasticity.h"

/* ----------------------------------------------
 * Linear Elasticity
 * ---------------------------------------------- */

static value linearelasticity_referenceproperty;
static value linearelasticity_weightbyreferenceproperty;
static value linearelasticity_poissonproperty;
static value linearelasticity_cacheproperty;

typedef struct {
    double *cache; /* packed columns: inv(G0) then reference measure */
    int stride;
    int nel;
    int gdim;
    grade grade;
    double lambda; /* Lamé coefficients */
    double mu;
} linearelasticityref;

/** Calculates the Gram matrix */
void linearelasticity_calculategram(objectmatrix *vert, int dim, int nv, int *vid, objectmatrix *gram) {
    int gdim=nv-1; // Dimension of Gram matrix
    double *x[nv], // Positions of vertices
            s[gdim][nv]; // Side vectors

    for (int j=0; j<nv; j++) matrix_getcolumnptr(vert, vid[j], &x[j]); // Get vertices
    for (int j=1; j<nv; j++) functional_vecsub(dim, x[j], x[0], s[j-1]); // u_i = X_i - X_0
    // <u_i, u_j>
    for (int i=0; i<nv-1; i++) for (int j=0; j<nv-1; j++) gram->elements[i+j*gdim]=functional_vecdot(dim, s[i], s[j]);
}

/** Caches the inv(G0) and measure from the reference mesh at grade g.
 * @details The cache is a matrix with stride rows and nel columns.
 *          Each column is one element: the first gdim*gdim entries are inv(G0),
 *          and the last entry is the measure. */
static bool linearelasticity_buildcache(vm *v, objectmesh *refmesh, grade g, value *out) {
    objectsparse *conn=mesh_getconnectivityelement(refmesh, 0, g);
    if (!conn) conn=mesh_addgrade(refmesh, g);
    if (!conn) MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) g);

    int nel=(int) mesh_nelements(conn);
    if (nel<=0) MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) g);

    int nv0=0, *vid0=NULL;
    if (!mesh_getconnectivity(conn, 0, &nv0, &vid0) || nv0<2) MORPHO_FAIL(v, FUNCTIONAL_ARGS);
    int gdim=nv0-1;
    int stride=gdim*gdim+1;

    value cacheval=morpho_wrapandbind(v, (object *) matrix_new(stride, nel, false));
    if (MORPHO_ISNIL(cacheval)) return false;
    objectmatrix *cache=MORPHO_GETMATRIX(cacheval);

    for (elementid id=0; id<nel; id++) {
        int nv=0, *vid=NULL;
        if (!mesh_getconnectivity(conn, id, &nv, &vid) || nv!=nv0) MORPHO_FAIL(v, FUNCTIONAL_ARGS);

        double *col=cache->elements+(size_t) id*(size_t) stride;
        objectmatrix gram=MORPHO_STATICMATRIX(col, gdim, gdim);
        linearelasticity_calculategram(refmesh->vert, refmesh->dim, nv, vid, &gram);
        if (!functional_matinv(gdim, col)) MORPHO_FAIL(v, FUNCTIONAL_ARGS);
        if (!functional_elementsize(v, refmesh, g, id, nv, vid, &col[gdim*gdim])) {
            MORPHO_FAIL(v, FUNCTIONAL_ARGS);
        }
    }

    *out=cacheval;
    return true;
}

/** Cache geometry of reference mesh */
static bool linearelasticity_fillcache(vm *v, objectinstance *self) {
    value refmeshval=MORPHO_NIL, gradeval=MORPHO_NIL, cacheval=MORPHO_NIL;
    if (!objectinstance_getpropertyinterned(self, linearelasticity_referenceproperty, &refmeshval) ||
        !MORPHO_ISMESH(refmeshval) ||
        !objectinstance_getpropertyinterned(self, functional_gradeproperty, &gradeval) ||
        !MORPHO_ISINTEGER(gradeval)) {
        MORPHO_FAIL(v, FUNCTIONAL_ARGS);
    }

    if (!linearelasticity_buildcache(v, MORPHO_GETMESH(refmeshval),
                                     MORPHO_GETINTEGERVALUE(gradeval), &cacheval)) {
        return false;
    }
    objectinstance_setproperty(self, linearelasticity_cacheproperty, cacheval);
    return true;
}

/** Bind the cached geometry of the reference mesh */
static bool linearelasticity_prepareref(vm *v, objectinstance *self, linearelasticityref *ref) {
    value poisson=MORPHO_NIL, cacheval=MORPHO_NIL;
    grade g=0;
    double nu=0.0;

    if (!functional_readgrade(self, &g) || g<1) MORPHO_FAIL(v, FUNCTIONAL_ARGS);
    if (!objectinstance_getpropertyinterned(self, linearelasticity_poissonproperty, &poisson) ||
        !morpho_valuetofloat(poisson, &nu)) {
        MORPHO_FAIL(v, FUNCTIONAL_ARGS);
    }

    int stride=g*g+1; /* simplices: gdim == grade */
    if (!objectinstance_getpropertyinterned(self, linearelasticity_cacheproperty, &cacheval) ||
        !MORPHO_ISMATRIX(cacheval)) {
        MORPHO_FAIL(v, FUNCTIONAL_ARGS);
    }

    objectmatrix *cache=MORPHO_GETMATRIX(cacheval);
    if (cache->nrows!=stride || cache->ncols<=0) MORPHO_FAIL(v, FUNCTIONAL_ARGS);

    ref->cache=cache->elements;
    ref->stride=cache->nrows;
    ref->nel=cache->ncols;
    ref->gdim=g;
    ref->grade=g;
    ref->mu=0.5/(1+nu);
    ref->lambda=nu/(1+nu)/(1-2*nu);
    return true;
}

/** Calculate the linear elastic energy from cached inv(G0) and measure. */
bool linearelasticity_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    linearelasticityref *info=(linearelasticityref *) ref;
    int gdim=nv-1;

    if (gdim!=info->gdim || id<0 || id>=info->nel) return false;

    double *col=info->cache+(size_t) id*(size_t) info->stride;
    double *invG0=col;
    double weight=col[gdim*gdim];

    double gramdefel[gdim*gdim], rel[gdim*gdim], cgel[gdim*gdim];
    objectmatrix gramdef=MORPHO_STATICMATRIX(gramdefel, gdim, gdim);
    linearelasticity_calculategram(mesh->vert, mesh->dim, nv, vid, &gramdef);

    functional_matmul(gdim, gdim, gdim, gramdefel, invG0, rel); /* G G0^{-1} */

    for (int i=0; i<gdim*gdim; i++) cgel[i]=0.5*rel[i];
    for (int i=0; i<gdim; i++) cgel[i+i*gdim]-=0.5;

    double trcg=0.0, trcgcg=0.0;
    for (int i=0; i<gdim; i++) trcg+=cgel[i+i*gdim];
    functional_matmul(gdim, gdim, gdim, cgel, cgel, rel);
    for (int i=0; i<gdim; i++) trcgcg+=rel[i+i*gdim];

    *out=weight*(info->mu*trcgcg+0.5*info->lambda*trcg*trcg);
    return true;
}

static void _linearelasticity_initmesh(objectinstance *self, value meshval) {
    objectinstance_setproperty(self, linearelasticity_referenceproperty, meshval);
    functional_setgrade(self, mesh_maxgrade(MORPHO_GETMESH(meshval)));
    objectinstance_setproperty(self, linearelasticity_poissonproperty, MORPHO_FLOAT(0.3));
}

value LinearElasticity_init__mesh(vm *v, int nargs, value *args) {
    objectinstance *self=MORPHO_GETINSTANCE(MORPHO_SELF(args));
    _linearelasticity_initmesh(self, MORPHO_GETARG(args, 0));
    if (!linearelasticity_fillcache(v, self)) return MORPHO_NIL;
    return MORPHO_NIL;
}

value LinearElasticity_init__mesh_int(vm *v, int nargs, value *args) {
    objectinstance *self=MORPHO_GETINSTANCE(MORPHO_SELF(args));
    _linearelasticity_initmesh(self, MORPHO_GETARG(args, 0));
    functional_setgrade(self, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1)));
    if (!linearelasticity_fillcache(v, self)) return MORPHO_NIL;
    return MORPHO_NIL;
}

value LinearElasticity_update(vm *v, int nargs, value *args) {
    if (!linearelasticity_fillcache(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)))) return MORPHO_NIL;
    return MORPHO_NIL;
}

value LinearElasticity_update__mesh(vm *v, int nargs, value *args) {
    objectinstance *self=MORPHO_GETINSTANCE(MORPHO_SELF(args));
    grade g=0;
    value cacheval=MORPHO_NIL;

    if (!functional_readgrade(self, &g) || g<1) MORPHO_RAISE(v, FUNCTIONAL_ARGS);
    if (!linearelasticity_buildcache(v, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), g, &cacheval)) {
        return MORPHO_NIL;
    }
    objectinstance_setproperty(self, linearelasticity_referenceproperty, MORPHO_GETARG(args, 0));
    objectinstance_setproperty(self, linearelasticity_cacheproperty, cacheval);
    return MORPHO_NIL;
}

static bool _LinearElasticity_bindref(vm *v, objectinstance *self, functional_mapinfo *info, linearelasticityref *ref) {
    if (!linearelasticity_prepareref(v, self, ref)) return false;
    if (info->g < 0) info->g = ref->grade;
    info->ref = ref;
    info->integrand = linearelasticity_integrand;
    return true;
}

FUNCTIONAL_MD_REF_INTEGRAND_COST(LinearElasticity, linearelasticityref, ref.grade, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_TOTAL_COST(LinearElasticity, linearelasticityref, ref.grade, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(LinearElasticity, linearelasticityref, ref.grade, NULL, SYMMETRY_ADD)

MORPHO_BEGINCLASS(LinearElasticity)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Mesh)", LinearElasticity_init__mesh, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Mesh, Int)", LinearElasticity_init__mesh_int, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_UPDATE_METHOD, "()", LinearElasticity_update, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_UPDATE_METHOD, "(Mesh)", LinearElasticity_update__mesh, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),

FUNCTIONAL_MD_INTEGRAND_METHODS(LinearElasticity),
FUNCTIONAL_MD_TOTAL_METHODS(LinearElasticity),
FUNCTIONAL_MD_GRADIENT_METHODS(LinearElasticity)
MORPHO_ENDCLASS

void elasticity_initialize(void) {
    linearelasticity_referenceproperty=builtin_internsymbolascstring(LINEARELASTICITY_REFERENCE_PROPERTY);
    linearelasticity_weightbyreferenceproperty=builtin_internsymbolascstring(LINEARELASTICITY_WTBYREF_PROPERTY);
    linearelasticity_poissonproperty=builtin_internsymbolascstring(LINEARELASTICITY_POISSON_PROPERTY);
    linearelasticity_cacheproperty=builtin_internsymbolascstring(LINEARELASTICITY_CACHE_PROPERTY);

    value objclass = builtin_findclassfromcstring(OBJECT_CLASSNAME);

    builtin_addclass(LINEARELASTICITY_CLASSNAME, MORPHO_GETCLASSDEFINITION(LinearElasticity), objclass);
}

#endif
