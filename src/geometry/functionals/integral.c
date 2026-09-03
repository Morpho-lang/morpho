/** @file integral.c
 *  @author T J Atherton
 *
 *  @brief LineIntegral, AreaIntegral and VolumeIntegral
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
#include <limits.h>
#include <string.h>
#include <math.h>
#include "optimize.h"
#include "integrate.h"
#include "integral.h"
#include "jump.h"
#include "elasticity.h"

#define INTEGRAL_MAXSPECIALS 8
#define INTEGRAL_SPECIALFLAGS (MORPHO_FN_THREADLOCAL | MORPHO_FN_ALLOCATES | MORPHO_FN_THROWS)
static value _specialfns[INTEGRAL_MAXSPECIALS];
static unsigned _specialbits[INTEGRAL_MAXSPECIALS];
static unsigned int nspecials=0;

/** Register a special for uses analysis, refreshing the function-table entry. */
static void integral_trackspecial(char *name, unsigned bit) {
    value selector=builtin_internsymbolascstring(name);
    value entry=builtin_findfunction(selector);

    for (unsigned int i=0; i<nspecials; i++) {
        if (_specialbits[i]==bit) {
            _specialfns[i]=entry;
            return;
        }
    }

    if (nspecials<INTEGRAL_MAXSPECIALS) {
        _specialfns[nspecials]=entry;
        _specialbits[nspecials]=bit;
        nspecials++;
    } else UNREACHABLE("Too many special functions in integral.c");
}

/** Add (or overload) an integral special with a signature. */
void integral_addspecial(char *name, builtinfunction fn, unsigned bit, char *signature) {
    morpho_addfunction(name, signature, fn, INTEGRAL_SPECIALFLAGS, NULL);
    integral_trackspecial(name, bit);
}

static unsigned integral_fnuses(vm *v, value integrand) {
    unsigned uses=0;
    bool hit[INTEGRAL_MAXSPECIALS];

    if (optimize_fnaccessesarg(v, integrand, 0)) uses|=INTEGRAL_USES_X;
    optimize_fnloadsconstants(v, integrand, (int) nspecials, _specialfns, hit);
    for (unsigned int i=0; i<nspecials; i++) if (hit[i]) uses|=_specialbits[i];
    return uses;
}

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



int elementhandle;

/** Get the current element ref from thread-local storage in the VM */
objectintegralelementref *integral_getelementref(vm *v) {
    value elref=MORPHO_NIL;
    vm_gettlvar(v, elementhandle, &elref);
    if (MORPHO_ISINTEGRALELEMENTREF(elref)) return MORPHO_GETINTEGRALELEMENTREF(elref);
    
    return NULL;
}

/** Checks whether an existing matrix is the correct size and allocates if not. */
objectmatrix *integral_ensurematrix(objectmatrix **slot, int nrows, int ncols) {
    if (!*slot || (*slot)->nrows!=nrows || (*slot)->ncols!=ncols) {
        if (*slot) object_free((object *) *slot);
        *slot=matrix_new(nrows, ncols, false);
    }
    return *slot;
}

/** Resets the geometry flags and other geometry-related data. */
static void integral_resetgeometryflags(objectintegralelementref *elref) {
    elref->flags &= ~ELREF_GEOM;
    elref->lambda=NULL;
    elref->posn=NULL;
    elref->qinterpolated=NULL;
}

/** Initialize an elref to a cleared state. */
void _integral_initelref(objectintegralelementref *elref) {
    memset(elref, 0, sizeof(objectintegralelementref));
    object_init((object *) elref, OBJECT_INTEGRALELEMENTREF);
    elref->allowed=INTEGRAL_ALLOW_ALL;
    elref->target_field=-1;
    elref->freeze_grad=-1;
}

/** Bind an elref to a particular element and reset per-element geometry cache.
 * @warning Does not clear map policy (allowed, target_field). */
void _integral_bindelref(objectintegralelementref *elref, objectmesh *mesh, grade g, elementid id, int nv, int *vid, double **vertexposn, integralref *iref) {
    elref->mesh=mesh;
    elref->g=g;
    elref->id=id;
    elref->nv=nv;
    elref->vid=vid;
    elref->vertexposn=vertexposn;
    elref->iref=iref;
    elref->freeze_grad=-1;
    integral_resetgeometryflags(elref);
}

/** Fill n values with nil. */
static void _integral_nilvalues(value *p, int n) {
    for (int i=0; i<n; i++) p[i]=MORPHO_NIL;
}

static void integral_releasegeometry(objectintegralelementref *elref) {
    if (elref->invj) { object_free((object *) elref->invj); elref->invj=NULL; }
    if (elref->tangent) { object_free((object *) elref->tangent); elref->tangent=NULL; }
    if (elref->normal) { object_free((object *) elref->normal); elref->normal=NULL; }
    if (elref->jacobian) { object_free((object *) elref->jacobian); elref->jacobian=NULL; }
    if (elref->invjacobian) { object_free((object *) elref->invjacobian); elref->invjacobian=NULL; }
    if (elref->cgtensor) { object_free((object *) elref->cgtensor); elref->cgtensor=NULL; }
    if (elref->xgeom) { object_free((object *) elref->xgeom); elref->xgeom=NULL; }
    integral_resetgeometryflags(elref);
}

static void integral_freegradhess(int nfields, value *qgrad, value *qhess);

/** Release buffers and geometry; does not free the elref object itself. */
void integral_clearelref(objectintegralelementref *elref) {
    if (!elref) return;
    
    if (elref->flags & ELREF_HASINTEG) integrator_clear(&elref->integ);
    
    if (elref->quantities) {
        for (int i=0; i<elref->nfields; i++) {
            if (elref->quantities[i].vals) MORPHO_FREE(elref->quantities[i].vals);
            if (elref->quantities[i].findx) MORPHO_FREE(elref->quantities[i].findx);
        }
        MORPHO_FREE(elref->quantities);
        elref->quantities=NULL;
    }
    
    if (elref->qgrad && elref->qhess) integral_freegradhess(elref->nfields, elref->qgrad, elref->qhess);
    if (elref->qgrad) { MORPHO_FREE(elref->qgrad); elref->qgrad=NULL; }
    if (elref->qhess) { MORPHO_FREE(elref->qhess); elref->qhess=NULL; }
    
    integral_releasegeometry(elref);
}

/** Frees a heap elref and attached data. */
static void integral_freeelref(objectintegralelementref *elref) {
    if (!elref) return;
    integral_clearelref(elref);
    object_free((object *) elref);
}

bool integral_contextactive(vm *v) {
    return integral_getelementref(v) || jump_getinterfaceref(v);
}

/** Reject specials not in elref->allowed. */
bool integral_checkfastpath(vm *v, unsigned bit, const char *name) {
    objectintegralelementref *elref=integral_getelementref(v);
    if (!elref) return true;
    if (!(elref->allowed & bit)) MORPHO_FAILVARGS(v, INTEGRAL_FASTPATH, name);
    return true;
}

/* ---------
 * Elementid
 * --------- */

static value integral_elementid(vm *v, int nargs, value *args) {
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, ELEMENTID_FUNCTION);
    
    return MORPHO_INTEGER(elref->id);
}

/* --------
 * Tangent
 * -------- */

/** Evaluate the tangent vector */
static bool integral_evaluatetangent(vm *v) {
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref || elref->g!=1) MORPHO_FAILVARGS(v, INTEGRAL_SPCLFN, TANGENT_FUNCTION);
    
    int dim = elref->mesh->dim;
    objectmatrix *mtangent=integral_ensurematrix(&elref->tangent, dim, 1);
    if (!mtangent) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    
    functional_vecsub(dim, elref->vertexposn[1], elref->vertexposn[0], mtangent->elements);

    double tnorm=functional_vecnorm(dim, mtangent->elements);
    if (fabs(tnorm)>MORPHO_EPS) functional_vecscale(dim, 1.0/tnorm, mtangent->elements, mtangent->elements);
    
    elref->flags |= ELREF_HASTANGENT;
    return true;
}

static value integral_tangent(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_TANGENT, TANGENT_FUNCTION)) return MORPHO_NIL;
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref || elref->g!=1) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, TANGENT_FUNCTION);
    
    if (!(elref->flags & ELREF_HASTANGENT) && !integral_evaluatetangent(v)) return MORPHO_NIL;
    return MORPHO_OBJECT(elref->tangent);
}

/* --------
 * Normal
 * -------- */

/** Evaluates the normal vector */
static bool integral_evaluatenormal(vm *v) {
    objectintegralelementref *elref = integral_getelementref(v);
    
    if (!elref || elref->g!=2) MORPHO_FAILVARGS(v, INTEGRAL_SPCLFN, NORMAL_FUNCTION);
    
    int dim = elref->mesh->dim;
    double s0[dim], s1[dim];
    objectmatrix *mnormal=integral_ensurematrix(&elref->normal, dim, 1);
    if (!mnormal) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    
    functional_vecsub(dim, elref->vertexposn[1], elref->vertexposn[0], s0);
    functional_vecsub(dim, elref->vertexposn[2], elref->vertexposn[1], s1);
    functional_veccross(s0, s1, mnormal->elements);
    
    double nnorm=functional_vecnorm(dim, mnormal->elements);
    if (fabs(nnorm)>MORPHO_EPS) functional_vecscale(dim, 1.0/nnorm, mnormal->elements, mnormal->elements);
    
    elref->flags |= ELREF_HASNORMAL;
    return true;
}

static value integral_normal(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_NORMAL, NORMAL_FUNCTION)) return MORPHO_NIL;
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref || elref->g!=2) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, NORMAL_FUNCTION);
    
    if (!(elref->flags & ELREF_HASNORMAL) && !integral_evaluatenormal(v)) return MORPHO_NIL;
    return MORPHO_OBJECT(elref->normal);
}

/* --------
 * Gradient
 * -------- */

/** @brief Prepares an inverse jacobian matrix.
    @param[in] dim - dimension of physical space
    @param[in] g - grade of the object
    @param[in] x - list of vertex positions (grade+1 entries, each of length dim)
    @param[out] invj - inverse jacobian for the transformation (dim*g entries) */
bool integral_prepareinvjacobian(unsigned int dim, grade g, double **x, objectmatrix *invj) {
    bool success=false;
    
    // Construct the (dim x g) matrix of edge vectors
    double s[dim*g];
    for (int i=0; i<g; i++) functional_vecsub(dim, x[i+1], x[0], s + i*dim);
    
    if (g==dim) {
        memcpy(invj->elements, s, sizeof(double)*dim*dim);
        success=functional_matinv(dim, invj->elements);
    } else if (g==1) {
        double s01norm = functional_vecdot(dim, s, s);
        if (s01norm>0) {
            functional_vecscale(dim, 1.0/s01norm, s, invj->elements);
            success=true;
        }
    } else if (g==2 && dim==3) {
        double *s0 = s, *s1 = s+dim, s0xs1[dim], u[dim], v[dim*g];
        functional_veccross(s0, s1, s0xs1);
        double s0xs1norm = functional_vecnorm(dim, s0xs1);
        if (s0xs1norm>0) {
            double invs0xs1norm = 1/(s0xs1norm*s0xs1norm);
            functional_veccross(s1, s0xs1, u);
            functional_vecscale(dim, invs0xs1norm, u, v);
            functional_veccross(s0xs1, s0, u);
            functional_vecscale(dim, invs0xs1norm, u, v+dim);
            
            objectmatrix invjt = MORPHO_STATICMATRIX(v, dim, g);
            
            matrix_transpose(&invjt, invj);
            
            success=true;
        }
    }
    return success;
}

static bool integral_ensureinvj(vm *v, objectintegralelementref *elref) {
    if (elref->flags & ELREF_HASINVJ) return true;
    if (!integral_ensurematrix(&elref->invj, elref->g, elref->mesh->dim) ||
        !integral_prepareinvjacobian(elref->mesh->dim, elref->g, elref->vertexposn, elref->invj)) {
        MORPHO_FAIL(v, INTEGRAL_DFFEVL);
    }
    elref->flags |= ELREF_HASINVJ;
    return true;
}

bool integral_gradalloc(int dim, value prototype, value *out) {
    if (MORPHO_ISNIL(prototype)) { // Scalar
        objectmatrix *mgrad=matrix_new(dim, 1, false);
        if (mgrad) *out = MORPHO_OBJECT(mgrad);
        return mgrad;
    } else if (MORPHO_ISMATRIX(prototype)) {
        objectlist *mlst = object_newlist(0, NULL);
        if (mlst) *out = MORPHO_OBJECT(mlst);
        return mlst;
    } else UNREACHABLE("Field type not supported in grad");
    return false;
}

/** Allocate suitable storage for the hessian */
bool integral_hessalloc(int dim, value prototype, value *out) {
    if (MORPHO_ISNIL(prototype)) { // Scalar
        objectmatrix *mhess=matrix_new(dim, dim, false);
        if (mhess) *out = MORPHO_OBJECT(mhess);
        return mhess;
    } else if (MORPHO_ISMATRIX(prototype)) {
        objectlist *mlst = object_newlist(0, NULL);
        if (mlst) *out = MORPHO_OBJECT(mlst);
        return mlst;
    } else UNREACHABLE("Field type not supported in hess");
    return false;
}

/** Prepares the gradient sum to hold the component of the gradient */
bool integral_gradsuminit(int i, value prototype, value dest, value *sum) {
    if (MORPHO_ISLIST(dest)) {
        objectlist *lst = MORPHO_GETLIST(dest);
        
        if (i>=list_length(lst)) {
            objectmatrix *prmat = MORPHO_GETMATRIX(prototype);
            objectmatrix *new = matrix_new(prmat->nrows, prmat->ncols, true);
            if (!new) return false;
            *sum = MORPHO_OBJECT(new);
            list_append(lst, *sum);
        } else {
            matrix_zero(MORPHO_GETMATRIX(lst->val.data[i]));
            *sum = lst->val.data[i];
        }
    }
    return true;
}
 
/** Copies the component of the gradient into the relevant destination if needed */
bool integral_gradsumcopy(int i, value sum, value dest) {
    if (MORPHO_ISMATRIX(dest)) {
        return morpho_valuetofloat(sum, &MORPHO_GETMATRIX(dest)->elements[i]);
    } else return true;
}

/** Prepares the hessian sum to hold a component of the hessian */
bool integral_hesssuminit(int c, value prototype, value dest, value *sum) {
    if (MORPHO_ISLIST(dest)) {
        objectlist *lst = MORPHO_GETLIST(dest);
        
        if (c>=list_length(lst)) {
            objectmatrix *prmat = MORPHO_GETMATRIX(prototype);
            objectmatrix *new = matrix_new(prmat->nrows, prmat->ncols, true);
            if (!new) return false;
            *sum = MORPHO_OBJECT(new);
            list_append(lst, *sum);
        } else {
            matrix_zero(MORPHO_GETMATRIX(lst->val.data[c]));
            *sum = lst->val.data[c];
        }
    }
    return true;
}

/** Copies the component of the hessian into the relevant destination if needed */
bool integral_hesssumcopy(int i, int j, value sum, value dest) {
    if (MORPHO_ISMATRIX(dest)) {
        return morpho_valuetofloat(sum, &MORPHO_GETMATRIX(dest)->elements[j*MORPHO_GETMATRIX(dest)->nrows+i]);
    } else return true;
}

/** Evaluates the gradient of a field */
bool integral_evaluategradient(vm *v, value q, value *out) {
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) MORPHO_FAILVARGS(v, INTEGRAL_SPCLFN, GRAD_FUNCTION);
    
    /* Identify the field being referred to */
    int ifld, xfld=-1;
    for (ifld=0; ifld<elref->iref->nfields; ifld++) {
        if (MORPHO_ISFIELD(q) && MORPHO_ISSAME(elref->iref->originalfields[ifld], q)) break;
        else if (MORPHO_ISSAME(elref->qinterpolated[ifld], q)) {
            if (xfld>=0) MORPHO_FAIL(v, INTEGRAL_FLD);
            // @warning: This will fail if two fields happen to have the same value(!)
            xfld=ifld;
        }
    }
    if (xfld>=0) ifld = xfld;
    
    // Raise an error if we couldn't find it
    if (ifld>=elref->iref->nfields) MORPHO_FAIL(v, INTEGRAL_FLD);

    if (ifld==elref->target_field) elref->target_grad_used=true;

    if (elref->freeze_grad==ifld && MORPHO_ISOBJECT(elref->qgrad[ifld])) {
        *out=elref->qgrad[ifld];
        return true;
    }
    
    // Extract information from the field
    objectfield *fld = MORPHO_GETFIELD(elref->iref->fields[ifld]);
    int dim = elref->mesh->dim;
    
    // Allocate objects if need be. Don't bind these; these will be freed when the elref is cleared.
    if (!MORPHO_ISOBJECT(elref->qgrad[ifld])) {
        if (!integral_gradalloc(dim, fld->prototype, &elref->qgrad[ifld])) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    }
    
    if (!(MORPHO_ISFESPACE(fld->fnspc) && elref->quantities)) MORPHO_FAIL(v, INTEGRAL_DFFEVL);
    if (!integral_ensureinvj(v, elref)) return false;
    
    fespace *disc = MORPHO_GETFESPACE(fld->fnspc)->fespace;
    if (!FESPACE_HASGRADIENT(disc)) MORPHO_FAIL(v, INTEGRAL_DFFEVL);
    
    int nnodes = disc->nnodes;
    double gdata[nnodes * elref->g];
    objectmatrix gmat = MORPHO_STATICMATRIX(gdata, nnodes, elref->g);
    
    // Compute gradient in reference frame
    fespace_gradient(disc, elref->lambda, &gmat);
    
    // Compute matrix
    double fmatdata[nnodes * dim];
    objectmatrix fmat = MORPHO_STATICMATRIX(fmatdata, nnodes, dim);
    
    functional_matmul(nnodes, elref->g, dim, gdata, elref->invj->elements, fmatdata);
    
    for (int i=0; i<dim; i++) {
        value sum;
        
        if (integral_gradsuminit(i, fld->prototype, elref->qgrad[ifld], &sum) &&
            integrator_sumquantityweighted(nnodes, fmat.elements+i*nnodes, elref->quantities[ifld].vals, &sum)) {
            integral_gradsumcopy(i, sum, elref->qgrad[ifld]);
        } else MORPHO_FAIL(v, INTEGRAL_DFFEVL);
    }
    
    *out=elref->qgrad[ifld];
    return true;
}

static value integral_gradfn(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_GRAD, GRAD_FUNCTION)) return MORPHO_NIL;
    value out=MORPHO_NIL;
    if (nargs!=1) MORPHO_RAISE(v, INTEGRAL_FLD);
    integral_evaluategradient(v, MORPHO_GETARG(args, 0), &out);
    return out;
}

/** Evaluates the hessian of a field */
bool integral_evaluatehessian(vm *v, value q, value *out) {
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) MORPHO_FAILVARGS(v, INTEGRAL_SPCLFN, HESS_FUNCTION);
    
    int ifld, xfld=-1;
    for (ifld=0; ifld<elref->iref->nfields; ifld++) {
        if (MORPHO_ISFIELD(q) && MORPHO_ISSAME(elref->iref->originalfields[ifld], q)) break;
        else if (MORPHO_ISSAME(elref->qinterpolated[ifld], q)) {
            if (xfld>=0) MORPHO_FAIL(v, INTEGRAL_FLD);
            xfld=ifld;
        }
    }
    if (xfld>=0) ifld = xfld;
    
    if (ifld>=elref->iref->nfields) MORPHO_FAIL(v, INTEGRAL_FLD);
    
    objectfield *fld = MORPHO_GETFIELD(elref->iref->fields[ifld]);
    int dim = elref->mesh->dim;
    
    if (!MORPHO_ISOBJECT(elref->qhess[ifld])) {
        if (!integral_hessalloc(dim, fld->prototype, &elref->qhess[ifld])) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    }
    
    if (MORPHO_ISFESPACE(fld->fnspc) && elref->quantities) {
        if (!integral_ensureinvj(v, elref)) return false;
        
        fespace *disc = MORPHO_GETFESPACE(fld->fnspc)->fespace;
        if (!FESPACE_HASHESSIAN(disc)) MORPHO_FAIL(v, INTEGRAL_DFFEVL);
        int nnodes = disc->nnodes;
        double hdata[nnodes * elref->g * elref->g];
        objectmatrix hmat = MORPHO_STATICMATRIX(hdata, nnodes, elref->g*elref->g);
        
        fespace_hessian(disc, elref->lambda, &hmat);
        
        double fdata[nnodes * dim * dim];
        for (int j=0; j<dim; j++) {
            for (int i=0; i<dim; i++) {
                double *outcol = fdata + (j*dim+i)*nnodes;
                for (int n=0; n<nnodes; n++) {
                    double sum=0.0;
                    for (int qref=0; qref<elref->g; qref++) {
                        for (int pref=0; pref<elref->g; pref++) {
                            sum += hdata[(qref*elref->g+pref)*nnodes+n] *
                                   elref->invj->elements[i*elref->g+pref] *
                                   elref->invj->elements[j*elref->g+qref];
                        }
                    }
                    outcol[n]=sum;
                }
            }
        }
        
        for (int j=0; j<dim; j++) {
            for (int i=0; i<dim; i++) {
                int c = j*dim+i;
                value sum;
                
                if (integral_hesssuminit(c, fld->prototype, elref->qhess[ifld], &sum) &&
                    integrator_sumquantityweighted(nnodes, fdata+c*nnodes, elref->quantities[ifld].vals, &sum)) {
                    integral_hesssumcopy(i, j, sum, elref->qhess[ifld]);
                } else MORPHO_FAIL(v, INTEGRAL_DFFEVL);
            }
        }
        
        *out=elref->qhess[ifld];
        return true;
    }
    
    MORPHO_FAIL(v, INTEGRAL_DFFEVL);
}

static value integral_hessfn(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_HESS, HESS_FUNCTION)) return MORPHO_NIL;
    value out=MORPHO_NIL;
    if (nargs!=1) MORPHO_RAISE(v, INTEGRAL_FLD);
    integral_evaluatehessian(v, MORPHO_GETARG(args, 0), &out);
    return out;
}

/* -------------------
 * Cauchy green strain
 * ------------------- */

/** Evaluates the cg strain tensor */
static bool integral_evaluatecg(vm *v) {
    objectintegralelementref *elref = integral_getelementref(v);
    
    if (!elref || !elref->iref->mref) MORPHO_FAILVARGS(v, INTEGRAL_SPCLFN, CGTENSOR_FUNCTION);
    
    int gdim=elref->nv-1; // Dimension of Gram matrix
    objectmatrix *cg=integral_ensurematrix(&elref->cgtensor, gdim, gdim);
    if (!cg) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    
    double gramrefel[gdim*gdim], gramdefel[gdim*gdim], qel[gdim*gdim], rel[gdim*gdim];
    objectmatrix gramref = MORPHO_STATICMATRIX(gramrefel, gdim, gdim); // Gram matrices
    objectmatrix gramdef = MORPHO_STATICMATRIX(gramdefel, gdim, gdim); //
    objectmatrix r = MORPHO_STATICMATRIX(rel, gdim, gdim); // Intermediate calculations
    
    linearelasticity_calculategram(elref->iref->mref->vert, elref->mesh->dim, elref->nv, elref->vid, &gramref);
    linearelasticity_calculategram(elref->mesh->vert, elref->mesh->dim, elref->nv, elref->vid, &gramdef);
    
    memcpy(qel, gramrefel, sizeof(double)*(size_t)gdim*(size_t)gdim);
    if (!functional_matinv(gdim, qel)) return false;
    functional_matmul(gdim, gdim, gdim, gramdefel, qel, rel);

    if (matrix_identity(cg)!=LINALGERR_OK) return false;
    matrix_scale(cg, -0.5);
    matrix_axpy(0.5, &r, cg);
    
    elref->flags |= ELREF_HASCG;
    return true;
}

static value integral_cgfn(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_CG, CGTENSOR_FUNCTION)) return MORPHO_NIL;
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref || !elref->iref->mref) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, CGTENSOR_FUNCTION);
    
    if (!(elref->flags & ELREF_HASCG) && !integral_evaluatecg(v)) return MORPHO_NIL;
    return MORPHO_OBJECT(elref->cgtensor);
}

/* -------------------
 * Jacobian
 * ------------------- */

/*
 * A reference triangle is mapped to a target triangle through a
 * linear transformation (the pushforward); an inverse transformation (the pullback)
 * exists if the triangle is not degenerate. This function computes the forward
 * and inverse jacobians.
 */

void _fetchvertices(objectintegralelementref *elref, objectmesh *mesh, int nv, elementid *vid, double **x) {
    // Fetch reference vertices
    for (int j=0; j<nv; j++) matrix_getcolumnptr(elref->iref->mref->vert, vid[j], &x[j]);
}

void _edgevectors(grade g, int dim, double **x, double *out) {
    for (int i=0; i<g; i++) functional_vecsub(dim, x[i+1], x[0], out + i*dim);
}

/** Evaluates the jacobian and inverse jacobian */
static bool integral_evaluatejacobian(vm *v) {
    objectintegralelementref *elref = integral_getelementref(v);
    
    if (!elref) MORPHO_FAILVARGS(v, INTEGRAL_SPCLFN, JACOBIAN_FUNCTION);
    
    int dim = elref->mesh->dim;     // Dimension of the mesh
    objectmatrix *J=integral_ensurematrix(&elref->jacobian, dim, dim);
    objectmatrix *Jinv=integral_ensurematrix(&elref->invjacobian, dim, dim);
    if (!J || !Jinv) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    
    // Now compute them
    grade g = elref->g;             // Grade of the element
    int nv = elref->nv;             //
    
    double **X = elref->vertexposn; // Vertex positions of the target element
    double *x[nv];                  // Vertex positions of the reference element
    
    objectmesh *mref = elref->iref->mref; // Reference mesh
    if (mref) _fetchvertices(elref, mref, nv, elref->vid, x);
    
    // Construct matrix of edge vectors for target and reference elements
    double starget[dim*dim], sinv[dim*dim];
    objectmatrix Sinv = MORPHO_STATICMATRIX(sinv, dim, dim);
    
    _edgevectors(g, dim, X, starget);
    if (mref) {
        _edgevectors(g, dim, x, sinv);
        if (!functional_matinv(dim, sinv)) return false;
    } else {
        matrix_identity(&Sinv); // If no reference, the reference is the unit triangle
    }
    
    functional_matmul(dim, dim, dim, starget, sinv, J->elements); // J = S . s^-1
    memcpy(Jinv->elements, J->elements, sizeof(double)*dim*dim);
    if (!functional_matinv(dim, Jinv->elements)) return false;
    
    elref->flags |= ELREF_HASJACOBIAN;
    return true;
}

static value integral_jacobian(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_JACOBIAN, JACOBIAN_FUNCTION)) return MORPHO_NIL;
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, JACOBIAN_FUNCTION);
    
    if (!(elref->flags & ELREF_HASJACOBIAN) && !integral_evaluatejacobian(v)) return MORPHO_NIL;
    return MORPHO_OBJECT(elref->jacobian);
}

static value integral_invjacobian(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_INVJ, INVJACOBIAN_FUNCTION)) return MORPHO_NIL;
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, INVJACOBIAN_FUNCTION);
    
    if (!(elref->flags & ELREF_HASJACOBIAN) && !integral_evaluatejacobian(v)) return MORPHO_NIL;
    return MORPHO_OBJECT(elref->invjacobian);
}

/* ----------------------------------------------
 * Generic integral support functions
 * ---------------------------------------------- */

static value integral_functionproperty;
static value integral_referenceproperty;
static value integral_weightbyrefproperty;
static value integral_methodproperty;
static value integral_optimizeproperty;

/** Prepares an integral reference */
bool integral_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, integralref *ref) {
    bool success=false;
    value func=MORPHO_NIL;
    value mref=MORPHO_NIL;
    value wtbyref=MORPHO_NIL;
    value field=MORPHO_NIL;
    value method=MORPHO_NIL;
    value optimize=MORPHO_NIL;
    ref->nfields=0;
    ref->method=MORPHO_NIL;
    ref->mref=NULL;
    ref->g=g;
    ref->weightbyref=false;
    ref->optimize=true;
    ref->uses=0;

    if (objectinstance_getpropertyinterned(self, integral_functionproperty, &func) &&
        MORPHO_ISCALLABLE(func)) {
        ref->integrand=func;
        success=true;
    }
    if (objectinstance_getpropertyinterned(self, integral_referenceproperty, &mref) &&
        MORPHO_ISMESH(mref)) {
        ref->mref=MORPHO_GETMESH(mref);
    }
    if (objectinstance_getpropertyinterned(self, integral_weightbyrefproperty, &wtbyref)) {
        ref->weightbyref=!morpho_isfalse(wtbyref);
    }
    if (objectinstance_getpropertyinterned(self, integral_methodproperty, &method)) {
        ref->method=method;
    }
    if (objectinstance_getpropertyinterned(self, integral_optimizeproperty, &optimize) &&
        MORPHO_ISBOOL(optimize)) {
        ref->optimize=MORPHO_GETBOOLVALUE(optimize);
    }
    if (objectinstance_getpropertyinterned(self, functional_fieldproperty, &field) &&
        MORPHO_ISLIST(field)) {
        objectlist *list = MORPHO_GETLIST(field);
        ref->nfields=list->val.count;
        ref->fields=list->val.data;
        ref->originalfields=list->val.data;
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

/** Free cached field gradients and hessians for an element */
static void integral_freegradhess(int nfields, value *qgrad, value *qhess) {
    for (int i=0; i<nfields; i++) {
        if (MORPHO_ISLIST(qgrad[i])) {
            objectlist *l = MORPHO_GETLIST(qgrad[i]);
            for (int j=0; j<l->val.count; j++) morpho_freeobject(l->val.data[j]);
        }
        morpho_freeobject(qgrad[i]);
        if (MORPHO_ISLIST(qhess[i])) {
            objectlist *l = MORPHO_GETLIST(qhess[i]);
            for (int j=0; j<l->val.count; j++) morpho_freeobject(l->val.data[j]);
        }
        morpho_freeobject(qhess[i]);
    }
}

/** Ensure quantity.vals and findx can hold at least n entries. */
static bool _integral_ensurequantityvals(quantity *q, int n) {
    if (q->capacity>=n) return true;
    value *vals=MORPHO_REALLOC(q->vals, sizeof(value)*n);
    if (!vals) return false;
    fieldindx *findx=MORPHO_REALLOC(q->findx, sizeof(fieldindx)*n);
    if (!findx) { q->vals=vals; return false; } /* vals kept; capacity unchanged */
    q->vals=vals;
    q->findx=findx;
    q->capacity=n;
    return true;
}

/** Prepares quantity list. Prefers to reuse existing buffers if possible. */
bool integral_preparequantities(integralref *iref, int nv, int *vid, quantity *quantities) {
    for (int k=0; k<iref->nfields; k++) {
        objectfield *f=MORPHO_GETFIELD(iref->fields[k]);
        
        if (MORPHO_ISFESPACE(f->fnspc)) {
            fespace *disc=MORPHO_GETFESPACE(f->fnspc)->fespace;
            if (nv-1<disc->grade) {
                if (!fespace_lower(disc, nv-1, &disc)) return false;
            }
            
            quantities[k].nnodes=disc->nnodes;
            quantities[k].ifn=disc->ifn;
            if (!_integral_ensurequantityvals(&quantities[k], disc->nnodes)) return false;
            if (!fespace_doftofieldindx(f, disc, nv, vid, quantities[k].findx)) return false;
            
            for (int i=0; i<disc->nnodes; i++) {
                int dof;
                fieldindx *fx=&quantities[k].findx[i];
                if (!field_getindex(f, fx->g, fx->id, fx->indx, &dof)) return false;
                if (!field_getelementwithindex(f, dof, &quantities[k].vals[i])) return false;
            }
        } else {
            quantities[k].nnodes=nv;
            quantities[k].ifn=NULL;
            if (!_integral_ensurequantityvals(&quantities[k], nv)) return false;
            for (unsigned int i=0; i<nv; i++) {
                quantities[k].findx[i]=(fieldindx){ .g=MESH_GRADE_VERTEX, .id=vid[i], .indx=0 };
                if (!field_getelement(f, MESH_GRADE_VERTEX, vid[i], 0, &quantities[k].vals[i])) return false;
            }
        }
    }
    return true;
}

/** Clears a list of quantities */
void integral_clearquantities(int nq, quantity *quantities) {
    for (int k=0; k<nq; k++) {
        if (quantities[k].vals) MORPHO_FREE(quantities[k].vals);
        if (quantities[k].findx) MORPHO_FREE(quantities[k].findx);
    }
    memset(quantities, 0, sizeof(quantity)*nq);
}

bool integral_integrandfn(unsigned int dim, double *t, double *x, unsigned int nquantity, value *quantity, void *ref, unsigned int nout, double *fout) {
    objectintegralelementref *elref = ref;
    integralref *iref = elref->iref;
    vm *v = elref->v;
    objectmatrix posn = MORPHO_STATICMATRIX(x, dim, 1);
    value args[nquantity+1], out;

    if (nout!=1) return false;

    // The integrand function is called with the position and then interpolated quantities.
    args[0]=MORPHO_OBJECT(&posn);
    for (unsigned int i=0; i<nquantity; i++) args[i+1]=quantity[i];
    
    elref->lambda=t;
    elref->posn=x;
    elref->qinterpolated=quantity;
    
    if (morpho_call(v, iref->integrand, nquantity+1, args, &out)) {
        morpho_valuetofloat(out, fout);
        return true;
    }

    return false;
}

/** Bind elref to an element, compute its measure, and fill vertex coordinate pointers. */
static bool integral_bindelement(vm *v, objectmesh *mesh, elementid id, int nv, int *vid,
                                 integralref *iref, objectintegralelementref *elref, double **x) {
    _integral_bindelref(elref, mesh, iref->g, id, nv, vid, x, iref);
    if (!functional_elementsize(v, iref->weightbyref ? iref->mref : mesh, iref->g, id, nv, vid, &elref->elementsize))
        return false;
    for (int i=0; i<nv; i++) mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i]);
    return true;
}

/** Claim elref->quantities or qlocal and prepare DOF values/findx. If *localq, caller must detach. */
static bool integral_prepareelementquantities(integralref *iref, objectintegralelementref *elref,
                                              int nv, int *vid, quantity *qlocal,
                                              quantity **quantities, bool *localq) {
    *quantities=elref->quantities ? elref->quantities : qlocal;
    *localq=(*quantities==qlocal);
    if (*localq) {
        memset(qlocal, 0, sizeof(quantity)*(size_t)(iref->nfields+1));
        elref->quantities=*quantities;
    }
    return integral_preparequantities(iref, nv, vid, *quantities);
}

/** Release a stack-local quantities buffer claimed by prepareelementquantities. */
static void integral_detachquantities(objectintegralelementref *elref, int nfields, quantity *quantities, bool localq) {
    if (!localq) return;
    elref->quantities=NULL;
    integral_clearquantities(nfields, quantities);
}

/** Configure an integrator from an optional method dictionary. Nil method uses the default rule. */
static bool integral_configureintegrator(integrator *integ, error *err, grade g, value method) {
    if (MORPHO_ISDICTIONARY(method)) {
        return integrator_configurewithdictionary(integ, err, g, MORPHO_GETDICTIONARY(method));
    }
    return integrator_configure(integ, err, true, g, -1, NULL);
}

/** Integrate a callable over elements of the grade stored in the integral ref */
bool integral_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    integralref *iref = (integralref *) ref;
    grade g = iref->g;
    double *x[nv];
    bool success=false;
    objectintegralelementref stackelref;
    objectintegralelementref *elref = integral_getelementref(v);
    bool persistent = (elref && (elref->flags & ELREF_PERSISTENT));
    value qgrad_local[iref->nfields+1], qhess_local[iref->nfields+1];
    quantity quantities_local[iref->nfields+1];
    quantity *quantities=NULL;
    bool localquantities=false;
    
    if (!persistent) {
        _integral_initelref(&stackelref);
        elref = &stackelref;
        _integral_nilvalues(qgrad_local, iref->nfields);
        _integral_nilvalues(qhess_local, iref->nfields);
        elref->qgrad=qgrad_local;
        elref->qhess=qhess_local;
        vm_settlvar(v, elementhandle, MORPHO_OBJECT(elref));
    }
    
    elref->v=v;
    if (!integral_bindelement(v, mesh, id, nv, vid, iref, elref, x)) goto integral_integrand_cleanup;

    if (!integral_prepareelementquantities(iref, elref, nv, vid, quantities_local, &quantities, &localquantities))
        goto integral_integrand_cleanup;

    if (elref->flags & ELREF_CONFIGURED) {
        success=integrator_integrate(&elref->integ, integral_integrandfn, mesh->dim, x, iref->nfields, quantities, elref, 1, out);
    } else {
        double err;
        objectdictionary *method=MORPHO_ISDICTIONARY(iref->method) ? MORPHO_GETDICTIONARY(iref->method) : NULL;
        success=integrate(integral_integrandfn, method, morpho_geterror(v), mesh->dim, g, x, iref->nfields, quantities, elref, out, &err);
    }

    if (success) *out *= elref->elementsize;

integral_integrand_cleanup:
    if (!persistent) vm_settlvar(v, elementhandle, MORPHO_NIL);
    integral_detachquantities(elref, iref->nfields, quantities, localquantities);
    if (!persistent) {
        integral_releasegeometry(elref);
        integral_freegradhess(iref->nfields, elref->qgrad, elref->qhess);
    }
    
    return success;
}

void *_integral_zalloc(int n, size_t size) {
    size_t bytes=(size_t) n*size;
    void *p=MORPHO_MALLOC(bytes);
    if (p) memset(p, 0, bytes);
    return p;
}

static bool integral_taskstart(vm *v, functional_mapinfo *info) {
    integralref *iref=(integralref *) info->ref;
    objectintegralelementref *elref=NULL;
    
    if (integral_contextactive(v)) MORPHO_FAIL(v, INTEGRAL_NESTED);
    
    elref=(objectintegralelementref *) object_new(sizeof(objectintegralelementref), OBJECT_INTEGRALELEMENTREF);
    if (!elref) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    
    _integral_initelref(elref);
    elref->v=v;
    elref->iref=iref;
    integrator_init(&elref->integ);
    elref->flags |= ELREF_PERSISTENT | ELREF_HASINTEG;

    if (!integral_configureintegrator(&elref->integ, morpho_geterror(v), info->g, iref->method)) goto integral_taskstart_cleanup;
    elref->flags |= ELREF_CONFIGURED;

    elref->nfields=iref->nfields;
    if (iref->nfields>0) {
        elref->qgrad=_integral_zalloc(iref->nfields, sizeof(value));
        elref->qhess=_integral_zalloc(iref->nfields, sizeof(value));
        elref->quantities=_integral_zalloc(iref->nfields, sizeof(quantity));
        if (!elref->qgrad || !elref->qhess || !elref->quantities) goto integral_taskstart_cleanup;
    }
    
    vm_settlvar(v, elementhandle, MORPHO_OBJECT(elref));
    return true;

integral_taskstart_cleanup:
    if (!morpho_checkerror(morpho_geterror(v))) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    integral_freeelref(elref);
    return false;
}

static void integral_taskend(vm *v, functional_mapinfo *info) {
    objectintegralelementref *elref=integral_getelementref(v);
    if (!elref || !(elref->flags & ELREF_PERSISTENT)) return;
    vm_settlvar(v, elementhandle, MORPHO_NIL);
    integral_freeelref(elref);
}

/** One-pass shape gradient: I_ref * d(measure)/dx. */
static bool integral_gradient_fq(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    integralref *iref=(integralref *) ref;
    objectintegralelementref *elref=integral_getelementref(v);
    double E=0.0;

    if (iref->weightbyref) return true;
    if (elref) elref->allowed=INTEGRAL_USES_NONE;
    if (!integral_integrand(v, mesh, id, nv, vid, ref, &E)) return false;

    elref=integral_getelementref(v);
    if (!elref) return false;
    if (elref->elementsize<=MORPHO_EPS) return true;

    return functional_elementgradient_scale(v, mesh, iref->g, id, nv, vid, frc, E/elref->elementsize);
}

/** True if integrand is f(q)-only (no x, no Integral specials). */
static bool integral_checkfieldonly(unsigned uses) {
    return uses==INTEGRAL_USES_NONE;
}

static bool integral_field_hasgradient(objectfield *field) {
    if (!field || !MORPHO_ISFESPACE(field->fnspc)) return false;
    return FESPACE_HASGRADIENT(MORPHO_GETFESPACE(field->fnspc)->fespace);
}

/** Local fieldgradient: hess/jumpdn force global FD. Static grad() needs a gradient on the target. */
static bool integral_checklocalfieldgrad(unsigned uses, objectfield *field) {
    if (uses & ~(INTEGRAL_USES_X | INTEGRAL_FIELDGRAD_ALLOWED)) return false;
    if ((uses & INTEGRAL_USES_GRAD) && !integral_field_hasgradient(field)) return false;
    return true;
}

/** True if the shape gradient can be assembled from x, tangent() and normal(). */
static bool integral_checklocalshapegrad(unsigned uses, grade g, unsigned int dim) {
    return uses && !(uses & ~INTEGRAL_SHAPEGRAD_ALLOWED) &&
           !((uses & INTEGRAL_USES_NORMAL) && (g!=2 || dim!=3));
}

typedef struct {
    integralref *iref;
    objectfield *field;
    int ifield, ncomp;
    unsigned allowed;
} integral_fq_mapref;

static void _integral_fq_mapref(integral_fq_mapref *mref, integralref *iref, objectfield *field, int ifield, unsigned allowed) {
    mref->iref=iref;
    mref->field=field;
    mref->ifield=ifield;
    mref->ncomp=(int) field->psize;
    mref->allowed=allowed;
}

typedef struct {
    integral_fq_mapref *mref;
    objectintegralelementref *elref;
    int nnodes;
} integral_fq_local;

typedef struct {
    integral_fq_mapref local;
    functional_numericalfieldgradientref numerical;
    objectfield *fieldclone;
    functional_mapinfo *info;
    bool clone;
} integral_fieldgradient_taskref;

/* ----------------------------------------------
 * Local derivatives at a quadrature point
 * ---------------------------------------------- */

/** Central difference of the integrand wrt *p. Restores *p.
 * @details If qfloat is non-NULL, q is a Morpho Float and *qfloat is rewritten
 * after each step so the integrand sees the new value. */
static bool _integral_centdiff(unsigned int dim, double *lambda, double *x, unsigned int nq, value *quantity,
                              objectintegralelementref *elref, double *p, value *qfloat, double *df) {
    double f0=*p, eps=functional_fdstepsize(f0, 1), f[2];
    bool ok=true;
    for (int s=0; s<2 && ok; s++) {
        *p=f0+(s ? -eps : eps);
        if (qfloat) *qfloat=MORPHO_FLOAT(*p);
        ok=integral_integrandfn(dim, lambda, x, nq, quantity, elref, 1, &f[s]);
    }
    *p=f0;
    if (qfloat) *qfloat=MORPHO_FLOAT(f0);
    if (ok) *df=(f[0]-f[1])/(2.0*eps);
    return ok;
}

/** Central difference of the integrand wrt n entries of p. */
static bool integral_dfdvec(objectintegralelementref *elref, unsigned nq, value *quantity, double *p, int n, double *df) {
    unsigned dim=elref->mesh->dim;
    for (int i=0; i<n; i++) {
        if (!_integral_centdiff(dim, elref->lambda, elref->posn, nq, quantity, elref, &p[i], NULL, &df[i])) return false;
    }
    return true;
}

/** Estimate dfdq by central difference of the interpolated field value. */
static bool integral_fq_dfdq(unsigned int dim, double *lambda, double *x, unsigned int nq, value *quantity, objectintegralelementref *elref, value *qval, int ncomp, double *dfdq) {
    double *p, tmp; objectmatrix *m=NULL;

    if (MORPHO_ISFLOAT(*qval)) {
        if (ncomp!=1) return false;
        tmp=MORPHO_GETFLOATVALUE(*qval); p=&tmp;
    } else if (MORPHO_ISMATRIX(*qval)) {
        m=MORPHO_GETMATRIX(*qval);
        if ((int) m->nels!=ncomp) return false;
        p=m->elements;
    } else return false;

    for (int c=0; c<ncomp; c++) {
        if (!_integral_centdiff(dim, lambda, x, nq, quantity, elref, &p[c], m ? NULL : qval, &dfdq[c])) return false;
    }
    return true;
}

/** Flatten a grad() Matrix (or List of Matrices) to double pointers. */
static int integral_grad_ptrs(value gval, double **ptrs, int maxn) {
    int n=0;
    if (MORPHO_ISMATRIX(gval)) {
        objectmatrix *m=MORPHO_GETMATRIX(gval);
        for (unsigned int i=0; i<m->nels && n<maxn; i++) ptrs[n++]=&m->elements[i];
        return n;
    }
    if (MORPHO_ISLIST(gval)) {
        objectlist *lst=MORPHO_GETLIST(gval);
        for (unsigned int i=0; i<list_length(lst); i++) {
            value el;
            if (!list_getelement(lst, (int) i, &el) || !MORPHO_ISMATRIX(el)) return -1;
            objectmatrix *m=MORPHO_GETMATRIX(el);
            for (unsigned int k=0; k<m->nels && n<maxn; k++) ptrs[n++]=&m->elements[k];
        }
        return n;
    }
    return -1;
}

/** Estimate dfdgrad by central difference of the cached grad() result.
 * @details freeze_grad stops EvaluateGradient from overwriting the perturbation. */
static bool integral_fq_dfdgrad(unsigned int dim, double *lambda, double *x, unsigned int nq, value *quantity, objectintegralelementref *elref, int ifield, int ngrad, double *dfdg) {
    double *ptrs[ngrad];

    if (!elref || integral_grad_ptrs(elref->qgrad[ifield], ptrs, ngrad)!=ngrad) return false;
    for (int i=0; i<ngrad; i++) {
        if (!_integral_centdiff(dim, lambda, x, nq, quantity, elref, ptrs[i], NULL, &dfdg[i])) return false;
    }
    return true;
}

/** Shape-function gradients in physical coordinates.
 * @details out[k*nnodes+a] is dNa/dx_k. */
static bool integral_physical_gradNa(vm *v, objectintegralelementref *elref, objectfield *fld, int nnodes, double *out) {
    if (!elref || !fld || !MORPHO_ISFESPACE(fld->fnspc)) return false;
    fespace *disc=MORPHO_GETFESPACE(fld->fnspc)->fespace;
    if (!FESPACE_HASGRADIENT(disc) || disc->nnodes!=nnodes) return false;
    if (!integral_ensureinvj(v, elref)) return false;

    double gdata[nnodes * elref->g];
    objectmatrix gmat=MORPHO_STATICMATRIX(gdata, nnodes, elref->g);
    fespace_gradient(disc, elref->lambda, &gmat);
    functional_matmul(nnodes, elref->g, elref->mesh->dim, gdata, elref->invj->elements, out);
    return true;
}

/** Integrand for local fieldgradient assembly: (dfdq) Na [+ (dfdgrad) · ∇Na]. */
static bool integral_fq_vector_integrand(unsigned int dim, double *lambda, double *x, unsigned int nquantity, value *quantity, void *ref, unsigned int nout, double *fout) {
    integral_fq_local *s=(integral_fq_local *) ref;
    integral_fq_mapref *mref=s->mref;
    objectintegralelementref *elref=s->elref;
    vm *v=elref->v;
    int nnodes=s->nnodes, ncomp=mref->ncomp, ifield=mref->ifield, ngrad=(int) dim*ncomp;
    double Na[nnodes], dfdq[ncomp];

    if (!elref || (unsigned)(nnodes*ncomp)!=nout) return false;
    if (elref->quantities && elref->quantities[ifield].ifn) (elref->quantities[ifield].ifn)(lambda, Na);
    else for (int a=0; a<nnodes; a++) Na[a]=lambda[a];
    elref->lambda=lambda; elref->posn=x; elref->qinterpolated=quantity;

    elref->target_grad_used=false;
    if (!integral_fq_dfdq(dim, lambda, x, nquantity, quantity, elref, &quantity[ifield], ncomp, dfdq)) return false;

    if (!elref->target_grad_used) {
        for (int a=0; a<nnodes; a++) for (int c=0; c<ncomp; c++) fout[a*ncomp+c]=dfdq[c]*Na[a];
        return true;
    }

    elref->freeze_grad=ifield;
    double dfdg[ngrad], gNa[nnodes*(int) dim];
    bool ok=integral_fq_dfdgrad(dim, lambda, x, nquantity, quantity, elref, ifield, ngrad, dfdg);
    elref->freeze_grad=-1;
    if (!ok) return false;
    if (!integral_physical_gradNa(v, elref, mref->field, nnodes, gNa)) return false;

    for (int a=0; a<nnodes; a++) {
        for (int c=0; c<ncomp; c++) {
            double g=dfdq[c]*Na[a];
            for (unsigned int k=0; k<dim; k++) g+=dfdg[k*ncomp+c]*gNa[k*nnodes+a];
            fout[a*ncomp+c]=g;
        }
    }
    return true;
}

/** True if local fieldgradient has no more derivative directions than nodal FD. */
static bool integral_fieldgradient_preferlocal(bool usesgrad, unsigned int dim, int nnodes) {
    return 1u+(usesgrad?dim:0u)<=(unsigned) nnodes;
}

/** Add this element's fieldgradient into out via the local integrand, or numerical FD of the scalar integral if try refines or FD is cheaper. */
static bool integral_fieldgradient_fq_element(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out) {
    integral_fieldgradient_taskref *tref=ref;
    integral_fq_mapref *mref=&tref->local;
    integralref *iref=mref->iref;
    objectfield *grad=(objectfield *) out;
    objectintegralelementref *elref=integral_getelementref(v);
    double *x[nv];
    quantity qlocal[iref->nfields+1], *quantities=NULL;
    bool localq=false, ok=false;
    int ifield=mref->ifield, ncomp=mref->ncomp;

    if (!elref || !(elref->flags & ELREF_CONFIGURED)) return false;

    elref->v=v;
    elref->allowed=mref->allowed;
    elref->target_field=mref->ifield;
    if (!integral_bindelement(v, mesh, id, nv, vid, iref, elref, x)) return false;
    if (elref->elementsize<=MORPHO_EPS) return true;
    if (!integral_prepareelementquantities(iref, elref, nv, vid, qlocal, &quantities, &localq)) {
        integral_detachquantities(elref, iref->nfields, quantities, localq);
        return false;
    }

    int nnodes=quantities[ifield].nnodes;
    unsigned int nout=(unsigned)(nnodes*ncomp);
    double local[nout];
    integral_fq_local s={ .mref=mref, .elref=elref, .nnodes=nnodes };
    bool numerical=false;

    if (!elref->integ.adapt) {
        ok=integrator_integrate(&elref->integ, integral_fq_vector_integrand,
                          mesh->dim, x, iref->nfields, quantities, &s, nout, local);
    } else {
        double fval=0.0;
        elref->target_grad_used=false;
        integratortrystatus st=integrator_try(&elref->integ, integral_integrandfn,
                          mesh->dim, x, iref->nfields, quantities, elref, 1, &fval);
        if (st==INTEGRATOR_TRY_FAILED) {
            integral_detachquantities(elref, iref->nfields, quantities, localq);
            return false;
        }
        if (st==INTEGRATOR_TRY_ACCEPTED &&
            integral_fieldgradient_preferlocal(elref->target_grad_used, mesh->dim, nnodes)) {
            ok=integrator_apply(&elref->integ, integral_fq_vector_integrand, &s, nout, local);
        } else numerical=true;
    }

    if (numerical) {
        integral_detachquantities(elref, iref->nfields, quantities, localq);
        if (!tref->numerical.field &&
            !functional_preparenumericalfieldgradientref(v, tref->info, tref->clone, &tref->numerical, &tref->fieldclone)) return false;
        return functional_numericalfieldgradientmapfn(v, mesh, id, nv, vid, &tref->numerical, out);
    }

    for (int a=0; ok && a<nnodes; a++) {
        fieldindx *fx=&quantities[ifield].findx[a];
        unsigned int nentries=0; double *gentry=NULL;
        ok=field_getelementaslist(grad, fx->g, fx->id, fx->indx, &nentries, &gentry) &&
           ncomp<=(int) nentries;
        if (ok) for (int c=0; c<ncomp; c++) functional_accum(&gentry[c], elref->elementsize*local[a*ncomp+c]);
    }

    integral_detachquantities(elref, iref->nfields, quantities, localq);
    return ok;
}

/** Map fieldgradient with the local integrand on an accepted formula, or numerical FD if that formula is insufficient. */
static bool integral_mapfieldgradient(vm *v, functional_mapinfo *info, value *out) {
    int success=false, ntask=functional_ntasks(info), ifield=-1;
    functional_task task[ntask];
    integral_fieldgradient_taskref tref[ntask];
    integralref *iref=(integralref *) info->ref;
    varray_elementid imageids;
    objectfield *new=NULL;
    unsigned allowed=INTEGRAL_FIELDGRAD_ALLOWED;

    memset(tref, 0, sizeof(tref));

    for (int i=0; i<iref->nfields; i++)
        if (MORPHO_ISFIELD(iref->fields[i]) && MORPHO_GETFIELD(iref->fields[i])==info->field) { ifield=i; break; }
    if (ifield<0) return false;

    varray_elementidinit(&imageids);
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    new=object_newfield(info->mesh, info->field->prototype, info->field->fnspc, info->field->dof);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto cleanup; }
    field_zero(new);

    for (int i=0; i<ntask; i++) {
        _integral_fq_mapref(&tref[i].local, iref, info->field, ifield, allowed);
        tref[i].info=info;
        tref[i].clone=(ntask>1);
        task[i].ref=&tref[i];
        task[i].mapfn=integral_fieldgradient_fq_element;
        task[i].result=new;
    }
    if (!functional_map(ntask, task)) goto cleanup;
    success=true; *out=MORPHO_OBJECT(new);
cleanup:
    for (int i=0; i<ntask; i++) {
        functional_clearnumericalfieldgradientref(info, &tref[i].numerical, tref[i].fieldclone);
    }
    if (!success && new) object_free((object *) new);
    functional_cleanuptasks(v, ntask, task, &imageids);
    return success;
}

/** Add (I-uu^T) df / len onto dest (and optionally -dest2). */
static void integral_unit_project(unsigned int n, double *u, double *df, double len, double *dest, double *dest2) {
    if (len<=MORPHO_EPS) return;
    double udot=functional_vecdot(n, u, df);
    for (unsigned int k=0; k<n; k++) {
        double pk=(df[k]-u[k]*udot)/len;
        dest[k]+=pk;
        if (dest2) dest2[k]-=pk;
    }
}

/** Add lambda_a * df/dx onto each vertex. */
static bool integral_pullback_x(objectintegralelementref *elref, unsigned nq, value *quantity, double *lambda, double *x, double *fout) {
    unsigned dim=elref->mesh->dim;
    double df[dim];
    if (!integral_dfdvec(elref, nq, quantity, x, (int) dim, df)) return false;
    for (int a=0; a<elref->nv; a++) for (unsigned int k=0; k<dim; k++) fout[a*dim+k]+=lambda[a]*df[k];
    return true;
}

/** Add the unit-tangent force onto the two line vertices. */
static bool integral_pullback_tangent(objectintegralelementref *elref, unsigned nq, value *quantity, double *fout) {
    unsigned dim=elref->mesh->dim;
    double df[dim], e[dim];
    if (!integral_dfdvec(elref, nq, quantity, elref->tangent->elements, (int) dim, df)) return false;
    functional_vecsub(dim, elref->vertexposn[1], elref->vertexposn[0], e);
    integral_unit_project(dim, elref->tangent->elements, df, functional_vecnorm(dim, e), fout+dim, fout);
    return true;
}

/** Add the unit-normal force onto the three vertices of a triangle in 3-space. */
static bool integral_pullback_normal(objectintegralelementref *elref, unsigned nq, value *quantity, double *fout) {
    double df[3], s0[3]={0,0,0}, s1[3]={0,0,0}, u[3], w[3]={0,0,0};
    if (!integral_dfdvec(elref, nq, quantity, elref->normal->elements, 3, df)) return false;
    functional_vecsub(3, elref->vertexposn[1], elref->vertexposn[0], s0);
    functional_vecsub(3, elref->vertexposn[2], elref->vertexposn[1], s1);
    functional_veccross(s0, s1, u);
    integral_unit_project(3, elref->normal->elements, df, functional_vecnorm(3, u), w, NULL);
    functional_veccross(s1, w, u); for (int k=0; k<3; k++) fout[k]-=u[k];
    functional_veccross(s0, w, u); for (int k=0; k<3; k++) fout[6+k]-=u[k];
    functional_vecadd(3, s0, s1, s0);
    functional_veccross(s0, w, u); for (int k=0; k<3; k++) fout[3+k]+=u[k];
    return true;
}

/** Vertex force density from df/dx, df/dt and df/dn. */
static bool integral_xgeom_integrand(unsigned int dim, double *lambda, double *x, unsigned int nquantity, value *quantity, void *ref, unsigned int nout, double *fout) {
    objectintegralelementref *elref=ref;
    double f;

    if (!elref || nout!=(unsigned)(elref->nv*(int) dim)) return false;
    memset(fout, 0, nout*sizeof(double));
    elref->lambda=lambda; elref->posn=x; elref->qinterpolated=quantity;
    if (!integral_integrandfn(dim, lambda, x, nquantity, quantity, elref, 1, &f)) return false;

    if ((elref->iref->uses & INTEGRAL_USES_X) &&
        !integral_pullback_x(elref, nquantity, quantity, lambda, x, fout)) return false;
    if ((elref->flags & ELREF_HASTANGENT) &&
        !integral_pullback_tangent(elref, nquantity, quantity, fout)) return false;
    if ((elref->flags & ELREF_HASNORMAL) &&
        !integral_pullback_normal(elref, nquantity, quantity, fout)) return false;
    return true;
}

/** Copy mesh vertices into elref->xgeom and retarget x[] (and vertexposn) at those columns. */
static bool integral_takexgeom(objectintegralelementref *elref, int nv, double **x) {
    unsigned dim=elref->mesh->dim;
    if (!integral_ensurematrix(&elref->xgeom, (int) dim, nv)) return false;
    for (int i=0; i<nv; i++) {
        if (matrix_setcolumnptr(elref->xgeom, i, x[i])!=LINALGERR_OK ||
            matrix_getcolumnptr(elref->xgeom, i, &x[i])!=LINALGERR_OK) return false;
    }
    elref->vertexposn=x;
    return true;
}

/** Local FD of I(X) with J held fixed; perturbs elref->xgeom, not mesh->vert. */
static bool integral_numericalgrad_xgeom(objectintegralelementref *elref, double **x, quantity *quantities, double J, objectmatrix *frc) {
    unsigned dim=elref->mesh->dim;
    int nfields=elref->iref->nfields;
    double f[2];
    bool ok=true;

    for (int a=0; ok && a<elref->nv; a++) {
        for (unsigned int k=0; ok && k<dim; k++) {
            double x0=x[a][k], eps=functional_fdstepsize(x0, 1);
            for (int s=0; s<2 && ok; s++) { // Loop +/- eps
                x[a][k]=x0+(s ? -eps : eps);
                integral_resetgeometryflags(elref);
                ok=integrator_integrate(&elref->integ, integral_integrandfn, dim, x, nfields, quantities, elref, 1, f+s);
            }
            x[a][k]=x0;
            if (ok) ok=functional_addtoelement(frc, (MatrixIdx_t) k, elref->vid[a], J*(f[0]-f[1])/(2.0*eps));
        }
    }
    integral_resetgeometryflags(elref);
    return ok;
}

/** Shape gradient from x, tangent() and normal(): I_ref * d(measure)/dx plus J * ∫ G. */
static bool integral_gradient_xgeom(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    integralref *iref=ref;
    objectintegralelementref *elref=integral_getelementref(v);
    double *x[nv], iref_f=0.0;
    quantity qlocal[iref->nfields+1], *quantities=NULL;
    bool localq=false, ok=false;

    if (!elref || !(elref->flags & ELREF_CONFIGURED)) return false;
    elref->v=v;
    elref->allowed=iref->uses;
    if (!integral_bindelement(v, mesh, id, nv, vid, iref, elref, x)) return false;
    if (elref->elementsize<=MORPHO_EPS) return true;
    if (!integral_prepareelementquantities(iref, elref, nv, vid, qlocal, &quantities, &localq)) goto done;

    integratortrystatus st=integrator_try(&elref->integ, integral_integrandfn, mesh->dim, x, iref->nfields, quantities, elref, 1, &iref_f);
    if (st==INTEGRATOR_TRY_FAILED) goto done;

    if (st==INTEGRATOR_TRY_ACCEPTED) { // Analytical
        double local[nv*mesh->dim];
        objectmatrix G=MORPHO_STATICMATRIX(local, mesh->dim, nv);
        if (!integrator_apply(&elref->integ, integral_xgeom_integrand, elref, (unsigned) G.nels, G.elements)) goto done;
        for (int a=0; a<nv; a++)
            if (!functional_addtocolumn(frc, vid[a], elref->elementsize, G.elements+a*G.nrows)) goto done;
    } else { // Refine: finite difference
        if (!integrator_refine(&elref->integ) ||
            !integral_takexgeom(elref, nv, x) ||
            !integral_numericalgrad_xgeom(elref, x, quantities, elref->elementsize, frc)) goto done;
    }

    if (!iref->weightbyref &&
        !functional_elementgradient_scale(v, mesh, iref->g, id, nv, vid, frc, iref_f)) goto done;
    ok=true;
done:
    integral_detachquantities(elref, iref->nfields, quantities, localq);
    return ok;
}

/** Specialize a Metafunction kernel from known quadrature argument types.
 *  argtypes are Matrix for the position, then each Field's prototype type. */
static bool integral_reducemetafunction(vm *v, value mf, int nfields, value *fields, value *out) {
    value argtypes[nfields+1];
    value matrixclass;
    error err;

    if (!MORPHO_ISMETAFUNCTION(mf)) return false;

    matrixclass=builtin_findclassfromcstring(MATRIX_CLASSNAME);
    if (MORPHO_ISNIL(matrixclass)) return false;
    argtypes[0]=matrixclass;

    for (int i=0; i<nfields; i++) {
        objectfield *fld;
        value prototype;

        if (!MORPHO_ISFIELD(fields[i])) return false;
        fld=MORPHO_GETFIELD(fields[i]);
        prototype=fld->prototype;
        if (MORPHO_ISNIL(prototype)) {
            argtypes[i+1]=builtin_findclassfromcstring(FLOAT_CLASSNAME);
            if (MORPHO_ISNIL(argtypes[i+1])) return false;
        } else if (!value_type(prototype, &argtypes[i+1])) {
            return false;
        }
    }

    error_init(&err);
    if (!metafunction_reduce(MORPHO_GETMETAFUNCTION(mf), nfields+1, argtypes, &err, out)) {
        morpho_error(v, &err);
        error_clear(&err);
        return false;
    }
    error_clear(&err);
    return true;
}

/** Prepare stored Fields' FE spaces once per map. */
static bool integral_startfn(vm *v, functional_mapinfo *info) {
    integralref *ref = (integralref *) info->ref;
    return functional_preparefieldlist(v, ref->fields, ref->nfields, info->g);
}

/** Fill integralref and attach it to the map. */
static bool _Integral_bindref(vm *v, objectinstance *self, functional_mapinfo *info, integralref *ref) {
    grade g=0;

    if (integral_contextactive(v)) MORPHO_FAIL(v, INTEGRAL_NESTED);

    if (!functional_readgrade(self, &g) ||
        !integral_prepareref(self, info->mesh, (info->g < 0 ? g : info->g), info->sel, ref)) {
        MORPHO_FAIL(v, INTEGRAL_ARGS);
    }
    if (info->g < 0) info->g = g;
    info->ref = ref;
    info->integrand = integral_integrand;
    info->start = integral_startfn;
    info->taskstart = integral_taskstart;
    info->taskend = integral_taskend;
    ref->uses = integral_fnuses(v, ref->integrand);
    return true;
}

/** Select which gradient map to use. */
static functional_mapcallback *Integral_choosegradient(vm *v, objectinstance *self, functional_mapinfo *info) {
    integralref *ref = info->ref;

    if (!_Integral_bindref(v, self, info, ref)) return NULL;

    if (ref->optimize && integral_checkfieldonly(ref->uses)) {
        info->grad=integral_gradient_fq;
        return functional_mapgradient;
    }
    if (ref->optimize &&
        integral_checklocalshapegrad(ref->uses, ref->g, info->mesh->dim)) {
        info->grad=integral_gradient_xgeom;
        return functional_mapgradient;
    }
    return functional_mapnumericalgradient;
}

/** Select which fieldgradient map to use. */
static functional_mapcallback *Integral_choosefieldgradient(vm *v, objectinstance *self, functional_mapinfo *info) {
    integralref *ref = info->ref;

    if (!_Integral_bindref(v, self, info, ref)) return NULL;
    info->cloneref=integral_cloneref;
    info->freeref=integral_freeref;

    if (ref->optimize &&
        integral_checklocalfieldgrad(ref->uses, info->field)) {
        return integral_mapfieldgradient;
    }
    return functional_mapnumericalfieldgradient;
}

FUNCTIONAL_MD_REF_INTEGRAND(Integral, integralref, ref.g)
FUNCTIONAL_MD_REF_TOTAL(Integral, integralref, ref.g)
FUNCTIONAL_MD_REF_CHOOSEGRADIENT(Integral, integralref, ref.g, Integral_choosegradient)
FUNCTIONAL_MD_REF_HESSIAN(Integral, integralref, ref.g, NULL, SYMMETRY_NONE)
FUNCTIONAL_MD_REF_CHOOSEFIELDGRADIENT(Integral, integralref, ref.g, Integral_choosefieldgradient)

/** Initialize a Line/Area/Volume/Jump integral object */
value integral_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    int nparams = -1;
    int nfixed;
    value method=MORPHO_NIL;
    value mref=MORPHO_NIL;
    value wtbyref=MORPHO_NIL;
    value optimize=MORPHO_NIL;

    if (builtin_options(v, nargs, args, &nfixed, 4,
                        integral_methodproperty, &method,
                        integral_referenceproperty, &mref,
                        integral_weightbyrefproperty, &wtbyref,
                        integral_optimizeproperty, &optimize)) {
        if (MORPHO_ISDICTIONARY(method)) {
            objectinstance_setproperty(self, integral_methodproperty, method);
        } else if (!MORPHO_ISNIL(method)) MORPHO_RAISE(v, INTEGRAL_ARGS);

        if (MORPHO_ISMESH(mref)) objectinstance_setproperty(self, integral_referenceproperty, mref);
        if (MORPHO_ISBOOL(wtbyref)) objectinstance_setproperty(self, integral_weightbyrefproperty, wtbyref);
        if (MORPHO_ISBOOL(optimize)) {
            objectinstance_setproperty(self, integral_optimizeproperty, optimize);
        } else if (!MORPHO_ISNIL(optimize)) MORPHO_RAISE(v, INTEGRAL_ARGS);
    } else MORPHO_RAISE(v, INTEGRAL_ARGS);

    if (nfixed>1) {
        /* Remaining arguments should be fields */
        for (unsigned int i=1; i<nfixed; i++) {
            if (!MORPHO_ISFIELD(MORPHO_GETARG(args, i))) MORPHO_RAISE(v, INTEGRAL_ARGS);
        }
    }

    if (nfixed>0) {
        value f = MORPHO_GETARG(args, 0);
        value kernel = f;
        int nfields = nfixed>1 ? nfixed-1 : 0;
        value *fields = nfields>0 ? &MORPHO_GETARG(args, 1) : NULL;

        if (MORPHO_ISMETAFUNCTION(f)) {
            if (!integral_reducemetafunction(v, f, nfields, fields, &kernel)) return MORPHO_NIL;
            nparams=nfixed;
        } else if (morpho_countparameters(f, &nparams)) {
            /* keep kernel = f */
        } else MORPHO_RAISE(v, INTEGRAL_ARGS);

        objectinstance_setproperty(self, integral_functionproperty, kernel);
    }

    if (nparams!=nfixed) MORPHO_RAISE(v, INTEGRAL_ARGS);

    if (nfixed>1) {
        objectlist *list = object_newlist(nfixed-1, & MORPHO_GETARG(args, 1));
        if (list) objectinstance_setproperty(self, functional_fieldproperty, MORPHO_OBJECT(list));
        return morpho_wrapandbind(v, (object *) list);
    }

    return MORPHO_NIL;
}

value integral_initwithgrade(vm *v, int nargs, value *args, grade g) {
    objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), functional_gradeproperty, MORPHO_INTEGER(g));
    return integral_init(v, nargs, args);
}

value LineIntegral_init(vm *v, int nargs, value *args) {
    return integral_initwithgrade(v, nargs, args, MESH_GRADE_LINE);
}

value AreaIntegral_init(vm *v, int nargs, value *args) {
    return integral_initwithgrade(v, nargs, args, MESH_GRADE_AREA);
}

value VolumeIntegral_init(vm *v, int nargs, value *args) {
    return integral_initwithgrade(v, nargs, args, MESH_GRADE_VOLUME);
}

MORPHO_BEGINCLASS(LineIntegral)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(...)", LineIntegral_init, INTEGRAL_INITFLAGS),
FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS, INTEGRAL_ELEMFLAGS),
FUNCTIONAL_MD_TOTAL_METHODS_FLAGS(Integral, INTEGRAL_TOTALFLAGS),
FUNCTIONAL_MD_GRADIENT_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_HESSIAN_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS)
MORPHO_ENDCLASS

MORPHO_BEGINCLASS(AreaIntegral)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(...)", AreaIntegral_init, INTEGRAL_INITFLAGS),
FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS, INTEGRAL_ELEMFLAGS),
FUNCTIONAL_MD_TOTAL_METHODS_FLAGS(Integral, INTEGRAL_TOTALFLAGS),
FUNCTIONAL_MD_GRADIENT_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_HESSIAN_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS)
MORPHO_ENDCLASS

MORPHO_BEGINCLASS(VolumeIntegral)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(...)", VolumeIntegral_init, INTEGRAL_INITFLAGS),
FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS, INTEGRAL_ELEMFLAGS),
FUNCTIONAL_MD_TOTAL_METHODS_FLAGS(Integral, INTEGRAL_TOTALFLAGS),
FUNCTIONAL_MD_GRADIENT_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_HESSIAN_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS)
MORPHO_ENDCLASS

void integral_initialize(void) {
    integral_functionproperty=builtin_internsymbolascstring(INTEGRAL_FUNCTION_PROPERTY);
    integral_referenceproperty=builtin_internsymbolascstring(INTEGRAL_REFERENCE_PROPERTY);
    integral_weightbyrefproperty=builtin_internsymbolascstring(INTEGRAL_WTBYREF_PROPERTY);
    integral_methodproperty=builtin_internsymbolascstring(INTEGRAL_METHOD_PROPERTY);
    integral_optimizeproperty=builtin_internsymbolascstring(INTEGRAL_OPTIMIZE_PROPERTY);

    value objclass = builtin_findclassfromcstring(OBJECT_CLASSNAME);

    builtin_addclass(LINEINTEGRAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(LineIntegral), objclass);
    builtin_addclass(AREAINTEGRAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(AreaIntegral), objclass);
    builtin_addclass(VOLUMEINTEGRAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(VolumeIntegral), objclass);

    morpho_addfunction(ELEMENTID_FUNCTION, "Int ()", integral_elementid, MORPHO_FN_THREADLOCAL | MORPHO_FN_THROWS, NULL);

    integral_addspecial(GRAD_FUNCTION, integral_gradfn, INTEGRAL_USES_GRAD, "Matrix (Float)");
    integral_addspecial(GRAD_FUNCTION, integral_gradfn, INTEGRAL_USES_GRAD, "List (Matrix)");
    integral_addspecial(GRAD_FUNCTION, integral_gradfn, INTEGRAL_USES_GRAD, "(Field)");

    integral_addspecial(HESS_FUNCTION, integral_hessfn, INTEGRAL_USES_HESS, "Matrix (Float)");
    integral_addspecial(HESS_FUNCTION, integral_hessfn, INTEGRAL_USES_HESS, "List (Matrix)");
    integral_addspecial(HESS_FUNCTION, integral_hessfn, INTEGRAL_USES_HESS, "(Field)");

    integral_addspecial(TANGENT_FUNCTION, integral_tangent, INTEGRAL_USES_TANGENT, "Matrix ()");
    integral_addspecial(NORMAL_FUNCTION, integral_normal, INTEGRAL_USES_NORMAL, "Matrix ()");
    integral_addspecial(JACOBIAN_FUNCTION, integral_jacobian, INTEGRAL_USES_JACOBIAN, "Matrix ()");
    integral_addspecial(INVJACOBIAN_FUNCTION, integral_invjacobian, INTEGRAL_USES_INVJ, "Matrix ()");
    integral_addspecial(CGTENSOR_FUNCTION, integral_cgfn, INTEGRAL_USES_CG, "Matrix ()");

    morpho_defineerror(INTEGRAL_ARGS, ERROR_HALT, INTEGRAL_ARGS_MSG);
    morpho_defineerror(INTEGRAL_FLD, ERROR_HALT, INTEGRAL_FLD_MSG);
    morpho_defineerror(INTEGRAL_SPCLFN, ERROR_HALT, INTEGRAL_SPCLFN_MSG);
    morpho_defineerror(INTEGRAL_DFFEVL, ERROR_HALT, INTEGRAL_DFFEVL_MSG);
    morpho_defineerror(INTEGRAL_NESTED, ERROR_HALT, INTEGRAL_NESTED_MSG);
    morpho_defineerror(INTEGRAL_FASTPATH, ERROR_HALT, INTEGRAL_FASTPATH_MSG);

    objectintegralelementreftype=object_addtype(&objectintegralelementrefdefn);
    elementhandle=vm_addtlvar();
}

#endif
