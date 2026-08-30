/** @file jump.c
 *  @author T J Atherton
 *
 *  @brief Jump functional
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
#include <string.h>
#include <math.h>
#include "integrate.h"
#include "integral.h"
#include "jump.h"

/* ----------------------------------------------
 * Jump
 * ---------------------------------------------- */

typedef enum {
    JUMP_STRATEGY_CENTROID_MODE,
    JUMP_STRATEGY_QUADRATURE_MODE
} jumpstrategy;

struct jumpref_s {
    integralref integral;
    grade parentgrade;
    grade interfacegrade;
    objectsparse *interfaceparents;
    objectsparse *parentinterfaces;
    objectsparse *parentvertices;
    jumpstrategy strategy;
};

static bool jump_getadjacentparents(jumpref *ref, elementid interfaceid, int *nparents, int **parents);
static void jump_orderparents(int *parents, elementid *plusid, elementid *minusid);

typedef struct {
    functional_mapinfo *info;
    objectfield *field;
    fespace *disc;
    objectsparse *conn;
    objectsparse *parentvertices;
    jumpref *ref;
} jump_numericalfieldgradientref;

static bool jump_getelementvertices(objectsparse *conn, grade g, elementid id, int *nv, int **vid) {
    if (conn) return sparseccs_getrowindices(&conn->ccs, id, nv, vid);
    if (g==0) {
        *nv=1;
        *vid=&id;
        return true;
    }
    return false;
}

static bool jump_collectparentfieldentries(jump_numericalfieldgradientref *tref, elementid interfaceid, fieldindx *findxout, int *nentries) {
    jumpref *ref=tref->ref;
    int nparents=0, *parents=NULL;
    int plusnv=0, minusnv=0, *plusvid=NULL, *minusvid=NULL;

    *nentries=0;
    if (!jump_getadjacentparents(ref, interfaceid, &nparents, &parents)) return false;
    if (nparents!=2) return true;

    elementid plusid, minusid;
    jump_orderparents(parents, &plusid, &minusid);

    if (!mesh_getconnectivity(tref->parentvertices, plusid, &plusnv, &plusvid)) return false;
    if (!mesh_getconnectivity(tref->parentvertices, minusid, &minusnv, &minusvid)) return false;

    fieldindx findx[tref->disc->nnodes];
    if (!fespace_doftofieldindx(tref->field, tref->disc, plusnv, plusvid, findx)) return false;
    for (int i=0; i<tref->disc->nnodes; i++) {
        bool found=false;
        for (int j=0; j<*nentries; j++) {
            if (findxout[j].g==findx[i].g && findxout[j].id==findx[i].id && findxout[j].indx==findx[i].indx) { found=true; break; }
        }
        if (!found) {
            findxout[*nentries]=findx[i];
            (*nentries)++;
        }
    }

    if (!fespace_doftofieldindx(tref->field, tref->disc, minusnv, minusvid, findx)) return false;
    for (int i=0; i<tref->disc->nnodes; i++) {
        bool found=false;
        for (int j=0; j<*nentries; j++) {
            if (findxout[j].g==findx[i].g && findxout[j].id==findx[i].id && findxout[j].indx==findx[i].indx) { found=true; break; }
        }
        if (!found) {
            findxout[*nentries]=findx[i];
            (*nentries)++;
        }
    }

    return true;
}

static bool jump_numericalfieldgradientmapfn(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out) {
    jump_numericalfieldgradientref *tref=(jump_numericalfieldgradientref *) ref;
    fieldindx findx[2*tref->disc->nnodes];
    int nentries=0;

    if (!jump_collectparentfieldentries(tref, id, findx, &nentries)) return false;

    for (int k=0; k<nentries; k++) {
        if (!functional_numericalfieldgradentry(v, mesh, id, tref->field, findx[k].g, findx[k].id, findx[k].indx, nv, vid, tref->info->integrand, tref->ref, out)) return false;

        if (tref->info->dependencies) {
            varray_elementid dependencies;
            varray_elementidinit(&dependencies);
            if ((tref->info->dependencies)(tref->info, id, &dependencies)) {
                for (int j=0; j<dependencies.count; j++) {
                    int rnv=0, *rvid=NULL;
                    if (!jump_getelementvertices(tref->conn, tref->info->g, dependencies.data[j], &rnv, &rvid)) {
                        varray_elementidclear(&dependencies);
                        return false;
                    }
                    if (!functional_numericalfieldgradentry(v, mesh, dependencies.data[j], tref->field, findx[k].g, findx[k].id, findx[k].indx, rnv, rvid, tref->info->integrand, tref->ref, out)) {
                        varray_elementidclear(&dependencies);
                        return false;
                    }
                }
            }
            varray_elementidclear(&dependencies);
        }
    }

    return true;
}

static bool functional_mapjumpnumericalfieldgradient(vm *v, functional_mapinfo *info, objectsparse *parentvertices, void *baseref, value *out) {
    int success=false;
    int ntask=functional_ntasks(info);
    functional_task task[ntask];

    varray_elementid imageids;
    varray_elementidinit(&imageids);

    objectfield *new=NULL;
    objectfield *fieldclones[ntask];
    jump_numericalfieldgradientref tref[ntask];
    for (int i=0; i<ntask; i++) {
        fieldclones[i]=NULL;
        tref[i].ref=NULL;
    }

    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;

    new=object_newfield(info->mesh, info->field->prototype, info->field->fnspc, info->field->dof);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapjumpfieldgradient_cleanup; }
    field_zero(new);

    for (int i=0; i<ntask; i++) {
        tref[i].info=info;
        tref[i].conn=mesh_getconnectivityelement(info->mesh, 0, info->g);
        tref[i].parentvertices=parentvertices;
        tref[i].disc=NULL;

        // Serial maps perturb the original field in place; clone only for workers
        if (ntask>1) {
            fieldclones[i]=field_clone(info->field);
            tref[i].field=fieldclones[i];
            tref[i].ref=(jumpref *) ((info->cloneref) ? (info->cloneref)(baseref, info->field, fieldclones[i]) : baseref);
        } else {
            tref[i].field=info->field;
            tref[i].ref=(jumpref *) baseref;
        }
        if (!tref[i].ref) goto functional_mapjumpfieldgradient_cleanup;
        if (MORPHO_ISFESPACE(tref[i].field->fnspc)) tref[i].disc=MORPHO_GETFESPACE(tref[i].field->fnspc)->fespace;
        if (!tref[i].disc) goto functional_mapjumpfieldgradient_cleanup;

        task[i].ref=(void *) &tref[i];
        task[i].mapfn=jump_numericalfieldgradientmapfn;
        task[i].result=(void *) new;
    }

    if (!functional_map(ntask, task)) goto functional_mapjumpfieldgradient_cleanup;

    success=true;
    *out=MORPHO_OBJECT(new);

functional_mapjumpfieldgradient_cleanup:
    for (int i=0; i<ntask; i++) {
        if (fieldclones[i]) {
            if (info->freeref && tref[i].ref) (info->freeref)(tref[i].ref);
            else if (info->cloneref && tref[i].ref) MORPHO_FREE(tref[i].ref);
            object_free((object *) fieldclones[i]);
        }
    }
    if (!success && new) object_free((object *) new);
    functional_cleanuptasks(v, ntask, task, &imageids);
    return success;
}

size_t objectjumpinterfaceref_sizefn(object *obj) {
    return sizeof(objectjumpinterfaceref);
}

void objectjumpinterfaceref_printfn(object *obj, void *v) {
    morpho_printf(v, "<JumpInterfaceRef>");
}

objecttypedefn objectjumpinterfacerefdefn = {
    .printfn=objectjumpinterfaceref_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectjumpinterfaceref_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

objecttype objectjumpinterfacereftype;
int jumpinterfacehandle;

objectjumpinterfaceref *jump_getinterfaceref(vm *v) {
    value iref=MORPHO_NIL;
    vm_gettlvar(v, jumpinterfacehandle, &iref);
    if (MORPHO_ISJUMPINTERFACEREF(iref)) return MORPHO_GETJUMPINTERFACEREF(iref);
    
    return NULL;
}


static bool jump_preparetopology(vm *v, objectmesh *mesh, jumpref *ref) {
    ref->parentgrade=mesh_maxgrade(mesh);
    if (ref->parentgrade<1) MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) MESH_GRADE_LINE);

    ref->interfacegrade=ref->parentgrade-1;
    ref->interfaceparents=mesh_addconnectivityelement(mesh, ref->parentgrade, ref->interfacegrade);
    ref->parentinterfaces=mesh_addconnectivityelement(mesh, ref->interfacegrade, ref->parentgrade);
    ref->parentvertices=mesh_getconnectivityelement(mesh, 0, ref->parentgrade);

    if (!ref->parentvertices) MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) ref->parentgrade);
    if (!ref->interfaceparents || !ref->parentinterfaces) MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) ref->interfacegrade);

    return true;
}

static bool jump_preparestrategy(vm *v, jumpref *ref) {
    ref->strategy=JUMP_STRATEGY_CENTROID_MODE;

    if (!MORPHO_ISDICTIONARY(ref->integral.method)) return true;

    objectdictionary *dict=MORPHO_GETDICTIONARY(ref->integral.method);
    objectstring strategylabel = MORPHO_STATICSTRING(JUMP_STRATEGY_LABEL);
    value val=MORPHO_NIL;

    if (!dictionary_get(&dict->dict, MORPHO_OBJECT(&strategylabel), &val)) return true;
    if (!MORPHO_ISSTRING(val)) MORPHO_FAIL(v, FUNCTIONAL_ARGS);

    char *strategy=MORPHO_GETCSTRING(val);
    if (strcmp(strategy, JUMP_STRATEGY_CENTROID)==0) {
        ref->strategy=JUMP_STRATEGY_CENTROID_MODE;
        return true;
    }
    if (strcmp(strategy, JUMP_STRATEGY_QUADRATURE)==0) {
        ref->strategy=JUMP_STRATEGY_QUADRATURE_MODE;
        return true;
    }

    MORPHO_FAIL(v, FUNCTIONAL_ARGS);
}

/** Initialize a Jump object.
    Optional arguments match Integral: 'method', 'mref', 'weightbyreference' and 'optimize'. */
static value Jump_init(vm *v, int nargs, value *args) {
    value ret = integral_init(v, nargs, args);
    if (nargs>1 && MORPHO_ISFIELD(MORPHO_GETARG(args, 1))) {
        grade g = mesh_maxgrade(MORPHO_GETFIELD(MORPHO_GETARG(args, 1))->mesh);
        if (g>0) g--;
        objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), functional_gradeproperty, MORPHO_INTEGER(g));
    }
    return ret;
}

/** Prepare a jump reference.
    Shared functional metadata is handled by integral_prepareref; Jump only adds
    codimension-1 topology needed for interior-interface traversal. */
static bool jump_prepareref(vm *v, objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, jumpref *ref) {
    ref->parentgrade=0;
    ref->interfacegrade=0;
    ref->interfaceparents=NULL;
    ref->parentinterfaces=NULL;
    ref->parentvertices=NULL;
    ref->strategy=JUMP_STRATEGY_CENTROID_MODE;

    if (!integral_prepareref(self, mesh, g, sel, &ref->integral)) MORPHO_FAIL(v, INTEGRAL_ARGS);
    if (!jump_preparetopology(v, mesh, ref)) return false;
    if (!jump_preparestrategy(v, ref)) return false;
    objectinstance_setproperty(self, functional_gradeproperty, MORPHO_INTEGER(ref->interfacegrade));
    return true;
}

static bool jump_startfn(vm *v, functional_mapinfo *info) {
    jumpref *ref = (jumpref *) info->ref;
    /* Check the interface grade: Jump needs a trace, which CG0 does not provide. */
    return functional_preparefieldlist(v, ref->integral.fields, ref->integral.nfields, ref->interfacegrade);
}

/** Clone a jump reference with a substituted field. */
static void *jump_cloneref(void *ref, objectfield *field, objectfield *sub) {
    jumpref *nref = (jumpref *) ref;
    jumpref *clone = MORPHO_MALLOC(sizeof(jumpref));

    if (clone) {
        *clone = *nref;
        clone->integral.originalfields=nref->integral.originalfields;
        clone->integral.fields=MORPHO_MALLOC(sizeof(value)*clone->integral.nfields);
        if (!clone->integral.fields) { MORPHO_FREE(clone); return NULL; }

        for (int i=0; i<clone->integral.nfields; i++) {
            clone->integral.fields[i]=nref->integral.fields[i];
            if (MORPHO_ISFIELD(nref->integral.fields[i]) &&
                MORPHO_GETFIELD(nref->integral.fields[i])==field) {
                clone->integral.fields[i]=MORPHO_OBJECT(sub);
            }
        }
    }

    return clone;
}

/** Free a cloned jump reference. */
static void jump_freeref(void *ref) {
    jumpref *nref = (jumpref *) ref;
    MORPHO_FREE(nref->integral.fields);
    MORPHO_FREE(ref);
}

/** Get the adjacent parent elements for an interface. */
static bool jump_getadjacentparents(jumpref *ref, elementid interfaceid, int *nparents, int **parents) {
    if (!ref->interfaceparents) return false;
    return mesh_getconnectivity(ref->interfaceparents, interfaceid, nparents, parents);
}

/** Return mesh vertices outside the interface that still influence the jump term
    through the two adjacent parent elements. */
static bool jump_dependencies(functional_mapinfo *info, elementid id, varray_elementid *out) {
    jumpref *ref = (jumpref *) info->ref;
    int nparents=0, *parents=NULL;

    if (!jump_getadjacentparents(ref, id, &nparents, &parents)) return false;
    if (nparents!=2) return true;

    int interface_nv=0, *interface_vid=NULL;
    objectsparse *ifaceverts=NULL;
    int n=0;

    if (!functional_countelements(info->mesh, ref->interfacegrade, &n, &ifaceverts)) return false;
    (void) n;
    if (!ifaceverts) return false;
    if (!sparseccs_getrowindices(&ifaceverts->ccs, id, &interface_nv, &interface_vid)) return false;

    for (int p=0; p<nparents; p++) {
        int parent_nv=0, *parent_vid=NULL;
        if (!mesh_getconnectivity(ref->parentvertices, parents[p], &parent_nv, &parent_vid)) return false;
        for (int j=0; j<parent_nv; j++) {
            if (!functional_containsvertex(interface_nv, interface_vid, parent_vid[j])) {
                varray_elementidwriteunique(out, parent_vid[j]);
            }
        }
    }

    return true;
}

/** Return interface elements that share one of the adjacent parent elements.
    This is appropriate for coordinate gradients, but not for FE field gradients:
    in the FE case the local parent-element DOF collection already captures the
    two-sided support of the interface term, and the outer interface traversal
    accounts for neighboring interfaces exactly once. */
static bool jump_fielddependencies(functional_mapinfo *info, elementid id, varray_elementid *out) {
    jumpref *ref = (jumpref *) info->ref;
    int nparents=0, *parents=NULL;

    if (!jump_getadjacentparents(ref, id, &nparents, &parents)) return false;
    if (nparents!=2) return true;

    for (int p=0; p<nparents; p++) {
        int nifaces=0, *ifaces=NULL;
        if (!mesh_getconnectivity(ref->parentinterfaces, parents[p], &nifaces, &ifaces)) return false;
        for (int j=0; j<nifaces; j++) {
            if (ifaces[j]!=id) varray_elementidwriteunique(out, ifaces[j]);
        }
    }

    return true;
}

static bool jump_ensuresidequantities(objectintegralelementref *side, int nfields) {
    if (nfields<=0) return true;
    if (side->quantities) return true;
    side->quantities=_integral_zalloc(nfields, sizeof(quantity));
    if (!side->quantities) return false;
    side->nfields=nfields;
    return true;
}

static void jump_bindside(objectintegralelementref *side, objectmesh *mesh, grade g, elementid id, int nv, int *vid, integralref *iref) {
    double *lam=side->lambda;
    quantity *q=side->quantities;
    int nfields=side->nfields;
    _integral_bindelref(side, mesh, g, id, nv, vid, NULL, iref);
    side->lambda=lam;
    side->quantities=q;
    side->nfields=nfields;
}

static bool jump_ensuresidelambda(objectintegralelementref *side, int nv) {
    if (side->lambda) return true;
    side->lambda=MORPHO_MALLOC(sizeof(double)*nv);
    return side->lambda!=NULL;
}

static void jump_clearinterfaceref(objectjumpinterfaceref *iref, bool persistent) {
    if (persistent) return;
    if (iref->plus.lambda) { MORPHO_FREE(iref->plus.lambda); iref->plus.lambda=NULL; }
    if (iref->minus.lambda) { MORPHO_FREE(iref->minus.lambda); iref->minus.lambda=NULL; }
    integral_clearelref(&iref->iface);
    integral_clearelref(&iref->plus);
    integral_clearelref(&iref->minus);
}

static void jump_orderparents(int *parents, elementid *plusid, elementid *minusid) {
    if (parents[0]<parents[1]) {
        *plusid=parents[0]; *minusid=parents[1];
    } else {
        *plusid=parents[1]; *minusid=parents[0];
    }
}

static bool jump_getinterfacevertexpositions(objectmesh *mesh, int nv, int *vid, double **x) {
    for (int i=0; i<nv; i++) {
        if (!mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i])) return false;
    }
    return true;
}

static void jump_centroid(unsigned int dim, int nv, double **x, double *out) {
    for (unsigned int i=0; i<dim; i++) out[i]=0.0;
    for (int i=0; i<nv; i++) {
        for (unsigned int j=0; j<dim; j++) out[j]+=x[i][j];
    }
    functional_vecscale(dim, 1.0/nv, out, out);
}

static bool jump_getelementcentroid(objectmesh *mesh, int nv, int *vid, double *centroid) {
    int dim=mesh->dim;
    for (int i=0; i<dim; i++) centroid[i]=0.0;

    for (int i=0; i<nv; i++) {
        double *x=NULL;
        if (!mesh_getvertexcoordinatesaslist(mesh, vid[i], &x)) return false;
        for (int j=0; j<dim; j++) centroid[j]+=x[j];
    }

    functional_vecscale(dim, 1.0/nv, centroid, centroid);
    return true;
}

static bool jump_parentlambda(unsigned int dim, grade g, double **x, double *posn, double *lambda) {
    double invjdata[g*dim], sdata[dim];
    objectmatrix invj = MORPHO_STATICMATRIX(invjdata, g, dim);

    functional_vecsub(dim, posn, x[0], sdata);

    if (!integral_prepareinvjacobian(dim, g, x, &invj)) return false;
    functional_matmul(g, dim, 1, invjdata, sdata, lambda+1);

    lambda[0]=1.0;
    for (int i=1; i<g+1; i++) lambda[0]-=lambda[i];

    return true;
}

static bool jump_interpolatequantity(quantity *q, grade g, double *lambda, value *out) {
    int nnodes=q->nnodes;
    double wts[nnodes];

    if (q->ifn) {
        (q->ifn) (lambda, wts);
    } else {
        if (nnodes!=1) return false;
        wts[0]=1.0;
    }

    return integrator_sumquantityweighted(nnodes, wts, q->vals, out);
}

static bool jump_preparepointdata(objectjumpinterfaceref *iref, double *posn, value *qinterp) {
    jumpref *ref=iref->jref;
    double *xplus[iref->plus.nv], *xminus[iref->minus.nv];

    if (!jump_getinterfacevertexpositions(iref->iface.mesh, iref->plus.nv, iref->plus.vid, xplus) ||
        !jump_getinterfacevertexpositions(iref->iface.mesh, iref->minus.nv, iref->minus.vid, xminus) ||
        !jump_parentlambda(iref->iface.mesh->dim, ref->parentgrade, xplus, posn, iref->plus.lambda) ||
        !jump_parentlambda(iref->iface.mesh->dim, ref->parentgrade, xminus, posn, iref->minus.lambda)) return false;

    iref->iface.posn=posn;
    iref->iface.qinterpolated=qinterp;

    for (int i=0; i<ref->integral.nfields; i++) {
        if (!jump_interpolatequantity(&iref->plus.quantities[i], ref->parentgrade, iref->plus.lambda, &qinterp[i])) return false;
    }

    return true;
}

static bool jump_callintegrand(objectjumpinterfaceref *iref, double *posn, double *out) {
    jumpref *ref=iref->jref;
    value qinterp[ref->integral.nfields+1], args[ref->integral.nfields+1], outval=MORPHO_NIL;
    objectmatrix mposn = MORPHO_STATICMATRIX(posn, iref->iface.mesh->dim, 1);

    if (!jump_preparepointdata(iref, posn, qinterp)) return false;

    args[0]=MORPHO_OBJECT(&mposn);
    for (int i=0; i<ref->integral.nfields; i++) args[i+1]=qinterp[i];

    if (!morpho_call(iref->v, ref->integral.integrand, ref->integral.nfields+1, args, &outval)) return false;
    return morpho_valuetofloat(outval, out);
}

static bool jump_integrandfn(unsigned int dim, double *t, double *x, unsigned int nquantity, value *quantity, void *ref, unsigned int nout, double *fout) {
    objectjumpinterfaceref *iref = (objectjumpinterfaceref *) ref;
    if (nout!=1) return false;
    iref->iface.lambda=t;
    iref->iface.posn=x;
    return jump_callintegrand(iref, x, fout);
}

static bool jump_evaluatesidegradient(objectjumpinterfaceref *iref, int ifld, bool plus, double *grad) {
    objectfield *fld = MORPHO_GETFIELD(iref->jref->integral.fields[ifld]);
    objectintegralelementref *side = (plus ? &iref->plus : &iref->minus);
    int nv = side->nv;
    int *vid = side->vid;
    double *lambda = side->lambda;
    int dim = iref->iface.mesh->dim;
    grade g = iref->jref->parentgrade;

    if (!MORPHO_ISFESPACE(fld->fnspc) || !MORPHO_ISNIL(fld->prototype)) return false;

    fespace *disc = MORPHO_GETFESPACE(fld->fnspc)->fespace;
    if (!FESPACE_HASGRADIENT(disc)) return false;

    double *x[nv];
    if (!jump_getinterfacevertexpositions(iref->iface.mesh, nv, vid, x)) return false;

    side->vertexposn=x;
    if (!(side->flags & ELREF_HASINVJ)) {
        if (!integral_ensurematrix(&side->invj, g, dim) ||
            !integral_prepareinvjacobian(dim, g, x, side->invj)) return false;
        side->flags |= ELREF_HASINVJ;
    }

    int nnodes = disc->nnodes;
    double gdata[nnodes*g];
    double fdata[nnodes*dim];
    objectmatrix gmat = MORPHO_STATICMATRIX(gdata, nnodes, g);
    objectmatrix fmat = MORPHO_STATICMATRIX(fdata, nnodes, dim);

    fespace_gradient(disc, lambda, &gmat);
    functional_matmul(nnodes, g, dim, gdata, side->invj->elements, fdata);

    for (int i=0; i<dim; i++) {
        value sum=MORPHO_FLOAT(0.0);
        if (!integrator_sumquantityweighted(nnodes, fmat.elements+i*nnodes, side->quantities[ifld].vals, &sum)) return false;
        if (!morpho_valuetofloat(sum, &grad[i])) return false;
    }

    return true;
}

static bool jump_preparenormal(vm *v, objectjumpinterfaceref *iref) {
    int dim=iref->iface.mesh->dim;
    double pluscentroid[dim], minuscentroid[dim], d[dim];

    if (!jump_getelementcentroid(iref->iface.mesh, iref->plus.nv, iref->plus.vid, pluscentroid)) return false;
    if (!jump_getelementcentroid(iref->iface.mesh, iref->minus.nv, iref->minus.vid, minuscentroid)) return false;

    functional_vecsub(dim, minuscentroid, pluscentroid, d);

    objectmatrix *mnormal=integral_ensurematrix(&iref->iface.normal, dim, 1);
    if (!mnormal) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);

    for (int i=0; i<dim; i++) mnormal->elements[i]=0.0;

    if (iref->iface.g==0) {
        for (int i=0; i<dim; i++) mnormal->elements[i]=d[i];
    } else if (iref->iface.g==1) {
        double t[dim], n[dim];

        functional_vecsub(dim, iref->iface.vertexposn[1], iref->iface.vertexposn[0], t);
        double tnorm=functional_vecnorm(dim, t);
        if (tnorm<MORPHO_EPS) return false;
        functional_vecscale(dim, 1.0/tnorm, t, t);

        double dott=functional_vecdot(dim, d, t);
        functional_vecaddscale(dim, d, -dott, t, n);

        double nnorm=functional_vecnorm(dim, n);
        if (nnorm<MORPHO_EPS) {
            if (dim==2) {
                n[0]=-t[1];
                n[1]= t[0];
            } else {
                for (int i=0; i<dim; i++) n[i]=d[i];
            }
            nnorm=functional_vecnorm(dim, n);
        }

        if (nnorm<MORPHO_EPS) return false;
        functional_vecscale(dim, 1.0/nnorm, n, mnormal->elements);
    } else if (iref->iface.g==2) {
        if (dim!=3) return false;

        double s0[3], s1[3];
        functional_vecsub(3, iref->iface.vertexposn[1], iref->iface.vertexposn[0], s0);
        functional_vecsub(3, iref->iface.vertexposn[2], iref->iface.vertexposn[1], s1);
        functional_veccross(s0, s1, mnormal->elements);
    } else {
        return false;
    }

    double nnorm=functional_vecnorm(dim, mnormal->elements);
    if (nnorm<MORPHO_EPS) return false;

    if (functional_vecdot(dim, mnormal->elements, d)<0.0) {
        functional_vecscale(dim, -1.0, mnormal->elements, mnormal->elements);
    }

    nnorm=functional_vecnorm(dim, mnormal->elements);
    if (nnorm<MORPHO_EPS) return false;
    functional_vecscale(dim, 1.0/nnorm, mnormal->elements, mnormal->elements);

    iref->iface.flags |= ELREF_HASNORMAL;
    return true;
}

static bool jump_preparegeometry(vm *v, objectjumpinterfaceref *iref, double **vertexposn) {
    iref->iface.vertexposn=vertexposn;
    if (iref->iface.g==0) iref->iface.elementsize=1.0;
    else if (!functional_elementsize(v, iref->iface.mesh, iref->iface.g, iref->iface.id, iref->iface.nv, iref->iface.vid, &iref->iface.elementsize)) return false;

    if (iref->iface.g>0 && iref->iface.elementsize<MORPHO_EPS) return true;

    return jump_preparenormal(v, iref);
}

static bool jump_prepareinterfaceref(vm *v, objectmesh *mesh, jumpref *ref, elementid id, int nv, int *vid, double **vertexposn, int *parents, objectjumpinterfaceref *iref) {
    int plusnv=0, minusnv=0;
    int *plusvid=NULL, *minusvid=NULL;
    bool persistent=(iref->iface.flags & ELREF_PERSISTENT);

    iref->v=v;
    iref->jref=ref;

    jump_orderparents(parents, &iref->plus.id, &iref->minus.id);

    if (!mesh_getconnectivity(ref->parentvertices, iref->plus.id, &plusnv, &plusvid)) return false;
    if (!mesh_getconnectivity(ref->parentvertices, iref->minus.id, &minusnv, &minusvid)) return false;

    _integral_bindelref(&iref->iface, mesh, ref->interfacegrade, id, nv, vid, vertexposn, &ref->integral);
    jump_bindside(&iref->plus, mesh, ref->parentgrade, iref->plus.id, plusnv, plusvid, &ref->integral);
    jump_bindside(&iref->minus, mesh, ref->parentgrade, iref->minus.id, minusnv, minusvid, &ref->integral);

    if (!jump_ensuresidelambda(&iref->plus, plusnv) || !jump_ensuresidelambda(&iref->minus, minusnv)) {
        jump_clearinterfaceref(iref, persistent);
        MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    }

    if (!jump_ensuresidequantities(&iref->plus, ref->integral.nfields) ||
        !jump_ensuresidequantities(&iref->minus, ref->integral.nfields)) {
        jump_clearinterfaceref(iref, persistent);
        return false;
    }

    if (ref->integral.nfields>0) {
        if (!integral_preparequantities(&ref->integral, plusnv, plusvid, iref->plus.quantities) ||
            !integral_preparequantities(&ref->integral, minusnv, minusvid, iref->minus.quantities)) {
            jump_clearinterfaceref(iref, persistent);
            return false;
        }
    }

    if (!jump_preparegeometry(v, iref, vertexposn)) {
        jump_clearinterfaceref(iref, persistent);
        return false;
    }

    return true;
}

static void jump_initstackref(objectjumpinterfaceref *iref) {
    memset(iref, 0, sizeof(*iref));
    object_init((object *) iref, OBJECT_JUMPINTERFACEREF);
    _integral_initelref(&iref->plus);
    _integral_initelref(&iref->minus);
}

/** Basic Jump scan over codimension-1 entities.
    This currently only identifies interior interfaces by checking that they
    have exactly two adjacent parent elements. */
static bool jump_scan_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *r, double *out) {
    jumpref *ref = (jumpref *) r;
    int nparents=0, *parents=NULL;
    double *x[nv];
    objectjumpinterfaceref stackiref;
    objectjumpinterfaceref *iref;
    bool persistent=false;

    if (!jump_getadjacentparents(ref, id, &nparents, &parents)) return false;

    /* Boundary interfaces or malformed topology are ignored for now. */
    if (nparents!=2) { *out=0.0; return true; }

    if (!jump_getinterfacevertexpositions(mesh, nv, vid, x)) return false;

    iref=jump_getinterfaceref(v);
    if (iref && (iref->iface.flags & ELREF_PERSISTENT)) {
        persistent=true;
    } else {
        jump_initstackref(&stackiref);
        iref=&stackiref;
    }

    if (!jump_prepareinterfaceref(v, mesh, ref, id, nv, vid, x, parents, iref)) return false;

    if (iref->iface.g>0 && iref->iface.elementsize<MORPHO_EPS) {
        *out=0.0;
        jump_clearinterfaceref(iref, persistent);
        return true;
    }

    vm_settlvar(v, jumpinterfacehandle, MORPHO_OBJECT(iref));

    if (ref->strategy==JUMP_STRATEGY_CENTROID_MODE || ref->interfacegrade==0) {
        double posn[mesh->dim];
        jump_centroid(mesh->dim, nv, x, posn);
        if (!jump_callintegrand(iref, posn, out)) {
            if (!persistent) vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
            jump_clearinterfaceref(iref, persistent);
            return false;
        }
        *out *= iref->iface.elementsize;
    } else if (ref->strategy==JUMP_STRATEGY_QUADRATURE_MODE) {
        bool success;
        if (iref->iface.flags & ELREF_CONFIGURED) {
            success=integrator_integrate(&iref->iface.integ, jump_integrandfn, mesh->dim, x, 0, NULL, iref, 1, out);
        } else {
            double err=0.0;
            success=integrate(jump_integrandfn, MORPHO_GETDICTIONARY(ref->integral.method), morpho_geterror(v), mesh->dim, ref->interfacegrade, x, 0, NULL, iref, out, &err);
        }
        if (!success) {
            if (!persistent) vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
            jump_clearinterfaceref(iref, persistent);
            return false;
        }
        *out *= iref->iface.elementsize;
    } else {
        if (!persistent) vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
        jump_clearinterfaceref(iref, persistent);
        MORPHO_FAIL(v, JUMP_UNIMPL);
    }

    if (!persistent) vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
    jump_clearinterfaceref(iref, persistent);
    return true;
}


static bool jump_mapfieldgradient(vm *v, functional_mapinfo *info, value *out) {
    jumpref *ref = (jumpref *) info->ref;
    return functional_mapjumpnumericalfieldgradient(v, info, ref->parentvertices, ref, out);
}

static void jump_freeheapref(objectjumpinterfaceref *iref) {
    if (!iref) return;
    jump_clearinterfaceref(iref, false);
    object_free((object *) iref);
}

static bool jump_taskstart(vm *v, functional_mapinfo *info) {
    jumpref *ref=(jumpref *) info->ref;
    objectjumpinterfaceref *jiref=NULL;
    
    if (integral_contextactive(v)) MORPHO_FAIL(v, INTEGRAL_NESTED);
    
    jiref=(objectjumpinterfaceref *) object_new(sizeof(objectjumpinterfaceref), OBJECT_JUMPINTERFACEREF);
    if (!jiref) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    
    memset(jiref, 0, sizeof(*jiref));
    object_init((object *) jiref, OBJECT_JUMPINTERFACEREF);
    _integral_initelref(&jiref->plus);
    _integral_initelref(&jiref->minus);
    
    jiref->jref=ref;
    jiref->v=v;
    jiref->iface.flags |= ELREF_PERSISTENT | ELREF_HASINTEG;
    integrator_init(&jiref->iface.integ);
    
    if (ref->strategy==JUMP_STRATEGY_QUADRATURE_MODE && ref->interfacegrade>0 &&
        MORPHO_ISDICTIONARY(ref->integral.method)) {
        if (!integrator_configurewithdictionary(&jiref->iface.integ, morpho_geterror(v), ref->interfacegrade, MORPHO_GETDICTIONARY(ref->integral.method))) goto jump_taskstart_cleanup;
        jiref->iface.flags |= ELREF_CONFIGURED;
    }
    
    if (!jump_ensuresidequantities(&jiref->plus, ref->integral.nfields) ||
        !jump_ensuresidequantities(&jiref->minus, ref->integral.nfields)) goto jump_taskstart_cleanup;
    
    int lamn=ref->parentgrade+1;
    jiref->plus.lambda=MORPHO_MALLOC(sizeof(double)*lamn);
    jiref->minus.lambda=MORPHO_MALLOC(sizeof(double)*lamn);
    if (!jiref->plus.lambda || !jiref->minus.lambda) goto jump_taskstart_cleanup;
    
    vm_settlvar(v, jumpinterfacehandle, MORPHO_OBJECT(jiref));
    return true;
    
jump_taskstart_cleanup:
    if (!morpho_checkerror(morpho_geterror(v))) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    jump_freeheapref(jiref);
    return false;
}

static void jump_taskend(vm *v, functional_mapinfo *info) {
    objectjumpinterfaceref *iref=jump_getinterfaceref(v);
    if (!iref || !(iref->iface.flags & ELREF_PERSISTENT)) return;
    vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
    jump_freeheapref(iref);
}

/** Jump bindref: prepare already raises IntgrlArgs / FnctlELNtFnd / FnctlArgs.
 * Map grade is the interface grade from topology, not the caller's Int. */
static bool _Jump_bindref(vm *v, objectinstance *self, functional_mapinfo *info, jumpref *ref) {
    if (integral_contextactive(v)) MORPHO_FAIL(v, INTEGRAL_NESTED);
    if (!jump_prepareref(v, self, info->mesh, 0, info->sel, ref)) return false;
    info->g = ref->interfacegrade;
    info->ref = ref;
    info->integrand = jump_scan_integrand;
    info->start = jump_startfn;
    info->taskstart = jump_taskstart;
    info->taskend = jump_taskend;
    return true;
}

FUNCTIONAL_MD_REF_INTEGRAND(Jump, jumpref, ref.interfacegrade)
FUNCTIONAL_MD_REF_TOTAL(Jump, jumpref, ref.interfacegrade)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(Jump, jumpref, ref.interfacegrade, jump_dependencies, SYMMETRY_NONE)
FUNCTIONAL_MD_REF_FIELDGRADIENT_MAP(Jump, jumpref, ref.interfacegrade, jump_mapfieldgradient, jump_cloneref, jump_freeref)

static value integral_jumpdnfn(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_JUMPDN, JUMPDN_FUNCTION)) return MORPHO_NIL;
    objectjumpinterfaceref *iref = jump_getinterfaceref(v);
    if (!iref) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, JUMPDN_FUNCTION);

    value q = MORPHO_GETARG(args, 0);
    int ifld, xfld=-1;

    for (ifld=0; ifld<iref->jref->integral.nfields; ifld++) {
        if (MORPHO_ISFIELD(q) && MORPHO_ISSAME(iref->jref->integral.originalfields[ifld], q)) break;
        else if (iref->iface.qinterpolated && MORPHO_ISSAME(iref->iface.qinterpolated[ifld], q)) {
            if (xfld>=0) MORPHO_RAISE(v, INTEGRAL_FLD);
            xfld=ifld;
        }
    }
    if (xfld>=0) ifld=xfld;

    if (ifld>=iref->jref->integral.nfields) MORPHO_RAISE(v, INTEGRAL_FLD);

    double gradplus[iref->iface.mesh->dim], gradminus[iref->iface.mesh->dim];
    if (!jump_evaluatesidegradient(iref, ifld, true, gradplus) ||
        !jump_evaluatesidegradient(iref, ifld, false, gradminus)) MORPHO_RAISE(v, JUMP_UNIMPL);

    double jp = functional_vecdot(iref->iface.mesh->dim, gradplus, iref->iface.normal->elements);
    double jm = functional_vecdot(iref->iface.mesh->dim, gradminus, iref->iface.normal->elements);
    return MORPHO_FLOAT(jp-jm);
}

MORPHO_BEGINCLASS(Jump)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(...)", Jump_init, INTEGRAL_INITFLAGS),
FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(Jump, INTEGRAL_MAPFLAGS, INTEGRAL_ELEMFLAGS),
FUNCTIONAL_MD_TOTAL_METHODS_FLAGS(Jump, INTEGRAL_TOTALFLAGS),
FUNCTIONAL_MD_GRADIENT_METHODS_FLAGS(Jump, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS_FLAGS(Jump, INTEGRAL_MAPFLAGS)
MORPHO_ENDCLASS

void jump_initialize(void) {
    objectstring objclassname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objclassname));

    builtin_addclass(JUMP_CLASSNAME, MORPHO_GETCLASSDEFINITION(Jump), objclass);

    integral_addspecial(JUMPDN_FUNCTION, integral_jumpdnfn, INTEGRAL_USES_JUMPDN);

    morpho_defineerror(JUMP_UNIMPL, ERROR_HALT, JUMP_UNIMPL_MSG);

    objectjumpinterfacereftype=object_addtype(&objectjumpinterfacerefdefn);
    jumpinterfacehandle=vm_addtlvar();
}

#endif
