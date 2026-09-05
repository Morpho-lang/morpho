/** @file field.c
 *  @author T J Atherton
 *
 *  @brief Fields
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "field.h"
#include "morpho.h"
#include "classes.h"
#include "common.h"
#include "linalg.h"
#include "cmplx.h"
#include "sparse.h"
#include "geometry.h"
#include "fespace.h"
#include "platform.h"

value field_gradeoption;
value field_functionspaceoption;
static MorphoMutex field_poolmutex;

/* **********************************************************************
 * Field objects
 * ********************************************************************** */

objecttype objectfieldtype;
objecttype objectscalarfieldtype;
objecttype objectmatrixfieldtype;
objecttype objectcomplexfieldtype;
objecttype objectcomplexmatrixfieldtype;

/** Field object definitions */
void objectfield_printfn(object *obj, void *v) {
    objectclass *klass=object_getveneerclass(obj->type);
    morpho_printf(v, "<");
    if (klass) morpho_printvalue(v, klass->name);
    else morpho_printf(v, "Field");
    morpho_printf(v, ">");
}

void objectfield_markfn(object *obj, void *v) {
    objectfield *c = (objectfield *) obj;
    morpho_markvalue(v, c->prototype);
    morpho_markvalue(v, c->fnspc);
    morpho_markobject(v, (object *) c->mesh);
}

void objectfield_freefn(object *obj) {
    objectfield *f = (objectfield *) obj;
    
    if (f->dof) MORPHO_FREE(f->dof);
    if (f->offset) MORPHO_FREE(f->offset);
    objectpool_clear(&f->pool);
}

size_t objectfield_sizefn(object *obj) {
    objectfield *f=(objectfield *) obj;
    return sizeof(objectfield)+sizeof(double)*f->data.nels;
}

objecttypedefn objectfielddefn = {
    .printfn=objectfield_printfn,
    .markfn=objectfield_markfn,
    .freefn=objectfield_freefn,
    .sizefn=objectfield_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

static bool field_addpool(objectfield *f);

/* **********************************************************************
 * Element interface
 * ********************************************************************** */

/* -------------------------------------------------------
 * Scalar
 * ------------------------------------------------------- */

static unsigned int _scalar_dof(value prototype) { (void) prototype; return 1; }

static bool _scalar_materialize(objectfield *f, double *in, void *pool, value *out) {
    (void) pool;
    if (!f || f->psize<1 || !in || !out) return false;
    *out=MORPHO_FLOAT(in[0]);
    return true;
}

static bool _scalar_dematerialize(objectfield *f, value in, double *out) {
    if (!f || f->psize<1 || !out) return false;
    if (MORPHO_ISNIL(in)) { out[0]=0.0; return true; }
    return morpho_valuetofloat(in, out);
}

static fieldinterfacedefn scalardefn = {
    .doffn=_scalar_dof,
    .materialize=_scalar_materialize, .dematerialize=_scalar_dematerialize,
    .poolinit=NULL, .poolsize=0, .pooltypefn=NULL
};

/* -------------------------------------------------------
 * Complex
 * ------------------------------------------------------- */

static unsigned int _complex_dof(value prototype) { (void) prototype; return 2; }

static bool _complex_materialize(objectfield *f, double *in, void *pool, value *out) {
    if (!f || f->psize<2 || !in || !out) return false;
    if (pool) {
        objectcomplex *c=(objectcomplex *) pool;
        c->Z=MCBuild(in[0], in[1]);
        *out=MORPHO_OBJECT(c);
        return true;
    }
    objectcomplex *c=object_newcomplex(in[0], in[1]);
    if (!c) return false;
    *out=MORPHO_OBJECT(c);
    return true;
}

static bool _complex_dematerialize(objectfield *f, value in, double *out) {
    if (!f || f->psize<2 || !out) return false;
    if (MORPHO_ISCOMPLEX(in)) {
        MorphoComplex z=MORPHO_GETCOMPLEX(in)->Z;
        out[0]=creal(z);
        out[1]=cimag(z);
        return true;
    }
    if (MORPHO_ISNUMBER(in)) {
        if (!morpho_valuetofloat(in, out)) return false;
        out[1]=0.0;
        return true;
    }
    return false;
}

static objecttype _complex_pooltype(value prototype) {
    (void) prototype;
    return OBJECT_COMPLEX;
}

static fieldinterfacedefn complexdefn = {
    .doffn=_complex_dof,
    .materialize=_complex_materialize, .dematerialize=_complex_dematerialize,
    .poolinit=NULL, .poolsize=sizeof(objectcomplex), .pooltypefn=_complex_pooltype
};

/* -------------------------------------------------------
 * Matrix
 * ------------------------------------------------------- */

static unsigned int _matrix_dof(value prototype) {
    return (unsigned int) MORPHO_GETMATRIX(prototype)->nels;
}

static bool _matrix_materialize(objectfield *f, double *in, void *pool, value *out) {
    if (!f || f->psize<1 || !in || !out || !matrix_isamatrix(f->prototype)) return false;
    if (pool) {
        *out=MORPHO_OBJECT(pool);
        return true;
    }
    objectmatrix *proto=MORPHO_GETMATRIX(f->prototype);
    objectmatrix *m=matrix_newwithtype(OBJECT_GETTYPE(proto), proto->nrows, proto->ncols, proto->nvals, false);
    if (!m) return false;
    memcpy(m->elements, in, sizeof(double)*f->psize);
    *out=MORPHO_OBJECT(m);
    return true;
}

static bool _matrix_dematerialize(objectfield *f, value in, double *out) {
    if (!f || !out || !matrix_isamatrix(in)) return false;
    objectmatrix *m=MORPHO_GETMATRIX(in);
    if (m->nels!=f->psize) return false;
    memcpy(out, m->elements, sizeof(double)*f->psize);
    return true;
}

static void _matrix_poolinit(objectfield *f, void *slot, double *el) {
    objectmatrix *m=(objectmatrix *) slot;
    objectmatrix *proto=MORPHO_GETMATRIX(f->prototype);
    m->elements=el;
    m->ncols=proto->ncols;
    m->nrows=proto->nrows;
    m->nvals=proto->nvals;
    m->nels=m->ncols*m->nrows*m->nvals;
}

static objecttype _matrix_pooltype(value prototype) {
    return OBJECT_GETTYPE(MORPHO_GETMATRIX(prototype));
}

static fieldinterfacedefn matrixdefn = {
    .doffn=_matrix_dof,
    .materialize=_matrix_materialize, .dematerialize=_matrix_dematerialize,
    .poolinit=_matrix_poolinit, .poolsize=sizeof(objectmatrix), .pooltypefn=_matrix_pooltype
};

fieldinterfacedefn *field_getinterface(value prototype) {
    if (MORPHO_ISNIL(prototype) || MORPHO_ISNUMBER(prototype)) return &scalardefn;
    if (MORPHO_ISCOMPLEX(prototype)) return &complexdefn;
    if (matrix_isamatrix(prototype)) return &matrixdefn;
    return NULL;
}

/** Object type for a prototype; used so each kind has its own veneer class.
    Field itself is the parent veneer and is never allocated. */
static objecttype field_typeforprototype(value prototype) {
    if (MORPHO_ISCOMPLEX(prototype)) return OBJECT_COMPLEXFIELD;
    if (MORPHO_ISCOMPLEXMATRIX(prototype)) return OBJECT_COMPLEXMATRIXFIELD;
    if (matrix_isamatrix(prototype)) return OBJECT_MATRIXFIELD;
    return OBJECT_SCALARFIELD;
}

/** Class name for a Field kind; used in constructor errors. */
static const char *field_kindname(objecttype t) {
    if (t==OBJECT_SCALARFIELD) return SCALARFIELD_CLASSNAME;
    if (t==OBJECT_MATRIXFIELD) return MATRIXFIELD_CLASSNAME;
    if (t==OBJECT_COMPLEXFIELD) return COMPLEXFIELD_CLASSNAME;
    if (t==OBJECT_COMPLEXMATRIXFIELD) return COMPLEXMATRIXFIELD_CLASSNAME;
    return FIELD_CLASSNAME;
}

/** Named constructors must produce the advertised leaf type. OBJECT_FIELD skips the check (factory). */
static objectfield *_field_requirekind(vm *v, objectfield *f, objecttype expected) {
    if (!f || expected==OBJECT_FIELD || f->obj.type==expected) return f;
    morpho_runtimeerror(v, FIELD_KIND, field_kindname(expected));
    object_free((object *) f);
    return NULL;
}

bool field_isafield(value val) {
    if (!MORPHO_ISOBJECT(val)) return false;
    objecttype t=MORPHO_GETOBJECT(val)->type;
    return t==OBJECT_SCALARFIELD || t==OBJECT_MATRIXFIELD ||
           t==OBJECT_COMPLEXFIELD || t==OBJECT_COMPLEXMATRIXFIELD;
}

/* **********************************************************************
 * Object constructors
 * ********************************************************************** */

/** Checks if a prototype object is acceptable */
static bool field_checkprototype(value v) {
    return (MORPHO_ISNUMBER(v) || MORPHO_ISCOMPLEX(v) || matrix_isamatrix(v));
}

unsigned int field_sizeprototype(value prototype) {
    fieldinterfacedefn *iface=field_getinterface(prototype);
    return iface ? iface->doffn(prototype) : 1;
}

/** Determines the overall size of storage required for the field
 * @param[in] mesh - mesh to use
 * @param[in] prototype - prototype object
 * @param[in] ngrades - size of grade array
 * @param[in] dof - number of degrees of freedom per grade
 * @param[out] offsets - offsets into the store (ngrades + 1 elements)
 * @returns the overall size of storage required */
static unsigned int field_size(objectmesh *mesh, value prototype, unsigned int ngrades, unsigned int *dof, unsigned int *offsets) {
    unsigned int size = 0;
    unsigned int psize = field_sizeprototype(prototype);
    for (unsigned int i=0; i<ngrades; i++) offsets[i]=0;
    
    if (!dof) { // Assume 1 element per vertex
        size=offsets[1]=mesh_nvertices(mesh)*psize;
        for (grade i=2; i<ngrades; i++) offsets[i]=offsets[1];
    } else {
        for (grade i=0; i<ngrades; i++) {
            unsigned int nel=mesh_nelementsforgrade(mesh, i);
            offsets[i+1]=offsets[i]+nel*dof[i];
            size=offsets[i+1]*psize;
        }
    }
    
    return size;
}

/** Creates a new field
 * @param[in] mesh - Mesh the field is attached to
 * @param[in] prototype - a prototype object
 * @param[in] fnspc - function space, or nil
 * @param[in] shape -  (optional) number of degrees of freedom per entry in each grade (should be maxgrade entries) */
objectfield *object_newfield(objectmesh *mesh, value prototype, value fnspc, unsigned int *shape) {
    int ngrades=mesh_maxgrade(mesh)+1;

    unsigned int dof[ngrades]; // Extract shape from fespace or the provided function space
    if (MORPHO_ISFESPACE(fnspc)) {
        fespace *disc = MORPHO_GETFESPACE(fnspc)->fespace;
        for (grade g=1; g<=disc->grade; g++) {
            if (disc->shape[g]>0 && !mesh_getconnectivityelement(mesh, 0, g)) {
                mesh_addgrade(mesh, g);
            }
        }
        for (int i=0; i<=disc->grade; i++) dof[i]=disc->shape[i];
        for (int i=disc->grade+1; i<ngrades; i++) dof[i]=0;
    } else if (shape) {
        for (int i=0; i<ngrades; i++) dof[i]=shape[i];
    } else { // Default is simply functions on vertices
        for (unsigned int i=0; i<ngrades; i++) dof[i]=0;
        dof[0]=1;
    }

    unsigned int offset[ngrades+1];
    unsigned int size=field_size(mesh, prototype, ngrades, dof, offset);
    objectfield *new=NULL;
    unsigned int *ndof = MORPHO_MALLOC(sizeof(int)*ngrades);
    unsigned int *noffset = MORPHO_MALLOC(sizeof(unsigned int)*(ngrades+1));

    if (ndof && noffset) {
        new = (objectfield *) object_new(sizeof(objectfield)+sizeof(double)*size, field_typeforprototype(prototype));
    }

    if (new) {
        new->mesh=mesh;
        new->prototype=(MORPHO_ISNUMBER(prototype)? MORPHO_NIL : prototype);
        new->psize=field_sizeprototype(prototype);
        new->nelements=size/new->psize;
        new->ngrades=ngrades;
        new->fnspc=(MORPHO_ISFESPACE(fnspc) ? fnspc : MORPHO_NIL);

        new->offset=noffset;
        memcpy(noffset, offset, sizeof(unsigned int)*(ngrades+1));

        new->dof=ndof;
        memcpy(ndof, dof, sizeof(unsigned int)*ngrades);

        new->iface=field_getinterface(prototype);
        objectpool_init(&new->pool,
                        new->iface ? new->iface->poolsize : 0,
                        (new->iface && new->iface->pooltypefn) ? new->iface->pooltypefn(prototype) : 0,
                        (object *) new);

        /* Store is always a real Matrix of independent dofs (Re/Im packed as consecutive doubles). */
        new->data=(objectmatrix) MORPHO_STATICMATRIX(new->data.matrixdata, (MatrixIdx_t) size, 1);
        morpho_bindtoparent((object *) &new->data, (object *) new);

        memset(new->data.elements, 0, sizeof(double)*size);
        if (new->iface && !MORPHO_ISNIL(prototype)) {
            for (unsigned int i=0; i<new->nelements; i++) {
                new->iface->dematerialize(new, prototype, new->data.elements+i*new->psize);
            }
        }

        /* Pool views must exist before any worker calls getelement. */
        if (new->pool.size>0 && !field_addpool(new)) {
            object_free((object *) new);
            new=NULL;
        }

    } else { // Cleanup partially allocated structure
        if (noffset) MORPHO_FREE(noffset);
        if (ndof) MORPHO_FREE(ndof);
    }

    return new;
}

/** Applies an initialization function to every vertex */
static bool field_applyfunctiontovertices(vm *v, objectmesh *mesh, value fn, objectfield *field) {
    value coords[mesh->dim]; // Vertex coords
    value ret=MORPHO_NIL; // Return value
    int nv = mesh_nvertices(mesh);

    for (elementid i=0; i<nv; i++) { // for each vertex
        if (mesh_getvertexcoordinatesasvalues(mesh, i, coords)) {
            //get the vertex coordinates
            if (!morpho_call(v, fn, mesh->dim, coords, &ret)) return false;

            if (!field_setelement(field, MESH_GRADE_VERTEX, i, 0, ret)) MORPHO_FAIL(v, FIELD_OPRETURN);
        }
    }
    return true;
}

/** Applies an initialization function to every DOF in an element */
static bool field_applyfunctiontoelements(vm *v, objectmesh *mesh, value fn, value fnspc, objectfield *field) {
    if (!MORPHO_ISFESPACE(fnspc)) return false;
    fespace *disc = MORPHO_GETFESPACE(fnspc)->fespace;

    objectsparse *conn = mesh_getconnectivityelement(mesh, 0, disc->grade);
    if (!conn) conn = mesh_addconnectivityelement(mesh, 0, disc->grade);
    if (!conn) return false;
    elementid nel = mesh_nelements(conn);

    for (elementid id=0; id<nel; id++) {
        int nv, *vids;
        if (!mesh_getconnectivity(conn, id, &nv, &vids)) return false;

        double *x[nv]; // Fetch vertex positions
        for (int i=0; i<nv; i++) mesh_getvertexcoordinatesaslist(mesh, vids[i], &x[i]);

        fieldindx findx[disc->nnodes];
        if (!fespace_doftofieldindx(field, disc, nv, vids, findx)) return false;

        for (int i=0; i<disc->nnodes; i++) { // Loop over nodes
            int indx;
            if (!field_getindex(field, findx[i].g, findx[i].id, findx[i].indx, &indx)) return false;
            
            double lambda[nv], ll=0.0; // Convert node positions in reference element to barycentric coordinates
            for (int j=0; j<nv-1; j++) { lambda[j+1]=disc->nodes[i*disc->grade+j]; ll+=lambda[j+1]; }
            lambda[0]=1-ll;

            double xx[mesh->dim]; // Interpolate position in physical space using barycentric coordinates
            for (int j=0; j<mesh->dim; j++) xx[j]=0.0;
            for (int j=0; j<nv; j++) functional_vecaddscale(mesh->dim, xx, lambda[j], x[j], xx);

            value coords[mesh->dim], ret;
            for (int j=0; j<mesh->dim; j++) coords[j]=MORPHO_FLOAT(xx[j]);

            if (!morpho_call(v, fn, mesh->dim, coords, &ret)) return false;

            if (!field_setelementwithindex(field, indx, ret)) MORPHO_FAIL(v, FIELD_OPRETURN);
        }
    }

    return true;
}

/** Creates a field by applying a function to the vertices of a mesh
 * @param[in] v - virtual machine to use for function calls
 * @param[in] mesh - mesh to use
 * @param[in] fn - function to call
 * @param[in] expected - required leaf type, or OBJECT_FIELD to infer from the first return
 * @returns field object or NULL on failure */
static objectfield *field_newwithfunction(vm *v, objectmesh *mesh, value fn, value fnspc, objecttype expected) {
    value ret=MORPHO_NIL; // Return value
    unsigned int dim=mesh->dim;
    value coords[dim>0 ? dim : 1]; // Vertex coords
    objectfield *new = NULL;
    int handle = -1;

    /* Prototype from the first vertex, or the origin if the mesh is empty. */
    bool sampled=mesh_getvertexcoordinatesasvalues(mesh, 0, coords);
    if (!sampled && mesh_nvertices(mesh)==0) {
        for (unsigned int i=0; i<dim; i++) coords[i]=MORPHO_FLOAT(0.0);
        sampled=true;
    }
    if (sampled) {
        if (!morpho_call(v, fn, (int) dim, coords, &ret)) goto field_newwithfunction_cleanup;
        if (MORPHO_ISOBJECT(ret)) handle=morpho_retainobjects(v, 1, &ret);
    }

    new=object_newfield(mesh, ret, fnspc, NULL);
    /* Reject before applying fn to every DOF; _constructfield checks again as the canonical gate. */
    new=_field_requirekind(v, new, expected);
    if (!new) goto field_newwithfunction_cleanup;

    if (MORPHO_ISFESPACE(fnspc)) {
        if (!field_applyfunctiontoelements(v, mesh, fn, fnspc, new)) goto field_newwithfunction_cleanup;
    } else {
        if (!field_applyfunctiontovertices(v, mesh, fn, new)) goto field_newwithfunction_cleanup;
    }

    if (handle>=0) morpho_releaseobjects(v, handle);
    return new;

field_newwithfunction_cleanup:
    if (new) object_free((object *) new);
    if (handle>=0) morpho_releaseobjects(v, handle);
    return NULL;
}

/** Zeros a field */
void field_zero(objectfield *f) {
    memset(f->data.elements, 0, sizeof(double)*f->data.nels);
}

/** Adds and initializes the object pool.
    `pool` is published before slot init, so callers must wait for `ready`. */
static bool field_addpool(objectfield *f) {
    if (!f || f->pool.size==0) return false;
    if (f->pool.ready) return true;

    MorphoMutex_lock(&field_poolmutex);
    bool success=true;
    if (!f->pool.ready) {
        if (!objectpool_isalloc(&f->pool) &&
            !objectpool_alloc(&f->pool, f->nelements)) {
            success=false;
        } else if (f->iface && f->iface->poolinit) {
            for (size_t i=0; i<f->pool.nelements; i++) {
                f->iface->poolinit(f, objectpool_get(&f->pool, i), f->data.elements+i*f->psize);
            }
        }
        if (success) f->pool.ready=true;
    }
    MorphoMutex_unlock(&field_poolmutex);
    return success;
}

bool field_ensurepool(objectfield *f) {
    if (!f || f->pool.size==0) return true;
    return field_addpool(f);
}

/** Clones a field */
objectfield *field_clone(objectfield *f) {
    objectfield *new = object_newfield(f->mesh, f->prototype, f->fnspc, f->dof);
    if (new) memcpy(new->data.elements, f->data.elements, f->data.nels*sizeof(double));
    return new;
}

/* **********************************************************************
 * Field operations
 * ********************************************************************* */

static bool _field_getat(objectfield *field, int ix, value *out) {
    if (ix<0 || ix>=(int) field->nelements) return false;
    fieldinterfacedefn *iface=field->iface;
    if (!iface) return false;

    void *slot=NULL;
    if (field->pool.size>0) {
        if (!field->pool.ready && !field_addpool(field)) return false;
        slot=objectpool_get(&field->pool, (size_t) ix);
        if (!slot) return false;
    }
    return iface->materialize(field, field->data.elements+ix*field->psize, slot, out);
}

static bool _field_setat(objectfield *field, int ix, value val) {
    if (ix<0 || ix>=(int) field->nelements) return false;
    if (!field->iface) return false;
    return field->iface->dematerialize(field, val, field->data.elements+ix*field->psize);
}

/** Retrieve a value from a field object
 * @param[in] field - field to use
 * @param[in] grade - grade to access
 * @param[in] el - element id
 * @param[in] indx - index within the element
 * @param[out] out - the retrieved value
 * @return true on success */
bool field_getelement(objectfield *field, grade grade, elementid el, int indx, value *out) {
    unsigned int ix=field->offset[grade]+field->dof[grade]*el+indx;
    if (!(ix<field->offset[grade+1] && indx<field->dof[grade])) return false;
    return _field_getat(field, (int) ix, out);
}

/** Retrieve a value from a field object given a single index
 * @param[in] field - field to use
 * @param[in] indx - index within the element
 * @param[out] out - the retrieved value
 * @return true on success */
bool field_getelementwithindex(objectfield *field, int indx, value *out) {
    return _field_getat(field, indx, out);
}

/** Constructs a single index, suitable for use with fieldgetelementwithindex from the grade, element id and quantity number
 * @param[in] field - field to use
 * @param[in] grade - grade to access
 * @param[in] el - element id
 * @param[in] indx - index within the element
 * @param[out] out - the retrieved index
 * @return true on success */
static bool field_validateaccess(objectfield *field, grade grade, elementid el, int indx) {
    if (!field) return false;
    if (grade<0 || grade>=field->ngrades) return false;
    if (el<0) return false;
    if (indx<0 || indx>=field->dof[grade]) return false;
    return true;
}

bool field_getindex(objectfield *field, grade grade, elementid el, int indx, int *out) {
    if (!out || !field_validateaccess(field, grade, el, indx)) return false;

    int ix=field->offset[grade]+field->dof[grade]*el+indx;
    if (!(ix<field->offset[grade+1])) return false;

    *out=ix;
    return true;
}

/** Retrieve the list of doubles that represent an entry in a field
 * @param[in] field - field to use
 * @param[in] grade - grade to access
 * @param[in] el - element id
 * @param[in] indx - index within the element
 * @param[out] nentries - number of entries
 * @param[out] out - the retrieved list
 * @return true on success */
bool field_getelementaslist(objectfield *field, grade grade, elementid el, int indx, unsigned int *nentries, double **out) {
    unsigned int ix=field->offset[grade]+field->dof[grade]*el+indx;
    if (!(ix<field->offset[grade+1] && indx<field->dof[grade])) return false;

    *out = &field->data.elements[ix*field->psize];
    *nentries=field->psize;
    return true;
}

static bool field_getelementdofs(objectfield *field, fespace *disc, elementid el, fieldindx *findx) {
    if (!field || !disc || !findx) return false;

    objectsparse *conn = mesh_getconnectivityelement(field->mesh, 0, disc->grade);
    if (!conn) conn = mesh_addconnectivityelement(field->mesh, 0, disc->grade);
    if (!conn) return false;

    int nv, *vids;
    if (!mesh_getconnectivity(conn, el, &nv, &vids)) return false;
    if (nv!=disc->grade+1) return false;

    return fespace_doftofieldindx(field, disc, nv, vids, findx);
}

bool field_evalelement(objectfield *field, elementid el, double *lambda, value *out) {
    if (!field || !lambda || !out) return false;
    if (!MORPHO_ISFESPACE(field->fnspc)) return false;
    if (!field->iface) return false;

    fespace *disc = MORPHO_GETFESPACE(field->fnspc)->fespace;
    fieldindx findx[disc->nnodes];
    if (!field_getelementdofs(field, disc, el, findx)) return false;

    double wts[disc->nnodes];
    disc->ifn(lambda, wts);

    unsigned int ndof=field->psize;
    double accum[ndof];
    memset(accum, 0, sizeof(double)*ndof);
    for (int i=0; i<disc->nnodes; i++) {
        unsigned int nentries;
        double *entries;
        if (!field_getelementaslist(field, findx[i].g, findx[i].id, findx[i].indx, &nentries, &entries)) return false;
        for (unsigned int j=0; j<nentries; j++) accum[j] += wts[i]*entries[j];
    }
    return field->iface->materialize(field, accum, NULL, out);
}

/** Sets the value of an entry in a field object
 * @param[in] field - field to use
 * @param[in] grade - grade to access
 * @param[in] el - element id
 * @param[in] indx - index within the element
 * @param[in] val - value to set
 * @return true on success */
bool field_setelement(objectfield *field, grade grade, elementid el, int indx, value val) {
    unsigned int ix=field->offset[grade]+field->dof[grade]*el+indx;
    if (!(ix<field->offset[grade+1] && indx<field->dof[grade])) return false;
    return _field_setat(field, (int) ix, val);
}

/** Sets the value of an entry in a field object given a single index
 * @param[in] field - field to use
 * @param[in] ix - index of the element
 * @param[in] val - value to set
 * @return true on success */
bool field_setelementwithindex(objectfield *field, int ix, value val) {
    return _field_setat(field, ix, val);
}

/** Prototype is complex-valued (Complex or ComplexMatrix). Packing is interleaved re/im doubles. */
static bool _field_complexvalued(objectfield *f) {
    return f->obj.type==OBJECT_COMPLEXFIELD || f->obj.type==OBJECT_COMPLEXMATRIXFIELD;
}

/** Checks if two fields have the same shape */
static bool field_compareshape(objectfield *a, objectfield *b) {
    if (a->iface!=b->iface || a->psize!=b->psize ||
        a->nelements!=b->nelements || a->ngrades!=b->ngrades) return false;
    if (matrix_isamatrix(a->prototype) && matrix_isamatrix(b->prototype)) {
        objectmatrix *pa=MORPHO_GETMATRIX(a->prototype), *pb=MORPHO_GETMATRIX(b->prototype);
        if (pa->nrows!=pb->nrows || pa->ncols!=pb->ncols || pa->nvals!=pb->nvals) return false;
    }
    for (unsigned int i=0; i<a->ngrades; i++) {
        if (a->dof[i]!=b->dof[i]) return false;
    }
    return true;
}

/** Returns the number of degrees of freedom in a given grade */
unsigned int field_dofforgrade(objectfield *f, grade g) {
    if (!f || g<0 || (unsigned int) g>=f->ngrades) return 0;
    return f->dof[g];
}

/** Retrieve the lowest active grade. */
bool field_lowestgrade(objectfield *field, grade *g) {
    grade gg = MESH_GRADE_VERTEX;
    while (gg<(grade) field->ngrades && field->dof[gg]==0) gg++;
    if (gg>=(grade) field->ngrades) return false;
    *g = gg;
    return true;
}

/** Adds two fields together */
static bool field_add(objectfield *left, objectfield *right, objectfield *out) {
    return (matrix_copy(&left->data, &out->data)==LINALGERR_OK &&
            matrix_axpy(1.0, &right->data, &out->data)==LINALGERR_OK);
}

/** Subtracts one field from another */
static bool field_sub(objectfield *left, objectfield *right, objectfield *out) {
    return (matrix_copy(&left->data, &out->data)==LINALGERR_OK &&
            matrix_axpy(-1.0, &right->data, &out->data)==LINALGERR_OK);
}

/** Accumulate, i.e. a <- a + lambda*b */
static bool field_accumulate(objectfield *left, double lambda, objectfield *right) {
    return (matrix_axpy(lambda, &right->data, &left->data)==LINALGERR_OK);
}

static bool field_inner(objectfield *left, objectfield *right, double *out) {
    return (matrix_inner(&left->data, &right->data, out)==LINALGERR_OK);
}

/** Calls a function fn on every element of a field, optionally with other fields as arguments */
static bool field_op(vm *v, value fn, objectfield *f, int nargs, objectfield **args, value *out) {
    unsigned int nel = f->nelements;
    value ret=MORPHO_NIL;
    value fargs[nargs+1];
    objectfield *fld=NULL;
    int handle = -1;
    
    for (int i=0; i<nel; i++) {
        if (!field_getelementwithindex(f, i, &fargs[0])) return false;
        for (unsigned int k=0; k<nargs; k++) {
            if (!field_getelementwithindex(args[k], i, &fargs[k+1])) return false;
        }
        
        if (morpho_call(v, fn, nargs+1, fargs, &ret)) {
            if (!fld) {
                if (field_checkprototype(ret)) {
                    if (MORPHO_ISOBJECT(ret)) handle=morpho_retainobjects(v, 1, &ret);
                    fld=object_newfield(f->mesh, ret, f->fnspc, f->dof);
                    if (!fld) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
                } else MORPHO_FAIL(v, FIELD_OPRETURN);
            }
            
            if (!field_setelementwithindex(fld, i, ret)) return false;
        } else return false;
    }
    
    if (handle>=0) morpho_releaseobjects(v, handle);
    if (fld) *out = MORPHO_OBJECT(fld);
    
    return true;
}

/* **********************************************************************
 * Field veneer class
 * ********************************************************************* */

/** Fill a dof vector from grade= when the Field is a raw container (no space). */
static bool field_doffromgrade(vm *v, value grd, unsigned int ngrades, unsigned int *dof, unsigned int **shape) {
    *shape=NULL;
    if (MORPHO_ISNIL(grd)) return true;

    if (MORPHO_ISINTEGER(grd)) {
        int n=MORPHO_GETINTEGERVALUE(grd);
        if (n<0 || (unsigned int) n>=ngrades) MORPHO_FAIL(v, FIELD_ARGS);
        dof[n]=1;
    } else if (MORPHO_ISLIST(grd)) {
        objectlist *list=MORPHO_GETLIST(grd);
        if (!array_valuelisttoindices(list->val.count, list->val.data, dof)) return false;
    } else if (MORPHO_ISTUPLE(grd)) {
        objecttuple *tuple=MORPHO_GETTUPLE(grd);
        if (!array_valuelisttoindices(tuple->length, tuple->tuple, dof)) return false;
    } else MORPHO_FAIL(v, FIELD_ARGS);
    
    *shape=dof;
    return true;
}

/** Default linear space on the mesh's highest grade; vertex-only meshes have none. */
static bool field_attachlinear(vm *v, objectmesh *mesh, value *fnspc) {
    grade g=mesh_maxgrade(mesh);
    if (g==0) return true;
    objectfespace *obj=fespace_newlinear(g);
    if (!obj) MORPHO_FAILVARGS(v, FNSPC_NOTFOUND, FESPACE_CG1, (int) g);
    *fnspc=MORPHO_OBJECT(obj);
    return true;
}

/** Decide on the field layout from grade= and finiteelementspace=.  */
static bool field_layoutfromoptions(vm *v, objectmesh *mesh, value grd, value *fnspc,
                                   unsigned int ngrades, unsigned int *dof, unsigned int **shape) {
    *shape=NULL;
    value fs=*fnspc;

    if (MORPHO_ISFESPACE(fs)) return true; // User provided a FiniteElementSpace
    if (MORPHO_ISNIL(fs)) { // finiteelementspace=nil means act as a raw container
        *fnspc=MORPHO_NIL;
        return field_doffromgrade(v, grd, ngrades, dof, shape);
    } else if (!MORPHO_ISSAME(fs, MORPHO_FALSE)) MORPHO_FAIL(v, FIELD_ARGS); // User provided invalid finiteelementspace option

    *fnspc=MORPHO_NIL;
    if (MORPHO_ISLIST(grd) || MORPHO_ISTUPLE(grd)) return field_doffromgrade(v, grd, ngrades, dof, shape);

    grade g=0; // Check grade
    if (MORPHO_ISINTEGER(grd)) {
        g=MORPHO_GETINTEGERVALUE(grd);
        if (g<0) MORPHO_FAIL(v, FIELD_ARGS);
    } else if (!MORPHO_ISNIL(grd)) MORPHO_FAIL(v, FIELD_ARGS); // User provided invalid grade

    if (g==0) return field_attachlinear(v, mesh, fnspc); // grade ommitted and grade=0 are mapped to CG1

    objectfespace *obj=fespace_newfromname(FESPACE_CG0, g); // Default is a CG0 field
    if (!obj) MORPHO_FAILVARGS(v, FNSPC_NOTFOUND, FESPACE_CG0, g);
    *fnspc=MORPHO_OBJECT(obj);
    return true;
}

/** Common constructor: mesh is required; prototype or fn is the optional fill.
    expected is OBJECT_FIELD to infer the kind, or a leaf type that the result must match. */
static value _constructfield(vm *v, int nargs, value *args, objectmesh *mesh, value prototype, value fn, objecttype expected) {
    value grd=MORPHO_NIL, fnspc=MORPHO_FALSE; // Guard value to detect if fnspc is set
    builtin_options(v, nargs, args, NULL, 2, field_gradeoption, &grd, field_functionspaceoption, &fnspc);

    unsigned int ngrades=mesh_maxgrade(mesh)+1; // Count dofs
    unsigned int dof[ngrades];
    for (unsigned int i=0; i<ngrades; i++) dof[i]=0;

    unsigned int *shape=NULL; // Extract layout from optional arguments
    if (!field_layoutfromoptions(v, mesh, grd, &fnspc, ngrades, dof, &shape)) return MORPHO_NIL;

    objectfield *new=NULL; // Choose constructor pattern
    if (!MORPHO_ISNIL(fn) && (MORPHO_ISNIL(prototype) || MORPHO_ISNUMBER(prototype))) {
        new=field_newwithfunction(v, mesh, fn, fnspc, expected);
    } else {
        new=object_newfield(mesh, prototype, fnspc, shape);
        if (new && !MORPHO_ISNIL(fn)) {
            bool ok=MORPHO_ISFESPACE(fnspc) ?
                field_applyfunctiontoelements(v, mesh, fn, fnspc, new) :
                field_applyfunctiontovertices(v, mesh, fn, new);
            if (!ok) {
                object_free((object *) new);
                new=NULL;
            }
        }
    }
    new=_field_requirekind(v, new, expected);

    if (!new && MORPHO_ISOBJECT(fnspc)) object_freeifunmanaged(MORPHO_GETOBJECT(fnspc));
    return morpho_wrapandbindrecursive(v, (object *) new);
}

static value _construct(vm *v, int nargs, value *args, value proto, value fn, objecttype expected) {
    return _constructfield(v, nargs, args, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), proto, fn, expected);
}

value field_constructor__mesh(vm *v, int nargs, value *args) {
    return _construct(v, nargs, args, MORPHO_NIL, MORPHO_NIL, OBJECT_FIELD);
}

value field_constructor__mesh_proto(vm *v, int nargs, value *args) {
    return _construct(v, nargs, args, MORPHO_GETARG(args, 1), MORPHO_NIL, OBJECT_FIELD);
}

value field_constructor__mesh_fn(vm *v, int nargs, value *args) {
    return _construct(v, nargs, args, MORPHO_NIL, MORPHO_GETARG(args, 1), OBJECT_FIELD);
}

value scalarfield_constructor__mesh(vm *v, int nargs, value *args) {
    return _construct(v, nargs, args, MORPHO_NIL, MORPHO_NIL, OBJECT_SCALARFIELD);
}

value scalarfield_constructor__mesh_proto(vm *v, int nargs, value *args) {
    return _construct(v, nargs, args, MORPHO_GETARG(args, 1), MORPHO_NIL, OBJECT_SCALARFIELD);
}

value scalarfield_constructor__mesh_fn(vm *v, int nargs, value *args) {
    return _construct(v, nargs, args, MORPHO_NIL, MORPHO_GETARG(args, 1), OBJECT_SCALARFIELD);
}

value matrixfield_constructor__mesh_proto(vm *v, int nargs, value *args) {
    return _construct(v, nargs, args, MORPHO_GETARG(args, 1), MORPHO_NIL, OBJECT_MATRIXFIELD);
}

value matrixfield_constructor__mesh_fn(vm *v, int nargs, value *args) {
    return _construct(v, nargs, args, MORPHO_NIL, MORPHO_GETARG(args, 1), OBJECT_MATRIXFIELD);
}

value complexmatrixfield_constructor__mesh_proto(vm *v, int nargs, value *args) {
    return _construct(v, nargs, args, MORPHO_GETARG(args, 1), MORPHO_NIL, OBJECT_COMPLEXMATRIXFIELD);
}

value complexmatrixfield_constructor__mesh_fn(vm *v, int nargs, value *args) {
    return _construct(v, nargs, args, MORPHO_NIL, MORPHO_GETARG(args, 1), OBJECT_COMPLEXMATRIXFIELD);
}

/** ComplexField needs a Complex prototype so the result is not inferred as a ScalarField.
    wrapandbindrecursive GC-binds an unmanaged proto on success; free it if construction fails first. */
static value _constructcomplexfield(vm *v, int nargs, value *args, value proto, value fn) {
    if (MORPHO_ISNIL(proto)) return MORPHO_NIL;
    value out=_construct(v, nargs, args, proto, fn, OBJECT_COMPLEXFIELD);
    if (MORPHO_ISOBJECT(proto)) object_freeifunmanaged(MORPHO_GETOBJECT(proto));
    return out;
}

static value _newcomplexproto(double re, double im) {
    objectcomplex *z=object_newcomplex(re, im);
    return z ? MORPHO_OBJECT(z) : MORPHO_NIL;
}

value complexfield_constructor__mesh(vm *v, int nargs, value *args) {
    return _constructcomplexfield(v, nargs, args, _newcomplexproto(0.0, 0.0), MORPHO_NIL);
}

value complexfield_constructor__mesh_proto(vm *v, int nargs, value *args) {
    value proto=MORPHO_GETARG(args, 1);
    if (MORPHO_ISNUMBER(proto)) {
        double x=0.0;
        if (!morpho_valuetofloat(proto, &x)) return MORPHO_NIL;
        return _constructcomplexfield(v, nargs, args, _newcomplexproto(x, 0.0), MORPHO_NIL);
    }
    return _construct(v, nargs, args, proto, MORPHO_NIL, OBJECT_COMPLEXFIELD);
}

value complexfield_constructor__mesh_fn(vm *v, int nargs, value *args) {
    return _constructcomplexfield(v, nargs, args, _newcomplexproto(0.0, 0.0), MORPHO_GETARG(args, 1));
}

/* ----------------------------------------------
 * Method implementations
 * ---------------------------------------------- */

static int field_indexfallbackwarned = 0;

/** Compatibility shim to find a fallback grade if grade 0 is empty; returns true on success */
static bool _indexresolvefallback(vm *v, objectfield *f, grade *g) {
    if (*g!=MESH_GRADE_VERTEX || field_dofforgrade(f, MESH_GRADE_VERTEX)!=0) return true;
    if (!field_lowestgrade(f, g)) return false; // Identify lowest grade
    if (MorphoAtomic_addint(&field_indexfallbackwarned, 1)==0) { // Warn if first use
        morpho_runtimewarning(v, FIELD_IDXFALLBACK);
    }
    return true;
}

static value _indexget(vm *v, objectfield *f, grade g, elementid el, int indx) {
    value out = MORPHO_NIL;
    if (!_indexresolvefallback(v, f, &g)) MORPHO_RAISE(v, FIELD_INDICESOUTSIDEBOUNDS);
    if (!field_getelement(f, g, el, indx, &out)) MORPHO_RAISE(v, FIELD_INDICESOUTSIDEBOUNDS);
    return out;
}

value Field_getindex__int(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    grade g;
    if (!field_lowestgrade(f, &g)) MORPHO_RAISE(v, FIELD_INDICESOUTSIDEBOUNDS);
    return _indexget(v, f, g, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)), 0);
}

value Field_getindex__int_int(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    return _indexget(v, f, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                          MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1)),
                          0);
}

value Field_getindex__int_int_int(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    return _indexget(v, f, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                          MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1)),
                          MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 2)));
}

static value _indexset(vm *v, objectfield *f, grade g, elementid el, int indx, value val) {
    if (!_indexresolvefallback(v, f, &g)) MORPHO_RAISE(v, FIELD_INCOMPATIBLEVAL);
    if (!field_setelement(f, g, el, indx, val)) MORPHO_RAISE(v, FIELD_INCOMPATIBLEVAL);
    return MORPHO_NIL;
}

value Field_setindex__int_x(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    grade g;
    if (!field_lowestgrade(f, &g)) MORPHO_RAISE(v, FIELD_INCOMPATIBLEVAL);
    return _indexset(v, f, g, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)), 0, MORPHO_GETARG(args, 1));
}

value Field_setindex__int_int_x(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    return _indexset(v, f, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                          MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1)),
                          0, MORPHO_GETARG(args, 2));
}

value Field_setindex__int_int_int_x(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    return _indexset(v, f, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                          MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1)),
                          MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 2)),
                          MORPHO_GETARG(args, 3));
}

/** Enumerate protocol */
value Field_enumerate__int(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    int i=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    value out=MORPHO_NIL;

    if (i<0) out=MORPHO_INTEGER(a->nelements);
    else if (i<(int) a->nelements) {
        if (!field_getelementwithindex(a, i, &out)) UNREACHABLE("Could not get field element.");
    }
    /* Note no need to bind as we are an object pool */

    return out;
}

/** Number of field elements */
value Field_count(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    return MORPHO_INTEGER(f->nelements);
}

/** Sum stored values. Scalars return a Float or Complex; matrix-valued fields return the summed matrix. */
value Field_sum(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    if (!f->iface) return MORPHO_NIL;

    double accum[f->psize];
    memset(accum, 0, sizeof(double)*f->psize);
    for (unsigned int i=0; i<f->nelements; i++) {
        cblas_daxpy((linalg_int_t) f->psize, 1.0, f->data.elements+i*f->psize, 1, accum, 1);
    }

    value out=MORPHO_NIL;
    if (!f->iface->materialize(f, accum, NULL, &out)) MORPHO_RAISE(v, ERROR_ALLOCATIONFAILED);
    if (MORPHO_ISOBJECT(out)) return morpho_wrapandbind(v, MORPHO_GETOBJECT(out));
    return out;
}

/** Field assign */
value Field_assign__field(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    objectfield *b=MORPHO_GETFIELD(MORPHO_GETARG(args, 0));

    if (!field_compareshape(a, b)) MORPHO_RAISE(v, FIELD_INCOMPATIBLEMATRICES);
    matrix_copy(&b->data, &a->data);

    return MORPHO_NIL;
}

value Field_assign__matrix(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));

    if (matrix_copy(b, &a->data)!=LINALGERR_OK) MORPHO_RAISE(v, FIELD_INCOMPATIBLEMATRICES);

    return MORPHO_NIL;
}

/** Perform a binary operation on the contents of two fields */
static value _field_binop(vm *v, objectfield *a, objectfield *b, bool (*op) (objectfield *, objectfield *, objectfield *)) {
    if (!field_compareshape(a, b)) MORPHO_RAISE(v, FIELD_INCOMPATIBLEMATRICES);

    objectfield *new = object_newfield(a->mesh, a->prototype, a->fnspc, a->dof);
    if (new) op(a, b, new);
    return morpho_wrapandbind(v, (object *) new);
}

/** Field add */
value Field_add__field(vm *v, int nargs, value *args) {
    return _field_binop(v, MORPHO_GETFIELD(MORPHO_SELF(args)),
                        MORPHO_GETFIELD(MORPHO_GETARG(args, 0)), field_add);
}

/** Returns a new field containing alpha*a + beta, applied to each Morpho element. */
static value _field_addscalar(vm *v, objectfield *a, double alpha, double beta) {
    objectfield *new=field_clone(a);
    if (new) {
        if (_field_complexvalued(a)) {
            matrix_scale(&new->data, alpha);
            double *el=new->data.elements;
            for (MatrixCount_t i=0; i<new->data.nels; i+=2) el[i]+=beta;
        } else {
            matrix_addscalar(&new->data, alpha, beta);
        }
    }
    return morpho_wrapandbind(v, (object *) new);
}

value Field_add__number(vm *v, int nargs, value *args) {
    double x;
    if (!morpho_valuetofloat(MORPHO_GETARG(args, 0), &x)) MORPHO_RAISE(v, VM_INVALIDARGS);
    return _field_addscalar(v, MORPHO_GETFIELD(MORPHO_SELF(args)), 1.0, x);
}

/** Add a complex scalar to every Morpho element (packed as re/im pairs). */
static value _field_addcomplex(vm *v, objectfield *a, MorphoComplex z) {
    if (!_field_complexvalued(a)) MORPHO_RAISE(v, FIELD_INCOMPATIBLEVAL);
    objectfield *new=field_clone(a);
    if (new) {
        double re=creal(z), im=cimag(z);
        double *el=new->data.elements;
        for (MatrixCount_t i=0; i<new->data.nels; i+=2) {
            el[i]+=re;
            el[i+1]+=im;
        }
    }
    return morpho_wrapandbind(v, (object *) new);
}

value Field_add__complex(vm *v, int nargs, value *args) {
    return _field_addcomplex(v, MORPHO_GETFIELD(MORPHO_SELF(args)), MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0))->Z);
}

/** Right add of nil or a number */
value Field_addr__nil(vm *v, int nargs, value *args) {
    return MORPHO_SELF(args);
}

value Field_addr__number(vm *v, int nargs, value *args) {
    return Field_add__number(v, nargs, args);
}

/** Field subtraction */
value Field_sub__field(vm *v, int nargs, value *args) {
    return _field_binop(v, MORPHO_GETFIELD(MORPHO_SELF(args)),
                        MORPHO_GETFIELD(MORPHO_GETARG(args, 0)), field_sub);
}

static value _field_neg(vm *v, objectfield *a) {
    objectfield *new=field_clone(a);
    if (new) matrix_scale(&new->data, -1.0);
    return morpho_wrapandbind(v, (object *) new);
}

/** Right subtract of nil or a number: x - field */
value Field_subr__nil(vm *v, int nargs, value *args) {
    return _field_neg(v, MORPHO_GETFIELD(MORPHO_SELF(args)));
}

value Field_subr__number(vm *v, int nargs, value *args) {
    double x;
    if (!morpho_valuetofloat(MORPHO_GETARG(args, 0), &x)) MORPHO_RAISE(v, VM_INVALIDARGS);
    return _field_addscalar(v, MORPHO_GETFIELD(MORPHO_SELF(args)), -1.0, x);
}

value Field_sub__number(vm *v, int nargs, value *args) {
    double x;
    if (!morpho_valuetofloat(MORPHO_GETARG(args, 0), &x)) MORPHO_RAISE(v, VM_INVALIDARGS);
    return _field_addscalar(v, MORPHO_GETFIELD(MORPHO_SELF(args)), 1.0, -x);
}

/** Field accumulate */
value Field_acc__number_field(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    objectfield *b=MORPHO_GETFIELD(MORPHO_GETARG(args, 1));

    if (!field_compareshape(a, b)) MORPHO_RAISE(v, FIELD_INCOMPATIBLEMATRICES);

    double lambda=1.0;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &lambda);
    field_accumulate(a, lambda, b);

    return MORPHO_NIL;
}

/** Field multiply by a scalar */
value Field_mul__number(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    double scale=1.0;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale);

    objectfield *new = field_clone(a);
    if (new) matrix_scale(&new->data, scale);
    return morpho_wrapandbind(v, (object *) new);
}

/** Multiply every Morpho element by a complex scalar. */
static value _field_mulcomplex(vm *v, objectfield *a, MorphoComplex z) {
    if (!_field_complexvalued(a)) MORPHO_RAISE(v, FIELD_INCOMPATIBLEVAL);
    objectfield *new=field_clone(a);
    if (new) {
        cblas_zscal((linalg_int_t) (new->data.nels/2),
                    (linalg_complexdouble_t *) &z,
                    (linalg_complexdouble_t *) new->data.elements, 1);
    }
    return morpho_wrapandbind(v, (object *) new);
}

value Field_mul__complex(vm *v, int nargs, value *args) {
    return _field_mulcomplex(v, MORPHO_GETFIELD(MORPHO_SELF(args)), MORPHO_GETCOMPLEX(MORPHO_GETARG(args, 0))->Z);
}

/** Field divide by a scalar */
value Field_div__number(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    double scale=1.0;
    morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale);
    if (fabs(scale)<MORPHO_EPS) MORPHO_RAISE(v, VM_DVZR);

    objectfield *new = field_clone(a);
    if (new) matrix_scale(&new->data, 1.0/scale);
    return morpho_wrapandbind(v, (object *) new);
}

/** Frobenius inner product */
value Field_inner__field(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    objectfield *b=MORPHO_GETFIELD(MORPHO_GETARG(args, 0));

    double prod=0.0;
    if (!field_inner(a, b, &prod)) MORPHO_RAISE(v, FIELD_INCOMPATIBLEMATRICES);

    return MORPHO_FLOAT(prod);
}

/** Frobenius norm of the store */
value Field_norm(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    return MORPHO_FLOAT(matrix_norm(&f->data, MATRIX_NORM_FROBENIUS));
}

/** Generalized operations */
value Field_op__callable(vm *v, int nargs, value *args) {
    objectfield *slf=MORPHO_GETFIELD(MORPHO_SELF(args));
    int nfields=nargs-1;
    objectfield *flds[nfields > 0 ? nfields : 1];
    value out=MORPHO_NIL;

    for (int i=0; i<nfields; i++) {
        if (!MORPHO_ISFIELD(MORPHO_GETARG(args, i+1))) MORPHO_RAISE(v, FIELD_OP);
        flds[i]=MORPHO_GETFIELD(MORPHO_GETARG(args, i+1));
    }

    if (!field_op(v, MORPHO_GETARG(args, 0), slf, nfields, nfields ? flds : NULL, &out)) return MORPHO_NIL;
    if (MORPHO_ISOBJECT(out)) return morpho_wrapandbind(v, MORPHO_GETOBJECT(out));
    return out;
}

/** Print the field */
value Field_print(vm *v, int nargs, value *args) {
    morpho_printvalue(v, MORPHO_SELF(args));
    return MORPHO_NIL;
}

/** Clones a field */
value Field_clone(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    return morpho_wrapandbind(v, (object *) field_clone(a));
}

/** Get the shape (number of dofs per grade) of a field */
value Field_shape(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));

    value shape[f->ngrades];
    for (unsigned int i=0; i<f->ngrades; i++) shape[i]=MORPHO_INTEGER(f->dof[i]);

    return morpho_wrapandbind(v, (object *) object_newtuple(f->ngrades, shape));
}

/** Get the functionspace used by a field */
value Field_fnspace(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    return f->fnspc;
}

/** Get a prototype used by the field */
value Field_prototype(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    return f->prototype;
}

/** Get the mesh associated with a field */
value Field_mesh(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    return MORPHO_OBJECT(f->mesh);
}

static bool field_readlambda_list(value arg, int nlambda, double *lambda) {
    objectlist *list = MORPHO_GETLIST(arg);
    if (list_length(list)!=(unsigned int) nlambda) return false;
    for (int i=0; i<nlambda; i++) {
        value el;
        if (!list_getelement(list, i, &el)) return false;
        if (!morpho_valuetofloat(el, &lambda[i])) return false;
    }
    return true;
}

static bool field_readlambda_matrix(value arg, int nlambda, double *lambda) {
    objectmatrix *mat = MORPHO_GETMATRIX(arg);
    if (mat->nrows!=(unsigned int) nlambda || mat->ncols!=1) return false;
    for (int i=0; i<nlambda; i++) lambda[i]=mat->elements[i];
    return true;
}

static value _evalelement(vm *v, objectfield *field, elementid el, bool (*readfn) (value, int, double *), value lambdaarg) {
    if (!MORPHO_ISFESPACE(field->fnspc)) return MORPHO_NIL;

    fespace *disc = MORPHO_GETFESPACE(field->fnspc)->fespace;
    int nlambda = disc->grade+1;
    double lambda[nlambda];
    value out = MORPHO_NIL;

    if (readfn(lambdaarg, nlambda, lambda) &&
        field_evalelement(field, el, lambda, &out) &&
        MORPHO_ISOBJECT(out)) {
        return morpho_wrapandbind(v, MORPHO_GETOBJECT(out));
    }

    return out;
}

value Field_evalelement__int_list(vm *v, int nargs, value *args) {
    return _evalelement(v, MORPHO_GETFIELD(MORPHO_SELF(args)),
                        MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                        field_readlambda_list, MORPHO_GETARG(args, 1));
}

value Field_evalelement__int_matrix(vm *v, int nargs, value *args) {
    return _evalelement(v, MORPHO_GETFIELD(MORPHO_SELF(args)),
                        MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                        field_readlambda_matrix, MORPHO_GETARG(args, 1));
}

value Field_elementdofs__int(vm *v, int nargs, value *args) {
    objectfield *field=MORPHO_GETFIELD(MORPHO_SELF(args));
    objectlist *list = NULL;
    if (!MORPHO_ISFESPACE(field->fnspc)) return MORPHO_NIL;

    fespace *disc = MORPHO_GETFESPACE(field->fnspc)->fespace;
    elementid el = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    fieldindx findx[disc->nnodes];
    if (!field_getelementdofs(field, disc, el, findx)) return MORPHO_NIL;

    list = object_newlist(0, NULL);
    if (!list) goto field_elementdofs_cleanup;

    for (int i=0; i<disc->nnodes; i++) {
        value entries[3] = {
            MORPHO_INTEGER(findx[i].g),
            MORPHO_INTEGER(findx[i].id),
            MORPHO_INTEGER(findx[i].indx)
        };
        objecttuple *tuple = object_newtuple(3, entries);
        if (!tuple) goto field_elementdofs_cleanup;
        list_append(list, MORPHO_OBJECT(tuple));
    }

    return morpho_wrapandbindrecursive(v, (object *) list);

field_elementdofs_cleanup:
    if (list) {
        for (unsigned int i=0; i<list->val.count; i++) {
            value el=list->val.data[i];
            if (MORPHO_ISOBJECT(el)) object_free(MORPHO_GETOBJECT(el));
        }
        object_free((object *) list);
    }

    MORPHO_RAISE(v, ERROR_ALLOCATIONFAILED);
}

/** Get the matrix that stores the Field */
value Field_linearize(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    return morpho_wrapandbind(v, (object *) matrix_clone(&f->data));
}

/** Directly the matrix that stores the Field
 @warning only use when you know what you're doing.  */
value Field_unsafelinearize(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    return MORPHO_OBJECT(&f->data);
}

MORPHO_BEGINCLASS(Field)
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "_ (Int)", Field_getindex__int, MORPHO_FN_PUREFN|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "_ (Int, Int)", Field_getindex__int_int, MORPHO_FN_PUREFN|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "_ (Int, Int, Int)", Field_getindex__int_int_int, MORPHO_FN_PUREFN|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int, _)", Field_setindex__int_x, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int, Int, _)", Field_setindex__int_int_x, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int, Int, Int, _)", Field_setindex__int_int_int_x, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ENUMERATE_METHOD, "_ (Int)", Field_enumerate__int, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int ()", Field_count, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_SUM_METHOD, "_ ()", Field_sum, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ASSIGN_METHOD, "(Field)", Field_assign__field, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ASSIGN_METHOD, "(Matrix)", Field_assign__matrix, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Field (Field)", Field_add__field, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Field (Int)", Field_add__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Field (Float)", Field_add__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Field (Complex)", Field_add__complex, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "Field (Nil)", Field_addr__nil, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "Field (Int)", Field_addr__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "Field (Float)", Field_addr__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "Field (Complex)", Field_add__complex, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "Field (Field)", Field_sub__field, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "Field (Int)", Field_sub__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "Field (Float)", Field_sub__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "Field (Nil)", Field_subr__nil, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "Field (Int)", Field_subr__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "Field (Float)", Field_subr__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ACC_METHOD, "(Int, Field)", Field_acc__number_field, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ACC_METHOD, "(Float, Field)", Field_acc__number_field, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "Field (Int)", Field_mul__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "Field (Float)", Field_mul__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "Field (Complex)", Field_mul__complex, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "Field (Int)", Field_mul__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "Field (Float)", Field_mul__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "Field (Complex)", Field_mul__complex, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "Field (Int)", Field_div__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "Field (Float)", Field_div__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MATRIX_INNER_METHOD, "Float (Field)", Field_inner__field, MORPHO_FN_PUREFN|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MATRIX_NORM_METHOD, "Float ()", Field_norm, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(FIELD_OP_METHOD, "Field (Callable, ...)", Field_op__callable, MORPHO_FN_REENTRANT|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "()", Field_print, MORPHO_FN_IO),
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, "Field ()", Field_clone, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(FIELD_SHAPE_METHOD, "Tuple ()", Field_shape, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(FIELD_FESPACE_METHOD, "_ ()", Field_fnspace, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(FIELD_PROTOTYPE_METHOD, "_ ()", Field_prototype, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(FIELD_MESH_METHOD, "Mesh ()", Field_mesh, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(FIELD_EVALELEMENT_METHOD, "_ (Int, List)", Field_evalelement__int_list, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES),
MORPHO_METHOD_SIGNATURE(FIELD_EVALELEMENT_METHOD, "_ (Int, Matrix)", Field_evalelement__int_matrix, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES),
MORPHO_METHOD_SIGNATURE(FIELD_ELEMENTDOFS_METHOD, "List (Int)", Field_elementdofs__int, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(FIELD_LINEARIZE_METHOD, "Matrix ()", Field_linearize, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(FIELD__LINEARIZE_METHOD, "Matrix ()", Field_unsafelinearize, MORPHO_FN_PUREFN)
MORPHO_ENDCLASS

/** Typed signatures for a Field kind; implementations stay on Field. */
#define FIELD_KIND_METHODS(cls, elt) \
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, elt " (Int)", Field_getindex__int, MORPHO_FN_PUREFN|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, elt " (Int, Int)", Field_getindex__int_int, MORPHO_FN_PUREFN|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, elt " (Int, Int, Int)", Field_getindex__int_int_int, MORPHO_FN_PUREFN|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_SUM_METHOD, elt " ()", Field_sum, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, cls " (" cls ")", Field_add__field, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, cls " (Field)", Field_add__field, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, cls " (Int)", Field_add__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, cls " (Float)", Field_add__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, cls " (Nil)", Field_addr__nil, MORPHO_FN_PUREFN), \
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, cls " (Int)", Field_addr__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, cls " (Float)", Field_addr__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, cls " (" cls ")", Field_sub__field, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, cls " (Field)", Field_sub__field, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, cls " (Int)", Field_sub__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, cls " (Float)", Field_sub__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, cls " (Nil)", Field_subr__nil, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, cls " (Int)", Field_subr__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, cls " (Float)", Field_subr__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, cls " (Int)", Field_mul__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, cls " (Float)", Field_mul__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, cls " (Int)", Field_mul__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, cls " (Float)", Field_mul__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, cls " (Int)", Field_div__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, cls " (Float)", Field_div__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, cls " ()", Field_clone, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(FIELD_EVALELEMENT_METHOD, elt " (Int, List)", Field_evalelement__int_list, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES), \
MORPHO_METHOD_SIGNATURE(FIELD_EVALELEMENT_METHOD, elt " (Int, Matrix)", Field_evalelement__int_matrix, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES)

#define FIELD_COMPLEX_ARITH(cls) \
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, cls " (Complex)", Field_add__complex, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, cls " (Complex)", Field_add__complex, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, cls " (Complex)", Field_mul__complex, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS), \
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, cls " (Complex)", Field_mul__complex, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS)

MORPHO_BEGINCLASS(ScalarField)
FIELD_KIND_METHODS("ScalarField", "Float")
MORPHO_ENDCLASS

MORPHO_BEGINCLASS(MatrixField)
FIELD_KIND_METHODS("MatrixField", "Matrix")
MORPHO_ENDCLASS

MORPHO_BEGINCLASS(ComplexField)
FIELD_KIND_METHODS("ComplexField", "Complex"),
FIELD_COMPLEX_ARITH("ComplexField")
MORPHO_ENDCLASS

MORPHO_BEGINCLASS(ComplexMatrixField)
FIELD_KIND_METHODS("ComplexMatrixField", "ComplexMatrix"),
FIELD_COMPLEX_ARITH("ComplexMatrixField")
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************* */

static void field_finalize(void) {
    MorphoMutex_clear(&field_poolmutex);
}

void field_initialize(void) {
    MorphoMutex_init(&field_poolmutex);
    morpho_addfinalizefn(field_finalize);

    objectfieldtype=object_addtype(&objectfielddefn);
    objectscalarfieldtype=object_addtype(&objectfielddefn);
    objectmatrixfieldtype=object_addtype(&objectfielddefn);
    objectcomplexfieldtype=object_addtype(&objectfielddefn);
    objectcomplexmatrixfieldtype=object_addtype(&objectfielddefn);
    
    field_gradeoption=builtin_internsymbolascstring(FIELD_GRADEOPTION);
    field_functionspaceoption=builtin_internsymbolascstring(FIELD_FESPACEOPTION);
    
#define FIELD_CONS_FLGS (MORPHO_FN_CONSTRUCTOR|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_OPTARGS)
    /* Field() is a factory: it never allocates OBJECT_FIELD, only a leaf kind. */
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh)", field_constructor__mesh, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh, Int)", field_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh, Float)", field_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh, Complex)", field_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh, Matrix)", field_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh, ComplexMatrix)", field_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh, Callable)", field_constructor__mesh_fn, FIELD_CONS_FLGS|MORPHO_FN_REENTRANT, NULL);

    morpho_addfunction(SCALARFIELD_CLASSNAME, "ScalarField (Mesh)", scalarfield_constructor__mesh, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(SCALARFIELD_CLASSNAME, "ScalarField (Mesh, Int)", scalarfield_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(SCALARFIELD_CLASSNAME, "ScalarField (Mesh, Float)", scalarfield_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(SCALARFIELD_CLASSNAME, "ScalarField (Mesh, Callable)", scalarfield_constructor__mesh_fn, FIELD_CONS_FLGS|MORPHO_FN_REENTRANT, NULL);

    morpho_addfunction(MATRIXFIELD_CLASSNAME, "MatrixField (Mesh, Matrix)", matrixfield_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(MATRIXFIELD_CLASSNAME, "MatrixField (Mesh, Callable)", matrixfield_constructor__mesh_fn, FIELD_CONS_FLGS|MORPHO_FN_REENTRANT, NULL);

    morpho_addfunction(COMPLEXFIELD_CLASSNAME, "ComplexField (Mesh)", complexfield_constructor__mesh, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(COMPLEXFIELD_CLASSNAME, "ComplexField (Mesh, Complex)", complexfield_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(COMPLEXFIELD_CLASSNAME, "ComplexField (Mesh, Int)", complexfield_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(COMPLEXFIELD_CLASSNAME, "ComplexField (Mesh, Float)", complexfield_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(COMPLEXFIELD_CLASSNAME, "ComplexField (Mesh, Callable)", complexfield_constructor__mesh_fn, FIELD_CONS_FLGS|MORPHO_FN_REENTRANT, NULL);

    morpho_addfunction(COMPLEXMATRIXFIELD_CLASSNAME, "ComplexMatrixField (Mesh, ComplexMatrix)", complexmatrixfield_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(COMPLEXMATRIXFIELD_CLASSNAME, "ComplexMatrixField (Mesh, Callable)", complexmatrixfield_constructor__mesh_fn, FIELD_CONS_FLGS|MORPHO_FN_REENTRANT, NULL);
    
    value objclass = builtin_findclassfromcstring(OBJECT_CLASSNAME);
    
    value fieldclass=builtin_addclass(FIELD_CLASSNAME, MORPHO_GETCLASSDEFINITION(Field), objclass);
    object_setveneerclass(OBJECT_FIELD, fieldclass);

    value scalarfieldclass=builtin_addclass(SCALARFIELD_CLASSNAME, MORPHO_GETCLASSDEFINITION(ScalarField), fieldclass);
    object_setveneerclass(OBJECT_SCALARFIELD, scalarfieldclass);

    value matrixfieldclass=builtin_addclass(MATRIXFIELD_CLASSNAME, MORPHO_GETCLASSDEFINITION(MatrixField), fieldclass);
    object_setveneerclass(OBJECT_MATRIXFIELD, matrixfieldclass);

    value complexfieldclass=builtin_addclass(COMPLEXFIELD_CLASSNAME, MORPHO_GETCLASSDEFINITION(ComplexField), fieldclass);
    object_setveneerclass(OBJECT_COMPLEXFIELD, complexfieldclass);

    value complexmatrixfieldclass=builtin_addclass(COMPLEXMATRIXFIELD_CLASSNAME, MORPHO_GETCLASSDEFINITION(ComplexMatrixField), fieldclass);
    object_setveneerclass(OBJECT_COMPLEXMATRIXFIELD, complexmatrixfieldclass);
    
    morpho_defineerror(FIELD_INDICESOUTSIDEBOUNDS, ERROR_HALT, FIELD_INDICESOUTSIDEBOUNDS_MSG);
    morpho_defineerror(FIELD_INCOMPATIBLEMATRICES, ERROR_HALT, FIELD_INCOMPATIBLEMATRICES_MSG);
    morpho_defineerror(FIELD_INCOMPATIBLEVAL, ERROR_HALT, FIELD_INCOMPATIBLEVAL_MSG);
    morpho_defineerror(FIELD_ARGS, ERROR_HALT, FIELD_ARGS_MSG);
    morpho_defineerror(FIELD_OP, ERROR_HALT, FIELD_OP_MSG);
    morpho_defineerror(FIELD_OPRETURN, ERROR_HALT, FIELD_OPRETURN_MSG);
    morpho_defineerror(FIELD_KIND, ERROR_HALT, FIELD_KIND_MSG);
    morpho_defineerror(FIELD_IDXFALLBACK, ERROR_WARNING, FIELD_IDXFALLBACK_MSG);
}

#endif
