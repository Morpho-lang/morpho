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

/** Field object definitions */
void objectfield_printfn(object *obj, void *v) {
    morpho_printf(v, "<Field>");
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
    if (f->pool) MORPHO_FREE(f->pool);
}

size_t objectfield_sizefn(object *obj) {
    return sizeof(objectfield)+(((objectfield *) obj)->ngrades * sizeof(int));
}

objecttypedefn objectfielddefn = {
    .printfn=objectfield_printfn,
    .markfn=objectfield_markfn,
    .freefn=objectfield_freefn,
    .sizefn=objectfield_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/* **********************************************************************
 * Object constructors
 * ********************************************************************** */

/** Checks if a prototype object is acceptable */
bool field_checkprototype(value v) {
    return (MORPHO_ISNUMBER(v) || MORPHO_ISMATRIX(v) || MORPHO_ISSPARSE(v));
}

unsigned int field_sizeprototype(value prototype) {
    unsigned int size = 1;
    
    if (MORPHO_ISMATRIX(prototype)) {
        objectmatrix *m = (MORPHO_GETMATRIX(prototype));
        size = m->ncols*m->nrows;
    }
    
    return size;
}

/** Determines the overall size of storage required for the field
 * @param[in] mesh - mesh to use
 * @param[in] prototype - prototype object
 * @param[in] ngrades - size of grade array
 * @param[in] dof - number of degrees of freedom per grade
 * @param[out] offsets - offsets into the store (ngrades + 1 elements)
 * @returns the overall size of storage required */
unsigned int field_size(objectmesh *mesh, value prototype, unsigned int ngrades, unsigned int *dof, unsigned int *offsets) {
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
 * @param[in] disc - a prototype object
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
        new = (objectfield *) object_new(sizeof(objectfield)+sizeof(double)*size, OBJECT_FIELD);
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

        new->pool=NULL;

        /* Initialize the store */
        object_init(&new->data.obj, OBJECT_MATRIX);
        new->data.ncols=1;
        new->data.nrows=size;
        new->data.nvals=1;
        new->data.nels=new->data.ncols*new->data.nrows*new->data.nvals;
        new->data.elements=new->data.matrixdata;

        if (MORPHO_ISMATRIX(prototype)) {
            objectmatrix *mat = MORPHO_GETMATRIX(prototype);
            int mel = mat->ncols*mat->nrows;
            for (unsigned int i=0; i<new->nelements; i++) {
                memcpy(new->data.elements+i*mel, mat->elements, sizeof(double)*mel);
            }
        } else if (MORPHO_ISNUMBER(prototype)) {
            double val;
            if (morpho_valuetofloat(prototype, &val)) { // Set all elements to the provided value
                for (unsigned int i=0; i<size; i++) new->data.elements[i]=val;
            }
        } else memset(new->data.elements, 0, sizeof(double)*size);

    } else { // Cleanup partially allocated structure
        if (noffset) MORPHO_FREE(noffset);
        if (ndof) MORPHO_FREE(ndof);
    }

    return new;
}

/** Applies an initialization function to every vertex */
bool field_applyfunctiontovertices(vm *v, objectmesh *mesh, value fn, objectfield *field) {
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
bool field_applyfunctiontoelements(vm *v, objectmesh *mesh, value fn, value fnspc, objectfield *field) {
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
 * @returns field object or NULL on failure */
objectfield *field_newwithfunction(vm *v, objectmesh *mesh, value fn, value fnspc) {
    value ret=MORPHO_NIL; // Return value
    value coords[mesh->dim]; // Vertex coords
    objectfield *new = NULL;
    int handle = -1;

    /* Use the first element to find a prototype **/
    if (mesh_getvertexcoordinatesasvalues(mesh, 0, coords)) {
        if (!morpho_call(v, fn, mesh->dim, coords, &ret)) goto field_newwithfunction_cleanup;
        if (MORPHO_ISOBJECT(ret)) handle=morpho_retainobjects(v, 1, &ret);
    }

    new=object_newfield(mesh, ret, fnspc, NULL);

    if (new) {
        if (MORPHO_ISFESPACE(fnspc)) {
            if (!field_applyfunctiontoelements(v, mesh, fn, fnspc, new)) goto field_newwithfunction_cleanup;
        } else {
            if (!field_applyfunctiontovertices(v, mesh, fn, new)) goto field_newwithfunction_cleanup;
        }
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
    memset(f->data.elements, 0, sizeof(double)*(f->data.nrows));
}

/** Adds the object pool. This is a collection of statically allocated objects.
    Serialized so workers indexing a matrix field cannot observe a half-built pool. */
bool field_addpool(objectfield *f) {
    if (f->pool) return true;
    if (!MORPHO_ISMATRIX(f->prototype)) return false;

    MorphoMutex_lock(&field_poolmutex);
    bool success=true;
    if (!f->pool) {
        unsigned int nel = f->nelements;
        objectmatrix *prototype=MORPHO_GETMATRIX(f->prototype);
        objectmatrix *m = MORPHO_MALLOC(sizeof(objectmatrix)*nel);
        if (!m) {
            success=false;
        } else {
            for (unsigned int i=0; i<nel; i++) {
                object_init(&m[i].obj, OBJECT_MATRIX);
                m[i].elements=f->data.elements+i*f->psize;
                m[i].ncols=prototype->ncols;
                m[i].nrows=prototype->nrows;
                m[i].nvals=prototype->nvals;
                m[i].nels=m[i].ncols*m[i].nrows*m[i].nvals;
            }
            f->pool=m;
        }
    }
    MorphoMutex_unlock(&field_poolmutex);
    return success;
}

/** Clones a field */
objectfield *field_clone(objectfield *f) {
    objectfield *new = object_newfield(f->mesh, f->prototype, f->fnspc, f->dof);
    if (new) memcpy(new->data.elements, f->data.elements, f->data.nrows*sizeof(double));
    return new;
}

/* **********************************************************************
 * Field operations
 * ********************************************************************* */

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
    
    if (MORPHO_ISNIL(field->prototype)) {
        *out=MORPHO_FLOAT(field->data.elements[ix]);
        return true;
    } else if (MORPHO_ISMATRIX(field->prototype)) {
        if (!field->pool) field_addpool(field);
        if (field->pool) {
            objectmatrix *mpool = (objectmatrix *) field->pool;
            *out = MORPHO_OBJECT(&mpool[ix]);
            return true;
        }
    }
    return false;
}

/** Retrieve a value from a field object given a single index
 * @param[in] field - field to use
 * @param[in] indx - index within the element
 * @param[out] out - the retrieved value
 * @return true on success */
bool field_getelementwithindex(objectfield *field, int indx, value *out) {
    if (MORPHO_ISNIL(field->prototype)) {
        *out=MORPHO_FLOAT(field->data.elements[indx]);
        return true;
    } else if (MORPHO_ISMATRIX(field->prototype)) {
        if (!field->pool) field_addpool(field);
        if (field->pool) {
            objectmatrix *mpool = (objectmatrix *) field->pool;
            *out = MORPHO_OBJECT(&mpool[indx]);
            return true;
        }
    }
    return false;
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
    bool success=false;
    unsigned int ix=field->offset[grade]+field->dof[grade]*el+indx;
    if (!(ix<field->offset[grade+1] && indx<field->dof[grade])) return false;
    
    if (MORPHO_ISNIL(field->prototype)) {
        *out = &field->data.elements[ix];
        *nentries=1;
        success=true;
    } else if (MORPHO_ISMATRIX(field->prototype)) {
        *out = &field->data.elements[ix*(field->psize)];
        *nentries=field->psize;
        success=true;
    }
    return success;
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

    fespace *disc = MORPHO_GETFESPACE(field->fnspc)->fespace;
    fieldindx findx[disc->nnodes];
    if (!field_getelementdofs(field, disc, el, findx)) return false;

    double wts[disc->nnodes];
    disc->ifn(lambda, wts);

    if (MORPHO_ISNIL(field->prototype)) {
        double accum = 0.0;
        for (int i=0; i<disc->nnodes; i++) {
            value q;
            double val;
            if (!field_getelement(field, findx[i].g, findx[i].id, findx[i].indx, &q)) return false;
            if (!morpho_valuetofloat(q, &val)) return false;
            accum += wts[i]*val;
        }
        *out = MORPHO_FLOAT(accum);
        return true;
    } else if (MORPHO_ISMATRIX(field->prototype)) {
        objectmatrix *proto = MORPHO_GETMATRIX(field->prototype);
        objectmatrix *accum = matrix_new(proto->nrows, proto->ncols, true);
        if (!accum) return false;

        for (int i=0; i<disc->nnodes; i++) {
            unsigned int nentries;
            double *entries;
            if (!field_getelementaslist(field, findx[i].g, findx[i].id, findx[i].indx, &nentries, &entries)) {
                object_free((object *) accum);
                return false;
            }
            for (unsigned int j=0; j<nentries; j++) accum->elements[j] += wts[i]*entries[j];
        }

        *out = MORPHO_OBJECT(accum);
        return true;
    }

    return false;
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
    
    if (MORPHO_ISNIL(field->prototype)) {
        if (MORPHO_ISNUMBER(val)) {
            return morpho_valuetofloat(val, &field->data.elements[ix]);
        }
    } else {
        unsigned int psize = field_sizeprototype(val);
        if (MORPHO_ISMATRIX(val)) {
            objectmatrix *m = MORPHO_GETMATRIX(val);
            if (psize==field->psize) {
                memcpy(field->data.elements+ix*psize, m->elements, psize*sizeof(double));
                return true;
            }
        }
    }
    return false;
}

/** Sets the value of an entry in a field object given a single index
 * @param[in] field - field to use
 * @param[in] ix - index of the element
 * @param[in] val - value to set
 * @return true on success */
bool field_setelementwithindex(objectfield *field, int ix, value val) {
    if (ix>=field->nelements) return false;
    
    if (MORPHO_ISNIL(field->prototype)) {
        if (MORPHO_ISNUMBER(val)) {
            return morpho_valuetofloat(val, &field->data.elements[ix]);
        }
    } else {
        unsigned int psize = field_sizeprototype(val);
        if (MORPHO_ISMATRIX(val)) {
            objectmatrix *m = MORPHO_GETMATRIX(val);
            if (psize==field->psize) {
                memcpy(field->data.elements+ix*psize, m->elements, psize*sizeof(double));
                return true;
            }
        }
    }
    return false;
}

/** Checks if two fields have the same shape */
bool field_compareshape(objectfield *a, objectfield *b) {
    if (a->data.nrows==b->data.nrows &&
        a->ngrades==b->ngrades) {
        for (unsigned int i=0; i<a->ngrades; i++) {
            if (a->dof[i]!=b->dof[i]) return false;
        }
        return true;
    }
    return false;
}

/** Returns the number of degrees of freedom in a given grade */
unsigned int field_dofforgrade(objectfield *f, grade g) {
    return (g<=f->ngrades ? f->dof[g] : 0);
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
bool field_add(objectfield *left, objectfield *right, objectfield *out) {
    return (matrix_copy(&left->data, &out->data)==LINALGERR_OK &&
            matrix_axpy(1.0, &right->data, &out->data)==LINALGERR_OK);
}

/** Subtracts one field from another */
bool field_sub(objectfield *left, objectfield *right, objectfield *out) {
    return (matrix_copy(&left->data, &out->data)==LINALGERR_OK &&
            matrix_axpy(-1.0, &right->data, &out->data)==LINALGERR_OK);
}

/** Accumulate, i.e. a <- a + lambda*b */
bool field_accumulate(objectfield *left, double lambda, objectfield *right) {
    return (matrix_axpy(lambda, &right->data, &left->data)==LINALGERR_OK);
}

bool field_inner(objectfield *left, objectfield *right, double *out) {
    return (matrix_inner(&left->data, &right->data, out)==LINALGERR_OK);
}

/** Calls a function fn on every element of a field, optionally with other fields as arguments */
bool field_op(vm *v, value fn, objectfield *f, int nargs, objectfield **args, value *out) {
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

/** Common constructor: mesh is required; prototype or fn is the optional fill. */
static value _constructfield(vm *v, int nargs, value *args, objectmesh *mesh, value prototype, value fn) {
    value grd=MORPHO_NIL, fnspc=MORPHO_FALSE; // Guard value to detect if fnspc is set
    builtin_options(v, nargs, args, NULL, 2, field_gradeoption, &grd, field_functionspaceoption, &fnspc);

    unsigned int ngrades=mesh_maxgrade(mesh)+1; // Count dofs
    unsigned int dof[ngrades];
    for (unsigned int i=0; i<ngrades; i++) dof[i]=0;

    unsigned int *shape=NULL; // Extract layout from optional arguments
    if (!field_layoutfromoptions(v, mesh, grd, &fnspc, ngrades, dof, &shape)) return MORPHO_NIL;

    objectfield *new=NULL; // Choose constructor pattern
    if (MORPHO_ISNIL(fn)) {
        new=object_newfield(mesh, prototype, fnspc, shape);
    } else {
        new=field_newwithfunction(v, mesh, fn, fnspc);
    }

    if (!new && MORPHO_ISOBJECT(fnspc)) object_freeifunmanaged(MORPHO_GETOBJECT(fnspc));
    return morpho_wrapandbindrecursive(v, (object *) new);
}

value field_constructor__mesh(vm *v, int nargs, value *args) {
    return _constructfield(v, nargs, args, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), MORPHO_NIL, MORPHO_NIL);
}

value field_constructor__mesh_proto(vm *v, int nargs, value *args) {
    return _constructfield(v, nargs, args, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), MORPHO_GETARG(args, 1), MORPHO_NIL);
}

value field_constructor__mesh_fn(vm *v, int nargs, value *args) {
    return _constructfield(v, nargs, args, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), MORPHO_NIL, MORPHO_GETARG(args, 1));
}

/* ----------------------------------------------
 * Method implementations
 * ---------------------------------------------- */

static value _indexget(vm *v, objectfield *f, grade g, elementid el, int indx) {
    value out = MORPHO_NIL;
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

/** Right add of nil or numeric zero */
value Field_addr__nil(vm *v, int nargs, value *args) {
    return MORPHO_SELF(args);
}

value Field_addr__number(vm *v, int nargs, value *args) {
    double x;
    if (!morpho_valuetofloat(MORPHO_GETARG(args, 0), &x) || fabs(x)>=MORPHO_EPS) MORPHO_RAISE(v, VM_INVALIDARGS);
    return MORPHO_SELF(args);
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

/** Right subtract of nil or integer zero */
value Field_subr__nil(vm *v, int nargs, value *args) {
    return _field_neg(v, MORPHO_GETFIELD(MORPHO_SELF(args)));
}

value Field_subr__int(vm *v, int nargs, value *args) {
    if (MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0))!=0) MORPHO_RAISE(v, VM_INVALIDARGS);
    return _field_neg(v, MORPHO_GETFIELD(MORPHO_SELF(args)));
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

/** Print the mesh */
value Field_print(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    if (!MORPHO_ISFIELD(self)) return Object_print(v, nargs, args);
    
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    morpho_printf(v, "<Field>\n");
    matrix_print(v, &f->data);
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
        morpho_bindobjects(v, 1, &out);
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
MORPHO_METHOD_SIGNATURE(MORPHO_ASSIGN_METHOD, "(Field)", Field_assign__field, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ASSIGN_METHOD, "(Matrix)", Field_assign__matrix, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Field (Field)", Field_add__field, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "Field (Nil)", Field_addr__nil, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "Field (Int)", Field_addr__number, MORPHO_FN_PUREFN|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADDR_METHOD, "Field (Float)", Field_addr__number, MORPHO_FN_PUREFN|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "Field (Field)", Field_sub__field, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "Field (Nil)", Field_subr__nil, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUBR_METHOD, "Field (Int)", Field_subr__int, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ACC_METHOD, "(Int, Field)", Field_acc__number_field, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ACC_METHOD, "(Float, Field)", Field_acc__number_field, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "Field (Int)", Field_mul__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_MUL_METHOD, "Field (Float)", Field_mul__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "Field (Int)", Field_mul__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_MULR_METHOD, "Field (Float)", Field_mul__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "Field (Int)", Field_div__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_DIV_METHOD, "Field (Float)", Field_div__number, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MATRIX_INNER_METHOD, "Float (Field)", Field_inner__field, MORPHO_FN_PUREFN|MORPHO_FN_THROWS),
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
    
    field_gradeoption=builtin_internsymbolascstring(FIELD_GRADEOPTION);
    field_functionspaceoption=builtin_internsymbolascstring(FIELD_FESPACEOPTION);
    
#define FIELD_CONS_FLGS (MORPHO_FN_CONSTRUCTOR|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_OPTARGS)
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh)", field_constructor__mesh, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh, Int)", field_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh, Float)", field_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh, Matrix)", field_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh, Sparse)", field_constructor__mesh_proto, FIELD_CONS_FLGS, NULL);
    morpho_addfunction(FIELD_CLASSNAME, "Field (Mesh, Callable)", field_constructor__mesh_fn, FIELD_CONS_FLGS|MORPHO_FN_REENTRANT, NULL);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    value fieldclass=builtin_addclass(FIELD_CLASSNAME, MORPHO_GETCLASSDEFINITION(Field), objclass);
    object_setveneerclass(OBJECT_FIELD, fieldclass);
    
    morpho_defineerror(FIELD_INDICESOUTSIDEBOUNDS, ERROR_HALT, FIELD_INDICESOUTSIDEBOUNDS_MSG);
    morpho_defineerror(FIELD_INCOMPATIBLEMATRICES, ERROR_HALT, FIELD_INCOMPATIBLEMATRICES_MSG);
    morpho_defineerror(FIELD_INCOMPATIBLEVAL, ERROR_HALT, FIELD_INCOMPATIBLEVAL_MSG);
    morpho_defineerror(FIELD_ARGS, ERROR_HALT, FIELD_ARGS_MSG);
    morpho_defineerror(FIELD_OP, ERROR_HALT, FIELD_OP_MSG);
    morpho_defineerror(FIELD_OPRETURN, ERROR_HALT, FIELD_OPRETURN_MSG);
}

#endif
