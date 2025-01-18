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
#include "matrix.h"

static value field_gradeoption;

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
 * Constructors
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
    for (unsigned int i=0; i<=ngrades; i++) offsets[i]=0;
    
    if (!dof) { // Assume 1 element per vertex
        size=offsets[1]=mesh_nvertices(mesh)*psize;
        for (grade i=2; i<=ngrades; i++) offsets[i]=offsets[1];
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
 * @param[in] dof -  umber of degrees of freedom per entry in each grade (should be maxgrade entries) */
objectfield *object_newfield(objectmesh *mesh, value prototype, unsigned int *dof) {
    int ngrades=mesh_maxgrade(mesh)+1;
    
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
        
        new->offset=noffset;
        memcpy(noffset, offset, sizeof(unsigned int)*(ngrades+1));
        
        new->dof=ndof;
        if (dof) {
            memcpy(ndof, dof, sizeof(unsigned int)*ngrades);
        } else {
            for (unsigned int i=0; i<ngrades; i++) ndof[i]=0;
            ndof[0]=1;
        }
        
        new->pool=NULL;
        
        /* Initialize the store */
        object_init(&new->data.obj, OBJECT_MATRIX);
        new->data.ncols=1;
        new->data.nrows=size;
        new->data.elements=new->data.matrixdata;
        
        if (MORPHO_ISMATRIX(prototype)) {
            objectmatrix *mat = MORPHO_GETMATRIX(prototype);
            int mel = mat->ncols*mat->nrows;
            for (unsigned int i=0; i<new->nelements; i++) {
                memcpy(new->data.elements+i*mel, mat->elements, sizeof(double)*mel);
            }
        } else if(MORPHO_ISNUMBER(prototype)){
            // if we have a number for our prototype set all the elements equal to it
            for (elementid i=0; i<mesh->vert->ncols; i++) {
                field_setelement(new, MESH_GRADE_VERTEX, i, 0, prototype);          
            }

        } else memset(new->data.elements, 0, sizeof(double)*size);
        
    } else { // Cleanup partially allocated structure
        if (noffset) MORPHO_FREE(noffset);
        if (ndof) MORPHO_FREE(ndof);
    }
    
    return new;
}

/** Creates a field by applying a function to the vertices of a mesh
 * @param[in] v - virtual machine to use for function calls
 * @param[in] mesh - mesh to use
 * @param[in] fn - function to call
 * @returns field object or NULL on failure */
objectfield *field_newwithfunction(vm *v, objectmesh *mesh, value fn) {
    objectmatrix *vert=mesh->vert;
    int nv = vert->ncols;
    value ret=MORPHO_NIL; // Return value
    value coords[mesh->dim]; // Vertex coords
    objectfield *new = NULL;
    int handle = -1;
    
    /* Use the first element to find a prototype **/
    if (mesh_getvertexcoordinatesasvalues(mesh, 0, coords)) {
        if (!morpho_call(v, fn, mesh->dim, coords, &ret)) goto field_newwithfunction_cleanup;
        if (MORPHO_ISOBJECT(ret)) handle=morpho_retainobjects(v, 1, &ret);
    }
    
    new=object_newfield(mesh, ret, NULL);
    
    if (new) {
        for (elementid i=0; i<nv; i++) {
            // for each element in the field
            if (mesh_getvertexcoordinatesasvalues(mesh, i, coords)) {
                //get the vertex coordinates
                if (!morpho_call(v, fn, mesh->dim, coords, &ret)){
                     // if the fn call fails go to clean up this should throw an error from morpho_call
                     goto field_newwithfunction_cleanup;
                     }
                if (!field_setelement(new, MESH_GRADE_VERTEX, i, 0, ret)) {
                    // if we can't set the field value to the ouptut of the function clean up
                    morpho_runtimeerror(v,FIELD_OPRETURN);
                    goto field_newwithfunction_cleanup;
                }
            }
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

/** Adds the object pool. This is a collection of statically allocated objects */
bool field_addpool(objectfield *f) {
    unsigned int nel = f->nelements;
    if (!f->pool && MORPHO_ISMATRIX(f->prototype)) {
        objectmatrix *prototype=MORPHO_GETMATRIX(f->prototype);
        f->pool=MORPHO_MALLOC(sizeof(objectmatrix)*nel);
        if (f->pool) {
            objectmatrix *m = (objectmatrix *) f->pool;
            for (unsigned int i=0; i<nel; i++) {
                object_init(&m[i].obj, OBJECT_MATRIX);
                m[i].elements=f->data.elements+i*f->psize;
                m[i].ncols=prototype->ncols;
                m[i].nrows=prototype->nrows;
            }
        }
        return true;
    }
    return false;
}

/** Clones a field */
objectfield *field_clone(objectfield *f) {
    objectfield *new = object_newfield(f->mesh, f->prototype, f->dof);
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

/** Adds two fields together */
bool field_add(objectfield *left, objectfield *right, objectfield *out) {
    return (matrix_add(&left->data, &right->data, &out->data)==MATRIX_OK);
}

/** Subtracts one field from another */
bool field_sub(objectfield *left, objectfield *right, objectfield *out) {
    return (matrix_sub(&left->data, &right->data, &out->data)==MATRIX_OK);
}

/** Accumulate, i.e. a <- a + lambda*b */
bool field_accumulate(objectfield *left, double lambda, objectfield *right) {
    return (matrix_accumulate(&left->data, lambda, &right->data)==MATRIX_OK);
}

bool field_inner(objectfield *left, objectfield *right, double *out) {
    return (matrix_inner(&left->data, &right->data, out)==MATRIX_OK);
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
                    fld=object_newfield(f->mesh, ret, f->dof);
                    if (!fld) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }
                } else {
                    morpho_runtimeerror(v, FIELD_OPRETURN); return false;
                }
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

/** Constructs a Field object */
value field_constructor(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectfield *new=NULL;
    objectmesh *mesh=NULL; // The mesh used by the object
    value fn = MORPHO_NIL; // A function to call
    value prototype=MORPHO_NIL; // Prototype object
    
    value grd = MORPHO_NIL;
    int nfixed;
    
    if (!builtin_options(v, nargs, args, &nfixed, 1, field_gradeoption, &grd))
        morpho_runtimeerror(v, FIELD_ARGS);
    
    for (unsigned int i=0; i<nfixed; i++) {
        if (MORPHO_ISMESH(MORPHO_GETARG(args, i))) mesh = MORPHO_GETMESH(MORPHO_GETARG(args, i)); // if the ith argument is a mesh get that mesh and assign it
        else if (morpho_iscallable(MORPHO_GETARG(args, i))) fn = MORPHO_GETARG(args, i); // if the ith argurment is a function to call put that in the fn spot
        else if (field_checkprototype(MORPHO_GETARG(args, i))) prototype = MORPHO_GETARG(args, i); //if the ith argument is a prototype put that in the prototype spot
    }
    
    if (!mesh) {
        // if we don't have a mesh return a nil and thorw and error
        morpho_runtimeerror(v,FIELD_MESHARG);
        return MORPHO_NIL;
    } 
    unsigned int ngrades = mesh_maxgrade(mesh)+1;
    unsigned int dof[ngrades];
    for (unsigned int i=0; i<ngrades; i++) dof[i]=0;
    
    /* Process optional grade argument */
    if (MORPHO_ISINTEGER(grd)) {
        dof[MORPHO_GETINTEGERVALUE(grd)]=1;
    } else if (MORPHO_ISLIST(grd)) {
        objectlist *list = MORPHO_GETLIST(grd);
        if (!array_valuelisttoindices(list->val.count, list->val.data, dof)) return MORPHO_NIL;
    }
    
    if (MORPHO_ISNIL(fn)) {
        new = object_newfield(mesh, prototype, (MORPHO_ISNIL(grd) ? NULL: dof));
    } else {
        new = field_newwithfunction(v, mesh, fn);
    }
    
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

/** Gets the field element with given indices */
value Field_getindex(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    unsigned int indx[nargs];
    value out = MORPHO_NIL;
    
    if (array_valuelisttoindices(nargs, args+1, indx)) {
        grade g = (nargs>1 ? indx[0] : MESH_GRADE_VERTEX);
        elementid el = (nargs>1 ? indx[1] : indx[0]);
        int elindx = (nargs>2 ? indx[2] : 0);
        
        /* If only one index is specified, increment g to the lowest nonempty grade */
        if (nargs==1 && f->dof) while (f->dof[g]==0 && g<f->ngrades) g++;
        
        if (!field_getelement(f, g, el, elindx, &out)) morpho_runtimeerror(v, FIELD_INDICESOUTSIDEBOUNDS);
    } else morpho_runtimeerror(v, FIELD_INVLDINDICES);
    
    return out;
}

/** Sets the field element with given indices */
value Field_setindex(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    unsigned int indx[nargs];
    value out = MORPHO_NIL;
    int nindices = nargs-1;
    
    if (array_valuelisttoindices(nindices, args+1, indx)) {
        grade g = (nindices>1 ? indx[0] : MESH_GRADE_VERTEX);
        elementid el = (nindices>1 ? indx[1] : indx[0]);
        int elindx = (nindices>2 ? indx[2] : 0);
        
        /* If only one index is specified, increment g to the lowest nonempty grade */
        if (nindices==1 && f->dof) while (f->dof[g]==0 && g<f->ngrades) g++;
        
        if (!field_setelement(f, g, el, elindx, MORPHO_GETARG(args, nargs-1))) {
            morpho_runtimeerror(v, FIELD_INCOMPATIBLEVAL);
        }
    } else morpho_runtimeerror(v, FIELD_INVLDINDICES);
    
    return out;
}

/** Enumerate protocol */
value Field_enumerate(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (nargs==1) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            int i=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
            
            if (i<0) out=MORPHO_INTEGER(a->nelements);
            else if (i<a->nelements) {
                if (!field_getelementwithindex(a, i, &out)) UNREACHABLE("Could not get field element.");
            }
            /* Note no need to bind as we are an object pool */
        }
    }
    
    return out;
}


/** Number of field elements */
value Field_count(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    
    return MORPHO_INTEGER(f->nelements);
}

/** Field assign */
value Field_assign(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
 
    if (nargs==1 && MORPHO_ISFIELD(MORPHO_GETARG(args, 0))) {
        objectfield *b=MORPHO_GETFIELD(MORPHO_GETARG(args, 0));
        
        if (field_compareshape(a, b)) {
            matrix_copy(&b->data, &a->data);
        } else morpho_runtimeerror(v, FIELD_INCOMPATIBLEMATRICES);
    } else if (nargs==1 && MORPHO_ISMATRIX(MORPHO_GETARG(args, 0))) {
        objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
        
        if (matrix_copy(b, &a->data)!=MATRIX_OK) morpho_runtimeerror(v, FIELD_INCOMPATIBLEMATRICES);
    } else morpho_runtimeerror(v, FIELD_ARITHARGS);
    
    return MORPHO_NIL;
}

/** Field add */
value Field_add(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISFIELD(MORPHO_GETARG(args, 0))) {
        objectfield *b=MORPHO_GETFIELD(MORPHO_GETARG(args, 0));
        
        if (field_compareshape(a, b)) {
            objectfield *new = object_newfield(a->mesh, a->prototype, a->dof);
            
            if (new) {
                out=MORPHO_OBJECT(new);
                field_add(a, b, new);
                morpho_bindobjects(v, 1, &out);
            }
        } else morpho_runtimeerror(v, FIELD_INCOMPATIBLEMATRICES);
    } else morpho_runtimeerror(v, FIELD_ARITHARGS);
    
    return out;
}

/** Right add */
value Field_addr(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
 
    if (nargs==1 && (MORPHO_ISNIL(MORPHO_GETARG(args, 0)) ||
                     MORPHO_ISNUMBER(MORPHO_GETARG(args, 0)))) {
        int i=0;
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) i=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        if (MORPHO_ISFLOAT(MORPHO_GETARG(args, 0))) i=(fabs(MORPHO_GETFLOATVALUE(MORPHO_GETARG(args, 0)))<MORPHO_EPS ? 0 : 1);
        
        if (i==0) {
            out=MORPHO_SELF(args);
        } else UNREACHABLE("Right addition to non-zero value.");
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}

/** Field subtraction */
value Field_sub(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISFIELD(MORPHO_GETARG(args, 0))) {
        objectfield *b=MORPHO_GETFIELD(MORPHO_GETARG(args, 0));
        
        if (field_compareshape(a, b)) {
            objectfield *new = object_newfield(a->mesh, a->prototype, a->dof);
            
            if (new) {
                out=MORPHO_OBJECT(new);
                field_sub(a, b, new);
                morpho_bindobjects(v, 1, &out);
            }
        } else morpho_runtimeerror(v, FIELD_INCOMPATIBLEMATRICES);
    } else morpho_runtimeerror(v, FIELD_ARITHARGS);
    
    return out;
}

/** Right subtract */
value Field_subr(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && (MORPHO_ISNIL(MORPHO_GETARG(args, 0)) ||
                     MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)))) {
        int i=(MORPHO_ISNIL(MORPHO_GETARG(args, 0)) ? 0 : MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)));
        
        if (i==0) {
            objectfield *new=field_clone(a);
            if (new) {
                out=MORPHO_OBJECT(new);
                matrix_scale(&new->data, -1.0);
                morpho_bindobjects(v, 1, &out);
            }
        } else morpho_runtimeerror(v, VM_INVALIDARGS);
    } else morpho_runtimeerror(v, VM_INVALIDARGS);
    
    return out;
}

/** Field accumulate */
value Field_acc(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
 
    if (nargs==2 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISFIELD(MORPHO_GETARG(args, 1))) {
        objectfield *b=MORPHO_GETFIELD(MORPHO_GETARG(args, 1));
        
        if (field_compareshape(a, b)) {
            double lambda=1.0;
            morpho_valuetofloat(MORPHO_GETARG(args, 0), &lambda);
            field_accumulate(a, lambda, b);
        } else morpho_runtimeerror(v, FIELD_INCOMPATIBLEMATRICES);
    } else morpho_runtimeerror(v, FIELD_ARITHARGS);
    
    return MORPHO_NIL;
}

/** Field multiply by a scalar */
value Field_mul(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double scale=1.0;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale)) {
            objectfield *new = field_clone(a);
            if (new) {
                out=MORPHO_OBJECT(new);
                matrix_scale(&new->data, scale);
                morpho_bindobjects(v, 1, &out);
            }
        }
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}

/** Field multiply by a scalar */
value Field_div(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        /* Division by a scalar */
        double scale=1.0;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale)) {
            if (fabs(scale)<MORPHO_EPS) MORPHO_RAISE(v, VM_DVZR);
            
            objectfield *new = field_clone(a);
            if (new) {
                out=MORPHO_OBJECT(new);
                matrix_scale(&new->data, 1.0/scale);
                morpho_bindobjects(v, 1, &out);
            }
        }
    }
    
    return out;
}

/** Frobenius inner product */
value Field_inner(vm *v, int nargs, value *args) {
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISFIELD(MORPHO_GETARG(args, 0))) {
        objectfield *b=MORPHO_GETFIELD(MORPHO_GETARG(args, 0));
        
        double prod=0.0;
        if (field_inner(a, b, &prod)) {
            out = MORPHO_FLOAT(prod);
        } else morpho_runtimeerror(v, FIELD_INCOMPATIBLEMATRICES);
    } else morpho_runtimeerror(v, FIELD_ARITHARGS);
    
    return out;
}

/** Generalized operations */
value Field_op(vm *v, int nargs, value *args) {
    objectfield *slf=MORPHO_GETFIELD(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    value fn=MORPHO_NIL;
    objectfield *flds[nargs];
    
    if (nargs>0) fn=MORPHO_GETARG(args, 0);
    if (morpho_iscallable(fn)) {
        for (unsigned int i=1; i<nargs; i++) {
            if (MORPHO_ISFIELD(MORPHO_GETARG(args, i))) flds[i-1]=MORPHO_GETFIELD(MORPHO_GETARG(args, i));
            else { morpho_runtimeerror(v, FIELD_OP); return MORPHO_NIL; }
        }
        
        if (field_op(v, fn, slf, nargs-1, flds, &out)) {
            morpho_bindobjects(v, 1, &out);
        }
    } else morpho_runtimeerror(v, FIELD_OP);
    
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
    value out=MORPHO_NIL;
    objectfield *a=MORPHO_GETFIELD(MORPHO_SELF(args));
    objectfield *new=field_clone(a);
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    return out;
}

/** Get the shape (number of dofs per grade) of a field */
value Field_shape(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));

    value shape[f->ngrades];
    for (unsigned int i=0; i<f->ngrades; i++) {
        shape[i]=MORPHO_INTEGER(f->dof[i]);
    }
    
    objectlist *new=object_newlist(f->ngrades, shape);
    if (new) {
        out = MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

/** Get the mesh associated with a field */
value Field_mesh(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    
    return MORPHO_OBJECT(f->mesh);
}

/** Get the matrix that stores the Field */
value Field_linearize(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    value out = MORPHO_NIL;
    
    objectmatrix *m=object_clonematrix(&f->data);
    if (m) {
        out = MORPHO_OBJECT(m);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

/** Directly the matrix that stores the Field
 @warning only use when you know what you're doing.  */
value Field_unsafelinearize(vm *v, int nargs, value *args) {
    objectfield *f=MORPHO_GETFIELD(MORPHO_SELF(args));
    
    return MORPHO_OBJECT(&f->data);
}

MORPHO_BEGINCLASS(Field)
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, Field_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, Field_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, Field_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Field_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ASSIGN_METHOD, Field_assign, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADD_METHOD, Field_add, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADDR_METHOD, Field_addr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUB_METHOD, Field_sub, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUBR_METHOD, Field_subr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ACC_METHOD, Field_acc, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_MUL_METHOD, Field_mul, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_MULR_METHOD, Field_mul, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_DIV_METHOD, Field_div, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_INNER_METHOD, Field_inner, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FIELD_OP_METHOD, Field_op, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Field_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Field_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FIELD_SHAPE_METHOD, Field_shape, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FIELD_MESH_METHOD, Field_mesh, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FIELD_LINEARIZE_METHOD, Field_linearize, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FIELD__LINEARIZE_METHOD, Field_unsafelinearize, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************* */

void field_initialize(void) {
    objectfieldtype=object_addtype(&objectfielddefn);
    
    field_gradeoption=builtin_internsymbolascstring(FIELD_GRADEOPTION);
    
    builtin_addfunction(FIELD_CLASSNAME, field_constructor, BUILTIN_FLAGSEMPTY);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    value fieldclass=builtin_addclass(FIELD_CLASSNAME, MORPHO_GETCLASSDEFINITION(Field), objclass);
    object_setveneerclass(OBJECT_FIELD, fieldclass);
    
    morpho_defineerror(FIELD_INDICESOUTSIDEBOUNDS, ERROR_HALT, FIELD_INDICESOUTSIDEBOUNDS_MSG);
    morpho_defineerror(FIELD_INVLDINDICES, ERROR_HALT, FIELD_INVLDINDICES_MSG);
    morpho_defineerror(FIELD_ARITHARGS, ERROR_HALT, FIELD_ARITHARGS_MSG);
    morpho_defineerror(FIELD_INCOMPATIBLEMATRICES, ERROR_HALT, FIELD_INCOMPATIBLEMATRICES_MSG);
    morpho_defineerror(FIELD_INCOMPATIBLEVAL, ERROR_HALT, FIELD_INCOMPATIBLEVAL_MSG);
    morpho_defineerror(FIELD_ARGS, ERROR_HALT, FIELD_ARGS_MSG);
    morpho_defineerror(FIELD_OP, ERROR_HALT, FIELD_OP_MSG);
    morpho_defineerror(FIELD_OPRETURN, ERROR_HALT, FIELD_OPRETURN_MSG);
    morpho_defineerror(FIELD_MESHARG, ERROR_HALT, FIELD_MESHARG_MSG);
}

#endif
