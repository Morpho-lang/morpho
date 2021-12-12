/** @file mesh.c
 *  @author T J Atherton
 *
 *  @brief Mesh class and associated functionality
 */

#include "object.h"
#include "builtin.h"
#include "mesh.h"
#include "file.h"
#include "varray.h"
#include "parse.h"
#include "veneer.h"
#include "sparse.h"
#include "matrix.h"

void mesh_link(objectmesh *mesh, object *obj);

DEFINE_VARRAY(elementid, elementid);

/* **********************************************************************
 * Create mesh objects
 * ********************************************************************** */

objectmesh *object_newmesh(unsigned int dim, unsigned int nv, double *v) {
    objectmesh *new = MORPHO_MALLOC(sizeof(objectmesh));

    if (new) {
        object_init((object *) new, OBJECT_MESH);

        new->dim=dim;
        new->conn=NULL;
        new->vert=object_newmatrix(dim, nv, false);
        new->link=NULL;
        if (new->vert) {
            mesh_link(new, (object *) new->vert);
            if (dim>0){
                memcpy(new->vert->elements, v, sizeof(double)*dim*nv);
            }
        }
    }

    return new;
}

/** Links an object to the mesh; used to keep track of unbound child objects */
void mesh_link(objectmesh *mesh, object *obj) {
    for (object *e = mesh->link; e!=NULL; e=e->next) if (e==obj) return;

    if (obj->status==OBJECT_ISUNMANAGED && obj->next==NULL) {
        obj->next=mesh->link;
        mesh->link=obj;
    }
}

/** Delinks an object from the mesh; used to keep track of unbound child objects */
void mesh_delink(objectmesh *mesh, object *obj) {
    if (mesh->link==obj) { // If the first, simply delink
        mesh->link=obj->next;
        return;
    }

    // Otherwise, search and delink once the object is found
    for (object *e = mesh->link; e!=NULL; e=e->next) {
        if (e->next==obj) {
            e->next=obj->next;
            break;
        }
    }
}

/* **********************************************************************
 * Manipulate mesh objects
 * ********************************************************************** */

/* -------------------------------------
 * Vertices
 * ------------------------------------- */

/** Gets vertex coordinates */
bool mesh_getvertexcoordinates(objectmesh *mesh, elementid id, double *out) {
    double *coords;
    if (matrix_getcolumn(mesh->vert, id, &coords)) {
        for (unsigned int i=0; i<mesh->dim; i++) out[i]=coords[i];
        return true;
    }
    return false;
}

/** Gets vertex coordinates as a list */
bool mesh_getvertexcoordinatesaslist(objectmesh *mesh, elementid id, double **out) {
    double *coords=NULL;
    if (matrix_getcolumn(mesh->vert, id, &coords)) {
        *out=coords;
    }
    return coords;
}

/** Gets vertex coordinates */
bool mesh_setvertexcoordinates(objectmesh *mesh, elementid id, double *x) {;
    return matrix_setcolumn(mesh->vert, id, x);
}

/** Gets vertex coordinates as a value list */
bool mesh_getvertexcoordinatesasvalues(objectmesh *mesh, elementid id, value *val) {
    double *x=NULL; // The vertex positions

    bool success=matrix_getcolumn(mesh->vert, id, &x);

    if (success) {
        for (unsigned int i=0; i<mesh->dim; i++) val[i]=MORPHO_FLOAT(x[i]);
    }

    return success;
}

/** Finds the nearest vertex to a point
 * @param[in] mesh - mesh to search
 * @param[in] x - position [should be of mesh->dim
 * @param[out] id - closest vertex
 * @param[out] separation (optional)
 * @returns true on success. */
bool mesh_nearestvertex(objectmesh *mesh, double *x, elementid *id, double *separation) {
    double *vx;
    double best=0, sep=0;
    elementid bestid=0;

    for (elementid i=0; i<mesh_nvertices(mesh); i++) {
        if (!matrix_getcolumn(mesh->vert, i, &vx)) return false;
        sep=0;
        for (int k=0; k<mesh->dim; k++) sep+=(vx[k]-x[k])*(vx[k]-x[k]);
        if (i==0 || sep<best) { best=sep; bestid = i; }
    }
    *id = bestid;
    if (separation) *separation=sqrt(best);
    return true;
}


/* -------------------------------------
 * The connectivity array
 * ------------------------------------- */

/** Ensures a mesh has a valid connectivity array */
bool mesh_checkconnectivity(objectmesh *mesh) {
    if (mesh->conn) return true;
    unsigned int dim[2]={mesh->dim+1, mesh->dim+1};
    mesh->conn=object_newarray(2, dim);

    return (mesh->conn);
}

/** Freezes mesh connectivity, converting subsidiary data structures to fixed but efficient versions */
void mesh_freezeconnectivity(objectmesh *mesh) {
    if (!mesh_checkconnectivity(mesh)) return;

    for (unsigned int i=0; i<mesh->dim+1; i++) {
        for (unsigned int j=0; j<mesh->dim+1; j++) {
            objectsparse *s=mesh_getconnectivityelement(mesh, i, j);
            if (s) sparse_checkformat(s, SPARSE_CCS, true, false);
        }
    }
}

/** Creates a new blank connectivity element */
objectsparse *mesh_newconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col) {
    objectsparse *out=NULL;
    value indx[2]={MORPHO_INTEGER(row),MORPHO_INTEGER(col)};

    out=object_newsparse(NULL, NULL);
    if (out) array_setelement(mesh->conn, 2, indx, MORPHO_OBJECT(out));

    if (out) mesh_link(mesh, (object *) out);

    return out;
}

/** Sets a connectivity element */
bool mesh_setconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col, objectsparse *el) {
    if (row==col) return false;
    value indx[2]={MORPHO_INTEGER(row),MORPHO_INTEGER(col)};
    if (mesh_checkconnectivity(mesh)) {
        value old = MORPHO_NIL;
        if (array_getelement(mesh->conn, 2, indx, &old) &&
            MORPHO_ISOBJECT(old)) {
            object *oel = MORPHO_GETOBJECT(old);
            mesh_delink(mesh, oel);
            if (oel->status==OBJECT_ISUNMANAGED) object_free(oel);
        }

        value val = MORPHO_NIL;
        if (el) val = MORPHO_OBJECT(el);

        if (array_setelement(mesh->conn, 2, indx, val) == ARRAY_OK) {
            if (el && el->obj.status==OBJECT_ISUNMANAGED) mesh_link(mesh, (object *) el);
        }
    }
    return false;
}

/** Gets the connectivity matrix corresponding to (row, col) */
objectsparse *mesh_getconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col) {
    objectsparse *out=NULL;
    value indx[2]={MORPHO_INTEGER(row),MORPHO_INTEGER(col)};
    value matrix=MORPHO_NIL;

    if (mesh->conn) array_getelement(mesh->conn, 2, indx, &matrix);
    if (MORPHO_ISSPARSE(matrix)) {
        out=MORPHO_GETSPARSE(matrix);
    }

    return out;
}

/** How many vertices are in a matrix? */
elementid mesh_nvertices(objectmesh *mesh) {
    elementid out = 0;
    if (mesh->vert) out = mesh->vert->ncols;
    return out;
}

/** How many elements are in a connectivity matrix? */
elementid mesh_nelements(objectsparse *conn) {
    return conn->ccs.ncols;
}

/** How many elements exist in a given grade? */
elementid mesh_nelementsforgrade(objectmesh *mesh, grade g) {
    elementid count=0;
    if (g==MESH_GRADE_VERTEX) {
        count=mesh_nvertices(mesh);
    } else {
        objectsparse *conn = mesh_getconnectivityelement(mesh, 0, g);
        if (conn) count=mesh_nelements(conn);
    }
    return count;
}

/** Maximum grade in the mesh */
grade mesh_maxgrade(objectmesh *mesh) {
    for (grade g=mesh->dim; g>0; g--) {
        if (mesh_getconnectivityelement(mesh, 0, g)) return g;
    }

    return 0;
}

/** Gets connectivitiy infornation for a given element
 * @param[in] conn - the connectivity matrix for the element
 * @param[in] id - the element id
 * @param[out] nentries - number of entries
 * @param[out] entries - a list of entries
 * @returns true on success, false otherwise */
bool mesh_getconnectivity(objectsparse *conn, elementid id, int *nentries, int **entries) {
    sparse_checkformat(conn, SPARSE_CCS, true, false);

    if (conn) {
        return sparseccs_getrowindices(&conn->ccs, id, nentries, entries);
    }

    return false;
}


/* ------------------------------------------
 * Functions to modify the connectivity array
 * ------------------------------------------ */

/* Data structure to store ntuples of elementids and determine if a
   particular ntuple is already present. */
typedef struct {
    int n; // number of elements in a tuple
    elementid maxval; // Maximum elementid that will be used
    int *elementoffset; // Offset to tuples based on first element of the tuple
    varray_elementid tuples; // data store
} ntuplelist;

/* Initialize an ntuplelist data structure */
void ntuplelist_init(ntuplelist *list, int n, elementid maxval) {
    list->n=n;
    list->maxval=maxval;
    list->elementoffset=MORPHO_MALLOC(sizeof(int)*(maxval+1));
    if (list->elementoffset) for (unsigned int i=0; i<=maxval; i++) list->elementoffset[i]=-1;
    varray_elementidinit(&list->tuples);
}

/* Clear ntuplelist data structure */
void ntuplelist_clear(ntuplelist *list) {
    if (list->elementoffset) MORPHO_FREE(list->elementoffset);
    varray_elementidclear(&list->tuples);
}

/* Add a tuple */
void ntuplelist_add(ntuplelist *list, elementid *tuple) {
    unsigned int posn = list->tuples.count;
    if (posn>INT_MAX) UNREACHABLE("Overflow in ntuplelist_add");
    varray_elementidadd(&list->tuples, tuple, list->n);
    if (list->elementoffset) { // Store an offset to the first
        varray_elementidwrite(&list->tuples, list->elementoffset[tuple[0]]);
        list->elementoffset[tuple[0]]=posn;
    }
}

/* Compare two tuples */
bool ntuplelist_compare(int n, elementid *t1, elementid *t2) {
    for (unsigned int i=0; i<n; i++) if (t1[i]!=t2[i]) return false;
    return true;
}

/* Determine whether a tuple is present in an ntuplelist */
bool ntuplelist_find(ntuplelist *list, elementid *tuple) {
    if (list->elementoffset) {
        for (int i=list->elementoffset[tuple[0]]; i>=0; ) {
            if (ntuplelist_compare(list->n, &list->tuples.data[i], tuple)) return true;
            i=list->tuples.data[i+list->n];
        }
    } else {
        for (unsigned int i=0; i<list->tuples.count; i+=list->n) {
            if (ntuplelist_compare(list->n, &list->tuples.data[i], tuple)) return true;
        }
    }
    return false;
}

objectsparse *mesh_addgrade(objectmesh *mesh, grade g) {
    /* Does the grade already exist? */
    objectsparse *el=mesh_getconnectivityelement(mesh, 0, g);
    if (el) return el;
    grade maxG = mesh_maxgrade(mesh);
    grade h;
    /* Otherwise, find the next available grade above it */
    for (h=g+1; (h<=maxG) && (!el); h++) {
        el=mesh_getconnectivityelement(mesh, 0, h);
    }
    /* if this grade doesn't exist and we can't find the next available
       grade above it return NULL */
    if (!el) return NULL;

    /* Create a new sparse matrix */
    objectsparse *new=object_newsparse(NULL, NULL);
    if (!new) return NULL;

    /* Create an ntuplelist to keep track of elements already created */
    ntuplelist list;
    int n = g+1; // Number of elements in the tuple
    elementid maxvid = mesh_nvertices(mesh)-1; // Highest vertexid
    ntuplelist_init(&list, n, maxvid);

    int nel, *entries;
    elementid newid = 0;
    /* Loop over elements in the higher grade */
    for (elementid id=0; id<el->ccs.ncols; id++) {
        /* Get the associated connectivity */
        if (!mesh_getconnectivity(el, id, &nel, &entries)) break;
        
        // Initialize n-tuple and counters with [0,1,2...]
        elementid tuple[n]; // Store the tuple
        int counter[n], cmax[n]; // Counters
        for (unsigned int i=0; i<n; i++) {
            counter[i]=i; tuple[i]=entries[i]; cmax[i]=nel-n+i;
        }
        
        if (!ntuplelist_find(&list, tuple)) { // Check if the first tuple exists
            ntuplelist_add(&list, tuple);
            for (unsigned int i=0; i<n; i++) sparsedok_insert(&new->dok, tuple[i], newid, MORPHO_NIL);
            newid++;
        }
        
        /* Generate tuples */
        int k;
        while (counter[0]<cmax[0]) {
            counter[n-1]++; // Increment last counter
            for (k=n-1; k>=0 && counter[k]>cmax[k]; k--) counter[k-1]++; // Carry
            if (k<n-1) for (unsigned int i=k+1; i<n; i++) counter[i]=counter[i-1]+1;
         
            // Set up tuple from counter
            for (unsigned int i=0; i<n; i++) tuple[i]=entries[counter[i]];
            
            if (!ntuplelist_find(&list, tuple)) { // Check if we have the tuple
                ntuplelist_add(&list, tuple);
                for (unsigned int i=0; i<n; i++) sparsedok_insert(&new->dok, tuple[i], newid, MORPHO_NIL);
                newid++;
            }
        }
        
    }

    ntuplelist_clear(&list);
    
    mesh_setconnectivityelement(mesh, 0, g, new);
    mesh_link(mesh, (object *) new);
    mesh_freezeconnectivity(mesh);

    return new;
}

/** Adds a missing grade */
objectsparse *mesh_addgradeold(objectmesh *mesh, grade g) {
    if (g>1) UNREACHABLE("mesh_addgrade only supports adding grade 1.");
    /* Does the grade already exist? */
    objectsparse *el=mesh_getconnectivityelement(mesh, 0, g);
    if (el) return el;
    grade maxG = mesh_maxgrade(mesh);
    grade h;
    /* Otherwise, find the next available grade above it */
    for (h=g+1; (h<=maxG) && (!el); h++) {
        el=mesh_getconnectivityelement(mesh, 0, h);
    }
    /* if this grade doesn't exist
    and we can't find the next available grade above it return NULL */
    if (!el){
        return NULL;
    }

    /* Create a new sparse matrix */
    objectsparse *new=object_newsparse(NULL, NULL);
    if (!new) return NULL;

    /* Create a temporary sparse matrix to hold connectivity information */
    objectsparse *temp=object_newsparse(NULL, NULL);
    if (!temp) {
        object_free((object *) new);
        return NULL;
    }

    int nentries, *entries;
    elementid newid=0;
    /* Loop over elements in the higher grade */
    if (temp) for (elementid id=0; id<el->ccs.ncols; id++) {
        /* Get the associated connectivity */
        if (mesh_getconnectivity(el, id, &nentries, &entries)) {
            /* Now loop over pairs of vertices in the element */
            /* Should be n-tuples */
            for (unsigned int j=0; j<nentries; j++) {
                for (unsigned int k=j+1; k<nentries; k++) {
                    /* Sort the indices */
                    int l=(entries[j]<entries[k] ? entries[j] : entries[k]);
                    int m=(entries[j]<entries[k] ? entries[k] : entries[j]);

                    /* Does this pair already exist? */
                    if (!sparse_getelement(temp, l, m, NULL)) {
                        /* Add the element to our new matrix */
                        sparsedok_insert(&new->dok, l, newid, MORPHO_NIL);
                        sparsedok_insert(&new->dok, m, newid, MORPHO_NIL);
                        /* Keep track of elements made */
                        sparsedok_insert(&temp->dok, l, m, MORPHO_NIL);
                        newid++;
                    }
                }
            }

        }
    }

    if (temp) object_free((object *) temp);

    mesh_setconnectivityelement(mesh, 0, g, new);
    mesh_link(mesh, (object *) new);
    mesh_freezeconnectivity(mesh);

    return new;
}


/** Internal function used for sorting ids */
static int mesh_compareid(const void *a, const void *b) {
    return *((int *) a) - *((int *) b);
}

/** Find elements that match grade g in a connectivity matrix .
 * @param[in] vmatrix - the (g, 0) connectivity matrix (i.e. the vertex raising matrix for grade g)
 * @param[in] g - the grade of interest
 * @param[in] nids - number of vertex ids to match
 * @param[in] ids - list of vertex ids to match
 * @param[in] maxmatches - maximum number of matches to find
 * @param[out] nmatches - the number of matches found
 * @param[out] matches - matched vertex ids
 * @returns true on success, false otherwise */
static bool mesh_matchelements(objectsparse *vmatrix, grade g, int nids, int *ids, int maxmatches, int *nmatches, int *matches) {
    int nentries[nids], *entries[nids], length=0, k=0;

    /* Obtain connectivity information from the columns of vertex connectivity matrix */
    for (unsigned int i=0; i<nids; i++) {
        if (!mesh_getconnectivity(vmatrix, ids[i], &nentries[i], &entries[i])) return false;
        length+=nentries[i];
    }

    /* Copy ids a single list */
    int sids[length+1]; sids[length]=-1;
    for (unsigned int i=0; i<nids; i++) {
        memcpy(sids+k, entries[i], nentries[i]*sizeof(int));
        k+=nentries[i];
    }

    /* and sort it */
    qsort(sids, length, sizeof(int), mesh_compareid);

    //for (unsigned int i=0; i<length; i++) printf("%u ", sids[i]);
    //printf("\n");

    /* Now look for repeated ids */
    k=0; *nmatches=0;
    for (unsigned int i=0; i<length; i++) {
        if (sids[i+1]==sids[i]) { k++; continue; } // Keep counting if the next one is the same
        if (k==g) { // if the number of repeats matches the grade we have a match
            if (*nmatches<maxmatches) matches[*nmatches]=sids[i];
            *nmatches+=1;
        }
        k=0; // Reset counter
    }

    return true;
}

/** Adds a missing grade lowering element */
static objectsparse *mesh_addlowermatrix(objectmesh *mesh, unsigned int row, unsigned int col) {
    objectsparse *new=NULL;

    /* Try to obtain the (row, 0) element (i.e. the grade raising matrix for the row) */
    objectsparse *traise=mesh_getconnectivityelement(mesh, row, 0);
    if (!traise) traise=mesh_addconnectivityelement(mesh, row, 0);

    /* Also need to obtain the (0, col) element (i.e. the grade definition) */
    objectsparse *tlower=mesh_getconnectivityelement(mesh, 0, col);

    if (traise && tlower) {
        /* Create a new sparse matrix */
        new=object_newsparse(NULL, NULL);
        if (!new) return NULL;

        int maxmatches = (col+1)*(col+2)/2+1; // Maximum number of elements for a given grade
        int nentries, *entries, nmatches, matches[maxmatches];

        /* Loop over elements in the higher grade */
        for (elementid rid=0; rid<tlower->ccs.ncols; rid++) {
            /* Get the associated connectivity */
            if (mesh_getconnectivity(tlower, rid, &nentries, &entries)) {
                if (mesh_matchelements(traise, row, nentries, entries, maxmatches, &nmatches, matches)) {
                    if (nmatches>=maxmatches) {
                        UNREACHABLE("Too many connections.");
                    }

                    for (unsigned int i=0; i<nmatches; i++) {
                        sparsedok_insert(&new->dok, matches[i], rid, MORPHO_NIL);
                    }
                }
            }
        }

        mesh_setconnectivityelement(mesh, row, col, new);
        mesh_freezeconnectivity(mesh);
    }

    return new;
}

/** Fill in a missing connectivity element */
objectsparse *mesh_addconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col) {
    /** Does the element already exist? */
    objectsparse *el=mesh_getconnectivityelement(mesh, row, col);
    if (el) return el;

    /* If not, what kind of element is it? */
    if (row==0) { /* First row */
        /* Can't add a missing grade; use addgrade instead*/
    } else if (row<col) { /* A grade lowering element */
        el=mesh_addlowermatrix(mesh, row, col);
    } else if (row!=col) { /* A grade raising element */
        /* Try to obtain the transpose */
        objectsparse *tlow=mesh_getconnectivityelement(mesh, col, row);
        if (!tlow) tlow=mesh_addconnectivityelement(mesh, col, row);

        if (tlow) {
            el=mesh_newconnectivityelement(mesh, row, col);
            sparse_transpose(tlow, el);
        }
    }

    return el;
}


/** Adds an element to a mesh
 * @param[in] mesh the mesh
 * @param[in] g grade of the element
 * @param[in] v elementids */
bool mesh_addelementwithvertices(objectmesh *mesh, grade g, elementid *v) {
    if (!mesh_checkconnectivity(mesh)) return false;

    objectsparse *m=mesh_getconnectivityelement(mesh, 0, g);
    if (!m) m=mesh_newconnectivityelement(mesh, 0, g);
    if (!m) return false;

    int eid=m->dok.ncols; // The new element is one after the last element

    bool success=true;
    for (unsigned int i=0; i<g+1; i++) {
        if (!sparsedok_insert(&m->dok, (int) v[i], eid, MORPHO_NIL)) success=false;
    }

    return success;
}

/** Resets connectivity elements other than the first row */
void mesh_resetconnectivity(objectmesh *m) {
    grade max = mesh_maxgrade(m);
    for (grade i=1; i<=max; i++) {
        for (grade j=0; j<=max; j++) {
            mesh_setconnectivityelement(m, i, j, NULL);
        }
    }
}

/* **********************************************************************
 * Clone
 * ********************************************************************** */

/** Clones a mesh object */
objectmesh *mesh_clone(objectmesh *mesh) {
    objectmesh *new = object_newmesh(mesh->dim, mesh->vert->ncols, mesh->vert->elements);

    if (new) {
        if (mesh->conn &&
            mesh_checkconnectivity(new)) {

            grade max = mesh_maxgrade(mesh);
            for (grade i=0; i<=max; i++) {
                for (grade j=0; j<=max; j++) {
                    objectsparse *conn=mesh_getconnectivityelement(mesh, i, j);

                    if (conn) {
                        objectsparse *cl=sparse_clone(conn);
                        if (cl) mesh_setconnectivityelement(new, i, j, cl);
                    }
                }
            }

        }
    }

    return new;
}

/* **********************************************************************
 * Symmetries
 * ********************************************************************** */

/** Adds a symmetry to a mesh. */
bool mesh_addsymmetry(vm *v, objectmesh *mesh, value symmetry, objectselection *sel) {
    value method=MORPHO_NIL;
    objectstring s = MORPHO_STATICSTRING(MESH_TRANSFORM_METHOD);

    objectsparse *sym=mesh_getconnectivityelement(mesh, MESH_GRADE_VERTEX, MESH_GRADE_VERTEX);

    double x[mesh->dim];
    objectmatrix posn = MORPHO_STATICMATRIX(x, mesh->dim, 1);

    elementid nv = mesh_nvertices(mesh);

    value arg = MORPHO_OBJECT(&posn);
    value ret = MORPHO_NIL;

    if (morpho_lookupmethod(symmetry, MORPHO_OBJECT(&s), &method)) {
        /* Loop over vertices */
        for (elementid i=0; i<nv; i++) {
            /* Read the vertex coordinates into x */
            if (!mesh_getvertexcoordinates(mesh, i, x)) return false;
            /* Call transformation */
            if (!morpho_invoke(v, symmetry, method, 1, &arg, &ret)) return false;

            if (MORPHO_ISMATRIX(ret)) {
                objectmatrix *newvert=MORPHO_GETMATRIX(ret);
                elementid nearest;
                double sep;
                if (!mesh_nearestvertex(mesh, newvert->elements, &nearest, &sep)) return false;

                if (sep<MESH_NEARESTPOINTEPS) {
                    /* Only add a symmetry matrix if we have a match */
                    if (!sym) {
                        sym=mesh_newconnectivityelement(mesh, MESH_GRADE_VERTEX, MESH_GRADE_VERTEX);
                        sparsedok_setdimensions(&sym->dok, nv, nv);
                    }
                    if (!sym) return false;

                    sparse_setelement(sym, i, nearest, symmetry);
                }
            }
        }

    } else morpho_runtimeerror(v, MESH_ADDSYMMSNGTRNSFRM);

    return false;
}

/* Get a list of synonymous elements for a given element */
bool mesh_getsynonyms(objectmesh *mesh, grade g, elementid id, varray_elementid *synonymids) {
    objectsparse *sym = mesh_getconnectivityelement(mesh, g, g);
    if (sym) {
        synonymids->count=0;
        void *ctr=sparsedok_loopstart(&sym->dok);
        int row, col;
        while (sparsedok_loop(&sym->dok, &ctr, &row, &col)) {
            if (id==row) varray_elementidwriteunique(synonymids, col);
            if (id==col) varray_elementidwriteunique(synonymids, row);
        }
    }

    return true;
}

void varray_elementidwriteunique(varray_elementid *list, elementid id) {
    for (unsigned int i=0; i<list->count; i++) if (list->data[i]==id) return;
    varray_elementidwrite(list, id);
}

/** Insert ids for a given element
 * @param[in] conn - Connectivity matrix
 * @param[in] id - id to insert
 * @param[in] ignoreid - whether or not to include id if it is found
 * @param[out] out - varray to hold output */
void mesh_insertidsforelement(objectsparse *conn, elementid id, bool ignore, elementid ignoreid, varray_elementid *out) {
    int nids, *entries;
    if (sparseccs_getrowindices(&conn->ccs, id, &nids, &entries)) {
        for (unsigned int i=0; i<nids; i++) {
            if (ignore && entries[i]==ignoreid) continue;
            varray_elementidwriteunique(out, entries[i]);
        }
    }
}

#define MAX_NEIGHBORS 64
int mesh_findneighbors(objectmesh *mesh, grade g, elementid id, grade target, varray_elementid *neighbors) {
    int nvert, *vids, vvid=id; // List of vertices in the element

    /* If the element is not a point, find all vertices associated with that point */
    if (g>0) {
        objectsparse *down = mesh_getconnectivityelement(mesh, 0, g);
        sparseccs_getrowindices(&down->ccs, id, &nvert, &vids);
    } else {
        nvert = 1; vids=&vvid;
    }

    objectsparse *conn = mesh_getconnectivityelement(mesh, target, 0);
    // Now find the neighboring elements
    if (conn && sparse_checkformat(conn, SPARSE_CCS, true, false)) {
        for (unsigned int k=0; k<nvert; k++) {
            mesh_insertidsforelement(conn, vids[k], g==target, id, neighbors);
        }
    }

    /* Now find any vertices that are related to an element vertex through symmetries */
    objectsparse *sym = mesh_getconnectivityelement(mesh, 0, 0);
    int nsymids=0, *symids;

    if (sym && sparse_checkformat(sym, SPARSE_CCS, true, false)) {
        for (unsigned int k=0; k<nvert; k++) { // Loop over vertices in the element
            // Is this vertex an image vertex of another element?
            if (sparseccs_getrowindices(&sym->ccs, vids[k], &nsymids, &symids)) {
                for (unsigned int k=0; k<nsymids; k++) {
                    mesh_insertidsforelement(conn, symids[k], g==target, id, neighbors);
                }
            }

            // Is this vertex a target vertex of any image vertices
            int nrids=0, rids[MAX_NEIGHBORS];
            if (sparseccs_getcolindicesforrow(&sym->ccs, id, MAX_NEIGHBORS, &nrids, rids)) {
                for (unsigned int k=0; k<nrids; k++) {
                    mesh_insertidsforelement(conn, rids[k], g==target, id, neighbors);
                }
            }
            if (nrids>=MAX_NEIGHBORS) UNREACHABLE("Too many neighbors.");

        }
    }

    return (neighbors->count);
}

/* **********************************************************************
 * Mesh loader
 * ********************************************************************** */

static unsigned int mesh_nsections = 4;
static char *mesh_sections[] = {MESH_VERTSECTION, MESH_EDGESECTION, MESH_FACESECTION, MESH_VOLSECTION};
static size_t mesh_slength[4];

/** Checks whether line matches a section marker.
 * @param[in] line line to match
 * @param[out] g if line matches, grade is updated
 * @returns true if line matched a section marker; false otherwise */
static bool mesh_checksection(char *line, grade *g) {
    /* Check if the line starts with a section marker */
    unsigned int i;
    for (i=0; i<mesh_nsections; i++) {
        if (strncmp(line, mesh_sections[i], mesh_slength[i])==0) {
            *g=i; return true;
        }
    }

    return false;
}

/** Loads a .mesh file. */
objectmesh *mesh_load(vm *v, char *file) {
    objectmesh *out = NULL;
    error err;

    /* Open the file */
    FILE *f = file_openrelative(file, "r");
    if (!f) {
        morpho_runtimeerror(v, MESH_FILENOTFOUND, file);
        return NULL;
    }

    grade g=-1; /* The current grade we're loading */
    int ndim=-1; /* Dimensionality of the mesh */
    int nv=0; /* Number of vertices */
    int fline=0;

    value val[5]; /* Values to read in per line */
    unsigned int n; /* Number of values read */

    dictionary vdict; // Record how the file's vertex ids map to ours.
    dictionary_init(&vdict);

    varray_double vert; // Hold the vertex positions
    varray_doubleinit(&vert);

    varray_char line; // Buffer to hold a line from the file
    varray_charinit(&line);

    /* Load vertices */
    do {
        line.count=0; // Reset buffer contents.
        file_readlineintovarray(f, &line);
        fline++;

        if (!mesh_checksection(line.data, &g)) {
            /* Convert the string to an array of values */
            if (parse_stringtovaluearray(line.data, 5, val, &n, &err)) {
                if (n>0 && g==0) {
                    /* Check dimensionality */
                    if (ndim<0) ndim=n-1;
                    else if (n-1!=ndim) {
                        morpho_runtimeerror(v, MESH_LOADVERTEXDIM, fline);
                        goto meshload_cleanup;
                    }

                    /* Add the vertex */
                    for (unsigned int k=0; k<ndim; k++) {
                        double coord;
                        if (!morpho_valuetofloat(val[k+1], &coord)) {
                            morpho_runtimeerror(v, MESH_LOADVERTEXCOORD, fline);
                            goto meshload_cleanup;
                        }
                        varray_doubleadd(&vert, &coord, 1);
                    }

                    /* Keey a record of the id in the file */
                    dictionary_insert(&vdict, val[0], MORPHO_INTEGER(nv));
                    nv++;
                }
            } else {
                morpho_runtimeerror(v, MESH_LOADPARSEERR, fline);
                goto meshload_cleanup;
            }
        }
    } while (!feof(f) && g<1);

    /* Create the mesh */
    out=object_newmesh(ndim, nv, vert.data);

    /* Now continue to parse the file for the remaining elements */
    while (!feof(f)) {
        line.count=0; // Reset buffer contents.
        file_readlineintovarray(f, &line);
        fline++;

        /* Check if this is a section header */
        if (line.count>0 && !mesh_checksection(line.data, &g)) {
            /* Convert the string to an array of values */
            if (parse_stringtovaluearray(line.data, 5, val, &n, &err)) {
                if (n>0) {
                    elementid vid[g+1];
                    /* Check number of vertices is consistent with the grade */
                    if (n-1!=g+1) {
                        morpho_runtimeerror(v, MESH_LOADVERTEXNUM, fline);
                        goto meshload_cleanup;
                    }

                    for (unsigned int i=0; i<g+1; i++) {
                        /* Look up our corresponding vertex id */
                        value vx;
                        if (dictionary_get(&vdict, val[i+1], &vx)) {
                            if (MORPHO_ISINTEGER(vx)) {
                                vid[i]=MORPHO_GETINTEGERVALUE(vx);
                            } else {
                                morpho_runtimeerror(v, MESH_LOADVERTEXID, fline);
                                goto meshload_cleanup;
                            }
                        } else {
                            morpho_runtimeerror(v, MESH_LOADVERTEXNOTFOUND, fline);
                            goto meshload_cleanup;
                        }
                    }

                    mesh_addelementwithvertices(out, g, vid);
                }
            }
        }

    }
    fclose(f);

    mesh_freezeconnectivity(out);

meshload_cleanup:

    varray_charclear(&line);
    dictionary_clear(&vdict);
    varray_doubleclear(&vert);

    return out;
}

/* **********************************************************************
 * Mesh exporter
 * ********************************************************************** */

bool mesh_save(objectmesh *m, char *file) {
    /* Open the file */
    FILE *f = file_openrelative(file, "w");
    if (!f) return false;

    /* Export vertices */
    fprintf(f, "%s\n\n", MESH_VERTSECTION);
    for (unsigned int i=0; i<mesh_nvertices(m); i++) {
        fprintf(f, "%i ", i+1); // Keep the mesh files 1-indexed

        for (unsigned int j=0; j<m->vert->nrows; j++) {
            double x;
            if (matrix_getelement(m->vert, j, i, &x)) {
                fprintf(f, "%g ", x);
            }
        }
        fprintf(f, "\n");
    }

    fprintf(f, "\n");

    for (grade g=1; g<m->dim; g++) {
        objectsparse *conn=mesh_getconnectivityelement(m, 0, g);

        if (conn) {
            fprintf(f, "%s\n\n", mesh_sections[g]);
            int nentries=0, *entries=NULL;
            int nel = mesh_nelements(conn);

            for (elementid id=0; id<nel; id++) {
                if (mesh_getconnectivity(conn, id, &nentries, &entries)) {
                    fprintf(f, "%i ", id+1);
                    for (int j=0; j<nentries; j++) {
                        fprintf(f, "%i ", entries[j]+1);
                    }
                    fprintf(f, "\n");
                }
            }
        }
    }

    fclose(f);
    return true;
}

/* **********************************************************************
 * Mesh veneer class
 * ********************************************************************** */

/** Constructs a Matrix object */
value mesh_constructor(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectmesh *new=NULL;

    if (nargs==1 && MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        new=mesh_load(v, MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)));
    } else if (nargs==0) {
        // empty mesh constructor
        new=object_newmesh(0, 0, NULL);
    }
    else {
        morpho_runtimeerror(v,MESH_CONSTRUCTORARGS);
    }

    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }

    return out;
}

/** Print the mesh */
value Mesh_save(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));

    if (nargs==1 && MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        mesh_save(m, MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)));
    }

    return MORPHO_NIL;
}

/** Print the mesh */
value Mesh_print(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    printf("<Mesh:");
    if (m->vert) printf(" %u vertices", mesh_nvertices(m));
    printf(">");
    return MORPHO_NIL;
}

/** Get the vertex matrix */
value Mesh_vertexmatrix(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (m->vert) out=MORPHO_OBJECT(m->vert);
    mesh_delink(m, (object *) m->vert);
    morpho_bindobjects(v, 1, &out);

    return out;
}

/** Set the vertex matrix */
value Mesh_setvertexmatrix(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));

    if (nargs==1 && MORPHO_ISMATRIX(MORPHO_GETARG(args, 0))) {
        objectmatrix *mat = MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));

        if (m->dim>0 && (mesh_nvertices(m)!=mat->ncols || m->vert->nrows!=mat->nrows)) {
            morpho_runtimeerror(v, MESH_VERTMTRXDIM);
        } else {
            if (m->dim==0) m->dim=mat->nrows;
            m->vert=mat;
        }
    }

    return MORPHO_NIL;
}

/** Get position of a vertex */
value Mesh_vertexposition(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        unsigned int id=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        double *vals;

        if (matrix_getcolumn(m->vert, id, &vals)) {
            objectmatrix *new=object_newmatrix(m->dim, 1, true);
            if (new) {
                matrix_setcolumn(new, 0, vals);
                out=MORPHO_OBJECT(new);
                morpho_bindobjects(v, 1, &out);
            }
        } else morpho_runtimeerror(v, MESH_INVLDID);
    } else morpho_runtimeerror(v, MESH_VRTPSNARGS);

    return out;
}

/** Set position of a vertex */
value Mesh_setvertexposition(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));

    if (nargs==2 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISMATRIX(MORPHO_GETARG(args, 1))) {
        unsigned int id=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        objectmatrix *mat = MORPHO_GETMATRIX(MORPHO_GETARG(args, 1));

        if (!matrix_setcolumn(m->vert, id, mat->elements)) morpho_runtimeerror(v, MESH_INVLDID);
    } else morpho_runtimeerror(v, MESH_STVRTPSNARGS);

    return MORPHO_NIL;
}

/** Gets a connectivity matrix */
value Mesh_connectivitymatrix(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    if (nargs==2 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISINTEGER(MORPHO_GETARG(args, 1))) {
        unsigned int row=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        unsigned int col=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));

        objectsparse *s=mesh_getconnectivityelement(m, row, col);
        if (!s && row>0 && row!=col) s=mesh_addconnectivityelement(m, row, col);
        if (s) {
            mesh_delink(m, (object *) s);
            out=MORPHO_OBJECT(s);
            morpho_bindobjects(v, 1, &out);
        }
    } else morpho_runtimeerror(v, MESH_CNNMTXARGS);

    return out;
}

/** Clears any connectivity matrices */
value Mesh_resetconnectivity(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));

    mesh_resetconnectivity(m);

    return MORPHO_NIL;
}

/** Adds a grade to a mesh */
value Mesh_addgrade(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        unsigned int g=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        if (g==0) return MORPHO_NIL;
        objectsparse *s=mesh_getconnectivityelement(m, 0, g);
        if (!s) {
            s=mesh_addgrade(m, g);
            if(!s){
                morpho_runtimeerror(v, MESH_ADDGRDOOB,g,mesh_maxgrade(m));
                return MORPHO_NIL;

            }
        }

    } else if (nargs==2 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)) &&
               MORPHO_ISSPARSE(MORPHO_GETARG(args, 1))) {
        unsigned int grade=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        objectsparse *s=MORPHO_GETSPARSE(MORPHO_GETARG(args, 1));

        if (s && grade>0) mesh_setconnectivityelement(m, 0, grade, s);
        mesh_freezeconnectivity(m);
    } else morpho_runtimeerror(v, MESH_ADDGRDARGS);

    return out;
}

/** Adds a symmetry to a mesh */
value Mesh_addsymmetry(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    value obj = MORPHO_NIL;
    objectselection *sel = NULL;

    if (nargs>0 && MORPHO_ISOBJECT(MORPHO_GETARG(args, 0))) {
        obj=MORPHO_GETARG(args, 0);
    } else morpho_runtimeerror(v, MESH_ADDSYMARGS);

    if (nargs>1) {
        if (MORPHO_ISSELECTION(MORPHO_GETARG(args, 1))) {
            sel = MORPHO_GETSELECTION(MORPHO_GETARG(args, 1));
        } else morpho_runtimeerror(v, MESH_ADDSYMARGS);
    }

    if (!MORPHO_ISNIL(obj)) {
        mesh_addsymmetry(v, m, obj, sel);
    }

    return MORPHO_NIL;
}

/* Returns the highest grade present */
value Mesh_maxgrade(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));

    return MORPHO_INTEGER(mesh_maxgrade(m));
}

/** Counts the number of elements for a given grade, or returns the number of vertices if no argument is supplied. */
value Mesh_count(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    grade g=0;
    value out=MORPHO_INTEGER(0);

    if (nargs>0 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        g = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    }

    if (g==0) {
        out = MORPHO_INTEGER(m->vert->ncols);
    } else {
        objectsparse *s = mesh_getconnectivityelement(m, 0, g);
        if (s) out = MORPHO_INTEGER(s->ccs.ncols);
    }

    return out;
}

/** Clones a mesh */
value Mesh_clone(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectmesh *a=MORPHO_GETMESH(MORPHO_SELF(args));
    objectmesh *new=mesh_clone(a);
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    return out;
}

MORPHO_BEGINCLASS(Mesh)
MORPHO_METHOD(MORPHO_PRINT_METHOD, Mesh_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SAVE_METHOD, Mesh_save, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MESH_VERTEXMATRIX_METHOD, Mesh_vertexmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MESH_SETVERTEXMATRIX_METHOD, Mesh_setvertexmatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MESH_VERTEXPOSITION_METHOD, Mesh_vertexposition, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MESH_SETVERTEXPOSITION_METHOD, Mesh_setvertexposition, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MESH_RESETCONNECTIVITY_METHOD, Mesh_resetconnectivity, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MESH_CONNECTIVITYMATRIX_METHOD, Mesh_connectivitymatrix, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MESH_ADDGRADE_METHOD, Mesh_addgrade, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MESH_ADDSYMMETRY_METHOD, Mesh_addsymmetry, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MESH_MAXGRADE_METHOD, Mesh_maxgrade, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Mesh_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Mesh_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void mesh_initialize(void) {
    for (unsigned int i=0; i<mesh_nsections; i++) mesh_slength[i]=strlen(mesh_sections[i]);

    builtin_addfunction(MESH_CLASSNAME, mesh_constructor, BUILTIN_FLAGSEMPTY);

    value meshclass=builtin_addclass(MESH_CLASSNAME, MORPHO_GETCLASSDEFINITION(Mesh), MORPHO_NIL);
    builtin_setveneerclass(OBJECT_MESH, meshclass);

    morpho_defineerror(MESH_FILENOTFOUND, ERROR_HALT, MESH_FILENOTFOUND_MSG);
    morpho_defineerror(MESH_VERTMTRXDIM, ERROR_HALT, MESH_VERTMTRXDIM_MSG);
    morpho_defineerror(MESH_LOADVERTEXDIM, ERROR_HALT, MESH_LOADVERTEXDIM_MSG);
    morpho_defineerror(MESH_LOADVERTEXCOORD, ERROR_HALT, MESH_LOADVERTEXCOORD_MSG);
    morpho_defineerror(MESH_LOADPARSEERR, ERROR_HALT, MESH_LOADPARSEERR_MSG);
    morpho_defineerror(MESH_LOADVERTEXNUM, ERROR_HALT, MESH_LOADVERTEXNUM_MSG);
    morpho_defineerror(MESH_LOADVERTEXID, ERROR_HALT, MESH_LOADVERTEXID_MSG);
    morpho_defineerror(MESH_LOADVERTEXNOTFOUND, ERROR_HALT, MESH_LOADVERTEXNOTFOUND_MSG);
    morpho_defineerror(MESH_STVRTPSNARGS, ERROR_HALT, MESH_STVRTPSNARGS_MSG);
    morpho_defineerror(MESH_VRTPSNARGS, ERROR_HALT, MESH_VRTPSNARGS_MSG);
    morpho_defineerror(MESH_INVLDID, ERROR_HALT, MESH_INVLDID_MSG);
    morpho_defineerror(MESH_CNNMTXARGS, ERROR_HALT, MESH_CNNMTXARGS_MSG);
    morpho_defineerror(MESH_ADDGRDARGS, ERROR_HALT, MESH_ADDGRDARGS_MSG);
    morpho_defineerror(MESH_ADDGRDOOB, ERROR_HALT, MESH_ADDGRDOOB_MSG);
    morpho_defineerror(MESH_ADDSYMARGS, ERROR_HALT, MESH_ADDSYMARGS_MSG);
    morpho_defineerror(MESH_ADDSYMMSNGTRNSFRM, ERROR_HALT, MESH_ADDSYMMSNGTRNSFRM_MSG);
    morpho_defineerror(MESH_CONSTRUCTORARGS, ERROR_HALT, MESH_CONSTRUCTORARGS_MSG);

}
