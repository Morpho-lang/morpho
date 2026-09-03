/** @file mesh.c
 *  @author T J Atherton
 *
 *  @brief Mesh class and associated functionality
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "morpho.h"
#include "classes.h"
#include "mesh.h"
#include "file.h"
#include "parse.h"
#include "lex.h"
#include "sparse.h"
#include "linalg.h"
#include "selection.h"

#include <limits.h>

DEFINE_VARRAY(elementid, elementid);

static bool mesh_nearestvertex(objectmesh *mesh, double *x, elementid *id, double *separation);

/** Bind a mesh-allocated object as a child; free it if bind fails. */
static bool mesh_attach(objectmesh *mesh, object *obj) {
    if (obj && morpho_bindtoparent(obj, (object *) mesh)) return true;
    if (obj) object_free(obj);
    return false;
}

/** Free an object if this mesh owns it as a child. */
static void mesh_release(object *obj) {
    if (morpho_ischildobject(obj)) object_free(obj);
}

/* **********************************************************************
 * Mesh object definitions
 * ********************************************************************** */

objecttype objectmeshtype;

/** Mesh object definitions */
void objectmesh_printfn(object *obj, void *v) {
    morpho_printf(v, "<Mesh>");
}

void objectmesh_markfn(object *obj, void *v) {
    objectmesh *c = (objectmesh *) obj;
    if (c->vert) morpho_markobject(v, (object *) c->vert);
    for (unsigned int i=0; i<c->nconn; i++) morpho_markvalue(v, c->conn[i]);
}

void objectmesh_freefn(object *obj) {
    objectmesh *m = (objectmesh *) obj;
    for (unsigned int i=0; i<m->nconn; i++) {
        if (MORPHO_ISOBJECT(m->conn[i])) mesh_release(MORPHO_GETOBJECT(m->conn[i]));
    }
    mesh_release((object *) m->vert);
}

size_t objectmesh_sizefn(object *obj) {
    objectmesh *m = (objectmesh *) obj;
    return sizeof(objectmesh)+sizeof(value)*m->nconn;
}

objecttypedefn objectmeshdefn = {
    .printfn=objectmesh_printfn,
    .markfn=objectmesh_markfn,
    .freefn=objectmesh_freefn,
    .sizefn=objectmesh_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/* **********************************************************************
 * Create mesh objects
 * ********************************************************************** */

/** Allocate a mesh of known dimension with an empty connectivity table. */
static objectmesh *mesh_new(unsigned int dim) {
    size_t nconn = (size_t) (dim+1)*(size_t) (dim+1);
    if (nconn>UINT_MAX) return NULL;

    objectmesh *new = (objectmesh *) object_new(sizeof(objectmesh)+sizeof(value)*nconn, OBJECT_MESH);
    if (new) {
        new->dim=dim;
        new->nconn=(unsigned int) nconn;
        new->vert=NULL;
        new->conn=new->conndata;
        for (unsigned int i=0; i<new->nconn; i++) new->conn[i]=MORPHO_NIL;
    }
    return new;
}

objectmesh *object_newmesh(unsigned int dim, unsigned int nv, double *v) {
    objectmesh *new = mesh_new(dim);
    if (!new) return NULL;

    new->vert=matrix_new(dim, nv, false);
    if (!mesh_attach(new, (object *) new->vert)) {
        new->vert=NULL; object_free((object *) new); return NULL;
    }
    if (dim>0 && nv>0 && v) memcpy(new->vert->elements, v, sizeof(double)*dim*nv);
    return new;
}

/** Clones a mesh object. Associated connectivity is also cloned */
static objectmesh *mesh_clone(objectmesh *mesh) {
    if (!mesh || !mesh->vert) return NULL;
    objectmesh *new = object_newmesh(mesh->dim, mesh->vert->ncols, mesh->vert->elements);
    if (!new) return NULL;

    for (unsigned int i=0; i<mesh->nconn; i++) {
        if (!MORPHO_ISSPARSE(mesh->conn[i])) continue;
        objectsparse *cl=sparse_clone(MORPHO_GETSPARSE(mesh->conn[i]));
        if (!mesh_attach(new, (object *) cl)) { object_free((object *) new); return NULL; }
        new->conn[i]=MORPHO_OBJECT(cl);
    }
    return new;
}

/* **********************************************************************
 * Manipulate mesh objects
 * ********************************************************************** */

/* -------------------------------------
 * Vertices
 * ------------------------------------- */

/** Gets vertex coordinates as a list */
bool mesh_getvertexcoordinatesaslist(objectmesh *mesh, elementid id, double **out) {
    return matrix_getcolumnptr(mesh->vert, id, out)==LINALGERR_OK;
}

/** Gets vertex coordinates */
static bool mesh_getvertexcoordinates(objectmesh *mesh, elementid id, double *out) {
    double *x;
    if (!mesh_getvertexcoordinatesaslist(mesh, id, &x)) return false;
    for (unsigned int i=0; i<mesh->dim; i++) out[i]=x[i];
    return true;
}

/** Gets vertex coordinates as a value list */
bool mesh_getvertexcoordinatesasvalues(objectmesh *mesh, elementid id, value *val) {
    double *x;
    if (!mesh_getvertexcoordinatesaslist(mesh, id, &x)) return false;
    for (unsigned int i=0; i<mesh->dim; i++) val[i]=MORPHO_FLOAT(x[i]);
    return true;
}

/** Replace vertex coordinates. The mesh always owns its vertex matrix as a child. */
static bool mesh_setvertexmatrix(objectmesh *mesh, objectmatrix *vert) {
    if (!mesh || !vert || vert->nrows!=(int) mesh->dim) return false;
    if (mesh->vert==vert) return true;

    if (mesh->vert && mesh->vert->ncols==vert->ncols) {
        return matrix_copy(vert, mesh->vert)==LINALGERR_OK;
    }

    objectmatrix *store=matrix_clone(vert);
    if (!mesh_attach(mesh, (object *) store)) return false;
    mesh_release((object *) mesh->vert);
    mesh->vert=store;
    return true;
}

/* -------------------------------------
 * The connectivity array
 * ------------------------------------- */

/** Slot in the (dim+1) x (dim+1) connectivity table, or NULL if out of range. */
static value *_connslotptr(objectmesh *mesh, unsigned int row, unsigned int col) {
    if (row>mesh->dim || col>mesh->dim) return NULL;
    return &mesh->conn[row+col*(mesh->dim+1)];
}

/** Gets the connectivity matrix corresponding to (row, col) */
objectsparse *mesh_getconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col) {
    value *slot=_connslotptr(mesh, row, col);
    if (slot && MORPHO_ISSPARSE(*slot)) return MORPHO_GETSPARSE(*slot);
    return NULL;
}

/** Sets a connectivity element. Clear the element by setting el to NULL. */
static bool mesh_setconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col, objectsparse *el) {
    value *slot=_connslotptr(mesh, row, col);
    if (!slot) return false;

    if (MORPHO_ISOBJECT(*slot)) {
        object *oel = MORPHO_GETOBJECT(*slot);
        if (el && oel==(object *) el) return true;
    }

    if (el && morpho_ischildobject((object *) el)) {
        el=sparse_clone(el);
        if (!mesh_attach(mesh, (object *) el)) return false;
    } else if (el && el->obj.status==OBJECT_ISUNMANAGED &&
               !morpho_bindtoparent((object *) el, (object *) mesh)) {
        return false;
    }

    if (MORPHO_ISOBJECT(*slot)) mesh_release(MORPHO_GETOBJECT(*slot));
    *slot = el ? MORPHO_OBJECT(el) : MORPHO_NIL;
    return true;
}

/** Creates a new blank connectivity matrix */
static objectsparse *mesh_newconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col) {
    objectsparse *out=object_newsparse(NULL, NULL);
    if (out && !mesh_setconnectivityelement(mesh, row, col, out)) { object_free((object *) out); return NULL; }
    return out;
}

/** Gets connectivity information for a given element */
bool mesh_getconnectivity(objectsparse *conn, elementid id, int *nentries, int **entries) {
    sparse_checkformat(conn, SPARSE_CCS, true, false);
    if (conn) return sparseccs_getrowindices(&conn->ccs, id, nentries, entries);
    return false;
}

/** Freezes mesh connectivity, converting subsidiary data structures to fixed but efficient versions */
void mesh_freezeconnectivity(objectmesh *mesh) {
    for (unsigned int i=0; i<mesh->dim+1; i++) {
        for (unsigned int j=0; j<mesh->dim+1; j++) {
            objectsparse *s=mesh_getconnectivityelement(mesh, i, j);
            if (s) sparse_checkformat(s, SPARSE_CCS, true, false);
        }
    }
}

/** Resets connectivity elements other than the first row */
static void mesh_resetconnectivity(objectmesh *m) {
    grade max = mesh_maxgrade(m);
    for (grade i=1; i<=max; i++) {
        for (grade j=0; j<=max; j++) mesh_setconnectivityelement(m, i, j, NULL);
    }
}

/* ------------------------------------------
 * Add and remove grades
 * ------------------------------------------ */

/** Insert a unique n-tuple of vertex ids as a new element. */
static bool _addelementidtuple(dictionary *seen, unsigned int n, elementid *ids, objectsparse *conn, elementid *newid) {
    value tvals[n];
    for (unsigned int i=0; i<n; i++) tvals[i]=MORPHO_INTEGER(ids[i]);

    objecttuple probe = MORPHO_STATICTUPLE(tvals, n);
    if (dictionary_get(seen, MORPHO_OBJECT(&probe), NULL)) return true;

    objecttuple *key = object_newtuple(n, tvals);
    if (!key || !dictionary_insert(seen, MORPHO_OBJECT(key), MORPHO_NIL)) {
        if (key) object_free((object *) key);
        return false;
    }

    for (unsigned int i=0; i<n; i++) if (!sparsedok_insert(&conn->dok, ids[i], *newid, MORPHO_NIL)) return false;
    (*newid)++;
    return true;
}

/** Adds connectivity information for a grade g */
objectsparse *mesh_addgrade(objectmesh *mesh, grade g) {
    objectsparse *el=mesh_getconnectivityelement(mesh, 0, g), *out=NULL;
    if (el) return el;

    grade maxG=mesh_maxgrade(mesh);
    for (grade h=g+1; (h<=maxG) && (!el); h++) el=mesh_getconnectivityelement(mesh, 0, h);
    if (!el) return NULL;

    objectsparse *new=object_newsparse(NULL, NULL);
    if (!new) return NULL;

    dictionary seen;
    dictionary_init(&seen);
    int n=g+1, nel, *entries, k;
    elementid newid=0;

    for (elementid id=0; id<el->ccs.ncols; id++) {
        if (!mesh_getconnectivity(el, id, &nel, &entries)) goto mesh_addgrade_cleanup;
        elementid tuple[n];
        int counter[n], cmax[n];
        for (unsigned int i=0; i<n; i++) { counter[i]=i; tuple[i]=entries[i]; cmax[i]=nel-n+i; }
        if (!_addelementidtuple(&seen, n, tuple, new, &newid)) goto mesh_addgrade_cleanup;

        while (counter[0]<cmax[0]) {
            counter[n-1]++;
            for (k=n-1; k>=0 && counter[k]>cmax[k]; k--) counter[k-1]++;
            if (k<n-1) for (unsigned int i=k+1; i<n; i++) counter[i]=counter[i-1]+1;
            for (unsigned int i=0; i<n; i++) tuple[i]=entries[counter[i]];
            if (!_addelementidtuple(&seen, n, tuple, new, &newid)) goto mesh_addgrade_cleanup;
        }
    }

    if (mesh_setconnectivityelement(mesh, 0, g, new)) {
        mesh_freezeconnectivity(mesh);
        out=new; new=NULL;
    }

mesh_addgrade_cleanup:
    dictionary_freecontents(&seen, true, false);
    dictionary_clear(&seen);
    if (new) object_free((object *) new);
    return out;
}

/** Removes connectivity information for grade g */
static void mesh_removegrade(objectmesh *mesh, grade g) {
    objectsparse *el=mesh_getconnectivityelement(mesh, 0, g);
    grade maxg = mesh_maxgrade(mesh);
    if (el && g<=maxg) {
        mesh_setconnectivityelement(mesh, 0, g, NULL);
        mesh_resetconnectivity(mesh);
    }
}

/** Internal function used for sorting ids */
static int _compareid(const void *a, const void *b) {
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
bool mesh_matchelements(objectsparse *vmatrix, grade g, int nids, int *ids, int maxmatches, int *nmatches, int *matches) {
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

    qsort(sids, length, sizeof(int), _compareid); // and sort it

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
    objectsparse *traise=mesh_getconnectivityelement(mesh, row, 0);
    if (!traise) traise=mesh_addconnectivityelement(mesh, row, 0);
    objectsparse *tlower=mesh_getconnectivityelement(mesh, 0, col);
    if (!traise || !tlower) return NULL;

    objectsparse *new=object_newsparse(NULL, NULL);
    if (!new) return NULL;

    int maxmatches = (col+1)*(col+2)/2+1, nentries, *entries, nmatches, matches[maxmatches];
    for (elementid rid=0; rid<tlower->ccs.ncols; rid++) {
        if (!mesh_getconnectivity(tlower, rid, &nentries, &entries)) continue;
        if (!mesh_matchelements(traise, row, nentries, entries, maxmatches, &nmatches, matches)) continue;
        if (nmatches>=maxmatches) UNREACHABLE("Too many connections.");
        for (unsigned int i=0; i<nmatches; i++) sparsedok_insert(&new->dok, matches[i], rid, MORPHO_NIL);
    }

    if (!mesh_setconnectivityelement(mesh, row, col, new)) { object_free((object *) new); return NULL; }
    mesh_freezeconnectivity(mesh);
    return new;
}

/** Fill in a missing connectivity element */
objectsparse *mesh_addconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col) {
    objectsparse *el=mesh_getconnectivityelement(mesh, row, col);
    if (el) return el;

    if (row>0 && row<col) { /* Grade lowering */
        el=mesh_addlowermatrix(mesh, row, col);
    } else if (row>col) { /* Grade raising: transpose the lowering matrix */
        objectsparse *tlow=mesh_getconnectivityelement(mesh, col, row);
        if (!tlow) tlow=mesh_addconnectivityelement(mesh, col, row);
        if (tlow) {
            el=mesh_newconnectivityelement(mesh, row, col);
            sparse_transpose(tlow, el);
        }
    }

    return el;
}

/** Adds an element to a mesh */
static bool mesh_addelementwithvertices(objectmesh *mesh, grade g, elementid *v) {
    objectsparse *m=mesh_getconnectivityelement(mesh, 0, g);
    if (!m) m=mesh_newconnectivityelement(mesh, 0, g);
    if (!m) return false;

    int eid=m->dok.ncols;
    bool success=true;
    for (unsigned int i=0; i<g+1 && success; i++) success=sparsedok_insert(&m->dok, (int) v[i], eid, MORPHO_NIL);
    return success;
}

/* **********************************************************************
 * Symmetries
 * ********************************************************************** */

/** How close two points can be before they're indistinct */
#define MESH_NEARESTPOINTEPS 1e-10

/** Adds a symmetry to a mesh. */
static bool mesh_addsymmetry(vm *v, objectmesh *mesh, value symmetry, objectselection *sel) {
    value method=MORPHO_NIL;
    objectstring s = MORPHO_STATICSTRING(MESH_TRANSFORM_METHOD);

    objectsparse *sym=mesh_getconnectivityelement(mesh, MESH_GRADE_VERTEX, MESH_GRADE_VERTEX);

    double x[mesh->dim];
    objectmatrix posn = MORPHO_STATICMATRIX(x, mesh->dim, 1);

    elementid nv = mesh_nvertices(mesh);

    value arg = MORPHO_OBJECT(&posn);
    value ret = MORPHO_NIL;
    (void) sel;

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

        return true;
    }

    MORPHO_FAIL(v, MESH_ADDSYMMSNGTRNSFRM);
}

/** Get a list of synonymous elements for a given element */
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

/* **********************************************************************
 * Utilities
 * ********************************************************************** */

/** How many vertices are in a matrix? */
elementid mesh_nvertices(objectmesh *mesh) {
    return mesh->vert ? mesh->vert->ncols : 0;
}

/** How many elements are in a connectivity matrix? */
elementid mesh_nelements(objectsparse *conn) {
    return conn->ccs.ncols;
}

/** How many elements exist in a given grade? */
elementid mesh_nelementsforgrade(objectmesh *mesh, grade g) {
    if (g==MESH_GRADE_VERTEX) return mesh_nvertices(mesh);
    objectsparse *conn = mesh_getconnectivityelement(mesh, 0, g);
    return conn ? mesh_nelements(conn) : 0;
}

/** Maximum grade in the mesh */
grade mesh_maxgrade(objectmesh *mesh) {
    for (grade g=mesh->dim; g>0; g--) {
        if (mesh_getconnectivityelement(mesh, 0, g)) return g;
    }

    return 0;
}

/** Direct search for the nearest vertex to a point */
static bool mesh_nearestvertex(objectmesh *mesh, double *x, elementid *id, double *separation) {
    double *vx;
    double best=0, sep=0;
    elementid bestid=0;

    for (elementid i=0; i<mesh_nvertices(mesh); i++) {
        if (!mesh_getvertexcoordinatesaslist(mesh, i, &vx)) return false;
        sep=0;
        for (int k=0; k<mesh->dim; k++) sep+=(vx[k]-x[k])*(vx[k]-x[k]);
        if (i==0 || sep<best) { best=sep; bestid = i; }
    }
    *id = bestid;
    if (separation) *separation=sqrt(best);
    return true;
}

/** Converts a point x to barycentric coordinates lambda */
static bool mesh_getbarycentriccoordinates(objectmesh *mesh, grade g, elementid id, double *x, double *lambda) {
    if (g<1 || g>mesh_maxgrade(mesh)) return false;

    objectsparse *conn=mesh_getconnectivityelement(mesh, 0, g);
    if (!conn) return false;

    int nentries, *entries; // Get the connectivity of the element
    if (!mesh_getconnectivity(conn, id, &nentries, &entries) || nentries!=g+1) return false;

    double *verts[4]; // Get the vertices of the element
    for (int i=0; i<nentries; i++) {
        if (!mesh_getvertexcoordinatesaslist(mesh, entries[i], &verts[i])) return false;
    }

    double gramdata[g*g], rhsdata[g];
    double dx[mesh->dim], edges[g][mesh->dim];
    objectmatrix gram = MORPHO_STATICMATRIX(gramdata, g, g);
    objectmatrix rhs = MORPHO_STATICMATRIX(rhsdata, g, 1);
    matrix_zero(&gram);
    matrix_zero(&rhs);

    for (unsigned int k=0; k<mesh->dim; k++) dx[k]=x[k]-verts[0][k];
    for (int i=0; i<g; i++) {
        for (unsigned int k=0; k<mesh->dim; k++) edges[i][k]=verts[i+1][k]-verts[0][k];
    }

    for (int i=0; i<g; i++) {
        double d=0.0;
        for (unsigned int k=0; k<mesh->dim; k++) d+=edges[i][k]*dx[k];
        rhs.elements[i]=d;
        for (int j=0; j<g; j++) {
            double d2=0.0;
            for (unsigned int k=0; k<mesh->dim; k++) d2+=edges[i][k]*edges[j][k];
            gram.elements[i+j*g]=d2;
        }
    }
    bool success=(matrix_solvesmall(&gram, &rhs)==LINALGERR_OK);
    if (success) {
        double sum=0.0;
        for (int i=0; i<g; i++) {
            lambda[i+1]=rhs.elements[i];
            sum+=rhs.elements[i];
        }
        lambda[0]=1.0-sum;
    }

    return success;
}

void varray_elementidwriteunique(varray_elementid *list, elementid id) {
    for (unsigned int i=0; i<list->count; i++) if (list->data[i]==id) return;
    varray_elementidwrite(list, id);
}

/** Append unique ids incident on `id` in a connectivity matrix. */
static void _appendincidentids(objectsparse *conn, elementid id, bool ignore, elementid ignoreid, varray_elementid *out) {
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
            _appendincidentids(conn, vids[k], g==target, id, neighbors);
        }
    }

    /* Now find any vertices that are related to an element vertex through symmetries */
    objectsparse *sym = mesh_getconnectivityelement(mesh, 0, 0);
    int nsymids=0, *symids;

    if (sym && sparse_checkformat(sym, SPARSE_CCS, true, false)) {
        for (unsigned int k=0; k<nvert; k++) { // Loop over vertices in the element
            // Is this vertex an image vertex of another element?
            if (sparseccs_getrowindices(&sym->ccs, vids[k], &nsymids, &symids)) {
                for (unsigned int s=0; s<nsymids; s++) {
                    _appendincidentids(conn, symids[s], g==target, id, neighbors);
                }
            }

            // Is this vertex a target vertex of any image vertices
            int nrids=0, rids[MAX_NEIGHBORS];
            if (sparseccs_getcolindicesforrow(&sym->ccs, id, MAX_NEIGHBORS, &nrids, rids)) {
                for (unsigned int r=0; r<nrids; r++) {
                    _appendincidentids(conn, rids[r], g==target, id, neighbors);
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

static char *mesh_sections[] = {MESH_VERTSECTION, MESH_EDGESECTION, MESH_FACESECTION, MESH_VOLSECTION};

#define MESH_LOADMAXVALS 8

enum {
    MESH_TOK_VERTICES = 1000,
    MESH_TOK_EDGES,
    MESH_TOK_FACES,
    MESH_TOK_VOLUMES
};

static bool mesh_lexnewline(lexer *l, token *tok, error *err) {
    (void) tok; (void) err;
    lex_newline(l);
    return true;
}

static tokendefn meshtokens[] = {
    { MESH_VERTSECTION, MESH_TOK_VERTICES, NULL },
    { MESH_EDGESECTION, MESH_TOK_EDGES,    NULL },
    { MESH_FACESECTION, MESH_TOK_FACES,    NULL },
    { MESH_VOLSECTION,  MESH_TOK_VOLUMES,  NULL },
    { "-",              TOKEN_MINUS,       NULL },
    { "\n",             TOKEN_NEWLINE,     mesh_lexnewline },
    { "",               TOKEN_NONE,        NULL }
};

typedef struct {
    grade g;
    int ndim, nv;
    dictionary vdict;
    varray_double vert;
    varray_elementid elgrade, elvert;
} meshload;

static void mesh_parseerror(parser *p, errorid id) {
    if (ERROR_SUCCEEDED(*p->err)) {
        morpho_writeerrorwithid(p->err, id, NULL, p->current.line, p->current.posn, p->current.line);
    }
}

static bool mesh_eol(parser *p) {
    return parse_checktoken(p, TOKEN_NEWLINE) || parse_checktoken(p, TOKEN_EOF);
}

static bool mesh_parsenumber(parser *p, value *out) {
    int sgn = parse_checktokenadvance(p, TOKEN_MINUS) ? -1 : 1;
    long i; double x;

    if (parse_checktokenadvance(p, TOKEN_INTEGER) && parse_tokentointeger(p, &i)) {
        *out = MORPHO_INTEGER(sgn*(int) i);
        return true;
    } else if (parse_checktokenadvance(p, TOKEN_NUMBER) && parse_tokentodouble(p, &x)) {
        *out = MORPHO_FLOAT(sgn*x);
        return true;
    }
    if (sgn<0) mesh_parseerror(p, MESH_LOADPARSEERR);
    return false;
}

/** Record a vertex (grade 0) or element (grade > 0) from a parsed line. */
static bool mesh_applyrecord(parser *p, meshload *load, int n, value *val) {
    if (n==0 || load->g<0) return true;

    if (load->g==0) {
        if (load->ndim<0) load->ndim=n-1;
        else if (n-1!=load->ndim) { mesh_parseerror(p, MESH_LOADVERTEXDIM); return false; }

        if (load->ndim>0) {
            double x[load->ndim];
            if (!morpho_valuestofloat(load->ndim, val+1, x)) {
                mesh_parseerror(p, MESH_LOADVERTEXCOORD);
                return false;
            }
            varray_doubleadd(&load->vert, x, load->ndim);
        }
        dictionary_insert(&load->vdict, val[0], MORPHO_INTEGER(load->nv++));
        return true;
    }

    if (n-1!=load->g+1) { mesh_parseerror(p, MESH_LOADVERTEXNUM); return false; }
    varray_elementidwrite(&load->elgrade, load->g);
    for (int i=0; i<load->g+1; i++) {
        value vx;
        if (!dictionary_get(&load->vdict, val[i+1], &vx)) { mesh_parseerror(p, MESH_LOADVERTEXNOTFOUND); return false; }
        if (!MORPHO_ISINTEGER(vx)) { mesh_parseerror(p, MESH_LOADVERTEXID); return false; }
        varray_elementidwrite(&load->elvert, MORPHO_GETINTEGERVALUE(vx));
    }
    return true;
}

static bool mesh_parsefile(parser *p, void *out) {
    meshload *load = out;

    while (!parse_checktoken(p, TOKEN_EOF)) {
        while (parse_checktokenadvance(p, TOKEN_NEWLINE));
        if (ERROR_FAILED(*p->err) || parse_checktoken(p, TOKEN_EOF)) break;

        tokentype t = p->current.type;
        if (t>=MESH_TOK_VERTICES && t<=MESH_TOK_VOLUMES) {
            load->g = (grade) (t-MESH_TOK_VERTICES);
            PARSE_CHECK(parse_advance(p));
            while (!mesh_eol(p)) PARSE_CHECK(parse_advance(p));
            continue;
        }

        value val[MESH_LOADMAXVALS];
        int n=0;
        while (!mesh_eol(p) && n<MESH_LOADMAXVALS) {
            if (!mesh_parsenumber(p, &val[n++])) {
                if (ERROR_SUCCEEDED(*p->err)) mesh_parseerror(p, MESH_LOADPARSEERR);
                return false;
            }
        }
        if (n>=MESH_LOADMAXVALS && !mesh_eol(p)) { mesh_parseerror(p, MESH_LOADPARSEERR); return false; }
        if (!mesh_applyrecord(p, load, n, val)) return false;
        if (parse_checktoken(p, TOKEN_NEWLINE)) PARSE_CHECK(parse_advance(p));
    }

    return ERROR_SUCCEEDED(*p->err);
}

/** Loads a .mesh file. */
static objectmesh *mesh_load(vm *v, char *file) {
    FILE *f = file_openrelative(file, "r");
    if (!f) { morpho_runtimeerror(v, MESH_FILENOTFOUND, file); return NULL; }

    error err; error_init(&err);
    varray_char src; varray_charinit(&src);
    meshload load = { .g=-1, .ndim=-1, .nv=0 };
    dictionary_init(&load.vdict);
    varray_doubleinit(&load.vert);
    varray_elementidinit(&load.elgrade);
    varray_elementidinit(&load.elvert);
    lexer l; parser p; objectmesh *out=NULL;

    if (!file_readintovarray(f, &src) || !src.data) {
        morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        goto mesh_load_cleanup;
    }

    lex_init(&l, src.data, 1);
    lex_settokendefns(&l, meshtokens);
    lex_setstringinterpolation(&l, false);
    parse_init(&p, &l, &err, &load);
    parse_setbaseparsefn(&p, mesh_parsefile);
    parse_setskipnewline(&p, false, TOKEN_NONE);

    if (!parse(&p)) { morpho_error(v, &err); goto mesh_load_lexcleanup; }

    out = object_newmesh(load.ndim<0 ? 0 : (unsigned int) load.ndim, load.nv, load.vert.data);
    for (unsigned int i=0, k=0; out && i<load.elgrade.count; i++) {
        grade g=load.elgrade.data[i];
        if (!mesh_addelementwithvertices(out, g, &load.elvert.data[k])) { object_free((object *) out); out=NULL; }
        k+=(unsigned int) g+1;
    }
    if (out) mesh_freezeconnectivity(out);

mesh_load_lexcleanup:
    parse_clear(&p); lex_clear(&l);
mesh_load_cleanup:
    fclose(f);
    varray_charclear(&src);
    dictionary_clear(&load.vdict);
    varray_doubleclear(&load.vert);
    varray_elementidclear(&load.elgrade);
    varray_elementidclear(&load.elvert);
    return out;
}

/* **********************************************************************
 * Mesh exporter
 * ********************************************************************** */

static bool mesh_save(objectmesh *m, char *file) {
    /* Open the file */
    FILE *f = file_openrelative(file, "w");
    if (!f) return false;

    fprintf(f, "%s\n\n", MESH_VERTSECTION); // Export vertices
    for (unsigned int i=0; i<mesh_nvertices(m); i++) {
        fprintf(f, "%i ", i+1); // Keep the mesh files 1-indexed

        for (unsigned int j=0; j<m->vert->nrows; j++) {
            double x;
            if (matrix_getelement(m->vert, j, i, &x) == LINALGERR_OK) fprintf(f, "%g ", x);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "\n");

    for (grade g=1; g<=m->dim; g++) { // Export connectivity, grade by grade
        objectsparse *conn=mesh_getconnectivityelement(m, 0, g);

        if (conn) {
            fprintf(f, "%s\n\n", mesh_sections[g]);
            int nentries=0, *entries=NULL;
            int nel = mesh_nelements(conn);

            for (elementid id=0; id<nel; id++) {
                if (mesh_getconnectivity(conn, id, &nentries, &entries)) {
                    fprintf(f, "%i ", id+1);
                    for (int j=0; j<nentries; j++) fprintf(f, "%i ", entries[j]+1);
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

value mesh_constructor(vm *v, int nargs, value *args) {
    return morpho_wrapandbind(v, (object *) object_newmesh(MESH_DEFAULTDIM, 0, NULL));
}

value mesh_constructor__int(vm *v, int nargs, value *args) {
    int dim=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    if (dim<0) MORPHO_RAISE(v, MESH_INVLDDIM);
    return morpho_wrapandbind(v, (object *) object_newmesh((unsigned int) dim, 0, NULL));
}

value mesh_constructor__matrix(vm *v, int nargs, value *args) {
    objectmatrix *mat=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
    return morpho_wrapandbind(v, (object *) object_newmesh(mat->nrows, mat->ncols, mat->elements));
}

value mesh_constructor__string(vm *v, int nargs, value *args) {
    return morpho_wrapandbind(v, (object *) mesh_load(v, MORPHO_GETCSTRING(MORPHO_GETARG(args, 0))));
}

value Mesh_save__string(vm *v, int nargs, value *args) {
    mesh_save(MORPHO_GETMESH(MORPHO_SELF(args)), MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)));
    return MORPHO_NIL;
}

value Mesh_print(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    if (!MORPHO_ISMESH(self)) return Object_print(v, nargs, args);

    objectmesh *m=MORPHO_GETMESH(self);
    morpho_printf(v, "<Mesh:");
    if (m->vert) morpho_printf(v, " %u vertices", mesh_nvertices(m));
    morpho_printf(v, ">");
    return MORPHO_NIL;
}

value Mesh_vertexmatrix(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    return m->vert ? MORPHO_OBJECT(m->vert) : MORPHO_NIL;
}

value Mesh_setvertexmatrix__matrix(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    objectmatrix *mat = MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
    if (mat->nrows!=(int) m->dim || (mesh_nvertices(m)>0 && mesh_nvertices(m)!=mat->ncols)) MORPHO_RAISE(v, MESH_VERTMTRXDIM);
    if (!mesh_setvertexmatrix(m, mat)) MORPHO_RAISE(v, ERROR_ALLOCATIONFAILED);
    return MORPHO_NIL;
}

value Mesh_vertexposition__int(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    unsigned int id=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    double *vals;

    if (matrix_getcolumnptr(m->vert, id, &vals)!=LINALGERR_OK) MORPHO_RAISE(v, MESH_INVLDID);
    objectmatrix *new=matrix_new(m->dim, 1, true);
    if (new) matrix_setcolumnptr(new, 0, vals);
    return morpho_wrapandbind(v, (object *) new);
}

value Mesh_setvertexposition__int_matrix(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    unsigned int id=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    objectmatrix *mat = MORPHO_GETMATRIX(MORPHO_GETARG(args, 1));

    if (matrix_setcolumnptr(m->vert, id, mat->elements)!=LINALGERR_OK) MORPHO_RAISE(v, MESH_INVLDID);
    return MORPHO_NIL;
}

value Mesh_connectivitymatrix__int_int(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    unsigned int row=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    unsigned int col=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));

    objectsparse *s=mesh_getconnectivityelement(m, row, col);
    if (!s && row>0 && row!=col) s=mesh_addconnectivityelement(m, row, col);
    return s ? MORPHO_OBJECT(s) : MORPHO_NIL;
}

value Mesh_resetconnectivity(vm *v, int nargs, value *args) {
    mesh_resetconnectivity(MORPHO_GETMESH(MORPHO_SELF(args)));
    return MORPHO_NIL;
}

value Mesh_addgrade__int(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    unsigned int g=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    if (g>0 && !mesh_getconnectivityelement(m, 0, g) && !mesh_addgrade(m, g)) MORPHO_RAISEVARGS(v, MESH_ADDGRDOOB, g, mesh_maxgrade(m));
    return MORPHO_NIL;
}

value Mesh_addgrade__int_sparse(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    unsigned int grade=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    objectsparse *s=MORPHO_GETSPARSE(MORPHO_GETARG(args, 1));
    if (s && grade>0 && !mesh_setconnectivityelement(m, 0, grade, s)) MORPHO_RAISE(v, ERROR_ALLOCATIONFAILED);
    mesh_freezeconnectivity(m);
    return MORPHO_NIL;
}

value Mesh_removegrade__int(vm *v, int nargs, value *args) {
    unsigned int g=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    if (g==0) return MORPHO_NIL;
    mesh_removegrade(MORPHO_GETMESH(MORPHO_SELF(args)), g);
    return MORPHO_NIL;
}

value Mesh_addsymmetry__x(vm *v, int nargs, value *args) {
    mesh_addsymmetry(v, MORPHO_GETMESH(MORPHO_SELF(args)), MORPHO_GETARG(args, 0), NULL);
    return MORPHO_NIL;
}

value Mesh_addsymmetry__x_selection(vm *v, int nargs, value *args) {
    mesh_addsymmetry(v, MORPHO_GETMESH(MORPHO_SELF(args)),
                     MORPHO_GETARG(args, 0),
                     MORPHO_GETSELECTION(MORPHO_GETARG(args, 1)));
    return MORPHO_NIL;
}

value Mesh_maxgrade(vm *v, int nargs, value *args) {
    return MORPHO_INTEGER(mesh_maxgrade(MORPHO_GETMESH(MORPHO_SELF(args))));
}

value Mesh_count(vm *v, int nargs, value *args) {
    return MORPHO_INTEGER(mesh_nelementsforgrade(MORPHO_GETMESH(MORPHO_SELF(args)), 0));
}

value Mesh_count__int(vm *v, int nargs, value *args) {
    return MORPHO_INTEGER(mesh_nelementsforgrade(MORPHO_GETMESH(MORPHO_SELF(args)),
                                                 MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0))));
}

value Mesh_barycentric__int_int_matrix(vm *v, int nargs, value *args) {
    objectmesh *m=MORPHO_GETMESH(MORPHO_SELF(args));
    grade g=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    elementid id=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    objectmatrix *x=MORPHO_GETMATRIX(MORPHO_GETARG(args, 2));

    if (x->nrows!=m->dim || x->ncols!=1) MORPHO_RAISE(v, MESH_BARYDIM);
    if (g<1 || g>mesh_maxgrade(m)) MORPHO_RAISE(v, MESH_BARYFAILED);

    objectmatrix *lambda=matrix_new(g+1, 1, true);
    if (lambda && !mesh_getbarycentriccoordinates(m, g, id, x->elements, lambda->elements)) {
        object_free((object *) lambda);
        MORPHO_RAISE(v, MESH_BARYFAILED);
    }
    return morpho_wrapandbind(v, (object *) lambda);
}

value Mesh_clone(vm *v, int nargs, value *args) {
    return morpho_wrapandbind(v, (object *) mesh_clone(MORPHO_GETMESH(MORPHO_SELF(args))));
}

MORPHO_BEGINCLASS(Mesh)
MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "()", Mesh_print, MORPHO_FN_IO),
MORPHO_METHOD_SIGNATURE(MORPHO_SAVE_METHOD, "(String)", Mesh_save__string, MORPHO_FN_IO|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MESH_VERTEXMATRIX_METHOD, "Matrix ()", Mesh_vertexmatrix, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MESH_SETVERTEXMATRIX_METHOD, "(Matrix)", Mesh_setvertexmatrix__matrix, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MESH_VERTEXPOSITION_METHOD, "Matrix (Int)", Mesh_vertexposition__int, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MESH_SETVERTEXPOSITION_METHOD, "(Int, Matrix)", Mesh_setvertexposition__int_matrix, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MESH_RESETCONNECTIVITY_METHOD, "()", Mesh_resetconnectivity, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MESH_CONNECTIVITYMATRIX_METHOD, "Sparse (Int, Int)", Mesh_connectivitymatrix__int_int, MORPHO_FN_MUTATES|MORPHO_FN_ALLOCATES),
MORPHO_METHOD_SIGNATURE(MESH_ADDGRADE_METHOD, "(Int)", Mesh_addgrade__int, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MESH_ADDGRADE_METHOD, "(Int, Sparse)", Mesh_addgrade__int_sparse, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MESH_REMOVEGRADE_METHOD, "(Int)", Mesh_removegrade__int, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MESH_ADDSYMMETRY_METHOD, "(_)", Mesh_addsymmetry__x, MORPHO_FN_MUTATES|MORPHO_FN_REENTRANT|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MESH_ADDSYMMETRY_METHOD, "(_, Selection)", Mesh_addsymmetry__x_selection, MORPHO_FN_MUTATES|MORPHO_FN_REENTRANT|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MESH_BARYCENTRIC_METHOD, "Matrix (Int, Int, Matrix)", Mesh_barycentric__int_int_matrix, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MESH_MAXGRADE_METHOD, "Int ()", Mesh_maxgrade, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int ()", Mesh_count, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int (Int)", Mesh_count__int, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, "Mesh ()", Mesh_clone, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void mesh_initialize(void) {
    objectmeshtype=object_addtype(&objectmeshdefn);

    morpho_addfunction(MESH_CLASSNAME, "Mesh ()", mesh_constructor, MORPHO_FN_CONSTRUCTOR|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS, NULL);
    morpho_addfunction(MESH_CLASSNAME, "Mesh (Int)", mesh_constructor__int, MORPHO_FN_CONSTRUCTOR|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS, NULL);
    morpho_addfunction(MESH_CLASSNAME, "Mesh (Matrix)", mesh_constructor__matrix, MORPHO_FN_CONSTRUCTOR|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS, NULL);
    morpho_addfunction(MESH_CLASSNAME, "Mesh (String)", mesh_constructor__string, MORPHO_FN_CONSTRUCTOR|MORPHO_FN_ALLOCATES|MORPHO_FN_IO|MORPHO_FN_THROWS, NULL);

    value objclass = builtin_findclassfromcstring(OBJECT_CLASSNAME);

    value meshclass=builtin_addclass(MESH_CLASSNAME, MORPHO_GETCLASSDEFINITION(Mesh), objclass);
    object_setveneerclass(OBJECT_MESH, meshclass);

    morpho_defineerror(MESH_FILENOTFOUND, ERROR_HALT, MESH_FILENOTFOUND_MSG);
    morpho_defineerror(MESH_VERTMTRXDIM, ERROR_HALT, MESH_VERTMTRXDIM_MSG);
    morpho_defineerror(MESH_INVLDDIM, ERROR_HALT, MESH_INVLDDIM_MSG);
    morpho_defineerror(MESH_LOADVERTEXDIM, ERROR_HALT, MESH_LOADVERTEXDIM_MSG);
    morpho_defineerror(MESH_LOADVERTEXCOORD, ERROR_HALT, MESH_LOADVERTEXCOORD_MSG);
    morpho_defineerror(MESH_LOADPARSEERR, ERROR_HALT, MESH_LOADPARSEERR_MSG);
    morpho_defineerror(MESH_LOADVERTEXNUM, ERROR_HALT, MESH_LOADVERTEXNUM_MSG);
    morpho_defineerror(MESH_LOADVERTEXID, ERROR_HALT, MESH_LOADVERTEXID_MSG);
    morpho_defineerror(MESH_LOADVERTEXNOTFOUND, ERROR_HALT, MESH_LOADVERTEXNOTFOUND_MSG);
    morpho_defineerror(MESH_INVLDID, ERROR_HALT, MESH_INVLDID_MSG);
    morpho_defineerror(MESH_BARYDIM, ERROR_HALT, MESH_BARYDIM_MSG);
    morpho_defineerror(MESH_BARYFAILED, ERROR_HALT, MESH_BARYFAILED_MSG);
    morpho_defineerror(MESH_ADDGRDOOB, ERROR_HALT, MESH_ADDGRDOOB_MSG);
    morpho_defineerror(MESH_ADDSYMMSNGTRNSFRM, ERROR_HALT, MESH_ADDSYMMSNGTRNSFRM_MSG);
}

#endif
