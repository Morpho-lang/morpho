/** @file mesh.h
 *  @author T J Atherton
 *
 *  @brief Mesh class and associated functionality
 */

#ifndef mesh_h
#define mesh_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "varray.h"
#include "sparse.h"

/* -------------------------------------------------------
 * Mesh object
 * ------------------------------------------------------- */

extern objecttype objectmeshtype;
#define OBJECT_MESH objectmeshtype

typedef struct {
    object obj;
    unsigned int dim;
    unsigned int nconn; /** Length of conn; (dim+1)^2 */
    objectmatrix *vert;
    value *conn;
    value conndata[]; /** (dim+1) x (dim+1) connectivity table */
} objectmesh;

/** Tests whether an object is a mesh */
#define MORPHO_ISMESH(val) object_istype(val, OBJECT_MESH)

/** Gets the object as a mesh */
#define MORPHO_GETMESH(val)   ((objectmesh *) MORPHO_GETOBJECT(val))

/** Creates a mesh object */
objectmesh *object_newmesh(unsigned int dim, unsigned int nv, double *v);

typedef int grade;
typedef int elementid;

DECLARE_VARRAY(elementid, elementid);

#define MESH_GRADE_VERTEX 0
#define MESH_GRADE_LINE 1
#define MESH_GRADE_AREA 2
#define MESH_GRADE_VOLUME 3

/** Default ambient dimension for Mesh() */
#define MESH_DEFAULTDIM 3

/* -------------------------------------------------------
 * Mesh class
 * ------------------------------------------------------- */

#define MESH_CLASSNAME "Mesh"

#define MESH_VERTSECTION "vertices"
#define MESH_EDGESECTION "edges"
#define MESH_FACESECTION "faces"
#define MESH_VOLSECTION  "volumes"

#define MESH_VERTEXMATRIX_METHOD           "vertexmatrix"
#define MESH_SETVERTEXMATRIX_METHOD        "setvertexmatrix"

#define MESH_VERTEXPOSITION_METHOD         "vertexposition"
#define MESH_SETVERTEXPOSITION_METHOD      "setvertexposition"

#define MESH_RESETCONNECTIVITY_METHOD      "resetconnectivity"
#define MESH_CONNECTIVITYMATRIX_METHOD     "connectivitymatrix"
#define MESH_ADDGRADE_METHOD               "addgrade"
#define MESH_REMOVEGRADE_METHOD            "removegrade"
#define MESH_MAXGRADE_METHOD               "maxgrade"
#define MESH_ADDSYMMETRY_METHOD            "addsymmetry"
#define MESH_BARYCENTRIC_METHOD            "barycentric"

#define MESH_TRANSFORM_METHOD              "transform"

/* -------------------------------------------------------
 * Mesh error messages
 * ------------------------------------------------------- */

#define MESH_VERTMTRXDIM                     "MshVrtMtrxDim"
#define MESH_VERTMTRXDIM_MSG                 "Vertex matrix dimensions inconsistent with mesh."

#define MESH_INVLDDIM                        "MshInvldDim"
#define MESH_INVLDDIM_MSG                    "Mesh dimension must be a non-negative integer."

#define MESH_LOADVERTEXDIM                   "MshLdVrtDim"
#define MESH_LOADVERTEXDIM_MSG               "Vertex has inconsistent dimensions at line %i."

#define MESH_LOADVERTEXCOORD                 "MshLdVrtCrd"
#define MESH_LOADVERTEXCOORD_MSG             "Vertex has nonnumerical coordinates at line %i."

#define MESH_LOADPARSEERR                    "MshLdPrsErr"
#define MESH_LOADPARSEERR_MSG                "Parse error in mesh file at line %i."

#define MESH_LOADVERTEXNUM                   "MshLdVrtNm"
#define MESH_LOADVERTEXNUM_MSG               "Element has incorrect number of vertices at line %i."

#define MESH_LOADVERTEXID                    "MshLdVrtId"
#define MESH_LOADVERTEXID_MSG                "Vertex id must be an integer at line %i."

#define MESH_LOADVERTEXNOTFOUND              "MshLdVrtNtFnd"
#define MESH_LOADVERTEXNOTFOUND_MSG          "Vertex not found at line %i."

#define MESH_FILENOTFOUND                    "MshFlNtFnd"
#define MESH_FILENOTFOUND_MSG                "Mesh file '%s' not found."

#define MESH_ADDGRDOOB                       "MshAddGrdOutOfBnds"
#define MESH_ADDGRDOOB_MSG                   "Cannot add elements of grade %d to mesh with max grade %d"

#define MESH_INVLDID                         "MshInvldId"
#define MESH_INVLDID_MSG                     "Invalid element id."

#define MESH_BARYDIM                         "MshBaryDim"
#define MESH_BARYDIM_MSG                     "Position matrix dimensions inconsistent with mesh."

#define MESH_BARYFAILED                      "MshBaryFailed"
#define MESH_BARYFAILED_MSG                  "Unable to compute barycentric coordinates for the given element."

#define MESH_ADDSYMMSNGTRNSFRM               "MshAddSymMsngTrnsfrm"
#define MESH_ADDSYMMSNGTRNSFRM_MSG           "Method 'addsymmetry' expects an object that provides a method 'transform'."

/* -------------------------------------------------------
 * Mesh interface
 * ------------------------------------------------------- */

elementid mesh_nvertices(objectmesh *mesh);
elementid mesh_nelements(objectsparse *conn);
elementid mesh_nelementsforgrade(objectmesh *mesh, grade g);
grade mesh_maxgrade(objectmesh *mesh);

bool mesh_getvertexcoordinatesaslist(objectmesh *mesh, elementid id, double **out);
bool mesh_getvertexcoordinatesasvalues(objectmesh *mesh, elementid id, value *val);

objectsparse *mesh_addgrade(objectmesh *mesh, grade g);
objectsparse *mesh_addconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col);
objectsparse *mesh_getconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col);
bool mesh_getconnectivity(objectsparse *conn, elementid id, int *nentries, int **entries);
bool mesh_matchelements(objectsparse *vmatrix, grade g, int nids, int *ids, int maxmatches, int *nmatches, int *matches);
void mesh_freezeconnectivity(objectmesh *mesh);

int mesh_findneighbors(objectmesh *mesh, grade g, elementid id, grade target, varray_elementid *neighbors);
bool mesh_getsynonyms(objectmesh *mesh, grade g, elementid id, varray_elementid *synonymids);

void varray_elementidwriteunique(varray_elementid *list, elementid id);

void mesh_initialize(void);

#endif

#endif /* mesh_h */
