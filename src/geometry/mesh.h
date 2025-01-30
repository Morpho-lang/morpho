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
#include "matrix.h"
#include "sparse.h"

/* -------------------------------------------------------
 * Mesh object
 * ------------------------------------------------------- */

extern objecttype objectmeshtype;
#define OBJECT_MESH objectmeshtype

typedef struct {
    object obj;
    unsigned int dim;
    objectmatrix *vert;
    objectarray *conn;
    object *link;
} objectmesh;

/** Tests whether an object is a mesh */
#define MORPHO_ISMESH(val) object_istype(val, OBJECT_MESH)

/** Gets the object as a mesh */
#define MORPHO_GETMESH(val)   ((objectmesh *) MORPHO_GETOBJECT(val))

/** Creates a mesh object */
objectmesh *object_newmesh(unsigned int dim, unsigned int nv, double *v);

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

#define MESH_TRANSFORM_METHOD              "transform"

typedef int grade;
typedef int elementid;

DECLARE_VARRAY(elementid, elementid);

#define MESH_GRADE_VERTEX 0
#define MESH_GRADE_LINE 1
#define MESH_GRADE_AREA 2
#define MESH_GRADE_VOLUME 3

#define MESH_VERTMTRXDIM                     "MshVrtMtrxDim"
#define MESH_VERTMTRXDIM_MSG                 "Vertex matrix dimensions inconsistent with mesh."

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

#define MESH_STVRTPSNARGS                    "MshStVrtPsnArgs"
#define MESH_STVRTPSNARGS_MSG                "Method 'setvertexposition' expects a vertex id and a position matrix as arguments."

#define MESH_VRTPSNARGS                      "MshVrtPsnArgs"
#define MESH_VRTPSNARGS_MSG                  "Method 'vertexposition' expects a vertex id as the argument."

#define MESH_CNNMTXARGS                      "MshCnnMtxArgs"
#define MESH_CNNMTXARGS_MSG                  "Method 'connectivitymatrix' expects integer arguments."

#define MESH_ADDGRDARGS                      "MshAddGrdArgs"
#define MESH_ADDGRDARGS_MSG                  "Method 'addgrade' expects either an integer grade and, optionally, a sparse connectivity matrix."

#define MESH_ADDGRDOOB                       "MshAddGrdOutOfBnds"
#define MESH_ADDGRDOOB_MSG                   "Cannot add elements of grade %d to mesh with max grade %d"

#define MESH_INVLDID                         "MshInvldId"
#define MESH_INVLDID_MSG                     "Invalid element id."

#define MESH_ADDSYMARGS                      "MshAddSymArgs"
#define MESH_ADDSYMARGS_MSG                  "Method 'addsymmetry' expects an object that provides a method 'transform' and an optional selection."

#define MESH_ADDSYMMSNGTRNSFRM               "MshAddSymMsngTrnsfrm"
#define MESH_ADDSYMMSNGTRNSFRM_MSG           "Method 'addsymmetry' expects an object that provides a method 'transform'."

#define MESH_ADDSYMNOMTCH                    "MshAddSymNoMtch"
#define MESH_ADDSYMNOMTCH_MSG                "Addsymmetry found no matching vertices."

#define MESH_CONSTRUCTORARGS                  "MshArgs"
#define MESH_CONSTRUCTORARGS_MSG              "Mesh expects either a single file name or no argurments"

/* Tolerances */

/** This controls how close two points can be before they're indistinct */
#define MESH_NEARESTPOINTEPS 1e-10

void varray_elementidwriteunique(varray_elementid *list, elementid id);

elementid mesh_nvertices(objectmesh *mesh);
elementid mesh_nelements(objectsparse *conn);
elementid mesh_nelementsforgrade(objectmesh *mesh, grade g);
grade mesh_maxgrade(objectmesh *mesh);

bool mesh_checkconnectivity(objectmesh *mesh);
objectsparse *mesh_newconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col);
objectsparse *mesh_addgrade(objectmesh *mesh, grade g);
objectsparse *mesh_addconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col);
objectsparse *mesh_getconnectivityelement(objectmesh *mesh, unsigned int row, unsigned int col);


bool mesh_getconnectivity(objectsparse *conn, elementid id, int *nentries, int **entries);
void mesh_freezeconnectivity(objectmesh *mesh);
void mesh_resetconnectivity(objectmesh *m);

bool mesh_getvertexcoordinates(objectmesh *mesh, elementid id, double *val);
bool mesh_getvertexcoordinatesaslist(objectmesh *mesh, elementid id, double **out);
bool mesh_getvertexcoordinatesasvalues(objectmesh *mesh, elementid id, value *val);

bool mesh_getsynonyms(objectmesh *mesh, grade g, elementid id, varray_elementid *synonymids);
int mesh_findneighbors(objectmesh *mesh, grade g, elementid id, grade target, varray_elementid *neighbors);

void mesh_initialize(void);

#endif

#endif /* mesh_h */
