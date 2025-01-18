/** @file field.h
 *  @author T J Atherton
 *
 *  @brief Fields
 */

#ifndef field_h
#define field_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "object.h"
#include "mesh.h"
#include "matrix.h"
#include <stdio.h>

/* -------------------------------------------------------
 * Field objects
 * ------------------------------------------------------- */

extern objecttype objectfieldtype;
#define OBJECT_FIELD objectfieldtype

typedef struct {
    object obj;
    objectmesh *mesh; /** The mesh the selection is referring to */
    
    unsigned int ngrades; /** Number of grades */
    unsigned int *dof; /** number of degrees of freedom per entry in each grade */
    unsigned int *offset; /** Offsets into the store for each grade */
    
    value prototype; /** Prototype object */
    unsigned int psize; /** Number of dofs per copy of the prototype */
    unsigned int nelements; /** Total number of elements in the fireld */
    void *pool; /** Pool of statically allocated objects */
    
    objectmatrix data; /** Underlying data store */
} objectfield;

/** Tests whether an object is a field */
#define MORPHO_ISFIELD(val) object_istype(val, OBJECT_FIELD)

/** Gets the object as a field */
#define MORPHO_GETFIELD(val)   ((objectfield *) MORPHO_GETOBJECT(val))

/** Creates an empty field object */
objectfield *object_newfield(objectmesh *mesh, value prototype, unsigned int *dof);

/* -------------------------------------------------------
 * Field class
 * ------------------------------------------------------- */

#define FIELD_CLASSNAME "Field"

#define FIELD_GRADEOPTION "grade"
#define FIELD_OP_METHOD      "op"
#define FIELD_SHAPE_METHOD   "shape"
#define FIELD_MESH_METHOD    "mesh"
#define FIELD_LINEARIZE_METHOD    "linearize"
#define FIELD__LINEARIZE_METHOD    "__linearize"

#define FIELD_INDICESOUTSIDEBOUNDS       "FldBnds"
#define FIELD_INDICESOUTSIDEBOUNDS_MSG   "Field index out of bounds."

#define FIELD_INVLDINDICES               "FldInvldIndx"
#define FIELD_INVLDINDICES_MSG           "Field indices must be numerical."

#define FIELD_ARITHARGS                  "FldInvldArg"
#define FIELD_ARITHARGS_MSG              "Field arithmetic methods expect a field or number as their argument."

#define FIELD_INCOMPATIBLEMATRICES       "FldIncmptbl"
#define FIELD_INCOMPATIBLEMATRICES_MSG   "Fields have incompatible shape."

#define FIELD_INCOMPATIBLEVAL            "FldIncmptblVal"
#define FIELD_INCOMPATIBLEVAL_MSG        "Assignment value has incompatible shape with field elements."

#define FIELD_ARGS                       "FldArgs"
#define FIELD_ARGS_MSG                   "Field allows 'grade' as an optional argument."

#define FIELD_OP                         "FldOp"
#define FIELD_OP_MSG                     "Method 'op' requires a callable object as the first argument; all other arguments must be fields of compatible shape."

#define FIELD_OPRETURN                   "FldOpFn"
#define FIELD_OPRETURN_MSG               "Could not construct a Field from the return value of the function passed to 'op'."

#define FIELD_MESHARG                    "FldMshArg"
#define FIELD_MESHARG_MSG                "Field expects a mesh as its first argurment"

objectfield *field_clone(objectfield *f);

void field_zero(objectfield *field);
bool field_addpool(objectfield *field);
unsigned int field_sizeprototype(value prototype);

unsigned int field_dofforgrade(objectfield *f, grade g);
bool field_getelement(objectfield *field, grade grade, elementid el, int indx, value *out);
bool field_getelementwithindex(objectfield *field, int indx, value *out);
bool field_getelementaslist(objectfield *field, grade grade, elementid el, int indx, unsigned int *nentries, double **out);

bool field_setelement(objectfield *field, grade grade, elementid el, int indx, value val);

void field_initialize(void);

#endif

#endif /* field_h */
