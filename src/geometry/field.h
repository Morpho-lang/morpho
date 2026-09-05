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
#include "objectpool.h"
#include "mesh.h"
#include "linalg.h"
#include <stdio.h>

/* -------------------------------------------------------
 * Field objects
 * ------------------------------------------------------- */

extern objecttype objectfieldtype;
extern objecttype objectscalarfieldtype;
extern objecttype objectmatrixfieldtype;
extern objecttype objectcomplexfieldtype;
extern objecttype objectcomplexmatrixfieldtype;
#define OBJECT_FIELD objectfieldtype
#define OBJECT_SCALARFIELD objectscalarfieldtype
#define OBJECT_MATRIXFIELD objectmatrixfieldtype
#define OBJECT_COMPLEXFIELD objectcomplexfieldtype
#define OBJECT_COMPLEXMATRIXFIELD objectcomplexmatrixfieldtype

typedef struct sfieldinterfacedefn fieldinterfacedefn;

/** A Field is a packed list of doubles (data) with a local interpretation
    of each slice, individually accessible through an object pool. */
typedef struct {
    object obj;
    objectmesh *mesh; /** The mesh the selection is referring to */
    
    unsigned int ngrades; /** Number of grades */
    unsigned int *dof; /** number of degrees of freedom per entry in each grade */
    unsigned int *offset; /** Offsets into the store for each grade */
    
    value prototype; /** Prototype object */
    fieldinterfacedefn *iface; /** Element interface for this prototype */
    unsigned int psize; /** Number of doubles per copy of the prototype */
    unsigned int nelements; /** Total number of elements in the field */
    objectpool pool; /** Pooled objects that interpret slices of data */
    
    value fnspc; /** Function space used */
    
    objectmatrix data; /** Packed dof store (always a real Matrix) */
} objectfield;

/** Tests whether a value is any Field kind */
bool field_isafield(value val);
#define MORPHO_ISFIELD(val) field_isafield(val)

/** Gets the object as a field */
#define MORPHO_GETFIELD(val)   ((objectfield *) MORPHO_GETOBJECT(val))

/** Creates an empty field object */
objectfield *object_newfield(objectmesh *mesh, value prototype, value fnspc, unsigned int *shape);

/* -------------------------------------------------------
 * Element interface
 * ------------------------------------------------------- */

/** Doubles per field element. Runs before the Field exists. */
typedef unsigned int (*field_doffn_t) (value prototype);

/** Wrap a dof slice as a Morpho value.
    @param in    packed doubles (a field element or a scratch buffer of the same kind)
    @param pool  pooled object for this index (CHILD of the Field), or NULL to allocate. */
typedef bool (*field_materializefn_t) (objectfield *f, double *in, void *pool, value *out);

/** Write a Morpho value into a dof slice. */
typedef bool (*field_dematerializefn_t) (objectfield *f, value in, double *out);

/** Finish one pool slot as a view onto el. NULL means headers only. */
typedef void (*field_poolinitfn_t) (objectfield *f, void *slot, double *el);

/** Object type of pool slots; needed because OBJECT_* ids are assigned at startup. */
typedef objecttype (*field_pooltypefn_t) (value prototype);

struct sfieldinterfacedefn {
    field_doffn_t doffn;
    field_materializefn_t materialize;
    field_dematerializefn_t dematerialize;
    field_poolinitfn_t poolinit;     /** NULL: headers only */
    size_t poolsize;                 /** 0 if this kind needs no pool */
    field_pooltypefn_t pooltypefn;   /** NULL if this kind needs no pool */
};

fieldinterfacedefn *field_getinterface(value prototype);

/* -------------------------------------------------------
 * Indexing a Field
 * ------------------------------------------------------- */

typedef struct {
    grade g;      // The grade
    elementid id; // The element
    int indx;     // Quantity index
} fieldindx;

/* -------------------------------------------------------
 * Field class
 * ------------------------------------------------------- */

extern value field_gradeoption;

#define FIELD_CLASSNAME "Field"
#define SCALARFIELD_CLASSNAME "ScalarField"
#define MATRIXFIELD_CLASSNAME "MatrixField"
#define COMPLEXFIELD_CLASSNAME "ComplexField"
#define COMPLEXMATRIXFIELD_CLASSNAME "ComplexMatrixField"

#define FIELD_GRADEOPTION "grade"
#define FIELD_FESPACEOPTION "finiteelementspace"

#define FIELD_OP_METHOD      "op"
#define FIELD_SHAPE_METHOD   "shape"
#define FIELD_FESPACE_METHOD   "finiteElementSpace"
#define FIELD_PROTOTYPE_METHOD   "prototype"
#define FIELD_MESH_METHOD    "mesh"
#define FIELD_EVALELEMENT_METHOD "evalElement"
#define FIELD_ELEMENTDOFS_METHOD "elementDofs"
#define FIELD_LINEARIZE_METHOD    "linearize"
#define FIELD__LINEARIZE_METHOD    "__linearize"

#define FIELD_INDICESOUTSIDEBOUNDS       "FldBnds"
#define FIELD_INDICESOUTSIDEBOUNDS_MSG   "Field index out of bounds."

#define FIELD_INCOMPATIBLEMATRICES       "FldIncmptbl"
#define FIELD_INCOMPATIBLEMATRICES_MSG   "Fields have incompatible shape."

#define FIELD_INCOMPATIBLEVAL            "FldIncmptblVal"
#define FIELD_INCOMPATIBLEVAL_MSG        "Assignment value has incompatible shape with field elements."

#define FIELD_ARGS                       "FldArgs"
#define FIELD_ARGS_MSG                   "Field allows 'grade' and 'finiteelementspace' as optional arguments."

#define FIELD_OP                         "FldOp"
#define FIELD_OP_MSG                     "Method 'op' requires fields as arguments after the function."

#define FIELD_OPRETURN                   "FldOpFn"
#define FIELD_OPRETURN_MSG               "Could not construct a Field from the return value of the function passed to 'op'."

#define FIELD_KIND                       "FldKind"
#define FIELD_KIND_MSG                   "Function returned a value incompatible with %s."

#define FIELD_IDXFALLBACK                "FldIdxAutoGrd"
#define FIELD_IDXFALLBACK_MSG            "Field indexed using the form f[0,id], but grade zero is empty. Using the lowest nonempty grade, but recommend using the form f[id] to select the grade automatically."

objectfield *field_clone(objectfield *f);

void field_zero(objectfield *field);
/** True if the Field needs no pool, or if the pool is allocated and initialized. */
bool field_ensurepool(objectfield *field);
unsigned int field_sizeprototype(value prototype);

unsigned int field_dofforgrade(objectfield *f, grade g);
bool field_lowestgrade(objectfield *field, grade *g);
bool field_getelement(objectfield *field, grade grade, elementid el, int indx, value *out);
bool field_getelementwithindex(objectfield *field, int indx, value *out);
bool field_getindex(objectfield *field, grade grade, elementid el, int indx, int *out);
bool field_getelementaslist(objectfield *field, grade grade, elementid el, int indx, unsigned int *nentries, double **out);
bool field_evalelement(objectfield *field, elementid el, double *lambda, value *out);

bool field_setelement(objectfield *field, grade grade, elementid el, int indx, value val);
bool field_setelementwithindex(objectfield *field, int ix, value val);

void field_initialize(void);

#endif

#endif /* field_h */
