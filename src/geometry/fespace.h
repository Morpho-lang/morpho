/** @file fespace.h
 *  @author T J Atherton
 *
 *  @brief Finite element fespaces and veneer class
 */

#ifndef fespace_h
#define fespace_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "geometry.h"
#include "elements.h"

/* -------------------------------------------------------
 * fespace type definitions
 * ------------------------------------------------------- */

/** @brief Interpolation functions are called to assign weights to the nodes given barycentric coordinates */
typedef void (*interpolationfn) (double *, double *);

/** @brief Gradient callbacks return d Xi[i] / d lambda[j] in column-major order:
 *         out[FESPACE_GRAD_INDEX(nnodes, j, i)] */
typedef void (*gradientfn) (double *, double *);

/** @brief Hessian callbacks return d^2 Xi[i] / d x[row] d x[col] in column-major tensor order:
 *         out[FESPACE_HESS_INDEX(nnodes, dim, row, col, i)] */
typedef void (*hessianfn) (double *, double *);

/** @brief Indexing helpers for derivative callback output layouts */
#define FESPACE_GRAD_INDEX(nnodes, component, node) ((component)*(nnodes)+(node))
#define FESPACE_HESS_INDEX(nnodes, dim, row, col, node) ((((col)*(dim))+(row))*(nnodes)+(node))

/** @brief Capability helpers for derivative evaluation */
#define FESPACE_HASGRADIENT(disc) ((disc)->gfn!=NULL)
#define FESPACE_HASHESSIAN(disc) ((disc)->hfn!=NULL)

/** @brief Element definitions comprise a sequence of instructions to map field degrees of freedom to local nodes */
typedef int eldefninstruction;

/** @brief Element definition opcodes. LINE/AREA/VOLUME values match the corresponding mesh grade. */
#define ELEMENT_LINE_OPCODE     1
#define ELEMENT_AREA_OPCODE     2
#define ELEMENT_VOLUME_OPCODE   3
#define ELEMENT_QUANTITY_OPCODE 255
#define ELEMENT_ENDDEFN         -1

/** @brief fespace definition structure */
typedef struct sfespace {
    char *name; /**  Name of the fespace */
    grade grade; /** Grade of element this fespace is defined on */
    unsigned int *shape; /** Number of degrees of freedom on each grade; must have grade+1 entries */
    int degree; /** Highest degree of polynomial represented by this element */
    int nnodes; /** Number of nodes for this element type */
    int nsubel; /** Number of subelements used by */
    double *nodes; /** Node positions */
    interpolationfn ifn; /** Interpolation function; receives barycentric coordinates as input and returns weights per node */
    gradientfn gfn; /** Gradient interpolation function in barycentric coordinates */
    hessianfn hfn; /** Hessian interpolation function in reference coordinates */
    eldefninstruction *eldefn; /** Element definition */
    struct sfespace **lower; /** Discretization to be used for interpolation on lower grades */
} fespace;

/* -------------------------------------------------------
 * fespace object type
 * ------------------------------------------------------- */

extern objecttype objectfespacetype;
#define OBJECT_FESPACE objectfespacetype

typedef struct {
    object obj;
    fespace *fespace;
} objectfespace;

/** Tests whether an object is a fespace */
#define MORPHO_ISFESPACE(val) object_istype(val, OBJECT_FESPACE)

/** Gets the object as a fespace */
#define MORPHO_GETFESPACE(val)   ((objectfespace *) MORPHO_GETOBJECT(val))

/** Name of a fespace definition, or NULL */
#define FESPACE_NAME(disc) ((disc) ? (disc)->name : NULL)

/** Name of a FiniteElementSpace value, or NULL if val is not a space */
#define MORPHO_GETFESPACENAME(val) (MORPHO_ISFESPACE(val) ? FESPACE_NAME(MORPHO_GETFESPACE(val)->fespace) : NULL)

/* -------------------------------------------------------
 * FiniteElementSpace veneer class
 * ------------------------------------------------------- */

#define FINITEELEMENTSPACE_CLASSNAME "FiniteElementSpace"

#define FINITEELEMENTSPACE_GRADE_METHOD "grade"
#define FINITEELEMENTSPACE_LAYOUT_METHOD "layout"
#define FINITEELEMENTSPACE_NODEELEMENTINDEX_METHOD "nodeElementIndex"
#define FINITEELEMENTSPACE_NODECOORDS_METHOD "nodeCoords"

/* -------------------------------------------------------
 * FiniteElementSpace error messages
 * ------------------------------------------------------- */

#define FNSPC_ARGS                       "FnSpcArgs"
#define FNSPC_ARGS_MSG                   "Function space must be initialized with a label and a grade."

#define FNSPC_NOTFOUND                   "FnSpcNtFnd"
#define FNSPC_NOTFOUND_MSG               "Function space '%s' on grade %i not found."

/* -------------------------------------------------------
 * fespace interface
 * ------------------------------------------------------- */

objectfespace *objectfespace_new(fespace *disc);

fespace *fespace_find(char *name, grade g);
fespace *fespace_findlinear(grade g);

objectfespace *fespace_newfromname(char *name, grade g);
objectfespace *fespace_newlinear(grade g);

bool fespace_doftofieldindx(objectfield *field, fespace *disc, int nv, int *vids, fieldindx *findx);
bool fespace_nodefieldindex(fespace *disc, int node, grade *g, int *sid, int *indx);

bool fespace_lower(fespace *disc, grade target, fespace **out);

bool fespace_getnodecoords(fespace *disc, int node, double *lambda);
bool fespace_layout(objectfield *field, fespace *disc, objectsparse **out);
void fespace_gradient(fespace *disc, double *lambda, objectmatrix *grad);
void fespace_hessian(fespace *disc, double *lambda, objectmatrix *hess);

void fespace_initialize(void);

#endif

#endif /* fespace_h */
