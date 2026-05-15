/** @file fespace.h
 *  @author T J Atherton
 *
 *  @brief Finite element fespaces
 */

#ifndef fespace_h
#define fespace_h

#include "geometry.h"

/* -------------------------------------------------------
 * Discretization type definitions
 * ------------------------------------------------------- */

/** @brief Interpolation functions are called to assign weights to the nodes given barycentric coordinates */
typedef void (*interpolationfn) (double *, double *);

/** @brief Hessian functions are called to assign second derivatives with respect to reference coordinates */
typedef void (*hessianfn) (double *, double *);

/** @brief Element definitions comprise a sequence of instructions to map field degrees of freedom to local nodes */
typedef int eldefninstruction;

/** @brief Discretization definitions */
typedef struct sfespace {
    char *name; /**  Name of the fespace */
    grade grade; /** Grade of element this fespace is defined on */
    unsigned int *shape; /** Number of degrees of freedom on each grade; must have grade+1 entries */
    int degree; /** Highest degree of polynomial represented by this element */
    int nnodes; /** Number of nodes for this element type */
    int nsubel; /** Number of subelements used by */
    double *nodes; /** Node positions */
    interpolationfn ifn; /** Interpolation function; receives barycentric coordinates as input and returns weights per node */
    interpolationfn gfn; /** Gradient interpolation function */
    hessianfn hfn; /** Hessian interpolation function in reference coordinates */
    eldefninstruction *eldefn; /** Element definition */
    struct sfespace **lower; /** Discretization to be used for interpolation on lower grades */
} fespace;

/* -------------------------------------------------------
 * Discretization object type
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

/* -------------------------------------------------------
 * FunctionSpace veneer class
 * ------------------------------------------------------- */

#define FINITEELEMENTSPACE_CLASSNAME "FiniteElementSpace"

#define FINITEELEMENTSPACE_LAYOUT_METHOD "layout"

/* -------------------------------------------------------
 * Discretization error messages
 * ------------------------------------------------------- */

#define FNSPC_ARGS                       "FnSpcArgs"
#define FNSPC_ARGS_MSG                   "Function space must be initialized with a label and a grade."

#define FNSPC_NOTFOUND                   "FnSpcNtFnd"
#define FNSPC_NOTFOUND_MSG               "Function space '%s' on grade %i not found."

/* -------------------------------------------------------
 * Discretization interface
 * ------------------------------------------------------- */

fespace *fespace_find(char *name, grade g);
fespace *fespace_findlinear(grade g);

bool fespace_doftofieldindx(objectfield *field, fespace *disc, int nv, int *vids, fieldindx *findx);

bool fespace_lower(fespace *disc, grade target, fespace **out);

bool fespace_layout(objectfield *field, fespace *disc, objectsparse **out);
void fespace_gradient(fespace *disc, double *lambda, objectmatrix *grad);
void fespace_hessian(fespace *disc, double *lambda, objectmatrix *hess);

void fespace_initialize(void);

#endif
