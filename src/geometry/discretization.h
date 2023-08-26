/** @file discretization.h
 *  @author T J Atherton
 *
 *  @brief Finite element discretizations
 */

#ifndef discretization_h
#define discretization_h

/* -------------------------------------------------------
 * Discretization type definitions
 * ------------------------------------------------------- */

/** @brief Interpolation functions are called to assign weights to the nodes given barycentric coordinates */
typedef void (*interpolationfn) (double *, double *);

/** @brief Element definitions comprise a sequence of instructions to map field degrees of freedom to local nodes */
typedef int eldefninstruction;

/** @brief Discretization definitions */
typedef struct {
    char *name; /**  Name of the discretization */
    grade grade; /** Grade of element this discretization is defined on */
    int *shape; /** Number of degrees of freedom on each grade; must have grade+1 entries */
    int degree; /** Highest degree of polynomial represented by this element */
    int nnodes; /** Number of nodes for this element type */
    int nsubel; /** Number of subelements used by */
    double *nodes; /** Node positions */
    interpolationfn ifn; /** Interpolation function; receives barycentric coordinates as input and returns weights per node */
    eldefninstruction *eldefn; /** Element definition */
} discretization;

/* -------------------------------------------------------
 * Discretization object type
 * ------------------------------------------------------- */

extern objecttype objectdiscretizationtype;
#define OBJECT_DISCRETIZATION objectdiscretizationtype

typedef struct {
    object obj;
    discretization *discretization;
} objectdiscretization;

/** Tests whether an object is a discretization */
#define MORPHO_ISDISCRETIZATION(val) object_istype(val, OBJECT_DISCRETIZATION)

/** Gets the object as a discretization */
#define MORPHO_GETDISCRETIZATION(val)   ((objectdiscretization *) MORPHO_GETOBJECT(val))

/* -------------------------------------------------------
 * FunctionSpace veneer class
 * ------------------------------------------------------- */

#define FUNCTIONSPACE_CLASSNAME "FunctionSpace"

#define FUNCTIONSPACE_LAYOUT_METHOD "layout"

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

void discretization_initialize(void);

#endif
