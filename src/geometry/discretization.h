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

/* -------------------------------------------------------
 * FunctionSpace veneer class
 * ------------------------------------------------------- */

#define FUNCTIONSPACE_CLASSNAME "FunctionSpace"

/* -------------------------------------------------------
 * Discretization error messages
 * ------------------------------------------------------- */

/* -------------------------------------------------------
 * Discretization interface
 * ------------------------------------------------------- */

void discretization_initialize(void);

#endif
