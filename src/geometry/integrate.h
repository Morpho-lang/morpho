/** @file integrate.h
 *  @author T J Atherton
 *
 *  @brief Numerical integration
*/

#ifndef integration_h
#define integration_h

#include <stdio.h>
#include "morpho.h"

#define INTEGRATE_ACCURACYGOAL 1e-6
#define INTEGRATE_ZEROCHECK 1e-15
#define INTEGRATE_MAXRECURSION 100
#define INTEGRATE_MAXITERATIONS 10000

/* -------------------------------------------------------
 * Integrator type definitions
 * ------------------------------------------------------- */

/* ----------------------------------
 * Integrands
 * ---------------------------------- */

/** Generic specification for an integrand.
 * @param[in] dim            - The dimension of the space
 * @param[in] lambda      - Barycentric coordinates for the element
 * @param[in] x                 - Coordinates of the point calculated from interpolation
 * @param[in] nquantity - Number of quantities
 * @param[in] quantity - List of quantities evaluated for the point, calculated from interpolation
 * @param[in] ref             - A reference passed by the caller (typically things constant over the domain
 * @returns value of the integrand at the appropriate point with interpolated quantities.
 */
typedef bool (integrandfunction) (unsigned int dim, double *lambda, double *x, unsigned int nquantity, value *quantity, void *ref, double *fout);

/* ----------------------------------
 * Quadrature rules define wts/nodes
 * ---------------------------------- */

typedef unsigned int quadrature_flags;

typedef struct {
    char *name; /** Identifier for the rule */
    int grade; /** Dimensionality of element the rule operates on */
    int order; /** Order of integrator */
    int nnodes; /** Number of nodes */
    int next; /** Number of extension points, or -1 for no extension */
    double *nodes; /** Nodes */
    double *weights; /** Weights */
} quadraturerule;

/** Indicate that a quadrature rule doesn't have extensions */
#define INTEGRATE_NOEXT -1

/* ----------------------------------
 * Subdivision rules
 * ---------------------------------- */

typedef struct {
    int grade;    /** Appropriate grade for the strategy */
    int npts;     /** Number of new pts created */
    double *pts;  /** New barycentric coordinates */
    int nels;     /** Number of new elements created */
    int *newels;  /** Indices of new elements */
    double *weights;  /** Weights of new elements */
} subdivisionrule;

/* --------------------------------
 * Quadrature work items
 * -------------------------------- */

typedef struct {
    double weight; /** Overall element weight */
    int elementid; /** Id of element on the element stack */
    //value **quantity;
    double val; /** Value of work item */
    double lval; /** Value of work item from lower order estimate */
    double err; /** Error estimate of work item */
} quadratureworkitem;

DECLARE_VARRAY(quadratureworkitem, quadratureworkitem)

/* ----------------------------------
 * Integrator
 * ---------------------------------- */

typedef struct {
    integrandfunction *integrand; /** Function to integrate */
    
    int dim; /** Dimension of points in embedded space */
    int nquantity; /** Number of quantities to interpolate */
    
    bool adapt; /** Enable adaptive integration */
    quadraturerule *rule;  /** Quadrature rule to use */
    quadraturerule *errrule; /** Additional rule for error estimation */
    
    subdivisionrule *subdivide; /** Subdivision rule to use */
    
    int workp; /** Index of largest item in the work list */
    int freep; /** Index of a free item in the work list */
    varray_quadratureworkitem worklist; /** Work list */
    varray_double vertexstack; /** Stack of vertices */
    varray_int elementstack; /** Stack of elements */
    
    double ztol; /** Tolerance for zero detection */
    double tol; /** Tolerance for relative error */
    int maxiterations; /** Maximum number of subdivisions to perform */
    
    double val; /** Estimated value of the integral */
    double err; /** Estimated error of the integral */
    
    error emsg; /** Store error messages from the integrator */
    
    void *ref; /** Reference to pass to integrand */
} integrator;

/* -------------------------------------------------------
 * Integrator interface
 * ------------------------------------------------------- */

bool integrate_integrate(integrandfunction *integrand, unsigned int dim, unsigned int grade, double **x, unsigned int nquantity, value **quantity, void *ref, double *out);

void integrate_test(void);

#endif /* integration_h */
