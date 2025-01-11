/** @file integrate.h
 *  @author T J Atherton
 *
 *  @brief Numerical integration
*/

#ifndef integration_h
#define integration_h

#include <stdio.h>
#include "morpho.h"
#include "dict.h"
#include "discretization.h"

#define INTEGRATE_RULELABEL "rule"
#define INTEGRATE_DEGREELABEL "degree"
#define INTEGRATE_ADAPTLABEL "adapt"

#define INTEGRATE_ACCURACYGOAL 1e-6
#define INTEGRATE_ZEROCHECK 1e-15
#define INTEGRATE_MAXRECURSION 100
#define INTEGRATE_MAXITERATIONS 1000

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

typedef struct quadraturerule_s quadraturerule;

/** @details A quadrature rule is defined by:
    - a set of nodes, given in barycentric coordinates (d+1 values per node)
    - and a set of weights
    - metadata
    The integrator is designed to work with rules which provide a higher order extension. */
struct quadraturerule_s {
    char *name; /** Identifier for the rule */
    int grade; /** Dimensionality of element the rule operates on */
    int order; /** Order of integrator */
    int nnodes; /** Number of nodes */
    double *nodes; /** Nodes */
    double *weights; /** Weights */
    quadraturerule *ext; /** Extension rule that uses same points */
};

/* ----------------------------------
 * Subdivision rules
 * ---------------------------------- */

/** @details A subdivision rule is defined by:
    - a set of new nodes to be created in the original element, given as barycentric coordinates
    - a list of vertex ids defining the new element (the original vertices are labelled 0...grade-1
    - a list of weights for the new elements (the fraction of the total d-volume of the original element)
      N.B. weights should sum to 1 (NOT the volume of the element
    - metadata */

typedef struct subdivisionrule_struct subdivisionrule;

struct subdivisionrule_struct {
    int grade;    /** Appropriate grade for the strategy */
    int npts;     /** Number of new pts created */
    double *pts;  /** New barycentric coordinates */
    int nels;     /** Number of new elements created */
    int *newels;  /** Indices of new elements */
    double *weights;  /** Weights of new elements */
    subdivisionrule *alt; /** Alternative subdivision rule */
} ;

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
 * Quantities
 * ---------------------------------- */

typedef struct {
    int nnodes;  /** Number of quantity values per element */
    value *vals; /** List of quantity values */
    interpolationfn ifn; /** Interpolation function */
    int ndof; /** Number of degrees of freedom (this will be filled out by the integrator) */
} quantity;

/* ----------------------------------
 * Integrator
 * ---------------------------------- */

typedef struct {
    integrandfunction *integrand; /** Function to integrate */
    void *ref; /** Reference to pass to integrand function */
    
    int dim; /** Dimension of points in embedded space */
    double **x; /** Vertices defining the element */
    
    int nbary; /** Number of barycentric coordinates */
    
    int nquantity; /** Number of quantities to interpolate */
    quantity *quantity; /** Quantity list */
    value *qval; /** Interpolated quantity values */
    
    quadraturerule *rule;  /** Quadrature rule to use */
    quadraturerule *errrule; /** Additional rule for error estimation */
    
    bool adapt; /** Enable adaptive integration */
    subdivisionrule *subdivide; /** Subdivision rule to use */
    
    varray_quadratureworkitem worklist; /** Work list */
    varray_double vertexstack; /** Stack of vertices */
    varray_int elementstack; /** Stack of elements */
    
    double ztol; /** Tolerance for zero detection */
    double tol; /** Tolerance for relative error */
    int maxiterations; /** Maximum number of subdivisions to perform */
    
    int niterations; /** Number of iterations performed */
    double val; /** Estimated value of the integral */
    double err; /** Estimated error of the integral */
    
    error emsg; /** Store error messages from the integrator */
} integrator;

/* -------------------------------------------------------
 * Integrator errors
 * ------------------------------------------------------- */

#define INTEGRATE_QDRTRMXSBDVSNS      "QdrtrMxSbdvns"
#define INTEGRATE_QDRTRMXSBDVSNS_MSG  "Maximum number of subdivisions reached in integrator."

#define INTEGRATE_QDRTRRLNTFND        "QdrtrRlNtFnd"
#define INTEGRATE_QDRTRRLNTFND_MSG    "Quadrature rule not found."

/* -------------------------------------------------------
 * Integrator interface
 * ------------------------------------------------------- */

bool integrate_integrate(integrandfunction *integrand, unsigned int dim, unsigned int grade, double **x, unsigned int nquantity, value **quantity, void *ref, double *out);

bool integrate(integrandfunction *integrand, objectdictionary *method, unsigned int dim, unsigned int grade, double **x, unsigned int nquantity, quantity *quantity, void *ref, double *out, double *err);

void integrate_test(void);

#endif /* integration_h */

