/** @file integrate.h
 *  @author T J Atherton
 *
 *  @brief Numerical integration
*/

#ifndef integration_h
#define integration_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include <stdio.h>
#include "morpho.h"
#include "dict.h"
#include "fespace.h"

#define INTEGRATE_RULELABEL "rule"
#define INTEGRATE_DEGREELABEL "degree"
#define INTEGRATE_ADAPTLABEL "adapt"
#define INTEGRATE_ERRORNORMLABEL "errornorm"
#define INTEGRATE_ERRORNORMMAX "max"
#define INTEGRATE_ERRORNORMSUM "sum"
#define INTEGRATE_TOLLABEL "tol"
#define INTEGRATE_HYBRID2D "hybrid2d"

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
 * @param[in] ref             - A reference passed by the caller (typically things constant over the domain)
 * @param[in] nout            - Number of output components (1 for a scalar integrand)
 * @param[out] fout            - Value of the integrand at the appropriate point with interpolated quantities
 * @returns true on success
 */
typedef bool (integrandfunction) (unsigned int dim, double *lambda, double *x, unsigned int nquantity, value *quantity, void *ref, unsigned int nout, double *fout);

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
    double val; /** L_inf of the work-item integral (used by the priority queue) */
    double lval; /** L_inf of the lower-order estimate */
    double err; /** Error estimate (max component) */
    double sval; /** Signed scalar integral when nout==1 (avoids the workvals pool) */
    double slval; /** Signed scalar lower-order estimate when nout==1 */
    int voff; /** Offset into integrator workvals for val[nout]/lval[nout]; -1 if unused or nout==1 */
} quadratureworkitem;

DECLARE_VARRAY(quadratureworkitem, quadratureworkitem)

/* ----------------------------------
 * Quantities
 * ---------------------------------- */

typedef struct {
    int nnodes;  /** Number of quantity values per element */
    int capacity; /** Allocated length of vals / findx */
    value *vals; /** List of quantity values */
    fieldindx *findx; /** DOF indices parallel to vals (filled by preparequantities) */
    interpolationfn ifn; /** Interpolation function */
    int ndof; /** Number of degrees of freedom (this will be filled out by the integrator) */
} quantity;

/* ----------------------------------
 * Integrator type definition
 * ---------------------------------- */

typedef struct integrator_s integrator;

/* ----------------------------------
 * Failure strategy functions
 * ---------------------------------- */

/** Optional on-fail policy. May switch the starting rule; return true to retry. */
typedef bool (integratorfailurestrategyfn) (integrator *integrate);

/* ----------------------------------
 * Integrator
 * ---------------------------------- */

struct integrator_s {
    integrandfunction *integrand; /** Function to integrate */
    void *ref; /** Reference to pass to integrand function */
    
    int dim; /** Dimension of points in embedded space */
    double **x; /** Vertices defining the element */
    
    int nbary; /** Number of barycentric coordinates */
    
    int nquantity; /** Number of quantities to interpolate */
    quantity *quantity; /** Quantity list */
    value *qval; /** Interpolated quantity values stored for reuse */
    int qvalcapacity; /** Allocated length of qval */
    
    quadraturerule *rule;  /** Current starting quadrature rule */
    quadraturerule *baserule; /** Rule selected at configure; reset restores this */
    quadraturerule *errrule; /** Additional rule for error estimation */
    quadraturerule *acceptedrule; /** Rule that produced the current integral value (p-ext / strategy) */
    integratorfailurestrategyfn *strategy; /** Called if the current rule misses the tolerance; NULL means p-extension then h-adapt */
    
    bool skipcentroid; /** Skip node 0 on the next reference-element evaluation */
    double *fcentroid; /** Cached integrand at node 0 (size nout) */
    unsigned int fcentroidcap;
    
    bool errnormmax; /** Selects error norm: if set, stops on max e_K < tol |I_root|; otherwise stops on sum e_K / |I| */
    double rootscale; /** |I| after the last root rule, used by the max-norm stop. For nout>1, |I| means L_inf of the vector integral (norm-relative, not per-component relative accuracy). */
    
    bool adapt; /** Enable adaptive integration */
    subdivisionrule *subdivide; /** Subdivision rule to use */
    
    varray_quadratureworkitem worklist; /** Work list */
    quadratureworkitem rootwork; /** Root-element result from try; refine starts from this */
    varray_double vertexstack; /** Stack of vertices */
    varray_int elementstack; /** Stack of elements */
    varray_double workvals; /** Pool for per-work-item val[nout]/lval[nout] */
    
    unsigned int nout; /** Output dimension for this integrate call (1 = scalar) */
    double *vout; /** Caller-owned result buffer (size nout) */
    
    double ztol; /** Tolerance for zero detection */
    double tol; /** Relative tolerance on the integral (scalar value, or L_inf of the vector) */
    int maxiterations; /** Maximum number of subdivisions to perform */
    
    int niterations; /** Number of iterations performed */
    unsigned int nevals; /** Integrand node evaluations since the last reset */
    double val; /** Estimated value of the integral (scalar, or L_inf of vector) */
    double errest; /** Estimated error of the integral */
    
    error *err; /** Error structure to report errors */
};

/* -------------------------------------------------------
 * Integrator errors
 * ------------------------------------------------------- */

#define INTEGRATE_SBDVSNS             "IntgrtrSbdvns"
#define INTEGRATE_SBDVSNS_MSG         "Too many subdivisions in evaluating integral; possible singularity detected."

#define INTEGRATE_RLNTFND             "IntgrtrRlNtFnd"
#define INTEGRATE_RLNTFND_MSG         "Integrator quadrature rule '%s' not found."

#define INTEGRATE_RLUNAVLB            "IntgrtrRlUnavlb"
#define INTEGRATE_RLUNAVLB_MSG        "No quadrature rule is available that matches the provided integrator method dictionary."

#define INTEGRATE_MTHDTYP             "IntgrtrMthdTyp"
#define INTEGRATE_MTHDTYP_MSG         "Integrator method dictionary option '%s' must be a %s."

#define INTEGRATE_MTHDERRNRM_STRING   "String and either \"max\" or \"sum\""

/* -------------------------------------------------------
 * Integrator interface
 * ------------------------------------------------------- */

/* Easy repeated interface: init, configure once, integrate, then clear. */
void integrator_init(integrator *integrate);
bool integrator_configure(integrator *integrate, error *err, bool adapt, int grade, int order, char *name);
bool integrator_configurewithdictionary(integrator *integrate, error *err, grade g, objectdictionary *dict);
bool integrator_integrate(integrator *integrate, integrandfunction *integrand, int dim, double **x, unsigned int nquantity, quantity *quantity, void *ref, unsigned int nout, double *out);
void integrator_clear(integrator *integrate);

/* Expert staged replacement for integrator_integrate:
   1) Call integrator_try to evaluate the integrand on the base element.
   2) if ACCEPTED: call integrator_apply to evaluate a (potentially different) integrand using that formula.
   3) if REFINE: call integrator_refine to continue with h-refinement with the same integrand. */
typedef enum {
    INTEGRATOR_TRY_ACCEPTED,
    INTEGRATOR_TRY_REFINE,
    INTEGRATOR_TRY_FAILED
} integratortrystatus;

integratortrystatus integrator_try(integrator *integrate, integrandfunction *integrand, int dim, double **x, unsigned int nquantity, quantity *quantity, void *ref, unsigned int nout, double *out);
bool integrator_apply(integrator *integrate, integrandfunction *integrand, void *ref, unsigned int nout, double *out);
bool integrator_refine(integrator *integrate);

/* Easy interface for one-off integrals */
bool integrate(integrandfunction *integrand, objectdictionary *method, error *err, unsigned int dim, unsigned int grade, double **x, unsigned int nquantity, quantity *quantity, void *ref, double *out, double *errest);

/* Old interface */
bool integrate_integrate(integrandfunction *integrand, unsigned int dim, unsigned int grade, double **x, unsigned int nquantity, value **quantity, void *ref, double *out);

void integrate_initialize(void);

#endif

#endif /* integration_h */

