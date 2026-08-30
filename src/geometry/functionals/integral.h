/** @file integral.h
 *  @author T J Atherton
 *
 *  @brief LineIntegral, AreaIntegral and VolumeIntegral
 */

#ifndef morpho_functionals_integral_h
#define morpho_functionals_integral_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "functional.h"
#include "integrate.h"

/* -------------------------------------------------------
 * Integral veneer classes
 * ------------------------------------------------------- */

#define LINEINTEGRAL_CLASSNAME         "LineIntegral"
#define AREAINTEGRAL_CLASSNAME         "AreaIntegral"
#define VOLUMEINTEGRAL_CLASSNAME       "VolumeIntegral"

#define INTEGRAL_FUNCTION_PROPERTY     "function"
#define INTEGRAL_REFERENCE_PROPERTY    "reference"
#define INTEGRAL_WTBYREF_PROPERTY      "weightByReference"
#define INTEGRAL_METHOD_PROPERTY       "method"
#define INTEGRAL_OPTIMIZE_PROPERTY     "optimize"

/* -------------------------------------------------------
 * Special functions that can be used in integrands
 * ------------------------------------------------------- */

#define ELEMENTID_FUNCTION             "elementid"
#define TANGENT_FUNCTION               "tangent"
#define NORMAL_FUNCTION                "normal"
#define GRAD_FUNCTION                  "grad"
#define HESS_FUNCTION                  "hess"
#define CGTENSOR_FUNCTION              "cgtensor"
#define JACOBIAN_FUNCTION              "jacobian"
#define INVJACOBIAN_FUNCTION           "invjacobian"

/* -------------------------------------------------------
 * Integral error messages
 * ------------------------------------------------------- */

#define INTEGRAL_ARGS                  "IntgrlArgs"
#define INTEGRAL_ARGS_MSG              "Integral functionals require a callable argument, followed by zero or more Fields."

#define INTEGRAL_FLD                   "IntgrlFld"
#define INTEGRAL_FLD_MSG               "Can't identify field."

#define INTEGRAL_DFFEVL                "IntgrlDffEvl"
#define INTEGRAL_DFFEVL_MSG            "Derivative evaluation failed or is unsupported by the finite element space."

#define INTEGRAL_SPCLFN                "IntgrlSpclFn"
#define INTEGRAL_SPCLFN_MSG            "Special function '%s' can't be called outside of an Integral."

#define INTEGRAL_NESTED                "IntgrlNested"
#define INTEGRAL_NESTED_MSG            "Nested Integrals are not supported."

#define INTEGRAL_FASTPATH              "IntgrlFstPath"
#define INTEGRAL_FASTPATH_MSG          "Special function '%s' is not supported by this local derivative path."

typedef struct {
    value integrand;
    int nfields;
    value *fields;
    value *originalfields;
    value method;
    objectmesh *mref;
    grade g;
    bool weightbyref;
    bool optimize;
    unsigned uses;
} integralref;

#define INTEGRAL_ALLOW_ALL UINT_MAX

enum {
    INTEGRAL_USES_NONE     = 0,
    INTEGRAL_USES_X        = 1u << 0,
    INTEGRAL_USES_GRAD     = 1u << 1,
    INTEGRAL_USES_HESS     = 1u << 2,
    INTEGRAL_USES_TANGENT  = 1u << 3,
    INTEGRAL_USES_NORMAL   = 1u << 4,
    INTEGRAL_USES_JACOBIAN = 1u << 5,
    INTEGRAL_USES_INVJ     = 1u << 6,
    INTEGRAL_USES_CG       = 1u << 7,
    INTEGRAL_USES_JUMPDN   = 1u << 8
};

#define INTEGRAL_FIELDGRAD_ALLOWED (INTEGRAL_USES_GRAD | INTEGRAL_USES_TANGENT | \
    INTEGRAL_USES_NORMAL | INTEGRAL_USES_JACOBIAN | INTEGRAL_USES_INVJ | INTEGRAL_USES_CG)
#define INTEGRAL_SHAPEGRAD_ALLOWED (INTEGRAL_USES_X | INTEGRAL_USES_TANGENT | INTEGRAL_USES_NORMAL)

#define ELREF_PERSISTENT   (1u<<0)
#define ELREF_CONFIGURED   (1u<<1)
#define ELREF_HASTANGENT   (1u<<2)
#define ELREF_HASNORMAL    (1u<<3)
#define ELREF_HASJACOBIAN  (1u<<4)
#define ELREF_HASCG        (1u<<5)
#define ELREF_HASINVJ      (1u<<6)
#define ELREF_HASINTEG     (1u<<7)
#define ELREF_GEOM         (ELREF_HASTANGENT|ELREF_HASNORMAL|ELREF_HASJACOBIAN|ELREF_HASCG|ELREF_HASINVJ)

typedef struct {
    object obj;
    objectmesh *mesh;
    integralref *iref;
    vm *v;
    grade g;
    elementid id;
    int nv;
    unsigned flags;
    unsigned allowed;
    int target_field;
    bool target_grad_used;
    int freeze_grad;
    int *vid;
    double **vertexposn;
    double elementsize;
    double *lambda;
    double *posn;
    quantity *quantities;
    objectmatrix *invj;
    value *qgrad;
    value *qhess;
    value *qinterpolated;
    integrator integ;
    int nfields;
    objectmatrix *tangent;
    objectmatrix *normal;
    objectmatrix *jacobian;
    objectmatrix *invjacobian;
    objectmatrix *cgtensor;
    objectmatrix *xgeom;
} objectintegralelementref;

extern objecttype objectintegralelementreftype;
#define OBJECT_INTEGRALELEMENTREF objectintegralelementreftype
#define MORPHO_ISINTEGRALELEMENTREF(val) object_istype(val, OBJECT_INTEGRALELEMENTREF)
#define MORPHO_GETINTEGRALELEMENTREF(val) ((objectintegralelementref *) MORPHO_GETOBJECT(val))

void _integral_initelref(objectintegralelementref *elref);
void _integral_bindelref(objectintegralelementref *elref, objectmesh *mesh, grade g, elementid id, int nv, int *vid, double **vertexposn, integralref *iref);
void *_integral_zalloc(int n, size_t size);
void integral_clearelref(objectintegralelementref *elref);
objectintegralelementref *integral_getelementref(vm *v);
bool integral_contextactive(vm *v);
bool integral_checkfastpath(vm *v, unsigned bit, const char *name);
value integral_init(vm *v, int nargs, value *args);
bool integral_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, integralref *ref);
void integral_addspecial(char *name, builtinfunction fn, unsigned bit);
objectmatrix *integral_ensurematrix(objectmatrix **slot, int nrows, int ncols);
bool integral_prepareinvjacobian(unsigned int dim, grade g, double **x, objectmatrix *invj);
bool integral_preparequantities(integralref *iref, int nv, int *vid, quantity *quantities);

#define INTEGRAL_MAPFLAGS  (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_MAPFLAGS)
#define INTEGRAL_TOTALFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_TOTALFLAGS)
#define INTEGRAL_ELEMFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_ELEMFLAGS)
#define INTEGRAL_INITFLAGS (MORPHO_FN_MUTATES|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_OPTARGS)

void integral_initialize(void);

#endif

#endif /* morpho_functionals_integral_h */
