/** @file functional.h
 *  @author T J Atherton
 *
 *  @brief Functionals
 */

#ifndef functional_h
#define functional_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include <stdio.h>
#include "morpho.h"
#include "mesh.h"
#include "field.h"
#include "selection.h"

/* -------------------------------------------------------
 * Functionals
 * ------------------------------------------------------- */

/* Functional properties */
#define FUNCTIONAL_GRADE_PROPERTY             "grade"
#define FUNCTIONAL_FIELD_PROPERTY             "field"
#define SCALARPOTENTIAL_FUNCTION_PROPERTY     "function"
#define SCALARPOTENTIAL_GRADFUNCTION_PROPERTY "gradfunction"
#define LINEARELASTICITY_REFERENCE_PROPERTY   "reference"
#define LINEARELASTICITY_WTBYREF_PROPERTY     "weightByReference"
#define LINEARELASTICITY_POISSON_PROPERTY     "poissonratio"
#define HYDROGEL_A_PROPERTY                   "a"
#define HYDROGEL_B_PROPERTY                   "b"
#define HYDROGEL_C_PROPERTY                   "c"
#define HYDROGEL_D_PROPERTY                   "d"
#define HYDROGEL_PHIREF_PROPERTY              "phiref"
#define HYDROGEL_PHI0_PROPERTY                "phi0"
#define EQUIELEMENT_WEIGHT_PROPERTY           "weight"

#define NEMATIC_KSPLAY_PROPERTY               "ksplay"
#define NEMATIC_KTWIST_PROPERTY               "ktwist"
#define NEMATIC_KBEND_PROPERTY                "kbend"
#define NEMATIC_PITCH_PROPERTY                "pitch"
#define NEMATIC_DIRECTOR_PROPERTY             "director"

#define CURVATURE_INTEGRANDONLY_PROPERTY      "integrandonly"
#define CURVATURE_GEODESIC_PROPERTY           "geodesic"

#define INTEGRAL_METHOD_PROPERTY              "method"

/* Functional methods */
#define FUNCTIONAL_INTEGRAND_METHOD    "integrand"
#define FUNCTIONAL_TOTAL_METHOD        "total"
#define FUNCTIONAL_GRADIENT_METHOD     "gradient"
#define FUNCTIONAL_FIELDGRADIENT_METHOD     "fieldgradient"
#define FUNCTIONAL_HESSIAN_METHOD      "hessian"
#define FUNCTIONAL_INTEGRANDFORELEMENT_METHOD      "integrandForElement"

/* Special functions that can be used in integrands */
#define TANGENT_FUNCTION               "tangent"
#define NORMAL_FUNCTION                "normal"
#define GRAD_FUNCTION                  "grad"
#define CGTENSOR_FUNCTION              "cgtensor"

/* Functional names */
#define LENGTH_CLASSNAME               "Length"
#define AREA_CLASSNAME                 "Area"
#define AREAENCLOSED_CLASSNAME         "AreaEnclosed"
#define VOLUME_CLASSNAME               "Volume"
#define VOLUMEENCLOSED_CLASSNAME       "VolumeEnclosed"
#define SCALARPOTENTIAL_CLASSNAME      "ScalarPotential"
#define LINEARELASTICITY_CLASSNAME     "LinearElasticity"
#define HYDROGEL_CLASSNAME             "Hydrogel"
#define EQUIELEMENT_CLASSNAME          "EquiElement"
#define LINECURVATURESQ_CLASSNAME      "LineCurvatureSq"
#define LINETORSIONSQ_CLASSNAME        "LineTorsionSq"
#define MEANCURVATURESQ_CLASSNAME      "MeanCurvatureSq"
#define GAUSSCURVATURE_CLASSNAME       "GaussCurvature"
#define GRADSQ_CLASSNAME               "GradSq"
#define NORMSQ_CLASSNAME               "NormSq"
#define LINEINTEGRAL_CLASSNAME         "LineIntegral"
#define AREAINTEGRAL_CLASSNAME         "AreaIntegral"
#define VOLUMEINTEGRAL_CLASSNAME       "VolumeIntegral"
#define NEMATIC_CLASSNAME              "Nematic"
#define NEMATICELECTRIC_CLASSNAME      "NematicElectric"

/* Errors */
#define FUNC_INTEGRAND_MESH            "FnctlIntMsh"
#define FUNC_INTEGRAND_MESH_MSG        "Method 'integrand' requires a mesh as the argument."

#define FUNC_INTEGRAND_MESH            "FnctlIntMsh"
#define FUNC_INTEGRAND_MESH_MSG        "Method 'integrand' requires a mesh as the argument."

#define FUNC_ELNTFND                   "FnctlELNtFnd"
#define FUNC_ELNTFND_MSG               "Mesh does not provide elements of grade %u."

#define SCALARPOTENTIAL_FNCLLBL        "SclrPtFnCllbl"
#define SCALARPOTENTIAL_FNCLLBL_MSG    "ScalarPotential function is not callable."

#define LINEINTEGRAL_ARGS              "IntgrlArgs"
#define LINEINTEGRAL_ARGS_MSG          "Integral functionals require a callable argument, followed by zero or more Fields."

#define INTEGRAL_FLD                   "IntgrlFld"
#define INTEGRAL_FLD_MSG               "Can't identify field."

#define INTEGRAL_AMBGSFLD              "IntgrlAmbgsFld"
#define INTEGRAL_AMBGSFLD_MSG          "Field reference is ambigious: call with a Field object."

#define INTEGRAL_SPCLFN                "IntgrlSpclFn"
#define INTEGRAL_SPCLFN_MSG            "Special function '%s' must not be called outside of an Integral."

#define LINEINTEGRAL_NFLDS             "IntgrlNFlds"
#define LINEINTEGRAL_NFLDS_MSG         "Incorrect number of Fields provided for integrand function."

#define VOLUMEENCLOSED_ZERO            "VolEnclZero"
#define VOLUMEENCLOSED_ZERO_MSG        "VolumeEnclosed detected an element of zero size. Check that a mesh point is not coincident with the origin."

#define LINEARELASTICITY_REF           "LnElstctyRef"
#define LINEARELASTICITY_REF_MSG       "LinearElasticity requires a mesh as the argument."

#define LINEARELASTICITY_PRP           "LnElstctyPrp"
#define LINEARELASTICITY_PRP_MSG       "LinearElasticity requires properties 'reference' to be a mesh, 'grade' to be an integer grade and 'poissonratio' to be a number."

#define EQUIELEMENT_ARGS               "EquiElArgs"
#define EQUIELEMENT_ARGS_MSG           "EquiElement allows 'grade' and 'weight' as optional arguments."

#define HYDROGEL_ARGS                  "HydrglArgs"
#define HYDROGEL_ARGS_MSG              "Hydrogel requires a reference mesh and allows 'grade', 'a', 'b', 'c', 'd', 'phi0' and 'phiref' as optional arguments."

#define HYDROGEL_PRP                   "HydrglPrp"
#define HYDROGEL_PRP_MSG               "Hydrogel requires the first argument to be a mesh, 'grade' to be an integer grade, 'a', 'b', 'c' 'd', 'phiref' to be numbers and 'phi0' to be a number or a Field."

#define HYDROGEL_FLDGRD                "HydrglFldGrd"
#define HYDROGEL_FLDGRD_MSG            "Hydrogel has been given phi0 as a Field that lacks scalar elements in grade %u."

#define HYDROGEL_ZEEROREFELEMENT       "HydrglZrRfVl"
#define HYDROGEL_ZEEROREFELEMENT_MSG   "Reference element %u has tiny volume V=%g, V0=%g\n"

#define HYDROGEL_BNDS                  "HydrglBnds"
#define HYDROGEL_BNDS_MSG              "Phi outside bounds at element %u V=%g, V0=%g, phi=%g, 1-phi=%g\n"

#define GRADSQ_ARGS                    "GradSqArgs"
#define GRADSQ_ARGS_MSG                "GradSq requires a field as the argument."

#define NEMATIC_ARGS                   "NmtcArgs"
#define NEMATIC_ARGS_MSG               "Nematic requires a field as the argument."

#define NEMATICELECTRIC_ARGS           "NmtcElArgs"
#define NEMATICELECTRIC_ARGS_MSG       "NematicElectric requires the director and electric field or potential as arguments (in that order)."

#define FUNCTIONAL_ARGS                "FnctlArgs"
#define FUNCTIONAL_ARGS_MSG            "Invalid args passed to method."

/* -------------------------------------------------------
 * Functional types
 * ------------------------------------------------------- */

extern value functional_gradeproperty;
extern value functional_fieldproperty;

/** Symmetry behaviors */
typedef enum {
    SYMMETRY_NONE,
    SYMMETRY_ADD
} symmetrybhvr;

/** Integrand function */
typedef bool (functional_integrand) (vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out);

/** Gradient function */
typedef bool (functional_gradient) (vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc);

/** Field gradient function */
typedef bool (functional_fieldgradient) (vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectfield *frc);

struct s_functional_mapinfo; // Resolve circular typedef dependency

/** Clone reference function */
typedef void * (functional_cloneref) (void *ref, objectfield *field, objectfield *sub);

/** Free reference function */
typedef void (functional_freeref) (void *ref);

/** Dependencies function */
typedef bool (functional_dependencies) (struct s_functional_mapinfo *info, elementid id, varray_elementid *out);

typedef struct s_functional_mapinfo {
    objectmesh *mesh; // Mesh to use
    objectselection *sel; // Selection, if any
    objectfield *field; // Field, if any
    grade g; // Grade to use
    elementid id; // Element id at which to evaluate the integrand
    functional_integrand *integrand; // Integrand function
    functional_gradient *grad; // Gradient
    functional_fieldgradient *fieldgrad; // Field gradient
    functional_dependencies *dependencies; // Dependencies
    functional_cloneref *cloneref; // Clone a reference with a given field substituted
    functional_freeref *freeref; // Free a reference
    symmetrybhvr sym; // Symmetry behavior
    void *ref; // Reference to pass on
} functional_mapinfo;

bool functional_validateargs(vm *v, int nargs, value *args, functional_mapinfo *info);
void functional_symmetryimagelist(objectmesh *mesh, grade g, bool sort, varray_elementid *ids);
bool functional_symmetrysumforces(objectmesh *mesh, objectmatrix *frc);
bool functional_inlist(varray_elementid *list, elementid id);
bool functional_containsvertex(int nv, int *vid, elementid id);

bool functional_sumintegrand(vm *v, functional_mapinfo *info, value *out);
bool functional_mapintegrand(vm *v, functional_mapinfo *info, value *out);
bool functional_mapintegrandat(vm *v, functional_mapinfo *info, value *out);
bool functional_mapgradient(vm *v, functional_mapinfo *info, value *out);
bool functional_mapfieldgradient(vm *v, functional_mapinfo *info, value *out);
bool functional_mapnumericalgradient(vm *v, functional_mapinfo *info, value *out);
bool functional_mapnumericalfieldgradient(vm *v, functional_mapinfo *info, value *out);

void functional_vecadd(unsigned int n, double *a, double *b, double *out);
void functional_vecaddscale(unsigned int n, double *a, double lambda, double *b, double *out);
void functional_vecsub(unsigned int n, double *a, double *b, double *out);
void functional_vecscale(unsigned int n, double lambda, double *a, double *out);
double functional_vecnorm(unsigned int n, double *a);
double functional_vecdot(unsigned int n, double *a, double *b);
void functional_veccross(double *a, double *b, double *out);
void functional_veccross2d(double *a, double *b, double *out);

bool functional_elementsize(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, double *out);
bool functional_elementgradient_scale(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, objectmatrix *frc, double scale);
bool functional_elementgradient(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, objectmatrix *frc);

/* -------------------------------------------------------
 * Functional method macros
 * ------------------------------------------------------- */

/** Initialize a functional */
#define FUNCTIONAL_INIT(name, grade) value name##_init(vm *v, int nargs, value *args) { \
    objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), functional_gradeproperty, MORPHO_INTEGER(grade)); \
    return MORPHO_NIL; \
}

/** Evaluate an integrand */
#define FUNCTIONAL_INTEGRAND(name, grade, integrandfn) value name##_integrand(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        info.g = grade; info.integrand = integrandfn; \
        functional_mapintegrand(v, &info, &out); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    return out; \
}

/** Evaluate an integrand at an element */
#define FUNCTIONAL_INTEGRANDFORELEMENT(name, grade, integrandfn) value name##_integrandForElement(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        info.g = grade; info.integrand = integrandfn; \
        functional_mapintegrandforelement(v, &info, &out); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    return out; \
}

/** Evaluate a gradient */
#define FUNCTIONAL_GRADIENT(name, grade, gradientfn, symbhvr) \
value name##_gradient(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        info.g = grade; info.grad = gradientfn; info.sym = symbhvr; \
        functional_mapgradient(v, &info, &out); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    \
    return out; \
}

/** Evaluate a gradient */
#define FUNCTIONAL_NUMERICALGRADIENT(name, grade, integrandfn, symbhvr) \
value name##_gradient(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        info.g = grade; info.integrand = integrandfn; info.sym = symbhvr; \
        functional_mapnumericalgradient(v, &info, &out); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    \
    return out; \
}

/** Total an integrand */
#define FUNCTIONAL_TOTAL(name, grade, totalfn) \
value name##_total(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        info.g = grade; info.integrand = totalfn; \
        functional_sumintegrand(v, &info, &out); \
    } \
    \
    return out; \
}

/** Hessian */
#define FUNCTIONAL_HESSIAN(name, grade, totalfn) \
value name##_hessian(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        info.g = grade; info.integrand = totalfn; \
        functional_mapnumericalhessian(v, &info, &out); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    \
    return out; \
}

/* Alternative way of defining methods that use a reference */
#define FUNCTIONAL_METHOD(class, name, grade, reftype, prepare, integrandfn, integrandmapfn, deps, err, symbhvr) value class##_##name(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    reftype ref; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        if (prepare(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, grade, info.sel, &ref)) { \
            info.integrand = integrandmapfn; \
            info.dependencies = deps, \
            info.sym = symbhvr; \
            info.g = grade; \
            info.ref = &ref; \
            integrandfn(v, &info, &out); \
        } else morpho_runtimeerror(v, err); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    return out; \
}

/* -------------------------------------------------------
 * Initialization
 * ------------------------------------------------------- */

void functional_initialize(void);
void functional_finalize(void);

#endif

#endif /* functional_h */
