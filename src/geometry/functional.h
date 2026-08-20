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

#define JUMP_STRATEGY_LABEL                   "strategy"
#define JUMP_STRATEGY_QUADRATURE              "quadrature"
#define JUMP_STRATEGY_CENTROID                "centroid"

/* Functional methods */
#define FUNCTIONAL_INTEGRAND_METHOD    "integrand"
#define FUNCTIONAL_TOTAL_METHOD        "total"
#define FUNCTIONAL_GRADIENT_METHOD     "gradient"
#define FUNCTIONAL_FIELDGRADIENT_METHOD     "fieldgradient"
#define FUNCTIONAL_HESSIAN_METHOD      "hessian"
#define FUNCTIONAL_INTEGRANDFORELEMENT_METHOD      "integrandForElement"

/* Special functions that can be used in integrands */
#define ELEMENTID_FUNCTION             "elementid"
#define TANGENT_FUNCTION               "tangent"
#define NORMAL_FUNCTION                "normal"
#define GRAD_FUNCTION                  "grad"
#define HESS_FUNCTION                  "hess"
#define CGTENSOR_FUNCTION              "cgtensor"
#define JUMPDN_FUNCTION                "jumpdn"
#define JACOBIAN_FUNCTION              "jacobian"
#define INVJACOBIAN_FUNCTION           "invjacobian"

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
#define JUMP_CLASSNAME                 "Jump"
#define NEMATIC_CLASSNAME              "Nematic"
#define NEMATICELECTRIC_CLASSNAME      "NematicElectric"

/* Errors */
#define FUNC_ELNTFND                   "FnctlELNtFnd"
#define FUNC_ELNTFND_MSG               "Mesh does not provide elements of grade %u."

#define FUNC_FESPACE                   "FnctlFESpc"
#define FUNC_FESPACE_MSG               "This Field%s%s cannot be evaluated on grade %u elements."

#define SCALARPOTENTIAL_FNCLLBL        "SclrPtFnCllbl"
#define SCALARPOTENTIAL_FNCLLBL_MSG    "ScalarPotential function is not callable."

#define INTEGRAL_ARGS                  "IntgrlArgs"
#define INTEGRAL_ARGS_MSG              "Integral functionals require a callable argument, followed by zero or more Fields."

#define INTEGRAL_FLD                   "IntgrlFld"
#define INTEGRAL_FLD_MSG               "Can't identify field."

#define INTEGRAL_DFFEVL                "IntgrlDffEvl"
#define INTEGRAL_DFFEVL_MSG            "Derivative evaluation failed or is unsupported by the finite element space."

#define INTEGRAL_SPCLFN                "IntgrlSpclFn"
#define INTEGRAL_SPCLFN_MSG            "Special function '%s' can't be called outside of an Integral."

#define JUMP_UNIMPL                    "JumpUnimpl"
#define JUMP_UNIMPL_MSG                "Jump is not implemented yet."

#define VOLUMEENCLOSED_ZERO            "VolEnclZero"
#define VOLUMEENCLOSED_ZERO_MSG        "VolumeEnclosed detected an element of zero size. Check that a mesh point is not coincident with the origin."

#define HYDROGEL_FLDGRD                "HydrglFldGrd"
#define HYDROGEL_FLDGRD_MSG            "Hydrogel has been given phi0 as a Field that lacks scalar elements in grade %u."

#define HYDROGEL_ZEEROREFELEMENT       "HydrglZrRfVl"
#define HYDROGEL_ZEEROREFELEMENT_MSG   "Reference element %u has tiny volume V=%g, V0=%g\n"

#define HYDROGEL_BNDS                  "HydrglBnds"
#define HYDROGEL_BNDS_MSG              "Phi outside bounds at element %u V=%g, V0=%g, phi=%g, 1-phi=%g\n"

#define FUNCTIONAL_ARGS                "FnctlArgs"
#define FUNCTIONAL_ARGS_MSG            "Invalid arguments passed to this functional."

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

struct s_functional_mapinfo; // Resolve circular typedef dependency

/** Optional start function called once before a functional evaluation begins */
typedef bool (functional_start) (vm *v, struct s_functional_mapinfo *info);

/** Optional end function called once after a functional evaluation completes (success or failure) */
typedef bool (functional_end) (vm *v, struct s_functional_mapinfo *info);

/** Map callback used by functional_runmap */
typedef bool (functional_mapcallback) (vm *v, struct s_functional_mapinfo *info, value *out);

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
    functional_start *start; // Optional preflight hook (once per user call)
    functional_end *end; // Optional postflight hook (once per user call)
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
bool functional_mapgradient(vm *v, functional_mapinfo *info, value *out);
bool functional_mapnumericalgradient(vm *v, functional_mapinfo *info, value *out);
bool functional_mapnumericalfieldgradient(vm *v, functional_mapinfo *info, value *out);
bool functional_startmap(vm *v, functional_mapinfo *info);
bool functional_endmap(vm *v, functional_mapinfo *info);
bool functional_runmap(vm *v, functional_mapinfo *info, functional_mapcallback *mapfn, value *out);

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

bool functional_readgrade(objectinstance *self, grade *g);

/* -------------------------------------------------------
 * Functional method macros
 * ------------------------------------------------------- */

/** Initialize a functional */
#define FUNCTIONAL_INIT(name, grade) value name##_init(vm *v, int nargs, value *args) { \
    objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), functional_gradeproperty, MORPHO_INTEGER(grade)); \
    return MORPHO_NIL; \
}

/** Evaluate an integrand */
#define FUNCTIONAL_INTEGRAND(name, grade, integrandfn) \
    FUNCTIONAL_INTEGRAND_START(name, grade, NULL, NULL, integrandfn)

#define FUNCTIONAL_INTEGRAND_START(name, grade, startfn, endfn, integrandfn) value name##_integrand(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        info.g = grade; info.start = startfn; info.end = endfn; info.integrand = integrandfn; \
        functional_runmap(v, &info, functional_mapintegrand, &out); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    return out; \
}

/** Evaluate an integrand at an element */
#define FUNCTIONAL_INTEGRANDFORELEMENT(name, grade, integrandfn) \
    FUNCTIONAL_INTEGRANDFORELEMENT_START(name, grade, NULL, NULL, integrandfn)

#define FUNCTIONAL_INTEGRANDFORELEMENT_START(name, grade, startfn, endfn, integrandfn) value name##_integrandForElement(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        info.g = grade; info.start = startfn; info.end = endfn; info.integrand = integrandfn; \
        functional_runmap(v, &info, functional_mapintegrandforelement, &out); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    return out; \
}

/** Evaluate a gradient */
#define FUNCTIONAL_GRADIENT(name, grade, gradientfn, symbhvr) \
    FUNCTIONAL_GRADIENT_START(name, grade, NULL, NULL, gradientfn, symbhvr)

#define FUNCTIONAL_GRADIENT_START(name, grade, startfn, endfn, gradientfn, symbhvr) \
value name##_gradient(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        info.g = grade; info.start = startfn; info.end = endfn; info.grad = gradientfn; info.sym = symbhvr; \
        functional_runmap(v, &info, functional_mapgradient, &out); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    \
    return out; \
}

/** Evaluate a gradient */
#define FUNCTIONAL_NUMERICALGRADIENT(name, grade, integrandfn, symbhvr) \
    FUNCTIONAL_NUMERICALGRADIENT_START(name, grade, NULL, NULL, integrandfn, symbhvr)

#define FUNCTIONAL_NUMERICALGRADIENT_START(name, grade, startfn, endfn, integrandfn, symbhvr) \
value name##_gradient(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        info.g = grade; info.start = startfn; info.end = endfn; info.integrand = integrandfn; info.sym = symbhvr; \
        functional_runmap(v, &info, functional_mapnumericalgradient, &out); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    \
    return out; \
}

/** Total an integrand */
#define FUNCTIONAL_TOTAL(name, grade, totalfn) \
    FUNCTIONAL_TOTAL_START(name, grade, NULL, NULL, totalfn)

#define FUNCTIONAL_TOTAL_START(name, grade, startfn, endfn, totalfn) \
value name##_total(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        info.g = grade; info.start = startfn; info.end = endfn; info.integrand = totalfn; \
        functional_runmap(v, &info, functional_sumintegrand, &out); \
    } \
    \
    return out; \
}

/** Hessian */
#define FUNCTIONAL_HESSIAN(name, grade, totalfn) \
    FUNCTIONAL_HESSIAN_START(name, grade, NULL, NULL, totalfn)

#define FUNCTIONAL_HESSIAN_START(name, grade, startfn, endfn, totalfn) \
value name##_hessian(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        info.g = grade; info.start = startfn; info.end = endfn; info.integrand = totalfn; \
        functional_runmap(v, &info, functional_mapnumericalhessian, &out); \
    } \
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out); \
    \
    return out; \
}

/* Alternative way of defining methods that use a reference */
#define FUNCTIONAL_METHOD(class, name, grade, reftype, prepare, integrandfn, integrandmapfn, deps, err, symbhvr) \
    FUNCTIONAL_METHOD_START(class, name, grade, reftype, prepare, NULL, NULL, integrandfn, integrandmapfn, deps, err, symbhvr)

#define FUNCTIONAL_METHOD_START(class, name, grade, reftype, prepare, startfn, endfn, integrandfn, integrandmapfn, deps, err, symbhvr) value class##_##name(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    reftype ref; \
    value out=MORPHO_NIL; \
    \
    if (functional_validateargs(v, nargs, args, &info)) { \
        if (prepare(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, grade, info.sel, &ref)) { \
            info.integrand = integrandmapfn; \
            info.start = startfn; \
            info.end = endfn; \
            info.dependencies = deps, \
            info.sym = symbhvr; \
            info.g = grade; \
            info.ref = &ref; \
            functional_runmap(v, &info, integrandfn, &out); \
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
