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

#define FUNC_NOFESPACE                 "FnctlNoFESpc"
#define FUNC_NOFESPACE_MSG             "This Field has no finite element space; pass finiteelementspace=... or omit the opt-out."

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

#define INTEGRAL_NESTED                "IntgrlNested"
#define INTEGRAL_NESTED_MSG            "Nested Integrals are not supported."

#define JUMP_UNIMPL                    "JumpUnimpl"
#define JUMP_UNIMPL_MSG                "This Jump evaluation is not implemented yet."

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

/** Optional per-task start, called once on the task VM before the element loop */
typedef bool (functional_taskstart) (vm *v, struct s_functional_mapinfo *info);

/** Optional per-task end, called once on the task VM after the element loop (success or failure) */
typedef void (functional_taskend) (vm *v, struct s_functional_mapinfo *info);

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
    functional_taskstart *taskstart; // Optional per-task setup (once per map task)
    functional_taskend *taskend; // Optional per-task teardown (once per map task)
    functional_dependencies *dependencies; // Dependencies
    functional_cloneref *cloneref; // Clone a reference with a given field substituted
    functional_freeref *freeref; // Free a reference
    symmetrybhvr sym; // Symmetry behavior
    void *ref; // Reference to pass on
} functional_mapinfo;

void functional_symmetryimagelist(objectmesh *mesh, grade g, bool sort, varray_elementid *ids);
bool functional_symmetrysumforces(objectmesh *mesh, objectmatrix *frc);
bool functional_inlist(varray_elementid *list, elementid id);
bool functional_containsvertex(int nv, int *vid, elementid id);

bool functional_sumintegrand(vm *v, functional_mapinfo *info, value *out);
bool functional_mapintegrand(vm *v, functional_mapinfo *info, value *out);
bool functional_mapintegrandforelement(vm *v, functional_mapinfo *info, value *out);
bool functional_mapgradient(vm *v, functional_mapinfo *info, value *out);
bool functional_mapnumericalgradient(vm *v, functional_mapinfo *info, value *out);
bool functional_mapnumericalfieldgradient(vm *v, functional_mapinfo *info, value *out);
bool functional_mapnumericalhessian(vm *v, functional_mapinfo *info, value *out);
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
void functional_setgrade(objectinstance *self, grade g);

/* Helpers used by FUNCTIONAL_MD_* macros. TODO: Make static once shims removed */
void _functional_mapinfo(functional_mapinfo *info, objectmesh *mesh, objectselection *sel, objectfield *field);
value _functional_run(vm *v, functional_mapinfo *info, grade g, functional_mapcallback *mapfn, bool bind);
value _functional_integrand(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn);
value _functional_integrand_elem(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn);
value _functional_total(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn);
value _functional_gradient(vm *v, functional_mapinfo *info, grade g, functional_gradient *fn, symmetrybhvr sym);
value _functional_numericalgradient(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn, symmetrybhvr sym);
value _functional_hessian(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn);
bool functional_validateargs(vm *v, int nargs, value *args, functional_mapinfo *info);

/* -------------------------------------------------------
 * Functional method macros
 * ------------------------------------------------------- */

/** Initialize a functional */
#define FUNCTIONAL_INIT(name, grade) value name##_init(vm *v, int nargs, value *args) { \
    functional_setgrade(MORPHO_GETINSTANCE(MORPHO_SELF(args)), grade); \
    return MORPHO_NIL; \
}

/* -------------------------------------------------------
 * Multiple-dispatch method macros
 *
 * Wrappers only unpack Morpho arguments. Shared C helpers in
 * functional.c (_functional_integrand, _functional_total, ...)
 * apply the default grade and run the map.
 * ------------------------------------------------------- */

/* Builtin-function flags used in class-table signatures: map/hessian allocate
 * and may run multithreaded; total does not allocate; per-element is throws only. */
#define FUNCTIONAL_MD_MAPFLAGS  (MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
#define FUNCTIONAL_MD_TOTALFLAGS (MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
#define FUNCTIONAL_MD_ELEMFLAGS (MORPHO_FN_THROWS)

/* Code fragments to pull method args into mapinfo based on MD types. */
#define FUNCTIONAL_MD_INFO__MESH() \
    functional_mapinfo info; \
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), NULL, NULL)

#define FUNCTIONAL_MD_INFO__MESH_SEL() \
    functional_mapinfo info; \
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), MORPHO_GETSELECTION(MORPHO_GETARG(args, 1)), NULL)

#define FUNCTIONAL_MD_INFO__MESH_INT() \
    FUNCTIONAL_MD_INFO__MESH(); \
    info.id = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1))

#define FUNCTIONAL_MD_INFO__MESH_INT_INT() \
    FUNCTIONAL_MD_INFO__MESH(); \
    info.g = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1)); \
    info.id = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 2))

/* fieldgradient is Field-first: (Field), (Field, Mesh), (Field, Selection),
 * (Field, Mesh, Selection). Mesh is taken from the Field when omitted. */
#define FUNCTIONAL_MD_INFO__FIELD() \
    functional_mapinfo info; \
    _functional_mapinfo(&info, NULL, NULL, MORPHO_GETFIELD(MORPHO_GETARG(args, 0)))

#define FUNCTIONAL_MD_INFO__FIELD_MESH() \
    functional_mapinfo info; \
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 1)), NULL, MORPHO_GETFIELD(MORPHO_GETARG(args, 0)))

#define FUNCTIONAL_MD_INFO__FIELD_SEL() \
    functional_mapinfo info; \
    _functional_mapinfo(&info, NULL, MORPHO_GETSELECTION(MORPHO_GETARG(args, 1)), MORPHO_GETFIELD(MORPHO_GETARG(args, 0)))

#define FUNCTIONAL_MD_INFO__FIELD_MESH_SEL() \
    functional_mapinfo info; \
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 1)), MORPHO_GETSELECTION(MORPHO_GETARG(args, 2)), MORPHO_GETFIELD(MORPHO_GETARG(args, 0)))

/* Constructs a function cls_mthod__suffix with a defined setup macro followed by remaining args */
#define FUNCTIONAL_MD_WRAP(cls, method, suffix, setup, ...) \
value cls##_##method##__##suffix(vm *v, int nargs, value *args) { \
    setup; \
    return __VA_ARGS__; \
}

/* Builds cls_method__mesh [etc.] : run an INFO__* fragment, then return fn(v, &info, extra...).
 * extra... is typically the default grade and the integrand or gradient C function. */
#define FUNCTIONAL_MD__MESH(cls, method, fn, ...) \
    FUNCTIONAL_MD_WRAP(cls, method, mesh, FUNCTIONAL_MD_INFO__MESH(), fn(v, &info, __VA_ARGS__))

#define FUNCTIONAL_MD__MESH_SEL(cls, method, fn, ...) \
    FUNCTIONAL_MD_WRAP(cls, method, mesh_sel, FUNCTIONAL_MD_INFO__MESH_SEL(), fn(v, &info, __VA_ARGS__))

#define FUNCTIONAL_MD__MESH_INT(cls, method, fn, ...) \
    FUNCTIONAL_MD_WRAP(cls, method, mesh_int, FUNCTIONAL_MD_INFO__MESH_INT(), fn(v, &info, __VA_ARGS__))

#define FUNCTIONAL_MD__MESH_INT_INT(cls, method, fn, ...) \
    FUNCTIONAL_MD_WRAP(cls, method, mesh_int_int, FUNCTIONAL_MD_INFO__MESH_INT_INT(), fn(v, &info, __VA_ARGS__))

#define FUNCTIONAL_MD__FIELD(cls, method, fn, ...) \
    FUNCTIONAL_MD_WRAP(cls, method, field, FUNCTIONAL_MD_INFO__FIELD(), fn(v, &info, __VA_ARGS__))

#define FUNCTIONAL_MD__FIELD_SEL(cls, method, fn, ...) \
    FUNCTIONAL_MD_WRAP(cls, method, field_sel, FUNCTIONAL_MD_INFO__FIELD_SEL(), fn(v, &info, __VA_ARGS__))

/* Emits Mesh and Field-as-mesh overloads for one C helper. A Field argument
 * supplies its mesh (_functional_mapinfo). */
#define FUNCTIONAL_MD_OVERLOADS(cls, method, fn, ...) \
    FUNCTIONAL_MD__MESH(cls, method, fn, __VA_ARGS__) \
    FUNCTIONAL_MD__MESH_SEL(cls, method, fn, __VA_ARGS__) \
    FUNCTIONAL_MD__FIELD(cls, method, fn, __VA_ARGS__) \
    FUNCTIONAL_MD__FIELD_SEL(cls, method, fn, __VA_ARGS__)

/* Emit geometry-only Morpho entry points. INTEGRAND also adds (Mesh, Int) and
 * (Mesh, Int, Int) -> Float. Map methods accept Mesh or Field (mesh from Field).
 * Each wrapper calls the matching _functional_* helper with grade and kernel. */
#define FUNCTIONAL_MD_INTEGRAND(cls, grade, integrandfn) \
    FUNCTIONAL_MD_OVERLOADS(cls, integrand, _functional_integrand, grade, integrandfn) \
    FUNCTIONAL_MD__MESH_INT(cls, integrand, _functional_integrand_elem, grade, integrandfn) \
    FUNCTIONAL_MD__MESH_INT_INT(cls, integrand, _functional_integrand_elem, grade, integrandfn)

#define FUNCTIONAL_MD_TOTAL(cls, grade, integrandfn) \
    FUNCTIONAL_MD_OVERLOADS(cls, total, _functional_total, grade, integrandfn)

#define FUNCTIONAL_MD_GRADIENT(cls, grade, gradientfn, symbhvr) \
    FUNCTIONAL_MD_OVERLOADS(cls, gradient, _functional_gradient, grade, gradientfn, symbhvr)

#define FUNCTIONAL_MD_NUMERICALGRADIENT(cls, grade, integrandfn, symbhvr) \
    FUNCTIONAL_MD_OVERLOADS(cls, gradient, _functional_numericalgradient, grade, integrandfn, symbhvr)

#define FUNCTIONAL_MD_HESSIAN(cls, grade, integrandfn) \
    FUNCTIONAL_MD_OVERLOADS(cls, hessian, _functional_hessian, grade, integrandfn)

/* MORPHO_BEGINCLASS rows that register the C wrappers under Morpho signatures.
 * METHODS_FLAGS takes explicit flags; METHODS plugs in MAPFLAGS / TOTALFLAGS / ELEMFLAGS. */
#define FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(cls, mapflags, elemflags) \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Matrix (Mesh)", cls##_integrand__mesh, mapflags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Matrix (Mesh, Selection)", cls##_integrand__mesh_sel, mapflags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Matrix (Field)", cls##_integrand__field, mapflags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Matrix (Field, Selection)", cls##_integrand__field_sel, mapflags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Float (Mesh, Int)", cls##_integrand__mesh_int, elemflags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Float (Mesh, Int, Int)", cls##_integrand__mesh_int_int, elemflags)

#define FUNCTIONAL_MD_TOTAL_METHODS_FLAGS(cls, flags) \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_TOTAL_METHOD, "Float (Mesh)", cls##_total__mesh, flags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_TOTAL_METHOD, "Float (Mesh, Selection)", cls##_total__mesh_sel, flags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_TOTAL_METHOD, "Float (Field)", cls##_total__field, flags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_TOTAL_METHOD, "Float (Field, Selection)", cls##_total__field_sel, flags)

#define FUNCTIONAL_MD_GRADIENT_METHODS_FLAGS(cls, flags) \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_GRADIENT_METHOD, "Matrix (Mesh)", cls##_gradient__mesh, flags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_GRADIENT_METHOD, "Matrix (Mesh, Selection)", cls##_gradient__mesh_sel, flags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_GRADIENT_METHOD, "Matrix (Field)", cls##_gradient__field, flags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_GRADIENT_METHOD, "Matrix (Field, Selection)", cls##_gradient__field_sel, flags)

#define FUNCTIONAL_MD_HESSIAN_METHODS_FLAGS(cls, flags) \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_HESSIAN_METHOD, "Sparse (Mesh)", cls##_hessian__mesh, flags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_HESSIAN_METHOD, "Sparse (Mesh, Selection)", cls##_hessian__mesh_sel, flags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_HESSIAN_METHOD, "Sparse (Field)", cls##_hessian__field, flags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_HESSIAN_METHOD, "Sparse (Field, Selection)", cls##_hessian__field_sel, flags)

#define FUNCTIONAL_MD_FIELDGRADIENT_METHODS_FLAGS(cls, flags) \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_FIELDGRADIENT_METHOD, "Field (Field)", cls##_fieldgradient__field, flags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_FIELDGRADIENT_METHOD, "Field (Field, Mesh)", cls##_fieldgradient__field_mesh, flags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_FIELDGRADIENT_METHOD, "Field (Field, Selection)", cls##_fieldgradient__field_sel, flags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_FIELDGRADIENT_METHOD, "Field (Field, Mesh, Selection)", cls##_fieldgradient__field_mesh_sel, flags)

#define FUNCTIONAL_MD_INTEGRAND_METHODS(cls) \
    FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(cls, FUNCTIONAL_MD_MAPFLAGS, FUNCTIONAL_MD_ELEMFLAGS)

#define FUNCTIONAL_MD_TOTAL_METHODS(cls) \
    FUNCTIONAL_MD_TOTAL_METHODS_FLAGS(cls, FUNCTIONAL_MD_TOTALFLAGS)

#define FUNCTIONAL_MD_GRADIENT_METHODS(cls) \
    FUNCTIONAL_MD_GRADIENT_METHODS_FLAGS(cls, FUNCTIONAL_MD_MAPFLAGS)

#define FUNCTIONAL_MD_HESSIAN_METHODS(cls) \
    FUNCTIONAL_MD_HESSIAN_METHODS_FLAGS(cls, FUNCTIONAL_MD_MAPFLAGS)

#define FUNCTIONAL_MD_FIELDGRADIENT_METHODS(cls) \
    FUNCTIONAL_MD_FIELDGRADIENT_METHODS_FLAGS(cls, FUNCTIONAL_MD_MAPFLAGS)

/* -------------------------------------------------------
 * Prepare/ref multiple-dispatch macros
 *
 * Wrappers unpack Morpho args and pass self to a typed helper:
 *   value fn(vm *, objectinstance *, functional_mapinfo *)
 * Geometry-only FUNCTIONAL_MD_* macros above stay self-free.
 *
 * Each class supplies _##cls##_bindref (FUNCTIONAL_MD_REF_BIND wraps
 * FORCEGRADE with info->g). Grade for _functional_run may be a class
 * constant or ref.grade (the helper's local ref).
 * Constructors stay handwritten.
 * ------------------------------------------------------- */

/* Same WRAP/INFO as geometry-only, but the call is fn(v, self, &info) so the
 * helper can read instance properties and prepare a ref. */
#define FUNCTIONAL_MD_REF__MESH(cls, method, fn) \
    FUNCTIONAL_MD_WRAP(cls, method, mesh, FUNCTIONAL_MD_INFO__MESH(), fn(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info))

#define FUNCTIONAL_MD_REF__MESH_SEL(cls, method, fn) \
    FUNCTIONAL_MD_WRAP(cls, method, mesh_sel, FUNCTIONAL_MD_INFO__MESH_SEL(), fn(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info))

#define FUNCTIONAL_MD_REF__MESH_INT(cls, method, fn) \
    FUNCTIONAL_MD_WRAP(cls, method, mesh_int, FUNCTIONAL_MD_INFO__MESH_INT(), fn(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info))

#define FUNCTIONAL_MD_REF__MESH_INT_INT(cls, method, fn) \
    FUNCTIONAL_MD_WRAP(cls, method, mesh_int_int, FUNCTIONAL_MD_INFO__MESH_INT_INT(), fn(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info))

#define FUNCTIONAL_MD_REF__FIELD(cls, method, fn) \
    FUNCTIONAL_MD_WRAP(cls, method, field, FUNCTIONAL_MD_INFO__FIELD(), fn(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info))

#define FUNCTIONAL_MD_REF__FIELD_MESH(cls, method, fn) \
    FUNCTIONAL_MD_WRAP(cls, method, field_mesh, FUNCTIONAL_MD_INFO__FIELD_MESH(), fn(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info))

#define FUNCTIONAL_MD_REF__FIELD_SEL(cls, method, fn) \
    FUNCTIONAL_MD_WRAP(cls, method, field_sel, FUNCTIONAL_MD_INFO__FIELD_SEL(), fn(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info))

#define FUNCTIONAL_MD_REF__FIELD_MESH_SEL(cls, method, fn) \
    FUNCTIONAL_MD_WRAP(cls, method, field_mesh_sel, FUNCTIONAL_MD_INFO__FIELD_MESH_SEL(), fn(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info))

/* Emits Mesh and Field-as-mesh overloads for a self-taking helper. */
#define FUNCTIONAL_MD_REF_OVERLOADS(cls, method, fn) \
    FUNCTIONAL_MD_REF__MESH(cls, method, fn) \
    FUNCTIONAL_MD_REF__MESH_SEL(cls, method, fn) \
    FUNCTIONAL_MD_REF__FIELD(cls, method, fn) \
    FUNCTIONAL_MD_REF__FIELD_SEL(cls, method, fn)

/* Field-first overloads for fieldgradient. */
#define FUNCTIONAL_MD_REF_FIELD_OVERLOADS(cls, method, fn) \
    FUNCTIONAL_MD_REF__FIELD(cls, method, fn) \
    FUNCTIONAL_MD_REF__FIELD_MESH(cls, method, fn) \
    FUNCTIONAL_MD_REF__FIELD_SEL(cls, method, fn) \
    FUNCTIONAL_MD_REF__FIELD_MESH_SEL(cls, method, fn)

/* Defines _Cls_bindref: call prepare; on failure raise err; otherwise store
 * ref, integrand, and optional startfn on mapinfo. FORCEGRADE also writes
 * info->g (and passes that grade to prepare). BIND uses whatever grade is
 * already on mapinfo. START variants set info->start (FE-space prep, etc.). */
#define FUNCTIONAL_MD_REF_BIND_FORCEGRADE_START(cls, reftype, prepare, integrandfn, err, grade, startfn) \
static bool _##cls##_bindref(vm *v, objectinstance *self, functional_mapinfo *info, reftype *ref) { \
    if (!prepare(self, info->mesh, grade, info->sel, ref)) MORPHO_FAIL(v, err); \
    info->g = grade; \
    info->ref = ref; \
    info->integrand = integrandfn; \
    info->start = (startfn); \
    return true; \
}

#define FUNCTIONAL_MD_REF_BIND_FORCEGRADE(cls, reftype, prepare, integrandfn, err, grade) \
    FUNCTIONAL_MD_REF_BIND_FORCEGRADE_START(cls, reftype, prepare, integrandfn, err, grade, NULL)

#define FUNCTIONAL_MD_REF_BIND_START(cls, reftype, prepare, integrandfn, err, startfn) \
    FUNCTIONAL_MD_REF_BIND_FORCEGRADE_START(cls, reftype, prepare, integrandfn, err, info->g, startfn)

#define FUNCTIONAL_MD_REF_BIND(cls, reftype, prepare, integrandfn, err) \
    FUNCTIONAL_MD_REF_BIND_FORCEGRADE(cls, reftype, prepare, integrandfn, err, info->g)

/* Defines _Cls_method: stack-allocate reftype, bindref, set grad/deps/sym,
 * then _functional_run. HELPER is RUN with those three left unused. */
#define FUNCTIONAL_MD_REF_RUN(cls, method, reftype, grade, mapfn, bind, gradfn, deps, symbhvr) \
static value _##cls##_##method(vm *v, objectinstance *self, functional_mapinfo *info) { \
    reftype ref; \
    if (!_##cls##_bindref(v, self, info, &ref)) return MORPHO_NIL; \
    info->grad = (gradfn); \
    info->dependencies = (deps); \
    info->sym = (symbhvr); \
    return _functional_run(v, info, grade, mapfn, bind); \
}

#define FUNCTIONAL_MD_REF_HELPER(cls, method, reftype, grade, mapfn, bind) \
    FUNCTIONAL_MD_REF_RUN(cls, method, reftype, grade, mapfn, bind, NULL, NULL, SYMMETRY_NONE)

/* Emit prepare/ref Morpho entry points plus the _Cls_* helpers they call.
 * INTEGRAND covers the four signatures; TOTAL mesh/sel; GRADIENT sets
 * info->grad; NUMERICALGRADIENT/HESSIAN set dependencies; FIELDGRADIENT
 * is Field-first and sets cloneref/freeref. */
#define FUNCTIONAL_MD_REF_INTEGRAND(cls, reftype, grade) \
    FUNCTIONAL_MD_REF_HELPER(cls, integrand, reftype, grade, functional_mapintegrand, true) \
    FUNCTIONAL_MD_REF_HELPER(cls, integrand_elem, reftype, grade, functional_mapintegrandforelement, false) \
    FUNCTIONAL_MD_REF_OVERLOADS(cls, integrand, _##cls##_integrand) \
    FUNCTIONAL_MD_REF__MESH_INT(cls, integrand, _##cls##_integrand_elem) \
    FUNCTIONAL_MD_REF__MESH_INT_INT(cls, integrand, _##cls##_integrand_elem)

#define FUNCTIONAL_MD_REF_TOTAL(cls, reftype, grade) \
    FUNCTIONAL_MD_REF_HELPER(cls, total, reftype, grade, functional_sumintegrand, false) \
    FUNCTIONAL_MD_REF_OVERLOADS(cls, total, _##cls##_total)

#define FUNCTIONAL_MD_REF_GRADIENT(cls, reftype, grade, gradientfn, symbhvr) \
    FUNCTIONAL_MD_REF_RUN(cls, gradient, reftype, grade, functional_mapgradient, true, gradientfn, NULL, symbhvr) \
    FUNCTIONAL_MD_REF_OVERLOADS(cls, gradient, _##cls##_gradient)

#define FUNCTIONAL_MD_REF_NUMERICALGRADIENT(cls, reftype, grade, deps, symbhvr) \
    FUNCTIONAL_MD_REF_RUN(cls, gradient, reftype, grade, functional_mapnumericalgradient, true, NULL, deps, symbhvr) \
    FUNCTIONAL_MD_REF_OVERLOADS(cls, gradient, _##cls##_gradient)

#define FUNCTIONAL_MD_REF_HESSIAN(cls, reftype, grade, deps, symbhvr) \
    FUNCTIONAL_MD_REF_RUN(cls, hessian, reftype, grade, functional_mapnumericalhessian, true, NULL, deps, symbhvr) \
    FUNCTIONAL_MD_REF_OVERLOADS(cls, hessian, _##cls##_hessian)

/* fieldgradient sets cloneref/freeref so the numerical map can clone the
 * target Field. MAP takes a custom mapfn (Jump); the default is numerical.
 * clonefn/freefn are named so they are not substituted into info->cloneref. */
#define FUNCTIONAL_MD_REF_FIELDGRADIENT_RUN(cls, reftype, grade, mapfn, clonefn, freefn) \
static value _##cls##_fieldgradient(vm *v, objectinstance *self, functional_mapinfo *info) { \
    reftype ref; \
    if (!_##cls##_bindref(v, self, info, &ref)) return MORPHO_NIL; \
    info->cloneref = (clonefn); \
    info->freeref = (freefn); \
    return _functional_run(v, info, grade, mapfn, true); \
}

#define FUNCTIONAL_MD_REF_FIELDGRADIENT_MAP(cls, reftype, grade, mapfn, clonefn, freefn) \
    FUNCTIONAL_MD_REF_FIELDGRADIENT_RUN(cls, reftype, grade, mapfn, clonefn, freefn) \
    FUNCTIONAL_MD_REF_FIELD_OVERLOADS(cls, fieldgradient, _##cls##_fieldgradient)

#define FUNCTIONAL_MD_REF_FIELDGRADIENT(cls, reftype, grade, clonefn, freefn) \
    FUNCTIONAL_MD_REF_FIELDGRADIENT_MAP(cls, reftype, grade, functional_mapnumericalfieldgradient, clonefn, freefn)

/* -------------------------------------------------------
 * Compatibility shim
 *
 * TODO: Remove once external modules are updated to
 * FUNCTIONAL_MD_* / FUNCTIONAL_MD_REF_* and typed class tables.
 * ------------------------------------------------------- */

#define FUNCTIONAL_SHIM_LEGACY_REF(cls, method) \
value cls##_##method(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    if (!functional_validateargs(v, nargs, args, &info)) return MORPHO_NIL; \
    return _##cls##_##method(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info); \
}

#define FUNCTIONAL_SHIM_LEGACY_GEOM(name, method, fn, ...) \
value name##_##method(vm *v, int nargs, value *args) { \
    functional_mapinfo info; \
    if (!functional_validateargs(v, nargs, args, &info)) return MORPHO_NIL; \
    return fn(v, &info, __VA_ARGS__); \
}

#define FUNCTIONAL_INTEGRAND(name, grade, integrandfn) \
    FUNCTIONAL_MD_INTEGRAND(name, grade, integrandfn) \
    FUNCTIONAL_SHIM_LEGACY_GEOM(name, integrand, _functional_integrand, grade, integrandfn)

#define FUNCTIONAL_INTEGRANDFORELEMENT(name, grade, integrandfn) \
    FUNCTIONAL_SHIM_LEGACY_GEOM(name, integrandForElement, _functional_integrand_elem, grade, integrandfn)

#define FUNCTIONAL_TOTAL(name, grade, totalfn) \
    FUNCTIONAL_MD_TOTAL(name, grade, totalfn) \
    FUNCTIONAL_SHIM_LEGACY_GEOM(name, total, _functional_total, grade, totalfn)

#define FUNCTIONAL_GRADIENT(name, grade, gradientfn, symbhvr) \
    FUNCTIONAL_MD_GRADIENT(name, grade, gradientfn, symbhvr) \
    FUNCTIONAL_SHIM_LEGACY_GEOM(name, gradient, _functional_gradient, grade, gradientfn, symbhvr)

#define FUNCTIONAL_NUMERICALGRADIENT(name, grade, integrandfn, symbhvr) \
    FUNCTIONAL_MD_NUMERICALGRADIENT(name, grade, integrandfn, symbhvr) \
    FUNCTIONAL_SHIM_LEGACY_GEOM(name, gradient, _functional_numericalgradient, grade, integrandfn, symbhvr)

#define FUNCTIONAL_HESSIAN(name, grade, totalfn) \
    FUNCTIONAL_MD_HESSIAN(name, grade, totalfn) \
    FUNCTIONAL_SHIM_LEGACY_GEOM(name, hessian, _functional_hessian, grade, totalfn)

#define FUNCTIONAL_METHOD(class, name, grade, reftype, prepare, mapfn, kernel, deps, err, symbhvr) \
    FUNCTIONAL_METHOD_START(class, name, grade, reftype, prepare, NULL, NULL, mapfn, kernel, deps, err, symbhvr)

#define FUNCTIONAL_METHOD_START(class, name, grade, reftype, prepare, startfn, endfn, mapfn, kernel, deps, err, symbhvr) \
    FUNCTIONAL_SHIM_METHOD__##name(class, grade, reftype, prepare, startfn, mapfn, kernel, deps, err, symbhvr)

#define FUNCTIONAL_SHIM_METHOD__integrand(class, grade, reftype, prepare, startfn, mapfn, kernel, deps, err, symbhvr) \
    FUNCTIONAL_MD_REF_BIND_START(class, reftype, prepare, kernel, err, startfn) \
    FUNCTIONAL_MD_REF_INTEGRAND(class, reftype, grade) \
    FUNCTIONAL_SHIM_LEGACY_REF(class, integrand)

#define FUNCTIONAL_SHIM_METHOD__total(class, grade, reftype, prepare, startfn, mapfn, kernel, deps, err, symbhvr) \
    FUNCTIONAL_MD_REF_TOTAL(class, reftype, grade) \
    FUNCTIONAL_SHIM_LEGACY_REF(class, total)

/* -------------------------------------------------------
 * Initialization
 * ------------------------------------------------------- */

void functional_initialize(void);
void functional_finalize(void);

#endif

#endif /* functional_h */
