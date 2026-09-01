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
#include <string.h>
#include <math.h>
#include "morpho.h"
#include "mesh.h"
#include "field.h"
#include "selection.h"

/* -------------------------------------------------------
 * Functionals
 * ------------------------------------------------------- */

/* -------------------------------------------------------
 * Generic functional properties and methods
 * ------------------------------------------------------- */

#define FUNCTIONAL_GRADE_PROPERTY             "grade"
#define FUNCTIONAL_FIELD_PROPERTY             "field"

#define FUNCTIONAL_INTEGRAND_METHOD    "integrand"
#define FUNCTIONAL_TOTAL_METHOD        "total"
#define FUNCTIONAL_GRADIENT_METHOD     "gradient"
#define FUNCTIONAL_FIELDGRADIENT_METHOD     "fieldgradient"
#define FUNCTIONAL_HESSIAN_METHOD      "hessian"
#define FUNCTIONAL_INTEGRANDFORELEMENT_METHOD      "integrandForElement"
#define FUNCTIONAL_UPDATE_METHOD       "update"

/* -------------------------------------------------------
 * Generic functional error messages
 * ------------------------------------------------------- */

#define FUNC_ELNTFND                   "FnctlELNtFnd"
#define FUNC_ELNTFND_MSG               "Mesh does not provide elements of grade %u."

#define FUNC_FESPACE                   "FnctlFESpc"
#define FUNC_FESPACE_MSG               "This Field%s%s cannot be evaluated on grade %u elements."

#define FUNC_NOFESPACE                 "FnctlNoFESpc"
#define FUNC_NOFESPACE_MSG             "This Field has no finite element space; pass finiteelementspace=... or omit the opt-out."

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

/** Field-gradient function */
typedef bool (functional_fieldgradient) (vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectfield *grad);

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

/** Choose which map callback to run (Integral/ScalarPotential). NULL if bind failed. */
typedef functional_mapcallback *(functional_mapchooser) (vm *v, objectinstance *self, struct s_functional_mapinfo *info);

/** Clone reference function */
typedef void * (functional_cloneref) (void *ref, objectfield *field, objectfield *sub);

/** Free reference function */
typedef void (functional_freeref) (void *ref);

/** Dependencies function */
typedef bool (functional_dependencies) (struct s_functional_mapinfo *info, elementid id, varray_elementid *out);

/** Workload cost hint. Remain serial if cost * nel < FORKWEIGHT * (nthreads-1).
 * Calibrated so Length.total (CHEAPEST) stays serial through a few thousand
 * edges at -w7 and parallelizes by ~10k; Area at 5k faces still forks. */
#define FUNCTIONAL_FORKWEIGHT 800

typedef enum {
    FUNCTIONAL_COST_CHEAPEST = 1,  /* Length, Area.total, NormSq, LinearElasticity.total */
    FUNCTIONAL_COST_CHEAP    = 4,  /* Curvature totals, NematicElectric.total */
    FUNCTIONAL_COST_REGULAR  = 10  /* Integrals, FD/shape gradients, Jump, Hydrogel, GradSq/Nematic.total */
} functional_cost;

typedef struct s_functional_mapinfo {
    objectmesh *mesh; // Mesh to use
    objectselection *sel; // Selection, if any
    objectfield *field; // Field, if any
    grade g; // Grade to use
    elementid id; // Element id at which to evaluate the integrand
    functional_integrand *integrand; // Integrand function
    functional_gradient *grad; // Gradient
    functional_fieldgradient *fieldgrad; // Analytic field gradient
    functional_start *start; // Optional preflight hook (once per user call)
    functional_end *end; // Optional postflight hook (once per user call)
    functional_taskstart *taskstart; // Optional per-task setup (once per map task)
    functional_taskend *taskend; // Optional per-task teardown (once per map task)
    functional_dependencies *dependencies; // Dependencies
    functional_cloneref *cloneref; // Clone a reference with a given field substituted
    functional_freeref *freeref; // Free a reference
    symmetrybhvr sym; // Symmetry behavior
    functional_cost cost; // Workload class for the dispatcher
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
bool functional_mapfieldgradient(vm *v, functional_mapinfo *info, value *out);
bool functional_mapnumericalgradient(vm *v, functional_mapinfo *info, value *out);
bool functional_mapnumericalfieldgradient(vm *v, functional_mapinfo *info, value *out);
bool functional_mapnumericalhessian(vm *v, functional_mapinfo *info, value *out);
bool functional_startmap(vm *v, functional_mapinfo *info);
bool functional_endmap(vm *v, functional_mapinfo *info);
bool functional_runmap(vm *v, functional_mapinfo *info, functional_mapcallback *mapfn, value *out);

/** Add two vectors */
static inline void functional_vecadd(unsigned int n, double *a, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]+b[i];
}

/** Add with scale */
static inline void functional_vecaddscale(unsigned int n, double *a, double lambda, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]+lambda*b[i];
}

/** Subtract two vectors */
static inline void functional_vecsub(unsigned int n, double *a, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]-b[i];
}

/** Scale a vector */
static inline void functional_vecscale(unsigned int n, double lambda, double *a, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=lambda*a[i];
}

/** Dot product */
static inline double functional_vecdot(unsigned int n, double *a, double *b) {
    double s=0.0;
    for (unsigned int i=0; i<n; i++) s+=a[i]*b[i];
    return s;
}

/** Euclidean norm */
static inline double functional_vecnorm(unsigned int n, double *a) {
    return sqrt(functional_vecdot(n, a, a));
}

/** 3D cross product  */
static inline void functional_veccross(double *a, double *b, double *out) {
    out[0]=a[1]*b[2]-a[2]*b[1];
    out[1]=a[2]*b[0]-a[0]*b[2];
    out[2]=a[0]*b[1]-a[1]*b[0];
}

/** 2D cross product  */
static inline void functional_veccross2d(double *a, double *b, double *out) {
    *out=a[0]*b[1]-a[1]*b[0];
}

/** In-place inverse of a 1×1, 2×2 or 3×3 column-major matrix. Returns false if singular.
 *  2×2 is a closed form; 3×3 is a looped cofactor. See benchmarks/matrix/. */
static inline bool functional_matinv2x2(double *e) {
    double det=e[0]*e[3]-e[2]*e[1];
    if (det==0.0) return false;
    double s=1.0/det, a00=e[0];
    e[0]=e[3]*s; e[1]=-e[1]*s; e[2]=-e[2]*s; e[3]=a00*s;
    return true;
}

static inline double _functional_minor3(double *a, unsigned int i, unsigned int j) {
    unsigned int r0=i?0:1, r1=i==2?1:2, c0=j?0:1, c1=j==2?1:2;
    return a[r0+c0*3]*a[r1+c1*3]-a[r0+c1*3]*a[r1+c0*3];
}

static inline bool functional_matinv3x3(double *e) {
    double a[9], c[9], det=0.0;
    memcpy(a, e, sizeof a);
    for (unsigned int j=0; j<3; j++) {
        for (unsigned int i=0; i<3; i++) {
            double m=_functional_minor3(a, i, j);
            c[i+j*3]=(i+j)%2 ? -m : m;
        }
    }
    det=a[0]*c[0]+a[3]*c[3]+a[6]*c[6];
    if (det==0.0) return false;
    det=1.0/det;
    for (unsigned int j=0; j<3; j++) {
        for (unsigned int i=0; i<3; i++) e[i+j*3]=c[j+i*3]*det;
    }
    return true;
}

static inline bool functional_matinv(unsigned int n, double *e) {
    if (n==1) { if (e[0]==0.0) return false; e[0]=1.0/e[0]; return true; }
    if (n==2) return functional_matinv2x2(e);
    if (n==3) return functional_matinv3x3(e);
    return false;
}

/** C <- A*B for small column-major matrices. C may alias A or B. */
static inline void _functional_matvec2(double *A, double *b, double *c) {
    c[0]=A[0]*b[0]+A[2]*b[1];
    c[1]=A[1]*b[0]+A[3]*b[1];
}

static inline void _functional_matvec3(double *A, double *b, double *c) {
    c[0]=A[0]*b[0]+A[3]*b[1]+A[6]*b[2];
    c[1]=A[1]*b[0]+A[4]*b[1]+A[7]*b[2];
    c[2]=A[2]*b[0]+A[5]*b[1]+A[8]*b[2];
}

static inline void functional_matmul2x2(double *A, double *B, double *C) {
    double t[4];
    for (int j=0; j<2; j++) _functional_matvec2(A, B+2*j, t+2*j);
    memcpy(C, t, sizeof t);
}

static inline void functional_matmul3x3(double *A, double *B, double *C) {
    double t[9];
    for (int j=0; j<3; j++) _functional_matvec3(A, B+3*j, t+3*j);
    memcpy(C, t, sizeof t);
}

static inline void functional_matmul(unsigned int m, unsigned int k, unsigned int n, double *A, double *B, double *C) {
    if (m==2 && k==2 && n==2) { functional_matmul2x2(A, B, C); return; }
    if (m==3 && k==3 && n==3) { functional_matmul3x3(A, B, C); return; }
    double t[m*n];
    for (unsigned int j=0; j<n; j++) {
        for (unsigned int i=0; i<m; i++) {
            double s=0.0;
            for (unsigned int p=0; p<k; p++) s+=A[i+p*m]*B[p+j*k];
            t[i+j*m]=s;
        }
    }
    memcpy(C, t, sizeof(double)*(size_t)m*(size_t)n);
}

bool functional_elementsize(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, double *out);
bool functional_elementgradient_scale(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, objectmatrix *frc, double scale);
bool functional_elementgradient(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, objectmatrix *frc);

bool functional_readgrade(objectinstance *self, grade *g);
void functional_setgrade(objectinstance *self, grade g);

double functional_fdstepsize(double x, int order);

/** Shared Field + grade ref used by GradSq, NormSq and Nematic. */
typedef struct {
    objectfield *field;
    grade grade;
} fieldref;

void functional_fespaceerror(vm *v, objectfield *field, grade g);
bool functional_preparefespacefield(vm *v, objectfield *field, grade g);
bool functional_preparefieldlist(vm *v, value *fields, int nfields, grade g);
bool fieldref_startfn(vm *v, functional_mapinfo *info);

#define FUNCTIONAL_FESPACE_FAIL(v, field, g) \
    { functional_fespaceerror(v, field, g); return false; }

bool functional_countelements(objectmesh *mesh, grade g, int *n, objectsparse **s);

void functional_accum(double *p, double inc);
bool functional_addtocolumn(objectmatrix *a, MatrixIdx_t col, double alpha, double *b);
bool functional_addtoelement(objectmatrix *a, MatrixIdx_t row, MatrixIdx_t col, double inc);

/* Internal map helpers used by Integral and Jump */
typedef bool (functional_mapfn) (vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out);
typedef bool (functional_processfn) (void *task);

typedef struct {
    elementid start, end;
    elementid id;
    elementid nel;
    varray_elementid *skip;
    unsigned int sindx;
    grade g;
    objectsparse *conn;
    functional_mapfn *mapfn;
    functional_processfn *processfn;
    vm *v;
    objectmesh *mesh;
    objectfield *field;
    objectselection *selection;
    functional_mapinfo *mapinfo;
    functional_taskstart *taskstart;
    functional_taskend *taskend;
    void *ref;
    void *result;
    void *out;
    bool usesubkernel;
    _MORPHO_PADDING;
} functional_task;

typedef struct sfespace fespace;

typedef struct {
    functional_mapinfo *info;
    objectfield *field;
    functional_integrand *integrand;
    fespace *disc;
    objectsparse *conn;
    void *ref;
} functional_numericalfieldgradientref;

int functional_ntasks(functional_mapinfo *info);
int functional_preparetasks(vm *v, functional_mapinfo *info, int ntask, functional_task *task, varray_elementid *imageids);
void functional_cleanuptasks(vm *v, int ntask, functional_task *task, varray_elementid *imageids);
bool functional_map(int ntasks, functional_task *tasks);
bool functional_numericalfieldgradentry(vm *v, objectmesh *mesh, elementid eid, objectfield *field, grade g, elementid i, int indx, int nv, int *vid, functional_integrand *integrand, void *ref, objectfield *grad);
bool functional_numericalfieldgradientmapfn(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out);
bool functional_preparenumericalfieldgradientref(vm *v, functional_mapinfo *info, bool clone, functional_numericalfieldgradientref *tref, objectfield **fieldclone);
void functional_clearnumericalfieldgradientref(functional_mapinfo *info, functional_numericalfieldgradientref *tref, objectfield *fieldclone);

/* Helpers used by FUNCTIONAL_MD_* macros. TODO: Make static once shims removed */
void _functional_mapinfo(functional_mapinfo *info, objectmesh *mesh, objectselection *sel, objectfield *field);
value _functional_run(vm *v, functional_mapinfo *info, grade g, functional_mapcallback *mapfn, bool bind);
value _functional_integrand(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn);
value _functional_integrand_elem(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn);
value _functional_total(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn);
value _functional_gradient(vm *v, functional_mapinfo *info, grade g, functional_gradient *fn, symmetrybhvr sym);
value _functional_fieldgradient(vm *v, functional_mapinfo *info, grade g, functional_fieldgradient *fn);
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

/* Constructs a function cls_method__suffix. WRAP_COST also sets info.cost. */
#define FUNCTIONAL_MD_WRAP(cls, method, suffix, setup, ...) \
value cls##_##method##__##suffix(vm *v, int nargs, value *args) { \
    setup; \
    return __VA_ARGS__; \
}

#define FUNCTIONAL_MD_WRAP_COST(cls, method, suffix, setup, wkld, ...) \
    FUNCTIONAL_MD_WRAP(cls, method, suffix, setup; info.cost=(wkld), __VA_ARGS__)

/* Geometry-only Morpho entry points. COST is the primitive (info.cost in WRAP).
 * Bare names forward FUNCTIONAL_COST_REGULAR. INTEGRAND also adds
 * (Mesh, Int) and (Mesh, Int, Int) -> Float. */
#define FUNCTIONAL_MD_OVERLOADS_COST(cls, method, fn, wkld, ...) \
    FUNCTIONAL_MD_WRAP_COST(cls, method, mesh, FUNCTIONAL_MD_INFO__MESH(), wkld, fn(v, &info, __VA_ARGS__)) \
    FUNCTIONAL_MD_WRAP_COST(cls, method, mesh_sel, FUNCTIONAL_MD_INFO__MESH_SEL(), wkld, fn(v, &info, __VA_ARGS__)) \
    FUNCTIONAL_MD_WRAP_COST(cls, method, field, FUNCTIONAL_MD_INFO__FIELD(), wkld, fn(v, &info, __VA_ARGS__)) \
    FUNCTIONAL_MD_WRAP_COST(cls, method, field_sel, FUNCTIONAL_MD_INFO__FIELD_SEL(), wkld, fn(v, &info, __VA_ARGS__))

#define FUNCTIONAL_MD_INTEGRAND_COST(cls, grade, integrandfn, wkld) \
    FUNCTIONAL_MD_OVERLOADS_COST(cls, integrand, _functional_integrand, wkld, grade, integrandfn) \
    FUNCTIONAL_MD_WRAP_COST(cls, integrand, mesh_int, FUNCTIONAL_MD_INFO__MESH_INT(), wkld, _functional_integrand_elem(v, &info, grade, integrandfn)) \
    FUNCTIONAL_MD_WRAP_COST(cls, integrand, mesh_int_int, FUNCTIONAL_MD_INFO__MESH_INT_INT(), wkld, _functional_integrand_elem(v, &info, grade, integrandfn))

#define FUNCTIONAL_MD_INTEGRAND(cls, grade, integrandfn) \
    FUNCTIONAL_MD_INTEGRAND_COST(cls, grade, integrandfn, FUNCTIONAL_COST_REGULAR)

#define FUNCTIONAL_MD_TOTAL_COST(cls, grade, integrandfn, wkld) \
    FUNCTIONAL_MD_OVERLOADS_COST(cls, total, _functional_total, wkld, grade, integrandfn)

#define FUNCTIONAL_MD_TOTAL(cls, grade, integrandfn) \
    FUNCTIONAL_MD_TOTAL_COST(cls, grade, integrandfn, FUNCTIONAL_COST_REGULAR)

#define FUNCTIONAL_MD_GRADIENT_COST(cls, grade, gradientfn, symbhvr, wkld) \
    FUNCTIONAL_MD_OVERLOADS_COST(cls, gradient, _functional_gradient, wkld, grade, gradientfn, symbhvr)

#define FUNCTIONAL_MD_GRADIENT(cls, grade, gradientfn, symbhvr) \
    FUNCTIONAL_MD_GRADIENT_COST(cls, grade, gradientfn, symbhvr, FUNCTIONAL_COST_REGULAR)

#define FUNCTIONAL_MD_NUMERICALGRADIENT_COST(cls, grade, integrandfn, symbhvr, wkld) \
    FUNCTIONAL_MD_OVERLOADS_COST(cls, gradient, _functional_numericalgradient, wkld, grade, integrandfn, symbhvr)

#define FUNCTIONAL_MD_NUMERICALGRADIENT(cls, grade, integrandfn, symbhvr) \
    FUNCTIONAL_MD_NUMERICALGRADIENT_COST(cls, grade, integrandfn, symbhvr, FUNCTIONAL_COST_REGULAR)

#define FUNCTIONAL_MD_HESSIAN_COST(cls, grade, integrandfn, wkld) \
    FUNCTIONAL_MD_OVERLOADS_COST(cls, hessian, _functional_hessian, wkld, grade, integrandfn)

#define FUNCTIONAL_MD_HESSIAN(cls, grade, integrandfn) \
    FUNCTIONAL_MD_HESSIAN_COST(cls, grade, integrandfn, FUNCTIONAL_COST_REGULAR)

/* MORPHO_BEGINCLASS rows that register the C wrappers under Morpho signatures.
 * METHODS_FLAGS takes explicit flags; METHODS plugs in MAPFLAGS / TOTALFLAGS / ELEMFLAGS. */
#define FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(cls, mapflags, elemflags) \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Field (Mesh)", cls##_integrand__mesh, mapflags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Field (Mesh, Selection)", cls##_integrand__mesh_sel, mapflags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Field (Field)", cls##_integrand__field, mapflags), \
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Field (Field, Selection)", cls##_integrand__field_sel, mapflags), \
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
#define FUNCTIONAL_MD_REF_WRAP(cls, method, suffix, setup, fn) \
    FUNCTIONAL_MD_WRAP(cls, method, suffix, setup, fn(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info))

#define FUNCTIONAL_MD_REF__MESH(cls, method, fn) \
    FUNCTIONAL_MD_REF_WRAP(cls, method, mesh, FUNCTIONAL_MD_INFO__MESH(), fn)

#define FUNCTIONAL_MD_REF__MESH_SEL(cls, method, fn) \
    FUNCTIONAL_MD_REF_WRAP(cls, method, mesh_sel, FUNCTIONAL_MD_INFO__MESH_SEL(), fn)

#define FUNCTIONAL_MD_REF__MESH_INT(cls, method, fn) \
    FUNCTIONAL_MD_REF_WRAP(cls, method, mesh_int, FUNCTIONAL_MD_INFO__MESH_INT(), fn)

#define FUNCTIONAL_MD_REF__MESH_INT_INT(cls, method, fn) \
    FUNCTIONAL_MD_REF_WRAP(cls, method, mesh_int_int, FUNCTIONAL_MD_INFO__MESH_INT_INT(), fn)

#define FUNCTIONAL_MD_REF__FIELD(cls, method, fn) \
    FUNCTIONAL_MD_REF_WRAP(cls, method, field, FUNCTIONAL_MD_INFO__FIELD(), fn)

#define FUNCTIONAL_MD_REF__FIELD_MESH(cls, method, fn) \
    FUNCTIONAL_MD_REF_WRAP(cls, method, field_mesh, FUNCTIONAL_MD_INFO__FIELD_MESH(), fn)

#define FUNCTIONAL_MD_REF__FIELD_SEL(cls, method, fn) \
    FUNCTIONAL_MD_REF_WRAP(cls, method, field_sel, FUNCTIONAL_MD_INFO__FIELD_SEL(), fn)

#define FUNCTIONAL_MD_REF__FIELD_MESH_SEL(cls, method, fn) \
    FUNCTIONAL_MD_REF_WRAP(cls, method, field_mesh_sel, FUNCTIONAL_MD_INFO__FIELD_MESH_SEL(), fn)

/* Emits Mesh and Field-as-mesh overloads for a self-taking helper.
 * Cost is set inside _Cls_method (RUN), not here. */
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

/* Defines _Cls_method: bindref, set cost/grad/deps/sym/clone, then _functional_run.
 * EMIT also registers Morpho overloads. HELPER is the no-grad/no-clone case. */
#define FUNCTIONAL_MD_REF_RUN(cls, method, reftype, grade, mapfn, bind, gradfn, deps, symbhvr, clonefn, freefn, wkld) \
static value _##cls##_##method(vm *v, objectinstance *self, functional_mapinfo *info) { \
    reftype ref; \
    if (!_##cls##_bindref(v, self, info, &ref)) return MORPHO_NIL; \
    info->grad = (gradfn); \
    info->dependencies = (deps); \
    info->sym = (symbhvr); \
    info->cloneref = (clonefn); \
    info->freeref = (freefn); \
    info->cost = (wkld); \
    return _functional_run(v, info, grade, mapfn, bind); \
}

#define FUNCTIONAL_MD_REF_EMIT(cls, method, reftype, grade, mapfn, bind, gradfn, deps, symbhvr, clonefn, freefn, overloads, wkld) \
    FUNCTIONAL_MD_REF_RUN(cls, method, reftype, grade, mapfn, bind, gradfn, deps, symbhvr, clonefn, freefn, wkld) \
    overloads(cls, method, _##cls##_##method)

#define FUNCTIONAL_MD_REF_HELPER(cls, method, reftype, grade, mapfn, bind, wkld) \
    FUNCTIONAL_MD_REF_RUN(cls, method, reftype, grade, mapfn, bind, NULL, NULL, SYMMETRY_NONE, NULL, NULL, wkld)

#define FUNCTIONAL_MD_REF_MAP_COST(cls, method, reftype, grade, mapfn, bind, gradfn, deps, symbhvr, wkld) \
    FUNCTIONAL_MD_REF_EMIT(cls, method, reftype, grade, mapfn, bind, gradfn, deps, symbhvr, NULL, NULL, FUNCTIONAL_MD_REF_OVERLOADS, wkld)

/* Chooser: stack-allocate ref, set cost, let choosefn bind and pick mapfn. */
#define FUNCTIONAL_MD_REF_RUN_CHOOSE(cls, method, reftype, grade, choosefn, bind, wkld) \
static value _##cls##_##method(vm *v, objectinstance *self, functional_mapinfo *info) { \
    reftype ref; \
    functional_mapcallback *mapfn; \
    info->ref = &ref; \
    info->cost = (wkld); \
    mapfn = (choosefn)(v, self, info); \
    if (!mapfn) return MORPHO_NIL; \
    return _functional_run(v, info, grade, mapfn, bind); \
}

#define FUNCTIONAL_MD_REF_CHOOSE_COST(cls, method, reftype, grade, choosefn, bind, overloads, wkld) \
    FUNCTIONAL_MD_REF_RUN_CHOOSE(cls, method, reftype, grade, choosefn, bind, wkld) \
    overloads(cls, method, _##cls##_##method)

/* Emit prepare/ref Morpho entry points plus the _Cls_* helpers they call.
 * COST is the primitive (info->cost in RUN). Bare names forward REGULAR.
 * INTEGRAND covers the four signatures; TOTAL mesh/sel; GRADIENT sets
 * info->grad; NUMERICALGRADIENT/HESSIAN set dependencies; FIELDGRADIENT
 * is Field-first (numerical sets cloneref/freeref; ANALYTICALFIELDGRADIENT
 * sets info->fieldgrad). */
#define FUNCTIONAL_MD_REF_INTEGRAND_COST(cls, reftype, grade, wkld) \
    FUNCTIONAL_MD_REF_HELPER(cls, integrand, reftype, grade, functional_mapintegrand, true, wkld) \
    FUNCTIONAL_MD_REF_HELPER(cls, integrand_elem, reftype, grade, functional_mapintegrandforelement, false, wkld) \
    FUNCTIONAL_MD_REF_OVERLOADS(cls, integrand, _##cls##_integrand) \
    FUNCTIONAL_MD_REF__MESH_INT(cls, integrand, _##cls##_integrand_elem) \
    FUNCTIONAL_MD_REF__MESH_INT_INT(cls, integrand, _##cls##_integrand_elem)

#define FUNCTIONAL_MD_REF_INTEGRAND(cls, reftype, grade) \
    FUNCTIONAL_MD_REF_INTEGRAND_COST(cls, reftype, grade, FUNCTIONAL_COST_REGULAR)

#define FUNCTIONAL_MD_REF_TOTAL_COST(cls, reftype, grade, wkld) \
    FUNCTIONAL_MD_REF_MAP_COST(cls, total, reftype, grade, functional_sumintegrand, false, NULL, NULL, SYMMETRY_NONE, wkld)

#define FUNCTIONAL_MD_REF_TOTAL(cls, reftype, grade) \
    FUNCTIONAL_MD_REF_TOTAL_COST(cls, reftype, grade, FUNCTIONAL_COST_REGULAR)

#define FUNCTIONAL_MD_REF_GRADIENT_COST(cls, reftype, grade, gradientfn, symbhvr, wkld) \
    FUNCTIONAL_MD_REF_MAP_COST(cls, gradient, reftype, grade, functional_mapgradient, true, gradientfn, NULL, symbhvr, wkld)

#define FUNCTIONAL_MD_REF_GRADIENT(cls, reftype, grade, gradientfn, symbhvr) \
    FUNCTIONAL_MD_REF_GRADIENT_COST(cls, reftype, grade, gradientfn, symbhvr, FUNCTIONAL_COST_REGULAR)

#define FUNCTIONAL_MD_REF_NUMERICALGRADIENT_COST(cls, reftype, grade, deps, symbhvr, wkld) \
    FUNCTIONAL_MD_REF_MAP_COST(cls, gradient, reftype, grade, functional_mapnumericalgradient, true, NULL, deps, symbhvr, wkld)

#define FUNCTIONAL_MD_REF_NUMERICALGRADIENT(cls, reftype, grade, deps, symbhvr) \
    FUNCTIONAL_MD_REF_NUMERICALGRADIENT_COST(cls, reftype, grade, deps, symbhvr, FUNCTIONAL_COST_REGULAR)

#define FUNCTIONAL_MD_REF_HESSIAN_COST(cls, reftype, grade, deps, symbhvr, wkld) \
    FUNCTIONAL_MD_REF_MAP_COST(cls, hessian, reftype, grade, functional_mapnumericalhessian, true, NULL, deps, symbhvr, wkld)

#define FUNCTIONAL_MD_REF_HESSIAN(cls, reftype, grade, deps, symbhvr) \
    FUNCTIONAL_MD_REF_HESSIAN_COST(cls, reftype, grade, deps, symbhvr, FUNCTIONAL_COST_REGULAR)

#define FUNCTIONAL_MD_REF_CHOOSEGRADIENT_COST(cls, reftype, grade, choosefn, wkld) \
    FUNCTIONAL_MD_REF_CHOOSE_COST(cls, gradient, reftype, grade, choosefn, true, FUNCTIONAL_MD_REF_OVERLOADS, wkld)

#define FUNCTIONAL_MD_REF_CHOOSEGRADIENT(cls, reftype, grade, choosefn) \
    FUNCTIONAL_MD_REF_CHOOSEGRADIENT_COST(cls, reftype, grade, choosefn, FUNCTIONAL_COST_REGULAR)

#define FUNCTIONAL_MD_REF_CHOOSEFIELDGRADIENT_COST(cls, reftype, grade, choosefn, wkld) \
    FUNCTIONAL_MD_REF_CHOOSE_COST(cls, fieldgradient, reftype, grade, choosefn, true, FUNCTIONAL_MD_REF_FIELD_OVERLOADS, wkld)

#define FUNCTIONAL_MD_REF_CHOOSEFIELDGRADIENT(cls, reftype, grade, choosefn) \
    FUNCTIONAL_MD_REF_CHOOSEFIELDGRADIENT_COST(cls, reftype, grade, choosefn, FUNCTIONAL_COST_REGULAR)

/* Numerical fieldgradient sets cloneref/freeref so the map can clone the
 * target Field. MAP takes a custom mapfn (Jump); the default is numerical.
 * ANALYTICALFIELDGRADIENT maps info->fieldgrad; a NULL kernel (including a
 * Field argument that is not reftype.field) yields a zero Field. reftype
 * must have an objectfield *field member. */
#define FUNCTIONAL_MD_REF_FIELDGRADIENT_MAP_COST(cls, reftype, grade, mapfn, clonefn, freefn, wkld) \
    FUNCTIONAL_MD_REF_EMIT(cls, fieldgradient, reftype, grade, mapfn, true, NULL, NULL, SYMMETRY_NONE, clonefn, freefn, FUNCTIONAL_MD_REF_FIELD_OVERLOADS, wkld)

#define FUNCTIONAL_MD_REF_FIELDGRADIENT_MAP(cls, reftype, grade, mapfn, clonefn, freefn) \
    FUNCTIONAL_MD_REF_FIELDGRADIENT_MAP_COST(cls, reftype, grade, mapfn, clonefn, freefn, FUNCTIONAL_COST_REGULAR)

#define FUNCTIONAL_MD_REF_FIELDGRADIENT_COST(cls, reftype, grade, clonefn, freefn, wkld) \
    FUNCTIONAL_MD_REF_FIELDGRADIENT_MAP_COST(cls, reftype, grade, functional_mapnumericalfieldgradient, clonefn, freefn, wkld)

#define FUNCTIONAL_MD_REF_FIELDGRADIENT(cls, reftype, grade, clonefn, freefn) \
    FUNCTIONAL_MD_REF_FIELDGRADIENT_COST(cls, reftype, grade, clonefn, freefn, FUNCTIONAL_COST_REGULAR)

#define FUNCTIONAL_MD_REF_ANALYTICALFIELDGRADIENT_COST(cls, reftype, grade, fieldgradfn, wkld) \
static value _##cls##_fieldgradient(vm *v, objectinstance *self, functional_mapinfo *info) { \
    reftype ref; \
    if (!_##cls##_bindref(v, self, info, &ref)) return MORPHO_NIL; \
    info->cost = (wkld); \
    return _functional_fieldgradient(v, info, grade, (ref.field==info->field ? (fieldgradfn) : NULL)); \
} \
    FUNCTIONAL_MD_REF_FIELD_OVERLOADS(cls, fieldgradient, _##cls##_fieldgradient)

#define FUNCTIONAL_MD_REF_ANALYTICALFIELDGRADIENT(cls, reftype, grade, fieldgradfn) \
    FUNCTIONAL_MD_REF_ANALYTICALFIELDGRADIENT_COST(cls, reftype, grade, fieldgradfn, FUNCTIONAL_COST_REGULAR)

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
