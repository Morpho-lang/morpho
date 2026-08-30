/** @file nematic.c
 *  @author T J Atherton
 *
 *  @brief Nematic and NematicElectric functionals
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "functional.h"
#include "morpho.h"
#include "classes.h"
#include "common.h"
#include "linalg.h"
#include "sparse.h"
#include "geometry.h"
#include <math.h>
#include "gradsq.h"
#include "nematic.h"

/* ----------------------------------------------
 * Nematic
 * ---------------------------------------------- */

static value nematic_ksplayproperty;
static value nematic_ktwistproperty;
static value nematic_kbendproperty;
static value nematic_pitchproperty;

typedef struct {
    double ksplay,ktwist,kbend,pitch;
    bool haspitch;
    objectfield *field;
    grade grade;
} nematicref;

static bool nematic_startfn(vm *v, functional_mapinfo *info) {
    nematicref *ref = (nematicref *) info->ref;
    return functional_preparefespacefield(v, ref->field, info->g);
}

/** Prepares the nematic reference */
bool nematic_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, nematicref *ref) {
    bool success=false, grdset=false;
    value field=MORPHO_NIL, grd=MORPHO_NIL;
    value val=MORPHO_NIL;
    ref->ksplay=1.0; ref->ktwist=1.0; ref->kbend=1.0; ref->pitch=0.0;
    ref->haspitch=false;

    if (objectinstance_getpropertyinterned(self, functional_fieldproperty, &field) &&
        MORPHO_ISFIELD(field)) {
        ref->field=MORPHO_GETFIELD(field);
        success=true;
    }
    if (objectinstance_getpropertyinterned(self, nematic_ksplayproperty, &val) && MORPHO_ISNUMBER(val)) {
        morpho_valuetofloat(val, &ref->ksplay);
    }
    if (objectinstance_getpropertyinterned(self, nematic_ktwistproperty, &val) && MORPHO_ISNUMBER(val)) {
        morpho_valuetofloat(val, &ref->ktwist);
    }
    if (objectinstance_getpropertyinterned(self, nematic_kbendproperty, &val) && MORPHO_ISNUMBER(val)) {
        morpho_valuetofloat(val, &ref->kbend);
    }
    if (objectinstance_getpropertyinterned(self, nematic_pitchproperty, &val) && MORPHO_ISNUMBER(val)) {
        morpho_valuetofloat(val, &ref->pitch);
        ref->haspitch=true;
    }

    if (objectinstance_getpropertyinterned(self, functional_gradeproperty, &grd) &&
        MORPHO_ISINTEGER(grd)) {
        ref->grade=MORPHO_GETINTEGERVALUE(grd);
        if (ref->grade>0) grdset=true;
    }
    if (!grdset) ref->grade=mesh_maxgrade(mesh);

    return success;
}

/** Clones the nematic reference with a given substitute field */
void *nematic_cloneref(void *ref, objectfield *field, objectfield *sub) {
    nematicref *nref = (nematicref *) ref;
    nematicref *clone = MORPHO_MALLOC(sizeof(nematicref));
    
    if (clone) {
        *clone = *nref;
        if (clone->field==field) clone->field=sub;
    }
    
    return clone;
}

/* Integrates two linear functions with values at vertices f[0]...f[2] and g[0]...g[2] */
double nematic_bcint(double *f, double *g) {
    return (f[0]*(2*g[0]+g[1]+g[2]) + f[1]*(g[0]+2*g[1]+g[2]) + f[2]*(g[0]+g[1]+2*g[2]))/12;
}

/* Integrates a linear vector function with values at vertices f[0]...f[2] */
double nematic_bcint1(double *f) {
    return (f[0] + f[1] + f[2])/3;
}

/* Integrates a linear vector function with values at vertices f[0]...f[n]
   Works for dimensions 1-3 at least */
double nematic_bcintf(unsigned int n, double *f) {
    double sum = 0;
    for (unsigned int i=0; i<n; i++) sum+=f[i];
    return sum/n;
}

/* Integrates a product of two linear functions with values at vertices
   f[0]...f[n] and g[0]...g[n].
   Works for dimensions 1-3 at least */
double nematic_bcintfg(unsigned int n, double *f, double *g) {
    double sum = 0;
    for (unsigned int i=0; i<n; i++) {
        for (unsigned int j=0; j<n; j++) sum+=f[i]*g[j];
        sum+=f[i]*g[i];
    }
    return sum/(n*(n+1));
}

/** Calculate the nematic energy */
bool nematic_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    nematicref *eref = ref;
    double size=0; // Length area or volume of the element
    double gradnnraw[eref->field->psize*3];
    double gradnn[eref->field->psize*3];
    double divnn, curlnn[3] = { 0.0, 0.0, 0.0 };
    
    for (int i=0; i<eref->field->psize*3; i++) { gradnn[i]=0.0; gradnnraw[i]=0.0; }

    if (!functional_elementsize(v, mesh, eref->grade, id, nv, vid, &size)) return false;

    // Get nematic director components
    double *nn[nv]; // Field value lists
    unsigned int nentries=0;
    for (unsigned int i=0; i<nv; i++) {
        if (!field_getelementaslist(eref->field, MESH_GRADE_VERTEX, vid[i], 0, &nentries, &nn[i])) return false;
    }

    // Evaluate gradients of the director
    if (eref->grade==2) {
        if (!gradsq_evaluategradient(mesh, eref->field, nv, vid, gradnnraw)) return
            false;
    } else if (eref->grade==3) {
        if (!gradsq_evaluategradient3d(mesh, eref->field, nv, vid, gradnnraw)) return
            false;
    }
    
    // Copy into 3x3 matrix
    for (int j=0; j<3; j++) for (int i=0; i<mesh->dim; i++) gradnn[3*j+i] = gradnnraw[mesh->dim*j+i];
    
    // Output of this is the matrix:
    // [ nx,x ny,x nz,x ] [ 0 3 6 ] <- indices
    // [ nx,y ny,y nz,y ] [ 1 4 7 ]
    // [ nx,z ny,z nz,z ] [ 2 5 8 ]
    objectmatrix gradnnmat = MORPHO_STATICMATRIX(gradnn, 3, 3);

    matrix_trace(&gradnnmat, &divnn);
    curlnn[0]=gradnn[7]-gradnn[5]; // nz,y - ny,z
    curlnn[1]=gradnn[2]-gradnn[6]; // nx,z - nz,x
    curlnn[2]=gradnn[3]-gradnn[1]; // ny,x - nx,y

    /* From components of the curl, construct the coefficients that go in front of integrals of
           nx^2, ny^2, nz^2, nx*ny, ny*nz, and nz*nx over the element. */
    double ctwst[6] = { curlnn[0]*curlnn[0], curlnn[1]*curlnn[1], curlnn[2]*curlnn[2],
                        2*curlnn[0]*curlnn[1], 2*curlnn[1]*curlnn[2], 2*curlnn[2]*curlnn[0]};

    double cbnd[6] = { ctwst[1] + ctwst[2], ctwst[0] + ctwst[2], ctwst[0] + ctwst[1],
                       -ctwst[3], -ctwst[4], -ctwst[5] };

    /* Calculate integrals of nx^2, ny^2, nz^2, nx*ny, ny*nz, and nz*nx over the element */
    double nnt[3][nv]; // The transpose of nn
    for (unsigned int i=0; i<nv; i++)
        for (unsigned int j=0; j<3; j++) nnt[j][i]=nn[i][j];

    double integrals[] = {  nematic_bcintfg(nv, nnt[0], nnt[0]),
                            nematic_bcintfg(nv, nnt[1], nnt[1]),
                            nematic_bcintfg(nv, nnt[2], nnt[2]),
                            nematic_bcintfg(nv, nnt[0], nnt[1]),
                            nematic_bcintfg(nv, nnt[1], nnt[2]),
                            nematic_bcintfg(nv, nnt[2], nnt[0])
    };

    /* Now we can calculate the components of splay, twist and bend */
    double splay=0.0, twist=0.0, bend=0.0, chol=0.0;

    /* Evaluate the three contributions to the integral */
    splay = 0.5*eref->ksplay*size*divnn*divnn;
    for (unsigned int i=0; i<6; i++) {
        twist += ctwst[i]*integrals[i];
        bend += cbnd[i]*integrals[i];
    }
    twist *= 0.5*eref->ktwist*size;
    bend *= 0.5*eref->kbend*size;

    if (eref->haspitch) {
        /* Cholesteric terms: 0.5 * k22 * [- 2 q (cx <nx> + cy <ny> + cz <nz>) + q^2] */
        for (unsigned i=0; i<3; i++) {
            chol += -2*curlnn[i]*nematic_bcintf(nv, nnt[i])*eref->pitch;
        }
        chol += (eref->pitch*eref->pitch);
        chol *= 0.5*eref->ktwist*size;
    }

    *out = splay+twist+bend+chol;

    return true;
}

value Nematic_init__field(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    value ksplay=MORPHO_FLOAT(1.0),
          ktwist=MORPHO_FLOAT(1.0),
          kbend=MORPHO_FLOAT(1.0);
    value pitch=MORPHO_NIL;

    builtin_options(v, nargs, args, NULL, 4,
                    nematic_ksplayproperty, &ksplay,
                    nematic_ktwistproperty, &ktwist,
                    nematic_kbendproperty, &kbend,
                    nematic_pitchproperty, &pitch);

    objectinstance_setproperty(self, nematic_ksplayproperty, ksplay);
    objectinstance_setproperty(self, nematic_ktwistproperty, ktwist);
    objectinstance_setproperty(self, nematic_kbendproperty, kbend);
    objectinstance_setproperty(self, nematic_pitchproperty, pitch);
    _gradsq_initfield(self, MORPHO_GETARG(args, 0));
    return MORPHO_NIL;
}

FUNCTIONAL_MD_REF_BIND_START(Nematic, nematicref, nematic_prepareref, nematic_integrand, FUNCTIONAL_ARGS, nematic_startfn)
FUNCTIONAL_MD_REF_INTEGRAND(Nematic, nematicref, ref.grade)
FUNCTIONAL_MD_REF_TOTAL(Nematic, nematicref, ref.grade)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(Nematic, nematicref, ref.grade, NULL, SYMMETRY_NONE)
FUNCTIONAL_MD_REF_FIELDGRADIENT(Nematic, nematicref, ref.grade, nematic_cloneref, NULL)

MORPHO_BEGINCLASS(Nematic)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Field)", Nematic_init__field, MORPHO_FN_MUTATES|MORPHO_FN_OPTARGS),

FUNCTIONAL_MD_INTEGRAND_METHODS(Nematic),
FUNCTIONAL_MD_TOTAL_METHODS(Nematic),
FUNCTIONAL_MD_GRADIENT_METHODS(Nematic),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS(Nematic)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * NematicElectric
 * ---------------------------------------------- */

typedef struct {
    objectfield *director;
    value field;
    grade grade;
} nematicelectricref;

static bool nematicelectric_startfn(vm *v, functional_mapinfo *info) {
    nematicelectricref *ref = (nematicelectricref *) info->ref;

    if (!functional_preparefespacefield(v, ref->director, info->g)) return false;
    if (MORPHO_ISFIELD(ref->field) &&
        !functional_preparefespacefield(v, MORPHO_GETFIELD(ref->field), info->g)) return false;

    return true;
}

/** Prepares the nematicelectric reference */
bool nematicelectric_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, nematicelectricref *ref) {
    bool success=false, grdset=false;
    ref->field=MORPHO_NIL;
    value fieldlist=MORPHO_NIL, grd=MORPHO_NIL;

    if (objectinstance_getpropertyinterned(self, functional_fieldproperty, &fieldlist) &&
        MORPHO_ISLIST(fieldlist)) {
        objectlist *lst = MORPHO_GETLIST(fieldlist);
        value director = MORPHO_NIL;
        list_getelement(lst, 0, &director);
        list_getelement(lst, 1, &ref->field);

        if (MORPHO_ISFIELD(director)) ref->director=MORPHO_GETFIELD(director);

        if (MORPHO_ISFIELD(ref->field) || MORPHO_ISMATRIX(ref->field)) success=true;
    }

    if (objectinstance_getpropertyinterned(self, functional_gradeproperty, &grd) &&
        MORPHO_ISINTEGER(grd)) {
        ref->grade=MORPHO_GETINTEGERVALUE(grd);
        if (ref->grade>0) grdset=true;
    }
    if (!grdset) ref->grade=mesh_maxgrade(mesh);

    return success;
}

/** Clones the nematic reference with a given substitute field */
void *nematicelectric_cloneref(void *ref, objectfield *field, objectfield *sub) {
    nematicelectricref *nref = (nematicelectricref *) ref;
    nematicelectricref *clone = MORPHO_MALLOC(sizeof(nematicelectricref));
    
    if (clone) {
        *clone = *nref;
        if (clone->director==field) clone->director=sub;
        if (MORPHO_ISFIELD(clone->field) &&
            MORPHO_GETFIELD(clone->field)==field) {
            clone->field=MORPHO_OBJECT(sub);
        }
    }
    
    return clone;
}

/** Calculate the integral (n.E)^2 energy, where E is calculated from the electric potential */
bool nematicelectric_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    nematicelectricref *eref = ref;
    double size=0; // Length area or volume of the element

    if (!functional_elementsize(v, mesh, eref->grade, id, nv, vid, &size)) return false;

    // Get nematic director components
    double *nn[nv]; // Field value lists
    unsigned int nentries=0;
    for (unsigned int i=0; i<nv; i++) {
        if (!field_getelementaslist(eref->director, MESH_GRADE_VERTEX, vid[i], 0, &nentries, &nn[i])) return false;
    }

    // The electric field ends up being constant over the element
    double ee[mesh->dim];
    if (MORPHO_ISFIELD(eref->field)) {
        if (eref->grade==2) {
            if (!gradsq_evaluategradient(mesh, MORPHO_GETFIELD(eref->field), nv, vid, ee)) return false;
        } else if (eref->grade==3) {
            if (!gradsq_evaluategradient3d(mesh, MORPHO_GETFIELD(eref->field), nv, vid, ee)) return false;
        }
    }

    /* Calculate integrals of nx^2, ny^2, nz^2, nx*ny, ny*nz, and nz*nx over the element */
    double nnt[mesh->dim][nv]; // The transpose of nn
    for (unsigned int i=0; i<nv; i++)
        for (unsigned int j=0; j<mesh->dim; j++) nnt[j][i]=nn[i][j];

    /* Calculate integral (n.e)^2 using the above results */
    double total = ee[0]*ee[0]*nematic_bcintfg(nv, nnt[0], nnt[0])+
                   ee[1]*ee[1]*nematic_bcintfg(nv, nnt[1], nnt[1])+
                   ee[2]*ee[2]*nematic_bcintfg(nv, nnt[2], nnt[2])+
                   2*ee[0]*ee[1]*nematic_bcintfg(nv, nnt[0], nnt[1])+
                   2*ee[1]*ee[2]*nematic_bcintfg(nv, nnt[1], nnt[2])+
                   2*ee[2]*ee[0]*nematic_bcintfg(nv, nnt[2], nnt[0]);

    *out = size*total;

    return true;
}

value NematicElectric_init__field_field(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    objectlist *new = object_newlist(2, &MORPHO_GETARG(args, 0));

    if (new) {
        objectinstance_setproperty(self, functional_fieldproperty, MORPHO_OBJECT(new));
        functional_setgrade(self, mesh_maxgrade(MORPHO_GETFIELD(MORPHO_GETARG(args, 0))->mesh));
    }

    return morpho_wrapandbind(v, (object *) new);
}

FUNCTIONAL_MD_REF_BIND_START(NematicElectric, nematicelectricref, nematicelectric_prepareref, nematicelectric_integrand, FUNCTIONAL_ARGS, nematicelectric_startfn)
FUNCTIONAL_MD_REF_INTEGRAND_COST(NematicElectric, nematicelectricref, ref.grade, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_REF_TOTAL_COST(NematicElectric, nematicelectricref, ref.grade, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(NematicElectric, nematicelectricref, ref.grade, NULL, SYMMETRY_NONE)
FUNCTIONAL_MD_REF_FIELDGRADIENT(NematicElectric, nematicelectricref, ref.grade, nematicelectric_cloneref, NULL)

MORPHO_BEGINCLASS(NematicElectric)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Field, Field)", NematicElectric_init__field_field, MORPHO_FN_MUTATES|MORPHO_FN_ALLOCATES),

FUNCTIONAL_MD_INTEGRAND_METHODS(NematicElectric),
FUNCTIONAL_MD_TOTAL_METHODS(NematicElectric),
FUNCTIONAL_MD_GRADIENT_METHODS(NematicElectric),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS(NematicElectric)
MORPHO_ENDCLASS

void nematic_initialize(void) {
    nematic_ksplayproperty=builtin_internsymbolascstring(NEMATIC_KSPLAY_PROPERTY);
    nematic_ktwistproperty=builtin_internsymbolascstring(NEMATIC_KTWIST_PROPERTY);
    nematic_kbendproperty=builtin_internsymbolascstring(NEMATIC_KBEND_PROPERTY);
    nematic_pitchproperty=builtin_internsymbolascstring(NEMATIC_PITCH_PROPERTY);

    objectstring objclassname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objclassname));

    builtin_addclass(NEMATIC_CLASSNAME, MORPHO_GETCLASSDEFINITION(Nematic), objclass);
    builtin_addclass(NEMATICELECTRIC_CLASSNAME, MORPHO_GETCLASSDEFINITION(NematicElectric), objclass);
}

#endif
