/** @file discretization.c
 *  @author T J Atherton
 *
 *  @brief Finite element discretizations
 */

#include "geometry.h"

/* **********************************************************************
 * Discretization objects
 * ********************************************************************** */

objecttype objectdiscretizationtype;

/** Field object definitions */
void objectdiscretization_printfn(object *obj, void *v) {
    objectdiscretization *disc=(objectdiscretization *) obj;
    morpho_printf(v, "<FunctionSpace %s>", disc->discretization->name);
}

size_t objectdiscretization_sizefn(object *obj) {
    return sizeof(objectdiscretization);
}

objecttypedefn objectdiscretizationdefn = {
    .printfn=objectdiscretization_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectdiscretization_sizefn
};

/* **********************************************************************
 * Discretization definitions
 * ********************************************************************** */

#define LINE_OPCODE 1
#define AREA_OPCODE 2
#define QUANTITY_OPCODE 255

#define LINE(id, v1, v2)      1, id, v1, v2           // Identify a grade 1 subelement given by two vertex indices
#define AREA(id, v1, v2, v3)  2, id, v1, v2, v3       // Identify a grade 2 subelement given by three vertex indices
#define QUANTITY(grade, id, qno) 128, grade, id, qno  // Fetch quantity from subelement of grade with id and quantity number
#define ENDDEFN -1

/* -------------------------------------------------------
 * CG1 element in 1D
 * ------------------------------------------------------- */

/*
 *   0 - 1    // One degree of freedom per vertex
 */

void cg1_1dinterpolate(double *lambda, double *wts) {
    wts[0]=lambda[1];
    wts[1]=lambda[0];
}

int cg1_1dshape[] = { 1, 0 };

double cg1_1dnodes[] = { 0.0, 1.0 };

eldefninstruction cg1_1ddefn[] = {
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    ENDDEFN
};

discretization cg1_1d = {
    .name = "CG1",
    .grade = 1,
    .shape = cg1_1dshape,
    .degree = 1,
    .nnodes = 2,
    .nsubel = 0,
    .nodes = cg1_1dnodes,
    .ifn = cg1_1dinterpolate,
    .eldefn = cg1_1ddefn
};

/* -------------------------------------------------------
 * CG2 element in 1D
 * ------------------------------------------------------- */

/*
 *   0 - 2 - 1    // One degree of freedom per vertex; one at the midpoint
 */

void cg2_1dinterpolate(double *lambda, double *wts) {
    double dl = (lambda[1]-lambda[0]);
    wts[0]=lambda[1]*dl;
    wts[1]=4*lambda[0]*lambda[1];
    wts[2]=-lambda[0]*dl;
}

int cg2_1dshape[] = { 1, 1 };

double cg2_1dnodes[] = { 0.0, 1.0, 0.5 };

eldefninstruction cg2_1ddefn[] = {
    LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(1,0,0), // Fetch quantity from line subelement
    ENDDEFN
};

discretization cg2_1d = {
    .name = "CG2",
    .grade = 1,
    .shape = cg2_1dshape,
    .degree = 2,
    .nnodes = 3,
    .ifn = cg2_1dinterpolate,
    .eldefn = cg2_1ddefn
};

/* -------------------------------------------------------
 * CG2 element in 2D
 * ------------------------------------------------------- */

/*   2
 *   |\
 *   5 4
 *   |  \
 *   0-3-1    // One degree of freedom per vertex; one at the midpoint
 */

void cg2_2dinterpolate(double *lambda, double *wts) {
    wts[0]=lambda[2]*(2*lambda[2]-1);
    wts[1]=lambda[0]*(2*lambda[0]-1);
    wts[2]=lambda[1]*(2*lambda[1]-1);
    wts[3]=4*lambda[0]*lambda[2];
    wts[4]=4*lambda[0]*lambda[1];
    wts[5]=4*lambda[1]*lambda[2];
}

int cg2_2dshape[] = { 1, 1, 0 };

double cg2_2dnodes[] = { 0.0, 0.0,
                         1.0, 0.0,
                         0.0, 1.0,
                         0.5, 0.0,
                         0.5, 0.5,
                         0.0, 0.5 };

eldefninstruction cg2_2deldefn[] = {
    LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    LINE(1,1,2),     // Identify line subelement with vertex indices (1,2)
    LINE(2,2,0),     // Identify line subelement with vertex indices (2,0)
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(0,2,0), // Fetch quantity on vertex 2
    QUANTITY(1,0,0), // Fetch quantity from line 0
    QUANTITY(1,1,0), // Fetch quantity from line 1
    QUANTITY(1,2,0), // Fetch quantity from line 2
    ENDDEFN
};

discretization cg2_2d = {
    .name = "CG2",
    .grade = 2,
    .shape = cg2_1dshape,
    .degree = 2,
    .nnodes = 6,
    .nodes = cg2_2dnodes,
    .ifn = cg2_2dinterpolate,
    .eldefn = cg2_2deldefn
};

discretization *discretizations[] = {
    &cg1_1d,
    &cg2_1d,
    &cg2_2d,
    NULL
};

/* **********************************************************************
 * Match DOFs to node numbers
 * ********************************************************************** */

void discretization_process(objectmesh *mesh, discretization *disc) {
    elementid subel[disc->nsubel+1];
    
    for (eldefninstruction *instr=disc->eldefn; instr!=NULL && *instr!=ENDDEFN; ) {
        switch(*instr) {
            case LINE_OPCODE:
                instr+=4;
                break;
            case AREA_OPCODE:
                instr+=5;
                break;
            case QUANTITY_OPCODE:
                instr+=4;
                break;
            default:
                UNREACHABLE("Error in finite element definition");
        }
    }
}

/* **********************************************************************
 * FunctionSpace class
 * ********************************************************************** */

/** Constructs a functionspace object */
value functionspace_constructor(vm *v, int nargs, value *args) {
    
}

MORPHO_BEGINCLASS(FunctionSpace)
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, NULL, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void discretization_initialize(void) {
    objectdiscretizationtype=object_addtype(&objectdiscretizationdefn);
    
    builtin_addfunction(FUNCTIONSPACE_CLASSNAME, functionspace_constructor, BUILTIN_FLAGSEMPTY);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    value functionspaceclass=builtin_addclass(FUNCTIONSPACE_CLASSNAME, MORPHO_GETCLASSDEFINITION(FunctionSpace), objclass);
    object_setveneerclass(OBJECT_DISCRETIZATION, functionspaceclass);
}
