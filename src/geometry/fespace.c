/** @file fespace.c
 *  @author T J Atherton
 *
 *  @brief Finite element fespaces
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "geometry.h"

/* **********************************************************************
 * Discretization objects
 * ********************************************************************** */

objecttype objectfespacetype;

/** Field object definitions */
void objectfespace_printfn(object *obj, void *v) {
    objectfespace *disc=(objectfespace *) obj;
    morpho_printf(v, "<FunctionSpace %s>", FESPACE_NAME(disc->fespace));
}

size_t objectfespace_sizefn(object *obj) {
    return sizeof(objectfespace);
}

objecttypedefn objectfespacedefn = {
    .printfn=objectfespace_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectfespace_sizefn
};

/** Creates a new fespace object
 * @param[in] fespace - fespace definition to use */
objectfespace *objectfespace_new(fespace *disc) {
    objectfespace *new = (objectfespace *) object_new(sizeof(objectfespace), OBJECT_FESPACE);
    if (new) new->fespace=disc;
    
    return new;
}

/* **********************************************************************
 * Discretization definitions
 * ********************************************************************** */

#define LINE_OPCODE 1
#define AREA_OPCODE 2
#define QUANTITY_OPCODE 255

#define LINE(id, v1, v2)      LINE_OPCODE, id, v1, v2           // Identify a grade 1 subelement given by two vertex indices
#define AREA(id, v1, v2, v3)  AREA_OPCODE, id, v1, v2, v3       // Identify a grade 2 subelement given by three vertex indices
#define QUANTITY(grade, id, qno) QUANTITY_OPCODE, grade, id, qno  // Fetch quantity from subelement of grade with id and quantity number
#define ENDDEFN -1

/* -------------------------------------------------------
 * CG1 element in 1D
 * ------------------------------------------------------- */

/*
 *   0 - 1    // One degree of freedom per vertex
 */

void cg1_1dinterpolate(double *lambda, double *wts) {
    wts[0]=lambda[0];
    wts[1]=lambda[1];
}

void cg1_1dgrad(double *lambda, double *grad) {
    double g[] =
    { 1, 0,
      0, 1 };
    memcpy(grad, g, sizeof(g));
}

unsigned int cg1_1dshape[] = { 1, 0 };

double cg1_1dnodes[] = { 0.0, 1.0 };

eldefninstruction cg1_1ddefn[] = {
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    ENDDEFN
};

fespace cg1_1d = {
    .name = "CG1",
    .grade = 1,
    .shape = cg1_1dshape,
    .degree = 1,
    .nnodes = 2,
    .nsubel = 0,
    .nodes = cg1_1dnodes,
    .ifn = cg1_1dinterpolate,
    .gfn = cg1_1dgrad,
    .eldefn = cg1_1ddefn,
    .lower = NULL
};

/* -------------------------------------------------------
 * CG2 element in 1D
 * ------------------------------------------------------- */

/*
 *   0 - 2 - 1    // One degree of freedom per vertex; one at the midpoint
 */

void cg2_1dinterpolate(double *lambda, double *wts) {
    double dl = (lambda[0]-lambda[1]);
    wts[0]=lambda[0]*dl;
    wts[1]=-lambda[1]*dl;
    wts[2]=4*lambda[0]*lambda[1];
}

void cg2_1dgrad(double *lambda, double *grad) {
    // Gij = d Xi[i] / d lambda[j]
    // Note this is in column-major order!
    double g[] =
    { 2*lambda[0]-lambda[1],            -lambda[1], 4*lambda[1],
                 -lambda[0], 2*lambda[1]-lambda[0], 4*lambda[0] };
    memcpy(grad, g, sizeof(g));
}


unsigned int cg2_1dshape[] = { 1, 1 };

double cg2_1dnodes[] = { 0.0, 1.0, 0.5 };

eldefninstruction cg2_1ddefn[] = {
    LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(1,0,0), // Fetch quantity from line subelement
    ENDDEFN
};

fespace cg2_1d = {
    .name = "CG2",
    .grade = 1,
    .shape = cg2_1dshape,
    .degree = 2,
    .nnodes = 3,
    .nsubel = 1,
    .nodes = cg2_1dnodes,
    .ifn = cg2_1dinterpolate,
    .gfn = cg2_1dgrad,
    .eldefn = cg2_1ddefn,
    .lower = NULL
};

/* -------------------------------------------------------
 * CG3 element in 1D
 * ------------------------------------------------------- */

/*
 *   0 - 2 - 3 - 1    // One degree of freedom per vertex; two on the line
 */

void cg3_1dinterpolate(double *lambda, double *wts) {
    double a = 4.5*lambda[0]*lambda[1];
    wts[0]=lambda[0]*(1-a);
    wts[1]=lambda[1]*(1-a);
    wts[2]=a*(2*lambda[0]-lambda[1]);
    wts[3]=a*(2*lambda[1]-lambda[0]);
}

void cg3_1dgrad(double *lambda, double *grad) {
    // Gij = d Xi[i] / d lambda[j]
    // Note this is in column-major order!
    double g[] =
    { 1-9*lambda[0]*lambda[1], -4.5*lambda[1]*lambda[1],
        4.5*(4*lambda[0]-lambda[1])*lambda[1], 9*lambda[1]*(lambda[1]-lambda[0]),
        
      -4.5*lambda[0]*lambda[0], 1-9*lambda[0]*lambda[1],
        9*lambda[0]*(lambda[0]-lambda[1]), 4.5*(4*lambda[1]-lambda[0])*lambda[0]
    };
    memcpy(grad, g, sizeof(g));
}

void cg3_1dhess(double *lambda, double *hess) {
    double x = lambda[1];

    #define H(node) hess[FESPACE_HESS_INDEX(4, 1, 0, 0, node)]
    H(0) = 18 - 27*x;
    H(1) = 27*x - 9;
    H(2) = -45 + 81*x;
    H(3) = 36 - 81*x;
    #undef H
}

unsigned int cg3_1dshape[] = { 1, 2 };

double cg3_1dnodes[] = { 0.0, 1.0, 1.0/3.0, 2.0/3.0 };

eldefninstruction cg3_1ddefn[] = {
    LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(1,0,0), // Fetch quantity from line subelement
    QUANTITY(1,0,1), // Fetch quantity from line subelement
    ENDDEFN
};

fespace cg3_1d = {
    .name = "CG3",
    .grade = 1,
    .shape = cg3_1dshape,
    .degree = 3,
    .nnodes = 4,
    .nsubel = 1,
    .nodes = cg3_1dnodes,
    .ifn = cg3_1dinterpolate,
    .gfn = cg3_1dgrad,
    .hfn = cg3_1dhess,
    .eldefn = cg3_1ddefn,
    .lower = NULL
};

/* -------------------------------------------------------
 * CG1 element in 2D
 * ------------------------------------------------------- */

/*   2
 *   |\
 *   0-1    // One degree of freedom per vertex
 */

void cg1_2dinterpolate(double *lambda, double *wts) {
    wts[0]=lambda[0];
    wts[1]=lambda[1];
    wts[2]=lambda[2];
}

void cg1_2dgrad(double *lambda, double *grad) {
    double g[] =
    { 1, 0, 0,
      0, 1, 0,
      0, 0, 1 };
    memcpy(grad, g, sizeof(g));
}

unsigned int cg1_2dshape[] = { 1, 0, 0 };

double cg1_2dnodes[] = { 0.0, 0.0,
                         1.0, 0.0,
                         0.0, 1.0 };

eldefninstruction cg1_2deldefn[] = {
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(0,2,0), // Fetch quantity on vertex 2
    ENDDEFN
};

fespace *cg1_2d_lower[] = {
    &cg1_1d,
    NULL
};

fespace cg1_2d = {
    .name = "CG1",
    .grade = 2,
    .shape = cg1_2dshape,
    .degree = 1,
    .nnodes = 3,
    .nsubel = 0,
    .nodes = cg1_2dnodes,
    .ifn = cg1_2dinterpolate,
    .gfn = cg1_2dgrad,
    .eldefn = cg1_2deldefn,
    .lower = cg1_2d_lower
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
    wts[0]=lambda[0]*(2*lambda[0]-1);
    wts[1]=lambda[1]*(2*lambda[1]-1);
    wts[2]=lambda[2]*(2*lambda[2]-1);
    wts[3]=4*lambda[0]*lambda[1];
    wts[4]=4*lambda[1]*lambda[2];
    wts[5]=4*lambda[2]*lambda[0];
}

void cg2_2dgrad(double *lambda, double *grad) {
    // Gij = d Xi[i] / d lambda[j]
    // Note this is in column-major order!
    double g[] =
    { 4*lambda[0]-1,             0,             0, 4*lambda[1],           0, 4*lambda[2],
                  0, 4*lambda[1]-1,             0, 4*lambda[0], 4*lambda[2],           0,
                  0,             0, 4*lambda[2]-1,           0, 4*lambda[1], 4*lambda[0] };
    memcpy(grad, g, sizeof(g));
}

void cg2_2dhess(double *lambda, double *hess) {
    // Hijq = d^2 Xi[i] / d x[j] d x[q] in column-major order
    #define H(row,col,node) hess[FESPACE_HESS_INDEX(6, 2, row, col, node)]
    H(0,0,0)=4;  H(0,0,1)=4;  H(0,0,2)=0;  H(0,0,3)=-8; H(0,0,4)=0;  H(0,0,5)=0;
    H(1,0,0)=4;  H(1,0,1)=0;  H(1,0,2)=0;  H(1,0,3)=-4; H(1,0,4)=4;  H(1,0,5)=-4;
    H(0,1,0)=4;  H(0,1,1)=0;  H(0,1,2)=0;  H(0,1,3)=-4; H(0,1,4)=4;  H(0,1,5)=-4;
    H(1,1,0)=4;  H(1,1,1)=0;  H(1,1,2)=4;  H(1,1,3)=0;  H(1,1,4)=0;  H(1,1,5)=-8;
    #undef H
}

unsigned int cg2_2dshape[] = { 1, 1, 0 };

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

fespace *cg2_2d_lower[] = {
    &cg2_1d,
    NULL
};

fespace cg2_2d = {
    .name = "CG2",
    .grade = 2,
    .shape = cg2_2dshape,
    .degree = 2,
    .nnodes = 6,
    .nsubel = 3,
    .nodes = cg2_2dnodes,
    .ifn = cg2_2dinterpolate,
    .gfn = cg2_2dgrad,
    .hfn = cg2_2dhess,
    .eldefn = cg2_2deldefn,
    .lower = cg2_2d_lower
};

/* -------------------------------------------------------
 * CG3 element in 2D
 * ------------------------------------------------------- */

/*   2
 *   | \
 *   7  6
 *   |   \
 *   8 10 5
 *   |     \
 *   0-3--4-1
 */

void cg3_2dinterpolate(double *lambda, double *wts) {
    wts[0]=lambda[0]*(1.0 + 4.5*(lambda[0]-1)*lambda[0]);
    wts[1]=lambda[1]*(1.0 + 4.5*(lambda[1]-1)*lambda[1]);
    wts[2]=lambda[2]*(1.0 + 4.5*(lambda[2]-1)*lambda[2]);
    wts[3]=4.5*lambda[0]*lambda[1]*(3*lambda[0]-1.0);
    wts[4]=4.5*lambda[0]*lambda[1]*(3*lambda[1]-1.0);
    wts[5]=4.5*lambda[1]*lambda[2]*(3*lambda[1]-1.0);
    wts[6]=4.5*lambda[1]*lambda[2]*(3*lambda[2]-1.0);
    wts[7]=4.5*lambda[0]*lambda[2]*(3*lambda[2]-1.0);
    wts[8]=4.5*lambda[0]*lambda[2]*(3*lambda[0]-1.0);
    wts[9]=27.0*lambda[0]*lambda[1]*lambda[2];
}

void cg3_2dgrad(double *lambda, double *grad) {
    // Gij = d Xi[i] / d lambda[j] in col. major order
    double g[] =
    { 1.0 + 4.5*lambda[0]*(3*lambda[0]-2), 0, 0,
        4.5*lambda[1]*(6*lambda[0]-1), 4.5*lambda[1]*(3*lambda[1]-1),
        0, 0,
        4.5*lambda[2]*(3*lambda[2]-1), 4.5*lambda[2]*(6*lambda[0]-1),
        27*lambda[1]*lambda[2],
        
      0, 1.0 + 4.5*lambda[1]*(3*lambda[1]-2), 0,
        4.5*lambda[0]*(3*lambda[0]-1), 4.5*lambda[0]*(6*lambda[1]-1),
        4.5*lambda[2]*(6*lambda[1]-1), 4.5*lambda[2]*(3*lambda[2]-1),
        0, 0,
        27*lambda[0]*lambda[2],
        
      0, 0, 1.0 + 4.5*lambda[2]*(3*lambda[2]-2),
        0, 0,
        4.5*lambda[1]*(3*lambda[1]-1), 4.5*lambda[1]*(6*lambda[2]-1),
        4.5*lambda[0]*(6*lambda[2]-1), 4.5*lambda[0]*(3*lambda[0]-1),
        27*lambda[0]*lambda[1],
    };
    memcpy(grad, g, sizeof(g));
}

void cg3_2dhess(double *lambda, double *hess) {
    double x = lambda[1];
    double y = lambda[2];

    // Hijq = d^2 Xi[i] / d x[j] d x[q] in column-major order
    #define H(row,col,node) hess[FESPACE_HESS_INDEX(10, 2, row, col, node)]
    H(0,0,0) = 18 - 27*x - 27*y;
    H(0,0,1) = 27*x - 9;
    H(0,0,2) = 0;
    H(0,0,3) = -45 + 81*x + 54*y;
    H(0,0,4) = 36 - 81*x - 27*y;
    H(0,0,5) = 27*y;
    H(0,0,6) = 0;
    H(0,0,7) = 0;
    H(0,0,8) = 27*y;
    H(0,0,9) = -54*y;

    H(1,0,0) = 18 - 27*x - 27*y;
    H(1,0,1) = 0;
    H(1,0,2) = 0;
    H(1,0,3) = -45.0/2 + 54*x + 27*y;
    H(1,0,4) = 9.0/2 - 27*x;
    H(1,0,5) = -9.0/2 + 27*x;
    H(1,0,6) = -9.0/2 + 27*y;
    H(1,0,7) = 9.0/2 - 27*y;
    H(1,0,8) = -45.0/2 + 27*x + 54*y;
    H(1,0,9) = 27 - 54*x - 54*y;

    H(0,1,0) = 18 - 27*x - 27*y;
    H(0,1,1) = 0;
    H(0,1,2) = 0;
    H(0,1,3) = -45.0/2 + 54*x + 27*y;
    H(0,1,4) = 9.0/2 - 27*x;
    H(0,1,5) = -9.0/2 + 27*x;
    H(0,1,6) = -9.0/2 + 27*y;
    H(0,1,7) = 9.0/2 - 27*y;
    H(0,1,8) = -45.0/2 + 27*x + 54*y;
    H(0,1,9) = 27 - 54*x - 54*y;

    H(1,1,0) = 18 - 27*x - 27*y;
    H(1,1,1) = 0;
    H(1,1,2) = 27*y - 9;
    H(1,1,3) = 27*x;
    H(1,1,4) = 0;
    H(1,1,5) = 0;
    H(1,1,6) = 27*x;
    H(1,1,7) = 36 - 27*x - 81*y;
    H(1,1,8) = -45 + 54*x + 81*y;
    H(1,1,9) = -54*x;
    #undef H
}

unsigned int cg3_2dshape[] = { 1, 2, 1 };

double cg3_2dnodes[] = { 0.0, 0.0,
                         1.0, 0.0,
                         0.0, 1.0,
                         0.3333333333333333,0.0,
                         0.6666666666666666,0.0,
                         0.6666666666666666,0.3333333333333333,
                         0.3333333333333333,0.6666666666666666,
                         0,0.6666666666666666,
                         0,0.3333333333333333,
                         0.3333333333333333,0.3333333333333333 };

eldefninstruction cg3_2deldefn[] = {
    LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    LINE(1,1,2),     // Identify line subelement with vertex indices (1,2)
    LINE(2,2,0),     // Identify line subelement with vertex indices (2,0)
    AREA(3,0,1,2),   // Identify area subelement with vertex indices (0,1,2)
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(0,2,0), // Fetch quantity on vertex 2
    QUANTITY(1,0,0), // Fetch quantity 0 from line 0
    QUANTITY(1,0,1), // Fetch quantity 1 from line 0
    QUANTITY(1,1,0), // Fetch quantity 0 from line 1
    QUANTITY(1,1,1), // Fetch quantity 1 from line 1
    QUANTITY(1,2,0), // Fetch quantity 0 from line 2
    QUANTITY(1,2,1), // Fetch quantity 1 from line 2
    QUANTITY(2,3,0), // Fetch quantity 0 from area 0
    ENDDEFN
};

fespace *cg3_2d_lower[] = {
    &cg3_1d,
    NULL
};

fespace cg3_2d = {
    .name = "CG3",
    .grade = 2,
    .shape = cg3_2dshape,
    .degree = 3,
    .nnodes = 10,
    .nsubel = 4,
    .nodes = cg3_2dnodes,
    .ifn = cg3_2dinterpolate,
    .gfn = cg3_2dgrad,
    .hfn = cg3_2dhess,
    .eldefn = cg3_2deldefn,
    .lower = cg3_2d_lower
};

/* -------------------------------------------------------
 * CG1 element in 3D
 * ------------------------------------------------------- */

/*   z=0    z=1
 *   2
 *   |\
 *   0-1    3 // One degree of freedom per vertex
 */

void cg1_3dinterpolate(double *lambda, double *wts) {
    wts[0]=lambda[0];
    wts[1]=lambda[1];
    wts[2]=lambda[2];
    wts[3]=lambda[3];
}

void cg1_3dgrad(double *lambda, double *grad) {
    double g[] =
    { 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1 };
    memcpy(grad, g, sizeof(g));
}

unsigned int cg1_3dshape[] = { 1, 0, 0, 0 };

double cg1_3dnodes[] = { 0.0, 0.0, 0.0,
                         1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0 };

eldefninstruction cg1_3deldefn[] = {
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(0,2,0), // Fetch quantity on vertex 2
    QUANTITY(0,3,0), // Fetch quantity on vertex 3
    ENDDEFN
};

fespace *cg1_3d_lower[] = {
    &cg1_2d,
    &cg1_1d,
    NULL
};

fespace cg1_3d = {
    .name = "CG1",
    .grade = 3,
    .shape = cg1_3dshape,
    .degree = 1,
    .nnodes = 4,
    .nsubel = 0,
    .nodes = cg1_3dnodes,
    .ifn = cg1_3dinterpolate,
    .gfn = cg1_3dgrad,
    .eldefn = cg1_3deldefn,
    .lower = cg1_3d_lower
};

/* -------------------------------------------------------
 * CG2 element in 3D
 * ------------------------------------------------------- */

/*   z=0       z=0.5     z=1
 *   2
 *   |\
 *   6 5       9
 *   |  \      | \
 *   0-4-1     7--8      3  - i.e. vertices
 */

void cg2_3dinterpolate(double *lambda, double *wts) {
    wts[0]=lambda[0]*(2*lambda[0]-1);
    wts[1]=lambda[1]*(2*lambda[1]-1);
    wts[2]=lambda[2]*(2*lambda[2]-1);
    wts[3]=lambda[3]*(2*lambda[3]-1);
    wts[4]=4*lambda[0]*lambda[1];
    wts[5]=4*lambda[1]*lambda[2];
    wts[6]=4*lambda[2]*lambda[0];
    wts[7]=4*lambda[0]*lambda[3];
    wts[8]=4*lambda[1]*lambda[3];
    wts[9]=4*lambda[2]*lambda[3];
}

void cg2_3dgrad(double *lambda, double *grad) { // TODO: FIX
    // Gij = d Xi[i] / d lambda[j]
    // Note this is in column-major order!
    double g[] =
    { 4*lambda[0]-1,             0,             0,             0,
        4*lambda[1],             0,   4*lambda[2],   4*lambda[3],            0,             0,
        
                  0, 4*lambda[1]-1,             0,             0,
        4*lambda[0],   4*lambda[2],             0,             0,  4*lambda[3],             0,
        
                  0,             0, 4*lambda[2]-1,             0,
                  0,   4*lambda[1],   4*lambda[0],             0,            0,   4*lambda[3],
        
                  0,             0,             0, 4*lambda[3]-1,
                  0,             0,             0,   4*lambda[0],  4*lambda[1],   4*lambda[2]
    };
    
    memcpy(grad, g, sizeof(g));
}

void cg2_3dhess(double *lambda, double *hess) {
    // Hijq = d^2 Xi[i] / d x[j] d x[q] in column-major tensor order
    #define H(row,col,node) hess[FESPACE_HESS_INDEX(10, 3, row, col, node)]
    H(0,0,0)=4;  H(0,0,1)=4;  H(0,0,2)=0;  H(0,0,3)=0;  H(0,0,4)=-8; H(0,0,5)=0;  H(0,0,6)=0;  H(0,0,7)=0;  H(0,0,8)=0; H(0,0,9)=0;
    H(1,0,0)=4;  H(1,0,1)=0;  H(1,0,2)=0;  H(1,0,3)=0;  H(1,0,4)=-4; H(1,0,5)=4;  H(1,0,6)=-4; H(1,0,7)=0;  H(1,0,8)=0; H(1,0,9)=0;
    H(2,0,0)=4;  H(2,0,1)=0;  H(2,0,2)=0;  H(2,0,3)=0;  H(2,0,4)=-4; H(2,0,5)=0;  H(2,0,6)=0;  H(2,0,7)=-4; H(2,0,8)=4; H(2,0,9)=0;

    H(0,1,0)=4;  H(0,1,1)=0;  H(0,1,2)=0;  H(0,1,3)=0;  H(0,1,4)=-4; H(0,1,5)=4;  H(0,1,6)=-4; H(0,1,7)=0;  H(0,1,8)=0; H(0,1,9)=0;
    H(1,1,0)=4;  H(1,1,1)=0;  H(1,1,2)=4;  H(1,1,3)=0;  H(1,1,4)=0;  H(1,1,5)=0;  H(1,1,6)=-8; H(1,1,7)=0;  H(1,1,8)=0; H(1,1,9)=0;
    H(2,1,0)=4;  H(2,1,1)=0;  H(2,1,2)=0;  H(2,1,3)=0;  H(2,1,4)=0;  H(2,1,5)=0;  H(2,1,6)=-4; H(2,1,7)=-4; H(2,1,8)=0; H(2,1,9)=4;

    H(0,2,0)=4;  H(0,2,1)=0;  H(0,2,2)=0;  H(0,2,3)=0;  H(0,2,4)=-4; H(0,2,5)=0;  H(0,2,6)=0;  H(0,2,7)=-4; H(0,2,8)=4; H(0,2,9)=0;
    H(1,2,0)=4;  H(1,2,1)=0;  H(1,2,2)=0;  H(1,2,3)=0;  H(1,2,4)=0;  H(1,2,5)=0;  H(1,2,6)=-4; H(1,2,7)=-4; H(1,2,8)=0; H(1,2,9)=4;
    H(2,2,0)=4;  H(2,2,1)=0;  H(2,2,2)=0;  H(2,2,3)=4;  H(2,2,4)=0;  H(2,2,5)=0;  H(2,2,6)=0;  H(2,2,7)=-8; H(2,2,8)=0; H(2,2,9)=0;
    #undef H
}

unsigned int cg2_3dshape[] = { 1, 1, 0, 0 };

double cg2_3dnodes[] = { 0,     0,   0,
                         1,     0,   0,
                         0,     1,   0,
                         0,     0,   1,
                         0.5,   0,   0,
                         0.5, 0.5,   0,
                         0,   0.5,   0,
                         0,     0, 0.5,
                         0.5,   0, 0.5,
                         0,   0.5, 0.5 };

eldefninstruction cg2_3deldefn[] = {
    LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    LINE(1,1,2),     // Identify line subelement with vertex indices (1,2)
    LINE(2,2,0),     // Identify line subelement with vertex indices (2,0)
    LINE(3,0,3),     // Identify line subelement with vertex indices (0,3)
    LINE(4,1,3),     // Identify line subelement with vertex indices (1,3)
    LINE(5,2,3),     // Identify line subelement with vertex indices (2,3)
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(0,2,0), // Fetch quantity on vertex 2
    QUANTITY(0,3,0), // Fetch quantity on vertex 3
    QUANTITY(1,0,0), // Fetch quantity from line 0
    QUANTITY(1,1,0), // Fetch quantity from line 1
    QUANTITY(1,2,0), // Fetch quantity from line 2
    QUANTITY(1,3,0), // Fetch quantity from line 3
    QUANTITY(1,4,0), // Fetch quantity from line 4
    QUANTITY(1,5,0), // Fetch quantity from line 5
    ENDDEFN
};

fespace *cg2_3d_lower[] = {
    &cg2_2d,
    &cg2_1d,
    NULL
};

fespace cg2_3d = {
    .name = "CG2",
    .grade = 3,
    .shape = cg2_3dshape,
    .degree = 2,
    .nnodes = 10,
    .nsubel = 6,
    .nodes = cg2_3dnodes,
    .ifn = cg2_3dinterpolate,
    .gfn = cg2_3dgrad,
    .hfn = cg2_3dhess,
    .eldefn = cg2_3deldefn,
    .lower = cg2_3d_lower
};

/* -------------------------------------------------------
 * CG3 element in 3D
 * ------------------------------------------------------- */

/*   z = 0 layer:           z = 1/3 layer:         z = 2/3 layer:         z = 1 layer:
 *
 *  2                         14
 *  | \                       .  .
 *  9   7                     .   .                    15
 *  |    \                    19   18                  . .
 *  8  16  6                  .     .                  .  .
 *  |       \                 .      .                 .   .
 *  0--4--5--1                10..17..12              11...13              3
 */

void cg3_3dinterpolate(double *lambda, double *wts) {
    int k=0;
    int edges[6][2] = { {0,1}, {1,2}, {2,0}, {0,3}, {1,3}, {2,3} };
    int faces[4][3] = { {0,1,2}, {0,3,1}, {1,3,2}, {2,3,0} };

    for (int i=0; i<4; i++) {
        wts[k++]=0.5*lambda[i]*(3*lambda[i]-1)*(3*lambda[i]-2);
    }

    for (int i=0; i<6; i++) {
        int a=edges[i][0], b=edges[i][1];
        wts[k++]=4.5*lambda[a]*lambda[b]*(3*lambda[a]-1);
        wts[k++]=4.5*lambda[a]*lambda[b]*(3*lambda[b]-1);
    }

    for (int i=0; i<4; i++) {
        int a=faces[i][0], b=faces[i][1], c=faces[i][2];
        wts[k++]=27.0*lambda[a]*lambda[b]*lambda[c];
    }
}

void cg3_3dgrad(double *lambda, double *grad) {
    int edges[6][2] = { {0,1}, {1,2}, {2,0}, {0,3}, {1,3}, {2,3} };
    int faces[4][3] = { {0,1,2}, {0,3,1}, {1,3,2}, {2,3,0} };

    memset(grad, 0, sizeof(double)*20*4);

    for (int j=0; j<4; j++) {
        grad[j*20+j]=13.5*lambda[j]*lambda[j]-9.0*lambda[j]+1.0;
    }

    int k=4;
    for (int i=0; i<6; i++) {
        int a=edges[i][0], b=edges[i][1];

        grad[a*20+k]=4.5*lambda[b]*(6*lambda[a]-1);
        grad[b*20+k]=4.5*lambda[a]*(3*lambda[a]-1);
        k++;

        grad[a*20+k]=4.5*lambda[b]*(3*lambda[b]-1);
        grad[b*20+k]=4.5*lambda[a]*(6*lambda[b]-1);
        k++;
    }

    for (int i=0; i<4; i++) {
        int a=faces[i][0], b=faces[i][1], c=faces[i][2];
        grad[a*20+k]=27.0*lambda[b]*lambda[c];
        grad[b*20+k]=27.0*lambda[a]*lambda[c];
        grad[c*20+k]=27.0*lambda[a]*lambda[b];
        k++;
    }
}

void cg3_3dhess(double *lambda, double *hess) {
    double x=lambda[1], y=lambda[2], z=lambda[3];

    #define H(row,col,node) hess[FESPACE_HESS_INDEX(20, 3, row, col, node)]
    H(0,0,0)=18-27*x-27*y-27*z; H(0,0,1)=27*x-9; H(0,0,2)=0; H(0,0,3)=0; H(0,0,4)=81*x+54*y+54*z-45; H(0,0,5)=-81*x-27*y-27*z+36; H(0,0,6)=27*y; H(0,0,7)=0; H(0,0,8)=0; H(0,0,9)=27*y; H(0,0,10)=27*z; H(0,0,11)=0; H(0,0,12)=27*z; H(0,0,13)=0; H(0,0,14)=0; H(0,0,15)=0; H(0,0,16)=-54*y; H(0,0,17)=-54*z; H(0,0,18)=0; H(0,0,19)=0;
    H(1,0,0)=18-27*x-27*y-27*z; H(1,0,1)=0; H(1,0,2)=0; H(1,0,3)=0; H(1,0,4)=54*x+27*y+27*z-22.5; H(1,0,5)=-27*x+4.5; H(1,0,6)=27*x-4.5; H(1,0,7)=27*y-4.5; H(1,0,8)=-27*y+4.5; H(1,0,9)=27*x+54*y+27*z-22.5; H(1,0,10)=27*z; H(1,0,11)=0; H(1,0,12)=0; H(1,0,13)=0; H(1,0,14)=0; H(1,0,15)=0; H(1,0,16)=-54*x-54*y-27*z+27; H(1,0,17)=-27*z; H(1,0,18)=27*z; H(1,0,19)=-27*z;
    H(2,0,0)=18-27*x-27*y-27*z; H(2,0,1)=0; H(2,0,2)=0; H(2,0,3)=0; H(2,0,4)=54*x+27*y+27*z-22.5; H(2,0,5)=-27*x+4.5; H(2,0,6)=0; H(2,0,7)=0; H(2,0,8)=0; H(2,0,9)=27*y; H(2,0,10)=27*x+27*y+54*z-22.5; H(2,0,11)=-27*z+4.5; H(2,0,12)=27*x-4.5; H(2,0,13)=27*z-4.5; H(2,0,14)=0; H(2,0,15)=0; H(2,0,16)=-27*y; H(2,0,17)=-54*x-27*y-54*z+27; H(2,0,18)=27*y; H(2,0,19)=-27*y;

    H(0,1,0)=18-27*x-27*y-27*z; H(0,1,1)=0; H(0,1,2)=0; H(0,1,3)=0; H(0,1,4)=54*x+27*y+27*z-22.5; H(0,1,5)=-27*x+4.5; H(0,1,6)=27*x-4.5; H(0,1,7)=27*y-4.5; H(0,1,8)=-27*y+4.5; H(0,1,9)=27*x+54*y+27*z-22.5; H(0,1,10)=27*z; H(0,1,11)=0; H(0,1,12)=0; H(0,1,13)=0; H(0,1,14)=0; H(0,1,15)=0; H(0,1,16)=-54*x-54*y-27*z+27; H(0,1,17)=-27*z; H(0,1,18)=27*z; H(0,1,19)=-27*z;
    H(1,1,0)=18-27*x-27*y-27*z; H(1,1,1)=0; H(1,1,2)=27*y-9; H(1,1,3)=0; H(1,1,4)=27*x; H(1,1,5)=0; H(1,1,6)=0; H(1,1,7)=27*x; H(1,1,8)=-27*x-81*y-27*z+36; H(1,1,9)=54*x+81*y+54*z-45; H(1,1,10)=27*z; H(1,1,11)=0; H(1,1,12)=0; H(1,1,13)=0; H(1,1,14)=27*z; H(1,1,15)=0; H(1,1,16)=-54*x; H(1,1,17)=0; H(1,1,18)=0; H(1,1,19)=-54*z;
    H(2,1,0)=18-27*x-27*y-27*z; H(2,1,1)=0; H(2,1,2)=0; H(2,1,3)=0; H(2,1,4)=27*x; H(2,1,5)=0; H(2,1,6)=0; H(2,1,7)=0; H(2,1,8)=-27*y+4.5; H(2,1,9)=27*x+54*y+27*z-22.5; H(2,1,10)=27*x+27*y+54*z-22.5; H(2,1,11)=-27*z+4.5; H(2,1,12)=0; H(2,1,13)=0; H(2,1,14)=27*y-4.5; H(2,1,15)=27*z-4.5; H(2,1,16)=-27*x; H(2,1,17)=-27*x; H(2,1,18)=27*x; H(2,1,19)=-27*x-54*y-54*z+27;

    H(0,2,0)=18-27*x-27*y-27*z; H(0,2,1)=0; H(0,2,2)=0; H(0,2,3)=0; H(0,2,4)=54*x+27*y+27*z-22.5; H(0,2,5)=-27*x+4.5; H(0,2,6)=0; H(0,2,7)=0; H(0,2,8)=0; H(0,2,9)=27*y; H(0,2,10)=27*x+27*y+54*z-22.5; H(0,2,11)=-27*z+4.5; H(0,2,12)=27*x-4.5; H(0,2,13)=27*z-4.5; H(0,2,14)=0; H(0,2,15)=0; H(0,2,16)=-27*y; H(0,2,17)=-54*x-27*y-54*z+27; H(0,2,18)=27*y; H(0,2,19)=-27*y;
    H(1,2,0)=18-27*x-27*y-27*z; H(1,2,1)=0; H(1,2,2)=0; H(1,2,3)=0; H(1,2,4)=27*x; H(1,2,5)=0; H(1,2,6)=0; H(1,2,7)=0; H(1,2,8)=-27*y+4.5; H(1,2,9)=27*x+54*y+27*z-22.5; H(1,2,10)=27*x+27*y+54*z-22.5; H(1,2,11)=-27*z+4.5; H(1,2,12)=0; H(1,2,13)=0; H(1,2,14)=27*y-4.5; H(1,2,15)=27*z-4.5; H(1,2,16)=-27*x; H(1,2,17)=-27*x; H(1,2,18)=27*x; H(1,2,19)=-27*x-54*y-54*z+27;
    H(2,2,0)=18-27*x-27*y-27*z; H(2,2,1)=0; H(2,2,2)=0; H(2,2,3)=27*z-9; H(2,2,4)=27*x; H(2,2,5)=0; H(2,2,6)=0; H(2,2,7)=0; H(2,2,8)=0; H(2,2,9)=27*y; H(2,2,10)=54*x+54*y+81*z-45; H(2,2,11)=-27*x-27*y-81*z+36; H(2,2,12)=0; H(2,2,13)=27*x; H(2,2,14)=0; H(2,2,15)=27*y; H(2,2,16)=0; H(2,2,17)=-54*x; H(2,2,18)=0; H(2,2,19)=-54*y;
    #undef H
}

unsigned int cg3_3dshape[] = { 1, 2, 1, 0 };

double cg3_3dnodes[] = {
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0/3.0, 0.0, 0.0,
    2.0/3.0, 0.0, 0.0,
    2.0/3.0, 1.0/3.0, 0.0,
    1.0/3.0, 2.0/3.0, 0.0,
    0.0, 2.0/3.0, 0.0,
    0.0, 1.0/3.0, 0.0,
    0.0, 0.0, 1.0/3.0,
    0.0, 0.0, 2.0/3.0,
    2.0/3.0, 0.0, 1.0/3.0,
    1.0/3.0, 0.0, 2.0/3.0,
    0.0, 2.0/3.0, 1.0/3.0,
    0.0, 1.0/3.0, 2.0/3.0,
    1.0/3.0, 1.0/3.0, 0.0,
    1.0/3.0, 0.0, 1.0/3.0,
    1.0/3.0, 1.0/3.0, 1.0/3.0,
    0.0, 1.0/3.0, 1.0/3.0
};

eldefninstruction cg3_3deldefn[] = {
    LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    LINE(1,1,2),     // Identify line subelement with vertex indices (1,2)
    LINE(2,2,0),     // Identify line subelement with vertex indices (2,0)
    LINE(3,0,3),     // Identify line subelement with vertex indices (0,3)
    LINE(4,1,3),     // Identify line subelement with vertex indices (1,3)
    LINE(5,2,3),     // Identify line subelement with vertex indices (2,3)
    AREA(6,0,1,2),   // Identify area subelement with vertex indices (0,1,2)
    AREA(7,0,3,1),   // Identify area subelement with vertex indices (0,3,1)
    AREA(8,1,3,2),   // Identify area subelement with vertex indices (1,3,2)
    AREA(9,2,3,0),   // Identify area subelement with vertex indices (2,3,0)
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(0,2,0), // Fetch quantity on vertex 2
    QUANTITY(0,3,0), // Fetch quantity on vertex 3
    QUANTITY(1,0,0), // Fetch quantity 0 from line 0
    QUANTITY(1,0,1), // Fetch quantity 1 from line 0
    QUANTITY(1,1,0), // Fetch quantity 0 from line 1
    QUANTITY(1,1,1), // Fetch quantity 1 from line 1
    QUANTITY(1,2,0), // Fetch quantity 0 from line 2
    QUANTITY(1,2,1), // Fetch quantity 1 from line 2
    QUANTITY(1,3,0), // Fetch quantity 0 from line 3
    QUANTITY(1,3,1), // Fetch quantity 1 from line 3
    QUANTITY(1,4,0), // Fetch quantity 0 from line 4
    QUANTITY(1,4,1), // Fetch quantity 1 from line 4
    QUANTITY(1,5,0), // Fetch quantity 0 from line 5
    QUANTITY(1,5,1), // Fetch quantity 1 from line 5
    QUANTITY(2,6,0), // Fetch quantity 0 from area 0
    QUANTITY(2,7,0), // Fetch quantity 0 from area 1
    QUANTITY(2,8,0), // Fetch quantity 0 from area 2
    QUANTITY(2,9,0), // Fetch quantity 0 from area 3
    ENDDEFN
};

fespace *cg3_3d_lower[] = {
    &cg3_2d,
    &cg3_1d,
    NULL
};

fespace cg3_3d = {
    .name = "CG3",
    .grade = 3,
    .shape = cg3_3dshape,
    .degree = 3,
    .nnodes = 20,
    .nsubel = 10,
    .nodes = cg3_3dnodes,
    .ifn = cg3_3dinterpolate,
    .gfn = cg3_3dgrad,
    .hfn = cg3_3dhess,
    .eldefn = cg3_3deldefn,
    .lower = cg3_3d_lower
};

/* -------------------------------------------------------
 * List of finite elements
 * ------------------------------------------------------- */

fespace *fespaces[] = {
    &cg1_1d,
    &cg2_1d,
    &cg3_1d,
    &cg1_2d,
    &cg2_2d,
    &cg3_2d,
    &cg1_3d,
    &cg2_3d,
    &cg3_3d,
    NULL
};

/* **********************************************************************
 * Discretization functions
 * ********************************************************************** */

/** Find a fespace definition based on a name and grade */
fespace *fespace_find(char *name, grade g) {
    for (int i=0; fespaces[i]!=NULL; i++) {
        if (strcmp(name, fespaces[i]->name)==0 &&
            g==fespaces[i]->grade) return fespaces[i];
    }
    return NULL;
}

/** Finds a linear fespace for a given grade */
fespace *fespace_findlinear(grade g) {
    for (int i=0; fespaces[i]!=NULL; i++) {
        if (fespaces[i]->grade && fespaces[i]->degree==1) return fespaces[i];
    }
    return NULL;
}

#define FETCH(instr) (*(instr++))

typedef struct {
    elementid id;
    bool reversed;
} fespacesubelement;

/** Remaps an edge-local quantity index if the matched line orientation is reversed. */
static int fespace_orientedquantity(fespace *disc, grade g, fespacesubelement *subel, int sid, int indx) {
    int stride = disc->nsubel+1;
    if (g!=MESH_GRADE_LINE || !subel[g*stride+sid].reversed) return indx;

    fespace *lower;
    if (!fespace_lower(disc, g, &lower)) return indx;

    int nedgeq = lower->shape[MESH_GRADE_LINE];
    if (indx<0 || indx>=nedgeq) return indx;

    return nedgeq-indx-1;
}

/** Returns the FE-local field index tuple (grade, subelement id, local dof index) for a node. */
bool fespace_nodefieldindex(fespace *disc, int node, grade *g, int *sid, int *indx) {
    if (node<0 || node>=disc->nnodes) return false;

    int k=0;
    for (eldefninstruction *instr=disc->eldefn; instr!=NULL && *instr!=ENDDEFN; ) {
        eldefninstruction op=FETCH(instr);
        switch(op) {
            case LINE_OPCODE:
            case AREA_OPCODE:
                FETCH(instr); // local subelement id
                for (int i=0; i<=op; i++) FETCH(instr); // local vertex ids
                break;
            case QUANTITY_OPCODE:
            {
                grade qg = FETCH(instr);
                int qsid = FETCH(instr);
                int qindx = FETCH(instr);
                if (k==node) {
                    if (g) *g=qg;
                    if (sid) *sid=qsid;
                    if (indx) *indx=qindx;
                    return true;
                }
                k++;
            }
                break;
            default:
                UNREACHABLE("Error in finite element definition");
        }
    }

    return false;
}

/** Steps through an element definition, generating subelements and identifying quantities */
bool fespace_doftofieldindx(objectfield *field, fespace *disc, int nv, int *vids, fieldindx *findx) {
    int stride = disc->nsubel+1;
    fespacesubelement subel[(disc->grade+1)*stride]; // Element IDs and orientation of subelements
    int sid, svids[nv], nmatch, k=0;
    
    objectsparse *vmatrix[disc->grade+1]; // Vertex->elementid connectivity matrices
    for (grade g=0; g<=disc->grade; g++) {
        vmatrix[g]=mesh_addconnectivityelement(field->mesh, g, 0);
        if (!vmatrix[g] && g>0 && disc->shape[g]>0) {
            mesh_addgrade(field->mesh, g);
            vmatrix[g]=mesh_addconnectivityelement(field->mesh, g, 0);
        }
    }
    objectsparse *lineconn = mesh_getconnectivityelement(field->mesh, 0, MESH_GRADE_LINE);
    
    for (eldefninstruction *instr=disc->eldefn; instr!=NULL && *instr!=ENDDEFN; ) {
        eldefninstruction op=FETCH(instr);
        switch(op) {
            case LINE_OPCODE: // Find an element defined by n vertices
            case AREA_OPCODE: // TODO: Need to cope with (mis) orientation of these subelements
            {
                sid = FETCH(instr);
                for (int i=0; i<=op; i++) svids[i] = vids[FETCH(instr)];
                
                fespacesubelement *matched = &subel[op*stride+sid];
                matched->id = -1;
                if (!mesh_matchelements(vmatrix[op], op, op+1, svids, 1, &nmatch, &matched->id)) return false;
                if (nmatch!=1 || matched->id<0) return false;

                matched->reversed=false;
                if (op==LINE_OPCODE) {
                    int nlinev, *linevids;
                    if (!lineconn || !mesh_getconnectivity(lineconn, matched->id, &nlinev, &linevids)) return false;
                    if (nlinev!=2) return false;

                    if (linevids[0]==svids[1] && linevids[1]==svids[0]) {
                        matched->reversed=true;
                    } else if (!(linevids[0]==svids[0] && linevids[1]==svids[1])) {
                        return false;
                    }
                }
            }
                break;
            case QUANTITY_OPCODE:
            {
                findx[k].g=FETCH(instr);
                int sid=FETCH(instr);
                findx[k].id=(findx[k].g==0 ? vids[sid]: subel[findx[k].g*stride+sid].id);
                findx[k].indx=fespace_orientedquantity(disc, findx[k].g, subel, sid, FETCH(instr));
                k++;
            }
                break;
            default:
                UNREACHABLE("Error in finite element definition");
        }
    }
    return true;
}

/** Searches a fespace's lower list to find a fespace to use on a lower grade */
bool fespace_lower(fespace *disc, grade target, fespace **out) {
    if (disc->lower) for (int i=0; disc->lower[i]!=NULL; i++) {
        if (disc->lower[i]->grade==target) {
            *out = disc->lower[i];
            return true;
        }
    }
    return false;
}

/** Returns the barycentric coordinates of a node in the reference element */
bool fespace_getnodecoords(fespace *disc, int node, double *lambda) {
    if (!disc || !lambda) return false;
    if (node<0 || node>=disc->nnodes) return false;

    double l0 = 1.0;

    for (int i=0; i<disc->grade; i++) {
        double li = disc->nodes[node*disc->grade+i];
        lambda[i+1] = li;
        l0 -= li;
    }

    lambda[0] = l0;

    return true;
}

/** Constructs a layout matrix that maps element ids (columns) to degree of freedom indices in a field */
bool fespace_layout(objectfield *field, fespace *disc, objectsparse **out) {
    objectsparse *conn = mesh_getconnectivityelement(field->mesh, 0, disc->grade);
    if (!conn) conn = mesh_addconnectivityelement(field->mesh, 0, disc->grade);
    elementid nel=mesh_nelements(conn);
    
    objectsparse *new = object_newsparse(NULL, NULL);
    if (!new) return false;
    sparseccs_resize(&new->ccs, field->nelements, nel, nel*disc->nnodes, NULL);
    
    for (elementid id=0; id<nel; id++) {
        int nv, *vids;
        if (!mesh_getconnectivity(conn, id, &nv, &vids)) goto fespace_layout_cleanup;
     
        new->ccs.cptr[id]=id*disc->nnodes;
        fieldindx findx[disc->nnodes];
        if (!fespace_doftofieldindx(field, disc, nv, vids, findx)) goto fespace_layout_cleanup;
        for (int i=0; i<disc->nnodes; i++) {
            if (!field_getindex(field, findx[i].g, findx[i].id, findx[i].indx, new->ccs.rix+new->ccs.cptr[id]+i)) goto fespace_layout_cleanup;
        }
    }
    new->ccs.cptr[nel]=nel*disc->nnodes; // Last column pointer points to next column
    
    *out=new;
    return true;
    
fespace_layout_cleanup:
    if (new) object_free((object *) new);
    return false;
}

/** @brief Calculates the gradient of the basis functions with respect to the reference coordinates.
 *  @param[in] disc - fespace to query
 *  @param[in] lambda - position in barycentric coordinates
 *  @param[out] grad - gradient of basis functions with respect to reference coordinates.
 *                     Layout is column-major by component:
 *                     grad->elements[FESPACE_GRAD_INDEX(disc->nnodes, component, node)].
 *  @pre FESPACE_HASGRADIENT(disc)
 */
void fespace_gradient(fespace *disc, double *lambda, objectmatrix *grad) {
    int nbary = disc->grade+1;
    
    // Compute gradients of the basis functions with respect to barycentric coordinates
    double gdata[disc->nnodes*nbary];
    (disc->gfn) (lambda, gdata);
    
    for (int i=0; i<disc->grade; i++) {
        functional_vecsub(disc->nnodes, gdata+(i+1)*disc->nnodes, gdata, grad->elements+i*disc->nnodes);
    }
}

/** @brief Calculates the Hessian of the basis functions with respect to the reference coordinates.
 *  @param[in] disc - fespace to query
 *  @param[in] lambda - position in barycentric coordinates
 *  @param[out] hess - Hessian of basis functions with respect to reference coordinates
 *                     in column-major tensor order:
 *                     hess->elements[FESPACE_HESS_INDEX(disc->nnodes, disc->grade, row, col, node)].
 *  @pre FESPACE_HASHESSIAN(disc)
 */
void fespace_hessian(fespace *disc, double *lambda, objectmatrix *hess) {
    if (disc->hfn) (disc->hfn) (lambda, hess->elements);
}

/* **********************************************************************
 * FunctionSpace class
 * ********************************************************************** */

/** Constructs a fespace object */
value fespace_constructor(vm *v, int nargs, value *args) {
    value grd=MORPHO_INTEGER(1);
    value out=MORPHO_NIL;
    int nfixed;
    
    if (!builtin_options(v, nargs, args, &nfixed, 1, field_gradeoption, &grd))
        morpho_runtimeerror(v, FNSPC_ARGS);
    
    if (nfixed==1 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISINTEGER(grd)) {
        char *label = MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)); 
        
        fespace *d=fespace_find(label, MORPHO_GETINTEGERVALUE(grd));
        
        if (d) {
            objectfespace *obj=objectfespace_new(d);
            out = morpho_wrapandbind(v, (object *) obj);
        } else morpho_runtimeerror(v, FNSPC_NOTFOUND, label, MORPHO_GETINTEGERVALUE(grd));
        
    } else morpho_runtimeerror(v, FNSPC_ARGS);
    
    return out;
}

value FiniteElementSpace_count(vm *v, int nargs, value *args) {
    objectfespace *slf = MORPHO_GETFESPACE(MORPHO_SELF(args));
    return MORPHO_INTEGER(slf->fespace->nnodes);
}

value FiniteElementSpace_grade(vm *v, int nargs, value *args) {
    objectfespace *slf = MORPHO_GETFESPACE(MORPHO_SELF(args));
    return MORPHO_INTEGER(slf->fespace->grade);
}

value FiniteElementSpace_layout(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectfespace *slf = MORPHO_GETFESPACE(MORPHO_SELF(args));
    if (nargs==1 && MORPHO_ISFIELD(MORPHO_GETARG(args, 0))) {
        objectfield *field = MORPHO_GETFIELD(MORPHO_GETARG(args, 0));
        objectsparse *new;
        
        if (fespace_layout(field, slf->fespace, &new)) out=morpho_wrapandbind(v, (object *) new);
    }
    return out;
}

value FiniteElementSpace_nodeelementindex(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectfespace *slf = MORPHO_GETFESPACE(MORPHO_SELF(args));

    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        grade g;
        int sid, indx;

        if (fespace_nodefieldindex(slf->fespace, i, &g, &sid, &indx)) {
            value entries[3] = { MORPHO_INTEGER(g), MORPHO_INTEGER(sid), MORPHO_INTEGER(indx) };
            objecttuple *new = object_newtuple(3, entries);
            out=morpho_wrapandbind(v, (object *) new);
        }
    }

    return out;
}

value FiniteElementSpace_nodecoords(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectfespace *slf = MORPHO_GETFESPACE(MORPHO_SELF(args));
    fespace *disc = slf->fespace;
    int nrows = disc->grade+1;

    if (nargs==0) {
        objectmatrix *new = matrix_new(nrows, disc->nnodes, true);
        if (!new) return MORPHO_NIL;

        for (int i=0; i<disc->nnodes; i++) {
            double lambda[nrows];
            if (!fespace_getnodecoords(disc, i, lambda)) {
                object_free((object *) new);
                return MORPHO_NIL;
            }
            matrix_setcolumnptr(new, i, lambda);
        }

        out=morpho_wrapandbind(v, (object *) new);
    } else if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        double lambda[nrows];

        if (fespace_getnodecoords(disc, i, lambda)) {
            objectmatrix *new = matrix_new(nrows, 1, true);
            if (new) matrix_setcolumnptr(new, 0, lambda);
            out=morpho_wrapandbind(v, (object *) new);
        }
    }

    return out;
}

MORPHO_BEGINCLASS(FiniteElementSpace)
MORPHO_METHOD(MORPHO_COUNT_METHOD, FiniteElementSpace_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FINITEELEMENTSPACE_GRADE_METHOD, FiniteElementSpace_grade, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FINITEELEMENTSPACE_LAYOUT_METHOD, FiniteElementSpace_layout, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FINITEELEMENTSPACE_NODEELEMENTINDEX_METHOD, FiniteElementSpace_nodeelementindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FINITEELEMENTSPACE_NODECOORDS_METHOD, FiniteElementSpace_nodecoords, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void fespace_initialize(void) {
    objectfespacetype=object_addtype(&objectfespacedefn);
    
    builtin_addfunction(FINITEELEMENTSPACE_CLASSNAME, fespace_constructor, MORPHO_FN_CONSTRUCTOR|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    value fespaceclass=builtin_addclass(FINITEELEMENTSPACE_CLASSNAME, MORPHO_GETCLASSDEFINITION(FiniteElementSpace), objclass);
    object_setveneerclass(OBJECT_FESPACE, fespaceclass);
    
    morpho_defineerror(FNSPC_ARGS, ERROR_HALT, FNSPC_ARGS_MSG);
    morpho_defineerror(FNSPC_NOTFOUND, ERROR_HALT, FNSPC_NOTFOUND_MSG);
}

#endif