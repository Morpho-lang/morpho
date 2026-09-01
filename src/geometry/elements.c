/** @file elements.c
 *  @author T J Atherton
 *
 *  @brief Finite element definitions
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "geometry.h"

/* **********************************************************************
 * Element definitions
 * ********************************************************************** */

#define ELEMENT_LINE(id, v1, v2)           ELEMENT_LINE_OPCODE, id, v1, v2              // Identify a grade 1 subelement given by two vertex indices
#define ELEMENT_AREA(id, v1, v2, v3)       ELEMENT_AREA_OPCODE, id, v1, v2, v3          // Identify a grade 2 subelement given by three vertex indices
#define ELEMENT_VOLUME(id, v1, v2, v3, v4) ELEMENT_VOLUME_OPCODE, id, v1, v2, v3, v4    // Identify a grade 3 subelement given by four vertex indices
#define ELEMENT_QUANTITY(grade, id, qno)   ELEMENT_QUANTITY_OPCODE, grade, id, qno      // Fetch quantity from subelement of grade with id and quantity number

void cg0_interpolate(double *lambda, double *wts) {
    wts[0]=1.0;
}

/* -------------------------------------------------------
 * CG0 element in 0D
 * ------------------------------------------------------- */

/*
 *   0    // One degree of freedom on the vertex
 */

void cg0_0dgrad(double *lambda, double *grad) {
    double g[] = { 0 };
    memcpy(grad, g, sizeof(g));
}

unsigned int cg0_0dshape[] = { 1 };

double cg0_0dnodes[] = { 1.0 };

eldefninstruction cg0_0ddefn[] = {
    ELEMENT_QUANTITY(0,0,0),
    ELEMENT_ENDDEFN
};

fespace cg0_0d = {
    .name = FESPACE_CG0,
    .grade = 0,
    .shape = cg0_0dshape,
    .degree = 0,
    .nnodes = 1,
    .nsubel = 0,
    .nodes = cg0_0dnodes,
    .ifn = cg0_interpolate,
    .gfn = cg0_0dgrad,
    .eldefn = cg0_0ddefn,
    .lower = NULL
};

/* -------------------------------------------------------
 * CG0 element in 1D
 * ------------------------------------------------------- */

/*
 *   0 ----- 1    // One degree of freedom on the line (centroid)
 */

void cg0_1dgrad(double *lambda, double *grad) {
    double g[] = { 0, 0 };
    memcpy(grad, g, sizeof(g));
}

unsigned int cg0_1dshape[] = { 0, 1 };

double cg0_1dnodes[] = { 0.5 };

eldefninstruction cg0_1ddefn[] = {
    ELEMENT_LINE(0,0,1),
    ELEMENT_QUANTITY(1,0,0),
    ELEMENT_ENDDEFN
};

fespace cg0_1d = {
    .name = FESPACE_CG0,
    .grade = 1,
    .shape = cg0_1dshape,
    .degree = 0,
    .nnodes = 1,
    .nsubel = 1,
    .nodes = cg0_1dnodes,
    .ifn = cg0_interpolate,
    .gfn = cg0_1dgrad,
    .eldefn = cg0_1ddefn,
    .lower = NULL
};

/* -------------------------------------------------------
 * CG0 element in 2D
 * ------------------------------------------------------- */

void cg0_2dgrad(double *lambda, double *grad) {
    double g[] = { 0, 0, 0 };
    memcpy(grad, g, sizeof(g));
}

unsigned int cg0_2dshape[] = { 0, 0, 1 };

double cg0_2dnodes[] = { 1.0/3.0, 1.0/3.0 };

eldefninstruction cg0_2ddefn[] = {
    ELEMENT_AREA(0,0,1,2),
    ELEMENT_QUANTITY(2,0,0),
    ELEMENT_ENDDEFN
};

fespace cg0_2d = {
    .name = FESPACE_CG0,
    .grade = 2,
    .shape = cg0_2dshape,
    .degree = 0,
    .nnodes = 1,
    .nsubel = 1,
    .nodes = cg0_2dnodes,
    .ifn = cg0_interpolate,
    .gfn = cg0_2dgrad,
    .eldefn = cg0_2ddefn,
    .lower = NULL
};

/* -------------------------------------------------------
 * CG0 element in 3D
 * ------------------------------------------------------- */

void cg0_3dgrad(double *lambda, double *grad) {
    double g[] = { 0, 0, 0, 0 };
    memcpy(grad, g, sizeof(g));
}

unsigned int cg0_3dshape[] = { 0, 0, 0, 1 };

double cg0_3dnodes[] = { 0.25, 0.25, 0.25 };

eldefninstruction cg0_3ddefn[] = {
    ELEMENT_VOLUME(0,0,1,2,3),
    ELEMENT_QUANTITY(3,0,0),
    ELEMENT_ENDDEFN
};

fespace cg0_3d = {
    .name = FESPACE_CG0,
    .grade = 3,
    .shape = cg0_3dshape,
    .degree = 0,
    .nnodes = 1,
    .nsubel = 1,
    .nodes = cg0_3dnodes,
    .ifn = cg0_interpolate,
    .gfn = cg0_3dgrad,
    .eldefn = cg0_3ddefn,
    .lower = NULL
};

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
    ELEMENT_QUANTITY(0,0,0), // Fetch quantity on vertex 0
    ELEMENT_QUANTITY(0,1,0), // Fetch quantity on vertex 1
    ELEMENT_ENDDEFN
};

fespace cg1_1d = {
    .name = FESPACE_CG1,
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
    ELEMENT_LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    ELEMENT_QUANTITY(0,0,0), // Fetch quantity on vertex 0
    ELEMENT_QUANTITY(0,1,0), // Fetch quantity on vertex 1
    ELEMENT_QUANTITY(1,0,0), // Fetch quantity from line subelement
    ELEMENT_ENDDEFN
};

fespace cg2_1d = {
    .name = FESPACE_CG2,
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
    ELEMENT_LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    ELEMENT_QUANTITY(0,0,0), // Fetch quantity on vertex 0
    ELEMENT_QUANTITY(0,1,0), // Fetch quantity on vertex 1
    ELEMENT_QUANTITY(1,0,0), // Fetch quantity from line subelement
    ELEMENT_QUANTITY(1,0,1), // Fetch quantity from line subelement
    ELEMENT_ENDDEFN
};

fespace cg3_1d = {
    .name = FESPACE_CG3,
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
    ELEMENT_QUANTITY(0,0,0), // Fetch quantity on vertex 0
    ELEMENT_QUANTITY(0,1,0), // Fetch quantity on vertex 1
    ELEMENT_QUANTITY(0,2,0), // Fetch quantity on vertex 2
    ELEMENT_ENDDEFN
};

fespace *cg1_2d_lower[] = {
    &cg1_1d,
    NULL
};

fespace cg1_2d = {
    .name = FESPACE_CG1,
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
    ELEMENT_LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    ELEMENT_LINE(1,1,2),     // Identify line subelement with vertex indices (1,2)
    ELEMENT_LINE(2,2,0),     // Identify line subelement with vertex indices (2,0)
    ELEMENT_QUANTITY(0,0,0), // Fetch quantity on vertex 0
    ELEMENT_QUANTITY(0,1,0), // Fetch quantity on vertex 1
    ELEMENT_QUANTITY(0,2,0), // Fetch quantity on vertex 2
    ELEMENT_QUANTITY(1,0,0), // Fetch quantity from line 0
    ELEMENT_QUANTITY(1,1,0), // Fetch quantity from line 1
    ELEMENT_QUANTITY(1,2,0), // Fetch quantity from line 2
    ELEMENT_ENDDEFN
};

fespace *cg2_2d_lower[] = {
    &cg2_1d,
    NULL
};

fespace cg2_2d = {
    .name = FESPACE_CG2,
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
    ELEMENT_LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    ELEMENT_LINE(1,1,2),     // Identify line subelement with vertex indices (1,2)
    ELEMENT_LINE(2,2,0),     // Identify line subelement with vertex indices (2,0)
    ELEMENT_AREA(3,0,1,2),   // Identify area subelement with vertex indices (0,1,2)
    ELEMENT_QUANTITY(0,0,0), // Fetch quantity on vertex 0
    ELEMENT_QUANTITY(0,1,0), // Fetch quantity on vertex 1
    ELEMENT_QUANTITY(0,2,0), // Fetch quantity on vertex 2
    ELEMENT_QUANTITY(1,0,0), // Fetch quantity 0 from line 0
    ELEMENT_QUANTITY(1,0,1), // Fetch quantity 1 from line 0
    ELEMENT_QUANTITY(1,1,0), // Fetch quantity 0 from line 1
    ELEMENT_QUANTITY(1,1,1), // Fetch quantity 1 from line 1
    ELEMENT_QUANTITY(1,2,0), // Fetch quantity 0 from line 2
    ELEMENT_QUANTITY(1,2,1), // Fetch quantity 1 from line 2
    ELEMENT_QUANTITY(2,3,0), // Fetch quantity 0 from area 0
    ELEMENT_ENDDEFN
};

fespace *cg3_2d_lower[] = {
    &cg3_1d,
    NULL
};

fespace cg3_2d = {
    .name = FESPACE_CG3,
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
    ELEMENT_QUANTITY(0,0,0), // Fetch quantity on vertex 0
    ELEMENT_QUANTITY(0,1,0), // Fetch quantity on vertex 1
    ELEMENT_QUANTITY(0,2,0), // Fetch quantity on vertex 2
    ELEMENT_QUANTITY(0,3,0), // Fetch quantity on vertex 3
    ELEMENT_ENDDEFN
};

fespace *cg1_3d_lower[] = {
    &cg1_2d,
    &cg1_1d,
    NULL
};

fespace cg1_3d = {
    .name = FESPACE_CG1,
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
    ELEMENT_LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    ELEMENT_LINE(1,1,2),     // Identify line subelement with vertex indices (1,2)
    ELEMENT_LINE(2,2,0),     // Identify line subelement with vertex indices (2,0)
    ELEMENT_LINE(3,0,3),     // Identify line subelement with vertex indices (0,3)
    ELEMENT_LINE(4,1,3),     // Identify line subelement with vertex indices (1,3)
    ELEMENT_LINE(5,2,3),     // Identify line subelement with vertex indices (2,3)
    ELEMENT_QUANTITY(0,0,0), // Fetch quantity on vertex 0
    ELEMENT_QUANTITY(0,1,0), // Fetch quantity on vertex 1
    ELEMENT_QUANTITY(0,2,0), // Fetch quantity on vertex 2
    ELEMENT_QUANTITY(0,3,0), // Fetch quantity on vertex 3
    ELEMENT_QUANTITY(1,0,0), // Fetch quantity from line 0
    ELEMENT_QUANTITY(1,1,0), // Fetch quantity from line 1
    ELEMENT_QUANTITY(1,2,0), // Fetch quantity from line 2
    ELEMENT_QUANTITY(1,3,0), // Fetch quantity from line 3
    ELEMENT_QUANTITY(1,4,0), // Fetch quantity from line 4
    ELEMENT_QUANTITY(1,5,0), // Fetch quantity from line 5
    ELEMENT_ENDDEFN
};

fespace *cg2_3d_lower[] = {
    &cg2_2d,
    &cg2_1d,
    NULL
};

fespace cg2_3d = {
    .name = FESPACE_CG2,
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
    ELEMENT_LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    ELEMENT_LINE(1,1,2),     // Identify line subelement with vertex indices (1,2)
    ELEMENT_LINE(2,2,0),     // Identify line subelement with vertex indices (2,0)
    ELEMENT_LINE(3,0,3),     // Identify line subelement with vertex indices (0,3)
    ELEMENT_LINE(4,1,3),     // Identify line subelement with vertex indices (1,3)
    ELEMENT_LINE(5,2,3),     // Identify line subelement with vertex indices (2,3)
    ELEMENT_AREA(6,0,1,2),   // Identify area subelement with vertex indices (0,1,2)
    ELEMENT_AREA(7,0,3,1),   // Identify area subelement with vertex indices (0,3,1)
    ELEMENT_AREA(8,1,3,2),   // Identify area subelement with vertex indices (1,3,2)
    ELEMENT_AREA(9,2,3,0),   // Identify area subelement with vertex indices (2,3,0)
    ELEMENT_QUANTITY(0,0,0), // Fetch quantity on vertex 0
    ELEMENT_QUANTITY(0,1,0), // Fetch quantity on vertex 1
    ELEMENT_QUANTITY(0,2,0), // Fetch quantity on vertex 2
    ELEMENT_QUANTITY(0,3,0), // Fetch quantity on vertex 3
    ELEMENT_QUANTITY(1,0,0), // Fetch quantity 0 from line 0
    ELEMENT_QUANTITY(1,0,1), // Fetch quantity 1 from line 0
    ELEMENT_QUANTITY(1,1,0), // Fetch quantity 0 from line 1
    ELEMENT_QUANTITY(1,1,1), // Fetch quantity 1 from line 1
    ELEMENT_QUANTITY(1,2,0), // Fetch quantity 0 from line 2
    ELEMENT_QUANTITY(1,2,1), // Fetch quantity 1 from line 2
    ELEMENT_QUANTITY(1,3,0), // Fetch quantity 0 from line 3
    ELEMENT_QUANTITY(1,3,1), // Fetch quantity 1 from line 3
    ELEMENT_QUANTITY(1,4,0), // Fetch quantity 0 from line 4
    ELEMENT_QUANTITY(1,4,1), // Fetch quantity 1 from line 4
    ELEMENT_QUANTITY(1,5,0), // Fetch quantity 0 from line 5
    ELEMENT_QUANTITY(1,5,1), // Fetch quantity 1 from line 5
    ELEMENT_QUANTITY(2,6,0), // Fetch quantity 0 from area 0
    ELEMENT_QUANTITY(2,7,0), // Fetch quantity 0 from area 1
    ELEMENT_QUANTITY(2,8,0), // Fetch quantity 0 from area 2
    ELEMENT_QUANTITY(2,9,0), // Fetch quantity 0 from area 3
    ELEMENT_ENDDEFN
};

fespace *cg3_3d_lower[] = {
    &cg3_2d,
    &cg3_1d,
    NULL
};

fespace cg3_3d = {
    .name = FESPACE_CG3,
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
    &cg0_0d,
    &cg0_1d,
    &cg1_1d,
    &cg2_1d,
    &cg3_1d,
    &cg0_2d,
    &cg1_2d,
    &cg2_2d,
    &cg3_2d,
    &cg0_3d,
    &cg1_3d,
    &cg2_3d,
    &cg3_3d,
    NULL
};

#endif /* MORPHO_INCLUDE_GEOMETRY */
