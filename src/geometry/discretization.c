/** @file discretization.c
 *  @author T J Atherton
 *
 *  @brief Finite element discretizations
 */

#include "geometry.h"

typedef void (*interpolationfn) (double *, double *);

typedef struct {
    grade grade;
    int *shape;
    int nnodes;
    interpolationfn ifn;
} discretization;

/* -------------------------------------------------------
 * CG1 elements
 * ------------------------------------------------------- */

void cg1interpolate(double *lambda, double *wts) {
    wts[0]=lambda[1];
    wts[1]=lambda[0];
}

int cg1shape[] = { 1, 0 };

discretization cg1 = {
    .grade = 1,
    .shape = cg1shape,
    .nnodes = 2,
    .ifn = cg1interpolate
};

/* **********************************************************************
 * Discretization objects
 * ********************************************************************** */

void discretization_initialize(void) {
}
