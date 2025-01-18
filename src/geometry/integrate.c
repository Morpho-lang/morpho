/** @file integrate.c
 *  @author T J Atherton
 *
 *  @brief Numerical integration
*/

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include <limits.h>
#include "integrate.h"
#include "morpho.h"
#include "classes.h"

#include "matrix.h"
#include "sparse.h"
#include "mesh.h"
#include "selection.h"
#include "functional.h"

bool integrate_recognizequantities(unsigned int nquantity, value *quantity, value *out) {
    if (nquantity>0) {
        for (unsigned int i=0; i<nquantity; i++) {
            if (MORPHO_ISFLOAT(quantity[i])) {
                out[i]=MORPHO_FLOAT(0);
            } else if (MORPHO_ISMATRIX(quantity[i])) {
                out[i]=MORPHO_OBJECT(object_clonematrix(MORPHO_GETMATRIX(quantity[i])));
            } else return false;
        }
    }
    return true;
}

/* **********************************************************************
 * Line integrals
 * ********************************************************************** */

static double gk[] = {
    /* Gauss 7 pt nodes [pt, Gauss wt, Kronrod wt] */
    -0.949107912342759,  0.129484966168870,  0.063092092629979,
    0.949107912342759,  0.129484966168870,  0.063092092629979,
    -0.741531185599394,  0.279705391489277,  0.140653259715525,
    0.741531185599394,  0.279705391489277,  0.140653259715525,
    -0.405845151377397,  0.381830050505119,  0.190350578064785,
    0.405845151377397,  0.381830050505119,  0.190350578064785,
    0.000000000000000,  0.417959183673469,  0.209482141084728,
    
    /* Kronrod extension [pt, Gauss wt, Kronrod wt] */
    -0.991455371120813,  0.0, 0.022935322010529,
    0.991455371120813,  0.0, 0.022935322010529,
    -0.864864423359769,  0.0, 0.104790010322250,
    0.864864423359769,  0.0, 0.104790010322250,
    -0.586087235467691,  0.0, 0.169004726639267,
    0.586087235467691,  0.0, 0.169004726639267,
    -0.207784955007898,  0.0, 0.204432940075298,
    0.207784955007898,  0.0, 0.204432940075298
};

unsigned int gknpts=15;
unsigned int gk1=7;
unsigned int gk2=15;

/* Linearly interpolate the position. t goes from [0,1] */
void integrate_interpolatepositionline(unsigned int dim, double *x[3], double t, double *xout) {
    double lambda[2] = {1-t,t};
    for (unsigned int j=0; j<dim; j++) {
        xout[j]=0;
        for (unsigned int k=0; k<2; k++) xout[j]+=lambda[k]*x[k][j];
    }
}

/* Interpolate any quantities. t goes from [0,1] */
void integrate_interpolatequantitiesline(unsigned int dim, double t, unsigned int nquantity, value *quantity[2], value *qout) {
    double lambda[2] = {1-t,t};
    
    for (unsigned int i=0; i<nquantity; i++) {
        if (MORPHO_ISFLOAT(quantity[0][i])) {
            double val = lambda[0]*MORPHO_GETFLOATVALUE(quantity[0][i])+
                         lambda[1]*MORPHO_GETFLOATVALUE(quantity[1][i]);
            qout[i]=MORPHO_FLOAT(val);
        } else if (MORPHO_ISMATRIX(quantity[0][i]) && MORPHO_ISMATRIX(quantity[1][i])) {
            objectmatrix *m0=MORPHO_GETMATRIX(quantity[0][i]),
                         *m1=MORPHO_GETMATRIX(quantity[1][i]),
                         *out=(MORPHO_ISMATRIX(qout[i]) ? MORPHO_GETMATRIX(qout[i]): NULL);
            
            if (!out) {
                out = object_clonematrix(m0);
                qout[i]=MORPHO_OBJECT(out);
            }
            
            for (unsigned int i=0; i<m0->ncols*m0->nrows; i++) {
                out->elements[i] = lambda[0]*m0->elements[i]+lambda[1]*m1->elements[i];
            }
        }
    }
}

/** Integrate over a line element
 * @param[in] function     - function to integrate
 * @param[in] dim                - Dimension of the vertices
 * @param[in] x                     - vertices of the line x[0] = {x,y,z} etc.
 * @param[in] nquantity   - number of quantities per vertex
 * @param[in] quantity     - List of quantities for each vertex.
 * @param[in] ref                 - a pointer to any data required by the function
 * @param[in] ge                   - Global estimate of the integral (used for recursion).
 * @param[out] out               - estimate of the integral
 * @returns True on success */
bool integrate_lineint(integrandfunction *function, unsigned int dim, double *x[2], unsigned int nquantity, value *quantity[2], value *q, void *ref, unsigned int recursiondepth, double ge, double *out) {
    double r[gknpts], r1=0.0, r2=0.0, eps;
    double xx[dim], gest=ge;
    double af=pow(0.5, (double) recursiondepth); // Length of whole line from recursion depth
    unsigned int i;
    bool success=false;
    double fout = 0;
    
    /* Try low order method for rapid results on low order functions */
    for (unsigned int i=0; i<gknpts; i++) {
        double tt=0.5*(1.0+gk[3*i]); // Convert [-1,1] to [0,1]
        integrate_interpolatepositionline(dim, x, tt, xx);
        if (nquantity)  integrate_interpolatequantitiesline(dim, tt, nquantity, quantity, q);
        if ((*function) (dim, &tt, xx, nquantity, q, ref,&fout)){
            r[i] = fout;
        }
        else {
            return false;
        }
    }
    
    for (i=0; i<gk1; i++) {
        r1+=r[i]*gk[3*i+1];
        r2+=r[i]*gk[3*i+2];
    }
    for (; i<gk2; i++) {
        r2+=r[i]*gk[3*i+2];
    }
    r1*=0.5; r2*=0.5;
    
    if (recursiondepth==0) gest=fabs(r2); // If at top level construct a global estimate of the integral
    
    eps=r2-r1;
    eps*=af;
    if (gest>MORPHO_EPS) eps/=gest; // Globally relative estimate using area factor
    
    //printf("Recursion depth %u: %g %g - %g\n",recursiondepth, r1, r2, eps);
    
    if (fabs(eps)<INTEGRATE_ACCURACYGOAL)  {
        *out=r2;
        return true;
    }
    
    if (recursiondepth>INTEGRATE_MAXRECURSION) {
        *out=r2;
        return false;
    }
    
    /* Bisect: */
    double *xn[2]; /* Will hold the vertices. */
    double xm[dim];
    double est;
    value qm[nquantity+1], *qn[2];
    
    /* New vertices s*/
    for (unsigned int i=0; i<dim; i++) {
        xm[i] = 0.5*(x[0][i]+x[1][i]);
    }
    /* Quantities */
    if (nquantity) {
        for (unsigned int i=0; i<nquantity; i++) qm[i]=MORPHO_NIL;
        integrate_interpolatequantitiesline(dim, 0.5, nquantity, quantity, qm);
    }
    
    r2=0.0;
    xn[0]=x[0]; xn[1]=xm;
    if (nquantity) { qn[0] = quantity[0]; qn[1] = qm; }
    if (!integrate_lineint(function, dim, xn, nquantity, qn, q, ref, recursiondepth+1, gest, &est)) goto integrate_lineint_cleanup;
        
    r2+=est;
    
    xn[0]=xm; xn[1]=x[1];
    if (nquantity) { qn[0] = qm; qn[1] = quantity[1]; }
    if (!integrate_lineint(function, dim, xn, nquantity, qn, q, ref, recursiondepth+1, gest, &est)) goto integrate_lineint_cleanup;
    
    r2+=est;
    r2*=0.5;
    
    *out = r2;
    success=true;
    
integrate_lineint_cleanup:
    /* Free interpolated quantities */
    for (unsigned int i=0; i<nquantity; i++) {
        if (MORPHO_ISOBJECT(qm[i])) object_free(MORPHO_GETOBJECT(qm[i]));
    }
    
    return success;
}


/* **********************************************************************
 * Area integrals
 * ********************************************************************** */

/* Points to evaluate the function at in Barycentric coordinates */
/* Adaptive rules based on Walkington, "Quadrature on Simplices of arbitrary dimension" */
static double pts[] = {
    0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
    0.6000000000000000, 0.2000000000000000, 0.2000000000000000,
    0.2000000000000000, 0.6000000000000000, 0.2000000000000000,
    0.2000000000000000, 0.2000000000000000, 0.6000000000000000,
    0.7142857142857143, 0.1428571428571429, 0.1428571428571429,
    0.1428571428571429, 0.7142857142857143, 0.1428571428571429,
    0.1428571428571429, 0.1428571428571429, 0.7142857142857143,
    0.4285714285714286, 0.4285714285714286, 0.1428571428571429,
    0.4285714285714286, 0.1428571428571429, 0.4285714285714286,
    0.1428571428571429, 0.4285714285714286, 0.4285714285714286,
    
    0.7777777777777778, 0.1111111111111111, 0.1111111111111111,
    0.1111111111111111, 0.7777777777777778, 0.1111111111111111,
    0.1111111111111111, 0.1111111111111111, 0.7777777777777778,
    0.3333333333333333, 0.5555555555555556, 0.1111111111111111,
    0.3333333333333333, 0.1111111111111111, 0.5555555555555556,
    0.5555555555555556, 0.3333333333333333, 0.1111111111111111,
    0.5555555555555556, 0.1111111111111111, 0.3333333333333333,
    0.1111111111111111, 0.3333333333333333, 0.5555555555555556,
    0.1111111111111111, 0.5555555555555556, 0.3333333333333333,
    0.3333333333333333, 0.3333333333333333, 0.3333333333333333
};

double w[] = {
    -0.5625, 0.5208333333333332, 0.5208333333333332, 0.5208333333333332,
    
    0.1265625, -0.5425347222222222, -0.5425347222222222, -0.5425347222222222,
    0.4168402777777778, 0.4168402777777778, 0.4168402777777778, 0.4168402777777778,
    0.4168402777777778, 0.4168402777777778
};

static double wts1[] = {-0.2812500000000000, 0.2604166666666667};
static double wts2[] = {0.06328125000000000, -0.2712673611111111, 0.2084201388888889};
static double wts3[] = {-0.007910156250000000, 0.1211015004960317, -0.3191433376736111,
    0.2059465680803571};
static unsigned int npts1 = 10;
static unsigned int npts2 = 20;

/* Linearly interpolate the position depending on the triangle */
void integrate_interpolatepositiontri(unsigned int dim, double *x[3], double *lambda, double *xout) {
    for (unsigned int j=0; j<dim; j++) {
        xout[j]=0;
        for (unsigned int k=0; k<3; k++) xout[j]+=lambda[k]*x[k][j];
    }
}

/* Interpolate any quantities. t goes from [0,1] */
void integrate_interpolatequantitiestri(unsigned int dim, double *lambda, unsigned int nquantity, value *quantity[3], value *qout) {
    
    for (unsigned int i=0; i<nquantity; i++) {
        if (MORPHO_ISFLOAT(quantity[0][i])) {
            double val = lambda[0]*MORPHO_GETFLOATVALUE(quantity[0][i])+
                         lambda[1]*MORPHO_GETFLOATVALUE(quantity[1][i])+
                         lambda[2]*MORPHO_GETFLOATVALUE(quantity[2][i]);
            qout[i]=MORPHO_FLOAT(val);
        } else if (MORPHO_ISMATRIX(quantity[0][i]) && MORPHO_ISMATRIX(quantity[1][i]) && MORPHO_ISMATRIX(quantity[2][i])) {
            objectmatrix *m0=MORPHO_GETMATRIX(quantity[0][i]),
                         *m1=MORPHO_GETMATRIX(quantity[1][i]),
                         *m2=MORPHO_GETMATRIX(quantity[2][i]),
                         *out=(MORPHO_ISMATRIX(qout[i]) ? MORPHO_GETMATRIX(qout[i]): NULL);
            
            if (!out) {
                out = object_clonematrix(m0);
                qout[i]=MORPHO_OBJECT(out);
            }
            
            for (unsigned int i=0; i<m0->ncols*m0->nrows; i++) {
                out->elements[i] = lambda[0]*m0->elements[i]+lambda[1]*m1->elements[i]+lambda[2]*m2->elements[i];
            }
        }
    }
}


/** Integrate over an area element
 * @param[in] function     - function to integrate
 * @param[in] dim                - Dimension of the vertices
 * @param[in] x                     - vertices of the line x[0] = {x,y,z} etc.
 * @param[in] nquantity   - number of quantities per vertex
 * @param[in] quantity     - List of quantities for each vertex.
 * @param[in] ref                 - a pointer to any data required by the function
 * @param[in] ge                   - Global estimate of the integral (used for recursion).
 * @param[out] out               - estimate of the integral
 * @returns True on success */
bool integrate_areaint(integrandfunction *function, unsigned int dim, double *x[3], unsigned int nquantity, value *quantity[3], value *q, void *ref, unsigned int recursiondepth, double ge, double *out) {
    double r[npts2], r1, rr, r2, rr2, r3, rr3, eps;
    double xx[dim], gest=ge;
    double af=pow(0.25, (double) recursiondepth); // Area of total triangle covered from recursion depth
    bool success=false;
    double fout = 0;
    /* Try low order method for rapid results on low order functions */
    for (unsigned int i=0; i<npts1; i++) {
        double *lambda=pts+3*i;
        integrate_interpolatepositiontri(dim, x, lambda, xx);
        if (nquantity)  integrate_interpolatequantitiestri(dim, lambda, nquantity, quantity, q);
        if ((*function) (dim, lambda, xx, nquantity, q, ref, &fout)) {
            r[i] = fout;
        } else{
            return false;
        }
        
    }
    rr=(r[1]+r[2]+r[3]);
    rr2=(r[4]+r[5]+r[6]+r[7]+r[8]+r[9]);
    r1 = wts1[0]*r[0] + wts1[1]*rr;
    r2 = wts2[0]*r[0] + wts2[1]*rr + wts2[2]*rr2;
    
    if (recursiondepth==0) gest=fabs(r2); // If at top level construct a global estimate of the integral

    eps=r2-r1;
    eps*=af;
    if (gest>MORPHO_EPS) eps/=gest; // Globally relative estimate using area factor

    if (fabs(eps)<INTEGRATE_ACCURACYGOAL)  { // Low order worked
        *out=2*r2;
        return true;
    }
    
    /* Extend order */
    for (unsigned int i=npts1; i<npts2; i++) {
        double *lambda=pts+3*i;
        integrate_interpolatepositiontri(dim, x, lambda, xx);
        if (nquantity)  integrate_interpolatequantitiestri(dim, lambda, nquantity, quantity, q);
        if ((*function) (dim, lambda, xx, nquantity, q, ref, &fout)){
            r[i] = fout;
        } else{
            return false;
        }
    }
    rr3=(r[10]+r[11]+r[12]+r[13]+r[14]+r[15]+r[16]+r[17]+r[18]+r[19]);
    r3 = wts3[0]*r[0] + wts3[1]*rr + wts3[2]*rr2 + wts3[3]*rr3;
    
    if (recursiondepth==0) gest=fabs(2*r3); // Use an improved estimate of the integral
    
    eps=r2-r3;
    eps*=af;
    if (gest>MORPHO_EPS) eps/=gest; // Globally relative estimate
    //printf("Estimates %lg %lg %lg, err=%g af=%g\n", r1,r2,r3, eps, af);
    if (fabs(eps)<INTEGRATE_ACCURACYGOAL) {
        *out=2*r3;
        return true;
    }
    
    if (recursiondepth>INTEGRATE_MAXRECURSION) {
        *out=2*r3;
        return false;
    }
    
    /* Quadrasect:
         *       2
         *      / \
         *   x20 - x12
         *    / \  / \
         *   0 - x01 - 1
         */
    double *xn[3]; /* Will hold the vertices. */
    double x01[dim], x12[dim], x20[dim]; /* Vertices from midpoints */
    double sub;
    value q01[nquantity+1], q12[nquantity+1], q20[nquantity+1], *qn[3];
    
    r3=0.0;
    /* New vertices s*/
    for (unsigned int i=0; i<dim; i++) {
        x01[i] = 0.5*(x[0][i]+x[1][i]);
        x12[i] = 0.5*(x[1][i]+x[2][i]);
        x20[i] = 0.5*(x[2][i]+x[0][i]);
    }
    /* Quantities */
    if (nquantity) {
        double ll[3];
        for (unsigned int i=0; i<nquantity; i++) { q01[i]=MORPHO_NIL; q12[i]=MORPHO_NIL; q20[i]=MORPHO_NIL; }
        ll[0]=0.5; ll[1]=0.5; ll[2]=0.0;
        integrate_interpolatequantitiestri(dim, ll, nquantity, quantity, q01);
        ll[0]=0.0; ll[1]=0.5; ll[2]=0.5;
        integrate_interpolatequantitiestri(dim, ll, nquantity, quantity, q12);
        ll[0]=0.5; ll[1]=0.0; ll[2]=0.5;
        integrate_interpolatequantitiestri(dim, ll, nquantity, quantity, q20);
    }
    
    xn[0]=x[0]; xn[1]=x01; xn[2]=x20;
    if (nquantity) { qn[0] = quantity[0]; qn[1] = q01; qn[2] = q20; }
    if (!integrate_areaint(function, dim, xn, nquantity, qn, q, ref, recursiondepth+1, gest, &sub)) goto integrate_areaint_cleanup;
    r3+=sub;
    
    xn[0]=x01; xn[1]=x[1]; xn[2]=x12;
    if (nquantity) { qn[0] = q01; qn[1] = quantity[1]; qn[2] = q12; }
    if (!integrate_areaint(function, dim, xn, nquantity, qn, q, ref, recursiondepth+1, gest, &sub)) goto integrate_areaint_cleanup;
    r3+=sub;
    
    xn[0]=x20; xn[1]=x12; xn[2]=x[2];
    if (nquantity) { qn[0] = q20; qn[1] = q12; qn[2] = quantity[2]; }
    if (!integrate_areaint(function, dim, xn, nquantity, qn, q, ref, recursiondepth+1, gest, &sub)) goto integrate_areaint_cleanup;
    r3+=sub;
    
    xn[0]=x01; xn[1]=x12; xn[2]=x20;
    if (nquantity) { qn[0] = q01; qn[1] = q12; qn[2] = q20; }
    if (!integrate_areaint(function, dim, xn, nquantity, qn, q, ref, recursiondepth+1, gest, &sub)) goto integrate_areaint_cleanup;
    r3+=sub;
    
    *out=0.25*r3;
    success=true;
    
integrate_areaint_cleanup:
    /* Free interpolated quantities */
    for (int j=0; j<3; j++) for (unsigned int i=0; i<nquantity; i++) {
        if (MORPHO_ISOBJECT(qn[j][i])) object_free(MORPHO_GETOBJECT(qn[j][i]));
    }
    
    return success;
}

/* **********************************************************************
 * Volume integrals
 * ********************************************************************** */

// Nodes and weights from Journal of Computational and Applied Mathematics, 236, 17, 4348-4364 (2012)

/*
static double v1[] = {
    0.2500000000000000,    0.2500000000000000,    0.2500000000000000,    0.2500000000000000,    1.0000000000000000
};

static unsigned int nv1 = 1;

static double v2[] = {
    0.5854101966249680,    0.1381966011250110,    0.1381966011250110,    0.1381966011250110,    0.2500000000000000,
    0.1381966011250110,    0.5854101966249680,    0.1381966011250110,    0.1381966011250110,    0.2500000000000000,
    0.1381966011250110,    0.1381966011250110,    0.5854101966249680,    0.1381966011250110,    0.2500000000000000,
    0.1381966011250110,    0.1381966011250110,    0.1381966011250110,    0.5854101966249680,    0.2500000000000000
};

static unsigned int nv2 = 4;

static double v3[] = {
    0.7784952948213300,    0.0738349017262234,    0.0738349017262234,    0.0738349017262234,    0.0476331348432089,
    0.0738349017262234,    0.7784952948213300,    0.0738349017262234,    0.0738349017262234,    0.0476331348432089,
    0.0738349017262234,    0.0738349017262234,    0.7784952948213300,    0.0738349017262234,    0.0476331348432089,
    0.0738349017262234,    0.0738349017262234,    0.0738349017262234,    0.7784952948213300,    0.0476331348432089,
    0.4062443438840510,    0.4062443438840510,    0.0937556561159491,    0.0937556561159491,    0.1349112434378610,
    0.4062443438840510,    0.0937556561159491,    0.4062443438840510,    0.0937556561159491,    0.1349112434378610,
    0.4062443438840510,    0.0937556561159491,    0.0937556561159491,    0.4062443438840510,    0.1349112434378610,
    0.0937556561159491,    0.4062443438840510,    0.4062443438840510,    0.0937556561159491,    0.1349112434378610,
    0.0937556561159491,    0.4062443438840510,    0.0937556561159491,    0.4062443438840510,    0.1349112434378610,
    0.0937556561159491,    0.0937556561159491,    0.4062443438840510,    0.4062443438840510,    0.1349112434378610
};

static unsigned int nv3 = 10;

static double v4[] = {
     0.9029422158182680,    0.0323525947272439,    0.0323525947272439,    0.0323525947272439,    0.0070670747944695,
     0.0323525947272439,    0.9029422158182680,    0.0323525947272439,    0.0323525947272439,    0.0070670747944695,
     0.0323525947272439,    0.0323525947272439,    0.9029422158182680,    0.0323525947272439,    0.0070670747944695,
     0.0323525947272439,    0.0323525947272439,    0.0323525947272439,    0.9029422158182680,    0.0070670747944695,
     0.2626825838877790,    0.6165965330619370,    0.0603604415251421,    0.0603604415251421,    0.0469986689718877,
     0.6165965330619370,    0.2626825838877790,    0.0603604415251421,    0.0603604415251421,    0.0469986689718877,
     0.2626825838877790,    0.0603604415251421,    0.6165965330619370,    0.0603604415251421,    0.0469986689718877,
     0.6165965330619370,    0.0603604415251421,    0.2626825838877790,    0.0603604415251421,    0.0469986689718877,
     0.2626825838877790,    0.0603604415251421,    0.0603604415251421,    0.6165965330619370,    0.0469986689718877,
     0.6165965330619370,    0.0603604415251421,    0.0603604415251421,    0.2626825838877790,    0.0469986689718877,
     0.0603604415251421,    0.2626825838877790,    0.6165965330619370,    0.0603604415251421,    0.0469986689718877,
     0.0603604415251421,    0.6165965330619370,    0.2626825838877790,    0.0603604415251421,    0.0469986689718877,
     0.0603604415251421,    0.2626825838877790,    0.0603604415251421,    0.6165965330619370,    0.0469986689718877,
     0.0603604415251421,    0.6165965330619370,    0.0603604415251421,    0.2626825838877790,    0.0469986689718877,
     0.0603604415251421,    0.0603604415251421,    0.2626825838877790,    0.6165965330619370,    0.0469986689718877,
     0.0603604415251421,    0.0603604415251421,    0.6165965330619370,    0.2626825838877790,    0.0469986689718877,
     0.3097693042728620,    0.3097693042728620,    0.3097693042728620,    0.0706920871814129,    0.1019369182898680,
     0.3097693042728620,    0.3097693042728620,    0.0706920871814129,    0.3097693042728620,    0.1019369182898680,
     0.3097693042728620,    0.0706920871814129,    0.3097693042728620,    0.3097693042728620,    0.1019369182898680,
     0.0706920871814129,    0.3097693042728620,    0.3097693042728620,    0.3097693042728620,    0.1019369182898680
};

static unsigned int nv4 = 20;
*/
 
static double v5[] = {
    0.9197896733368800,    0.0267367755543735,    0.0267367755543735,    0.0267367755543735,    0.0021900463965388,
    0.0267367755543735,    0.9197896733368800,    0.0267367755543735,    0.0267367755543735,    0.0021900463965388,
    0.0267367755543735,    0.0267367755543735,    0.9197896733368800,    0.0267367755543735,    0.0021900463965388,
    0.0267367755543735,    0.0267367755543735,    0.0267367755543735,    0.9197896733368800,    0.0021900463965388,
    0.1740356302468940,    0.7477598884818090,    0.0391022406356488,    0.0391022406356488,    0.0143395670177665,
    0.7477598884818090,    0.1740356302468940,    0.0391022406356488,    0.0391022406356488,    0.0143395670177665,
    0.1740356302468940,    0.0391022406356488,    0.7477598884818090,    0.0391022406356488,    0.0143395670177665,
    0.7477598884818090,    0.0391022406356488,    0.1740356302468940,    0.0391022406356488,    0.0143395670177665,
    0.1740356302468940,    0.0391022406356488,    0.0391022406356488,    0.7477598884818090,    0.0143395670177665,
    0.7477598884818090,    0.0391022406356488,    0.0391022406356488,    0.1740356302468940,    0.0143395670177665,
    0.0391022406356488,    0.1740356302468940,    0.7477598884818090,    0.0391022406356488,    0.0143395670177665,
    0.0391022406356488,    0.7477598884818090,    0.1740356302468940,    0.0391022406356488,    0.0143395670177665,
    0.0391022406356488,    0.1740356302468940,    0.0391022406356488,    0.7477598884818090,    0.0143395670177665,
    0.0391022406356488,    0.7477598884818090,    0.0391022406356488,    0.1740356302468940,    0.0143395670177665,
    0.0391022406356488,    0.0391022406356488,    0.1740356302468940,    0.7477598884818090,    0.0143395670177665,
    0.0391022406356488,    0.0391022406356488,    0.7477598884818090,    0.1740356302468940,    0.0143395670177665,
    0.4547545999844830,    0.4547545999844830,    0.0452454000155172,    0.0452454000155172,    0.0250305395686746,
    0.4547545999844830,    0.0452454000155172,    0.4547545999844830,    0.0452454000155172,    0.0250305395686746,
    0.4547545999844830,    0.0452454000155172,    0.0452454000155172,    0.4547545999844830,    0.0250305395686746,
    0.0452454000155172,    0.4547545999844830,    0.4547545999844830,    0.0452454000155172,    0.0250305395686746,
    0.0452454000155172,    0.4547545999844830,    0.0452454000155172,    0.4547545999844830,    0.0250305395686746,
    0.0452454000155172,    0.0452454000155172,    0.4547545999844830,    0.4547545999844830,    0.0250305395686746,
    0.5031186450145980,    0.2232010379623150,    0.2232010379623150,    0.0504792790607720,    0.0479839333057554,
    0.2232010379623150,    0.5031186450145980,    0.2232010379623150,    0.0504792790607720,    0.0479839333057554,
    0.2232010379623150,    0.2232010379623150,    0.5031186450145980,    0.0504792790607720,    0.0479839333057554,
    0.5031186450145980,    0.2232010379623150,    0.0504792790607720,    0.2232010379623150,    0.0479839333057554,
    0.2232010379623150,    0.5031186450145980,    0.0504792790607720,    0.2232010379623150,    0.0479839333057554,
    0.2232010379623150,    0.2232010379623150,    0.0504792790607720,    0.5031186450145980,    0.0479839333057554,
    0.5031186450145980,    0.0504792790607720,    0.2232010379623150,    0.2232010379623150,    0.0479839333057554,
    0.2232010379623150,    0.0504792790607720,    0.5031186450145980,    0.2232010379623150,    0.0479839333057554,
    0.2232010379623150,    0.0504792790607720,    0.2232010379623150,    0.5031186450145980,    0.0479839333057554,
    0.0504792790607720,    0.5031186450145980,    0.2232010379623150,    0.2232010379623150,    0.0479839333057554,
    0.0504792790607720,    0.2232010379623150,    0.5031186450145980,    0.2232010379623150,    0.0479839333057554,
    0.0504792790607720,    0.2232010379623150,    0.2232010379623150,    0.5031186450145980,    0.0479839333057554,
    0.2500000000000000,    0.2500000000000000,    0.2500000000000000,    0.2500000000000000,    0.0931745731195340
};

static unsigned int nv5 = 35;

static double v6[] = {
    0.9551438045408220,    0.0149520651530592,    0.0149520651530592,    0.0149520651530592,    0.0010373112336140,
    0.0149520651530592,    0.9551438045408220,    0.0149520651530592,    0.0149520651530592,    0.0010373112336140,
    0.0149520651530592,    0.0149520651530592,    0.9551438045408220,    0.0149520651530592,    0.0010373112336140,
    0.0149520651530592,    0.0149520651530592,    0.0149520651530592,    0.9551438045408220,    0.0010373112336140,
    0.7799760084415400,    0.1518319491659370,    0.0340960211962615,    0.0340960211962615,    0.0096016645399480,
    0.1518319491659370,    0.7799760084415400,    0.0340960211962615,    0.0340960211962615,    0.0096016645399480,
    0.7799760084415400,    0.0340960211962615,    0.1518319491659370,    0.0340960211962615,    0.0096016645399480,
    0.1518319491659370,    0.0340960211962615,    0.7799760084415400,    0.0340960211962615,    0.0096016645399480,
    0.7799760084415400,    0.0340960211962615,    0.0340960211962615,    0.1518319491659370,    0.0096016645399480,
    0.1518319491659370,    0.0340960211962615,    0.0340960211962615,    0.7799760084415400,    0.0096016645399480,
    0.0340960211962615,    0.7799760084415400,    0.1518319491659370,    0.0340960211962615,    0.0096016645399480,
    0.0340960211962615,    0.1518319491659370,    0.7799760084415400,    0.0340960211962615,    0.0096016645399480,
    0.0340960211962615,    0.7799760084415400,    0.0340960211962615,    0.1518319491659370,    0.0096016645399480,
    0.0340960211962615,    0.1518319491659370,    0.0340960211962615,    0.7799760084415400,    0.0096016645399480,
    0.0340960211962615,    0.0340960211962615,    0.7799760084415400,    0.1518319491659370,    0.0096016645399480,
    0.0340960211962615,    0.0340960211962615,    0.1518319491659370,    0.7799760084415400,    0.0096016645399480,
    0.3549340560639790,    0.5526556431060170,    0.0462051504150017,    0.0462051504150017,    0.0164493976798232,
    0.5526556431060170,    0.3549340560639790,    0.0462051504150017,    0.0462051504150017,    0.0164493976798232,
    0.3549340560639790,    0.0462051504150017,    0.5526556431060170,    0.0462051504150017,    0.0164493976798232,
    0.5526556431060170,    0.0462051504150017,    0.3549340560639790,    0.0462051504150017,    0.0164493976798232,
    0.3549340560639790,    0.0462051504150017,    0.0462051504150017,    0.5526556431060170,    0.0164493976798232,
    0.5526556431060170,    0.0462051504150017,    0.0462051504150017,    0.3549340560639790,    0.0164493976798232,
    0.0462051504150017,    0.3549340560639790,    0.5526556431060170,    0.0462051504150017,    0.0164493976798232,
    0.0462051504150017,    0.5526556431060170,    0.3549340560639790,    0.0462051504150017,    0.0164493976798232,
    0.0462051504150017,    0.3549340560639790,    0.0462051504150017,    0.5526556431060170,    0.0164493976798232,
    0.0462051504150017,    0.5526556431060170,    0.0462051504150017,    0.3549340560639790,    0.0164493976798232,
    0.0462051504150017,    0.0462051504150017,    0.3549340560639790,    0.5526556431060170,    0.0164493976798232,
    0.0462051504150017,    0.0462051504150017,    0.5526556431060170,    0.3549340560639790,    0.0164493976798232,
    0.5381043228880020,    0.2281904610687610,    0.2281904610687610,    0.0055147549744775,    0.0153747766513310,
    0.2281904610687610,    0.5381043228880020,    0.2281904610687610,    0.0055147549744775,    0.0153747766513310,
    0.2281904610687610,    0.2281904610687610,    0.5381043228880020,    0.0055147549744775,    0.0153747766513310,
    0.5381043228880020,    0.2281904610687610,    0.0055147549744775,    0.2281904610687610,    0.0153747766513310,
    0.2281904610687610,    0.5381043228880020,    0.0055147549744775,    0.2281904610687610,    0.0153747766513310,
    0.2281904610687610,    0.2281904610687610,    0.0055147549744775,    0.5381043228880020,    0.0153747766513310,
    0.5381043228880020,    0.0055147549744775,    0.2281904610687610,    0.2281904610687610,    0.0153747766513310,
    0.2281904610687610,    0.0055147549744775,    0.5381043228880020,    0.2281904610687610,    0.0153747766513310,
    0.2281904610687610,    0.0055147549744775,    0.2281904610687610,    0.5381043228880020,    0.0153747766513310,
    0.0055147549744775,    0.5381043228880020,    0.2281904610687610,    0.2281904610687610,    0.0153747766513310,
    0.0055147549744775,    0.2281904610687610,    0.5381043228880020,    0.2281904610687610,    0.0153747766513310,
    0.0055147549744775,    0.2281904610687610,    0.2281904610687610,    0.5381043228880020,    0.0153747766513310,
    0.1961837595745600,    0.3523052600879940,    0.3523052600879940,    0.0992057202494530,    0.0293520118375230,
    0.3523052600879940,    0.1961837595745600,    0.3523052600879940,    0.0992057202494530,    0.0293520118375230,
    0.3523052600879940,    0.3523052600879940,    0.1961837595745600,    0.0992057202494530,    0.0293520118375230,
    0.1961837595745600,    0.3523052600879940,    0.0992057202494530,    0.3523052600879940,    0.0293520118375230,
    0.3523052600879940,    0.1961837595745600,    0.0992057202494530,    0.3523052600879940,    0.0293520118375230,
    0.3523052600879940,    0.3523052600879940,    0.0992057202494530,    0.1961837595745600,    0.0293520118375230,
    0.1961837595745600,    0.0992057202494530,    0.3523052600879940,    0.3523052600879940,    0.0293520118375230,
    0.3523052600879940,    0.0992057202494530,    0.1961837595745600,    0.3523052600879940,    0.0293520118375230,
    0.3523052600879940,    0.0992057202494530,    0.3523052600879940,    0.1961837595745600,    0.0293520118375230,
    0.0992057202494530,    0.1961837595745600,    0.3523052600879940,    0.3523052600879940,    0.0293520118375230,
    0.0992057202494530,    0.3523052600879940,    0.1961837595745600,    0.3523052600879940,    0.0293520118375230,
    0.0992057202494530,    0.3523052600879940,    0.3523052600879940,    0.1961837595745600,    0.0293520118375230,
    0.5965649956210170,    0.1344783347929940,    0.1344783347929940,    0.1344783347929940,    0.0366291366405108,
    0.1344783347929940,    0.5965649956210170,    0.1344783347929940,    0.1344783347929940,    0.0366291366405108,
    0.1344783347929940,    0.1344783347929940,    0.5965649956210170,    0.1344783347929940,    0.0366291366405108,
    0.1344783347929940,    0.1344783347929940,    0.1344783347929940,    0.5965649956210170,    0.0366291366405108
};

static unsigned int nv6 = 56;

/* Linearly interpolate the position depending on the tetrahedron */
void integrate_interpolatepositionvol(unsigned int dim, double *x[4], double *lambda, double *xout) {
    for (unsigned int j=0; j<dim; j++) {
        xout[j]=0;
        for (unsigned int k=0; k<4; k++) xout[j]+=lambda[k]*x[k][j];
    }
}

/* Interpolate any quantities. */
void integrate_interpolatequantitiesvol(unsigned int dim, double *lambda, unsigned int nquantity, value *quantity[3], value *qout) {
    for (unsigned int i=0; i<nquantity; i++) {
        if (MORPHO_ISFLOAT(quantity[0][i])) {
            double val = lambda[0]*MORPHO_GETFLOATVALUE(quantity[0][i])+
                         lambda[1]*MORPHO_GETFLOATVALUE(quantity[1][i])+
                         lambda[2]*MORPHO_GETFLOATVALUE(quantity[2][i])+
                         lambda[3]*MORPHO_GETFLOATVALUE(quantity[3][i]);
            qout[i]=MORPHO_FLOAT(val);
        } else if (MORPHO_ISMATRIX(quantity[0][i]) && MORPHO_ISMATRIX(quantity[1][i]) && MORPHO_ISMATRIX(quantity[2][i]) && MORPHO_ISMATRIX(quantity[3][i])) {
            objectmatrix *m0=MORPHO_GETMATRIX(quantity[0][i]),
                         *m1=MORPHO_GETMATRIX(quantity[1][i]),
                         *m2=MORPHO_GETMATRIX(quantity[2][i]),
                         *m3=MORPHO_GETMATRIX(quantity[3][i]),
                         *out=(MORPHO_ISMATRIX(qout[i]) ? MORPHO_GETMATRIX(qout[i]): NULL);
            
            if (!out) {
                out = object_clonematrix(m0);
                qout[i]=MORPHO_OBJECT(out);
            }
            
            for (unsigned int i=0; i<m0->ncols*m0->nrows; i++) {
                out->elements[i] = lambda[0]*m0->elements[i]+lambda[1]*m1->elements[i]+lambda[2]*m2->elements[i]+lambda[3]*m3->elements[i];
            }
        }
    }
}

int nf = 0;

/** Integrate over an volume element given a specified integration rule
 * @param[in] function     - function to integrate
 * @param[in] nsamples     - number of sampling pts
 * @param[in] integrationrule - integration rule data
 * @param[in] dim                - Dimension of the vertices
 * @param[in] x                     - vertices of the line x[0] = {x,y,z} etc.
 * @param[in] nquantity   - number of quantities per vertex
 * @param[in] quantity     - List of quantities for each vertex.
 * @param[in] ref                 - a pointer to any data required by the function
 * @param[out] out               - estimate of the integral
 * @returns True on success */
bool integrate_integratevol(integrandfunction *function, unsigned int nsamples, double *integrationrule, unsigned int dim, double *x[4], unsigned int nquantity, value *quantity[3], value *q, void *ref, double *out) {
    double xx[dim];
    double r[nsamples], rout=0;
    double fout = 0;
    
    for (unsigned int i=0; i<nsamples; i++) {
        double *lambda=integrationrule+5*i;
        double w = integrationrule[5*i+4];
        
        integrate_interpolatepositionvol(dim, x, lambda, xx);
        if (nquantity) integrate_interpolatequantitiesvol(dim, lambda, nquantity, quantity, q);
        nf++;
        if ((*function) (dim, lambda, xx, nquantity, q, ref, &fout)) {
            r[i] = fout;
            rout+=w*r[i];
        } else{
            return false;
        }
        
    }
    
    *out = rout;
    return true;
}

/* Subdivision */
static unsigned int vsub[] =  { 1, 4, 7, 8,
                                0, 4, 7, 9,
                                0, 4, 8, 9,
                                4, 7, 8, 9,
                                0, 5, 7, 9,
                                0, 6, 8, 9,
                                2, 5, 7, 9,
                                3, 6, 8, 9 };

static unsigned int nvsub = 8;

/** Integrate over an volume element
 * @param[in] function     - function to integrate
 * @param[in] dim                - Dimension of the vertices
 * @param[in] x                     - vertices of the line x[0] = {x,y,z} etc.
 * @param[in] nquantity   - number of quantities per vertex
 * @param[in] quantity     - List of quantities for each vertex.
 * @param[in] ref                 - a pointer to any data required by the function
 * @param[in] ge                   - Global estimate of the integral (used for recursion).
 * @param[out] out               - estimate of the integral
 * @returns True on success */
bool integrate_volint(integrandfunction *function, unsigned int dim, double *x[4], unsigned int nquantity, value *quantity[4], value *q, void *ref, unsigned int recursiondepth, double ge, double *out) {
    double r1, r2, r3;
    double gest=ge;
    double af=pow(1.0/nvsub, (double) recursiondepth); // Volume of total tetrahedron calculated from recursion depth
    
    if (!integrate_integratevol(function, nv5, v5, dim, x, nquantity, quantity, q, ref, &r1)) return false;
    if (!integrate_integratevol(function, nv6, v6, dim, x, nquantity, quantity, q, ref, &r2)) return false;

    if (recursiondepth==0) gest=fabs(r2); // If at top level construct a global estimate of the integral

    double eps=r2-r1;
    eps*=af;
    if (gest>MORPHO_EPS) eps/=gest; // Globally relative estimate using volume factor
    
    if (fabs(eps)<INTEGRATE_ACCURACYGOAL)  { // We converged
        *out=r2;
        return true;
    }
    
    // Subdivision strategy
    double *xn[4]; /* Will hold the vertices. */
    double x01[dim], x02[dim], x03[dim], x12[dim], x13[dim], x23[dim]; /* New ertices from midpoints */
    double *xx[] = { x[0], x[1], x[2], x[3], x01, x02, x03, x12, x13, x23 }; // All vertices
    value q01[nquantity+1], q02[nquantity+1], q03[nquantity+1], q12[nquantity+1], q13[nquantity+1], q23[nquantity+1];
    value *qq[] = { quantity[0], quantity[1], quantity[2], quantity[3], q01, q02, q03, q12, q13, q23 }; // All vertices
    value *qn[4];
    
    r3=0.0;
    /* New vertices s*/
    for (unsigned int i=0; i<dim; i++) {
        x01[i] = 0.5*(x[0][i]+x[1][i]);
        x02[i] = 0.5*(x[0][i]+x[2][i]);
        x03[i] = 0.5*(x[0][i]+x[3][i]);
        x12[i] = 0.5*(x[1][i]+x[2][i]);
        x13[i] = 0.5*(x[1][i]+x[3][i]);
        x23[i] = 0.5*(x[2][i]+x[3][i]);
    }
    
    /* Quantities */
    if (nquantity) {
        double ll[4];
        for (unsigned int i=0; i<nquantity; i++) { q01[i]=MORPHO_NIL; q02[i]=MORPHO_NIL; q03[i]=MORPHO_NIL; q12[i]=MORPHO_NIL; q13[i]=MORPHO_NIL; q23[i]=MORPHO_NIL; }
        
        ll[0]=0.5; ll[1]=0.5; ll[2]=0.0; ll[3]=0.0;
        integrate_interpolatequantitiesvol(dim, ll, nquantity, quantity, q01);
        ll[0]=0.5; ll[1]=0.0; ll[2]=0.5; ll[3]=0.0;
        integrate_interpolatequantitiesvol(dim, ll, nquantity, quantity, q02);
        ll[0]=0.5; ll[1]=0.0; ll[2]=0.0; ll[3]=0.5;
        integrate_interpolatequantitiesvol(dim, ll, nquantity, quantity, q03);
        ll[0]=0.0; ll[1]=0.5; ll[2]=0.5; ll[3]=0.0;
        integrate_interpolatequantitiesvol(dim, ll, nquantity, quantity, q12);
        ll[0]=0.0; ll[1]=0.5; ll[2]=0.0; ll[3]=0.5;
        integrate_interpolatequantitiesvol(dim, ll, nquantity, quantity, q13);
        ll[0]=0.0; ll[1]=0.0; ll[2]=0.5; ll[3]=0.5;
        integrate_interpolatequantitiesvol(dim, ll, nquantity, quantity, q23);
    }
    
    double rr = 0.0;
    
    for (unsigned int i=0; i<nvsub; i++) {
        double sub;
        for (unsigned int j=0; j<4; j++) xn[j]=xx[vsub[4*i+j]];
        if (nquantity) for (unsigned int j=0; j<4; j++) qn[j]=qq[vsub[4*i+j]];
        
        if (!integrate_volint(function, dim, xn, nquantity, qn, q, ref, recursiondepth+1, gest, &sub)) goto integrate_volint_cleanup;
        
        rr+=sub;
    }
    
    *out=rr/nvsub;
    
integrate_volint_cleanup:
    
    return true;
    
}

/* **********************************************************************
 * Public interface
 * ********************************************************************** */

/** Integrate over an element - public interface.
 * @param[in] integrand   - integrand
 * @param[in] dim                - Dimension of the vertices
 * @param[in] grade            - Grade to integrate over
 * @param[in] x                     - vertices of the triangle x[0] = {x,y,z} etc.
 * @param[in] nquantity   - number of quantities per vertex
 * @param[in] quantity     - List of quantities for each endpoint.
 * @param[in] ref                - a pointer to any data required by the function
 * @param[out] out              - value of the integral
 * @returns true on success.
 */
bool integrate_integrate(integrandfunction *integrand, unsigned int dim, unsigned int grade, double **x, unsigned int nquantity, value **quantity, void *ref, double *out) {
    double result=0.0;
    value q[nquantity+1];
    bool success=false;
    
    for (unsigned int i=0; i<nquantity; i++) q[i]=MORPHO_NIL;
    if (quantity) integrate_recognizequantities(nquantity, quantity[0], q);
    
    /* Do the integration */
    switch (grade) {
        case 1:
            success=integrate_lineint(integrand, dim, x, nquantity, quantity, q, ref, 0, 0.0, &result);
            break;
        case 2:
            success=integrate_areaint(integrand, dim, x, nquantity, quantity, q, ref, 0, 0.0, &result);
            break;
        case 3:
            success=integrate_volint(integrand, dim, x, nquantity, quantity, q, ref, 0, 0.0, &result);
            break;
    }
    
    /* Free any quantities allocated */
    for (unsigned int i=0; i<nquantity; i++) {
        if (MORPHO_ISOBJECT(q[i])) object_free(MORPHO_GETOBJECT(q[i]));
    }
    
    *out = result;
    
    return success;
}

/* **********************************************************************
 * Testing code
 * ********************************************************************** */

double testintegrand(unsigned int dim, double *t, double *x, unsigned int nquantity, value *quantity, void *data) {
    //return pow(x[0]*(1.0-x[0]), 20.0);
    objectmatrix *m = MORPHO_GETMATRIX(quantity[0]);
    double tr=0;
    matrix_trace(m, &tr);
    return pow(tr,50);//pow(MORPHO_GETFLOATVALUE(quantity[0]),50);
}

/*void integrate_test(void) {
    double x0[3] = { 0,0,0 };
    double x1[3] = { 1,0,0 };
    //double *xx[2] = { x0, x1 };
    double m0s[] = {0, 0, 0, 0};
    double m1s[] = {0.5, 0, 0, 0.5};
    objectmatrix m0 = MORPHO_STATICMATRIX(m0s, 2, 2);
    objectmatrix m1 = MORPHO_STATICMATRIX(m1s, 2, 2);
    value v0[1] = { MORPHO_OBJECT(&m0) };
    value v1[1] = { MORPHO_OBJECT(&m1) };
    value *v[2] = { v0, v1 };
    double out;
    //integrate_integrate(testintegrand, 3, 1, xx, 1, v, NULL, &out);
    //printf("integral value: %g\n", out);
}*/


/* **********************************************************************
 * New integrator
 * ********************************************************************** */

/* **********************************************
 * Quadrature rules
 * ********************************************** */

/* --------------------------------
 * Simple midpoint-simpson rule
 * -------------------------------- */

double midpointnodes[] = {
    0.5, 0.5, // Midpoint
    
    0.0, 1.0, // } Simpsons extension
    1.0, 0.0, // }
};

double midpointweights[] = {
    1.0
};

double simpsonweights[] = {
    0.66666666666666667, 0.16666666666666667, 0.16666666666666667
};

quadraturerule simpson = {
    .name = "simpson",
    .grade = 1,
    .order = 3,
    .nnodes = 1,
    .nodes = midpointnodes,
    .weights = simpsonweights,
    .ext = NULL
};

quadraturerule midpoint = {
    .name = "midpoint",
    .grade = 1,
    .order = 1,
    .nnodes = 1,
    .nodes = midpointnodes,
    .weights = midpointweights,
    .ext = &simpson
};

/* --------------------------------
 * Gauss-Kronrod 1-3 rule
 * -------------------------------- */

double gk13nds[] = {
    0.50000000000000000000, 0.50000000000000000000,
    0.11270166537925831148, 0.88729833462074168852,
    0.88729833462074168852, 0.11270166537925831148
};

double g1wts[] = {
    1.0,
};

double k3wts[] = {
    0.4444444444444444444445, 0.2777777777777777777778,
    0.277777777777777777778
};

quadraturerule kronrod3 = {
    .name = "kronrod3",
    .grade = 1,
    .order = 3,
    .nnodes = 3,
    .nodes = gk13nds,
    .weights = k3wts,
    .ext = NULL
};

quadraturerule gauss1 = {
    .name = "gauss1",
    .grade = 1,
    .order = 1,
    .nnodes = 1,
    .nodes = gk13nds,
    .weights = g1wts,
    .ext = &kronrod3
};

/* --------------------------------
 * Gauss-Kronrod 2-5 rule
 * -------------------------------- */

double gk25nds[] = {
    0.21132486540518711775, 0.78867513459481288225,
    0.78867513459481288225, 0.21132486540518711775,
    
    0.037089950113724269217, 0.96291004988627573078,
    0.50000000000000000000, 0.50000000000000000000,
    0.96291004988627573078, 0.037089950113724269217
};

double gauss2wts[] = {
    0.5, 0.5, // Gauss weights
};

double kronrod5wts[] = {
    0.2454545454545454545455, // Kronrod extension
    0.245454545454545454546,
    0.098989898989898989899,
    0.3111111111111111111111,
    0.098989898989898989899
};

quadraturerule kronrod5 = {
    .name = "kronrod5",
    .grade = 1,
    .order = 7,
    .nnodes = 5,
    .nodes = gk25nds,
    .weights = kronrod5wts,
    .ext = NULL
};

quadraturerule gauss2 = {
    .name = "gauss2",
    .grade = 1,
    .order = 3,
    .nnodes = 2,
    .nodes = gk25nds,
    .weights = gauss2wts,
    .ext = &kronrod5
};

/* --------------------------------
 * Gauss-Kronrod 5-11 rule
 * -------------------------------- */

// Appears to be Mathematica's default integrator!

double gk511nds[] = {
    0.046910077030668003601,  0.9530899229693319964,
    0.23076534494715845448,   0.76923465505284154552,
    0.5,0.5,
    0.76923465505284154552,   0.23076534494715845448,
    0.9530899229693319964,    0.046910077030668003601,
    
    0.0079573199525787677519, 0.99204268004742123225,
    0.12291663671457538978,   0.87708336328542461022,
    0.36018479341910840329,   0.63981520658089159671,
    0.63981520658089159671,   0.36018479341910840329,
    0.87708336328542461022,   0.12291663671457538978,
    0.99204268004742123225,   0.0079573199525787677519
};

double gauss5wts[] = {
    0.118463442528094543757, 0.2393143352496832340206, 0.2844444444444444444444,
    0.2393143352496832340206, 0.118463442528094543757
};

double kronrod11wts[] = {
    0.0576166583112366970123, // Kronrod
    0.12052016961432379335,
    0.1414937089287456066021,
    0.12052016961432379335,
    0.0576166583112366970123,
    0.02129101837554091643225,
    0.093400398278246328734,
    0.136424900956279461171,
    0.136424900956279461171,
    0.0934003982782463287339,
    0.0212910183755409164322
};

quadraturerule kronrod11 = {
    .name = "kronrod11",
    .grade = 1,
    .order = 16,
    .nnodes = 11,
    .nodes = gk511nds,
    .weights = kronrod11wts,
    .ext = NULL
};

quadraturerule gauss5 = {
    .name = "gauss5",
    .grade = 1,
    .order = 9,
    .nnodes = 5,
    .nodes = gk511nds,
    .weights = gauss5wts,
    .ext = &kronrod11
};

/* --------------------------------
 * Gauss-Kronrod 7-15 rule
 * -------------------------------- */

double gk715nds[] = {
    0.0254460438286207377369, 0.9745539561713792622631, // Gauss nodes
    0.1292344072003027800681, 0.8707655927996972199320,
    0.2970774243113014165467, 0.7029225756886985834533,
    0.5, 0.5,
    0.7029225756886985834533, 0.2970774243113014165467,
    0.8707655927996972199320, 0.1292344072003027800681,
    0.9745539561713792622631, 0.0254460438286207377369,
    
    0.0042723144395936803966, 0.9957276855604063196035, // Kronrod extension
    0.0675677883201154636052, 0.9324322116798845363949,
    0.2069563822661544348530, 0.7930436177338455651471,
    0.3961075224960507661997, 0.6038924775039492338004,
    0.6038924775039492338004, 0.3961075224960507661997,
    0.7930436177338455651471, 0.2069563822661544348530,
    0.9324322116798845363949, 0.0675677883201154636052,
    0.9957276855604063196035, 0.0042723144395936803966
};

double gauss7wts[] = {
    0.0647424830844348466355, // Gauss weights
    0.1398526957446383339505,
    0.1909150252525594724752,
    0.20897959183673469387755,
    0.1909150252525594724752,
    0.13985269574463833395075,
    0.0647424830844348466353
};
    
double kronrod15wts[] = {
    0.0315460463149892766454,
    0.070326629857762959373,
    0.0951752890323927049567,
    0.104741070542363914007,
    0.095175289032392704957,
    0.0703266298577629593726,
    0.0315460463149892766454,
    
    0.0114676610052646124819,
    0.0523950051611250919200,
    0.0845023633196339514133,
    0.1022164700376494462071,
    0.1022164700376494462071,
    0.0845023633196339514133,
    0.05239500516112509192,
    0.01146766100526461248187
};

quadraturerule kronrod15 = {
    .name = "kronrod15",
    .grade = 1,
    .order = 22,
    .nnodes = 15,
    .nodes = gk715nds,
    .weights = kronrod15wts,
    .ext = NULL
};

quadraturerule gauss7 = {
    .name = "gauss7",
    .grade = 1,
    .order = 13,
    .nnodes = 7,
    .nodes = gk715nds,
    .weights = gauss7wts,
    .ext = &kronrod15
};

/* --------------------------------
 * Triangle
 * -------------------------------- */

/* Quadrature rules based on Walkington, "Quadrature on Simplices of arbitrary dimension" */

double tripts[] = {
    0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
    0.6000000000000000, 0.2000000000000000, 0.2000000000000000,
    0.2000000000000000, 0.6000000000000000, 0.2000000000000000,
    0.2000000000000000, 0.2000000000000000, 0.6000000000000000,
    
    0.7142857142857143, 0.1428571428571429, 0.1428571428571429,
    0.1428571428571429, 0.7142857142857143, 0.1428571428571429,
    0.1428571428571429, 0.1428571428571429, 0.7142857142857143,
    0.4285714285714286, 0.4285714285714286, 0.1428571428571429,
    0.4285714285714286, 0.1428571428571429, 0.4285714285714286,
    0.1428571428571429, 0.4285714285714286, 0.4285714285714286,
    
    0.7777777777777778, 0.1111111111111111, 0.1111111111111111,
    0.1111111111111111, 0.7777777777777778, 0.1111111111111111,
    0.1111111111111111, 0.1111111111111111, 0.7777777777777778,
    0.3333333333333333, 0.5555555555555556, 0.1111111111111111,
    0.3333333333333333, 0.1111111111111111, 0.5555555555555556,
    0.5555555555555556, 0.3333333333333333, 0.1111111111111111,
    0.5555555555555556, 0.1111111111111111, 0.3333333333333333,
    0.1111111111111111, 0.3333333333333333, 0.5555555555555556,
    0.1111111111111111, 0.5555555555555556, 0.3333333333333333,
    0.3333333333333333, 0.3333333333333333, 0.3333333333333333
};

double tri4wts[] = {
    -0.5625, 0.5208333333333332, 0.5208333333333332, 0.5208333333333332
};
    
double tri10wts[] = {
    0.1265625, -0.5425347222222222, -0.5425347222222222, -0.5425347222222222,
    0.4168402777777778, 0.4168402777777778, 0.4168402777777778, 0.4168402777777778,
    0.4168402777777778, 0.4168402777777778
};

double tri20wts[] = {
    -0.0158203125, 0.2422030009920634, 0.2422030009920634, 0.2422030009920634,
    -0.6382866753472222, -0.6382866753472222, -0.6382866753472222, -0.6382866753472222,
    -0.6382866753472222, -0.6382866753472222,
    
    0.4118931361607142, 0.4118931361607142, 0.4118931361607142, 0.4118931361607142,
    0.4118931361607142, 0.4118931361607142, 0.4118931361607142, 0.4118931361607142,
    0.4118931361607142, 0.4118931361607142
};

quadraturerule tri20 = {
    .name = "tri20",
    .grade = 2,
    .order = 8,
    .nnodes = 20,
    .nodes = tripts,
    .weights = tri20wts,
    .ext = NULL
};

quadraturerule tri10 = {
    .name = "tri10",
    .grade = 2,
    .order = 5,
    .nnodes = 10,
    .nodes = tripts,
    .weights = tri10wts,
    .ext = &tri20
};

quadraturerule tri4 = {
    .name = "tri4",
    .grade = 2,
    .order = 3, // 4pt rule is order 3,
    .nnodes = 4,
    .nodes = tripts,
    .weights = tri4wts,
    .ext = &tri10
};

// CUBTRI rule from D. P. Laurie, ACM Transactions on Mathematical Software, Vol 8, No. 2, June 1982,Pages 210-218

double cubtripts[] = {
    0.333333333333333333,0.333333333333333333,0.333333333333333333,
    0.797426985353087322,0.101286507323456339,0.101286507323456339,
    0.101286507323456339,0.101286507323456339,0.797426985353087322,
    0.101286507323456339,0.797426985353087322,0.101286507323456339,
    0.0597158717897698205,0.47014206410511509,0.47014206410511509,
    0.47014206410511509,0.47014206410511509,0.0597158717897698205,
    0.47014206410511509,0.0597158717897698205,0.47014206410511509,
    
    0.941038278231120867,0.0294808608844395667,0.0294808608844395667,
    0.0294808608844395667,0.0294808608844395667,0.941038278231120867,
    0.0294808608844395667,0.941038278231120867,0.0294808608844395667,
    0.535795346449899265,0.232102326775050368,0.232102326775050368,
    0.232102326775050368,0.232102326775050368,0.535795346449899265,
    0.232102326775050368,0.535795346449899265,0.232102326775050368,
    0.0294808608844395667,0.232102326775050368,0.738416812340510066,
    0.232102326775050368,0.0294808608844395667,0.738416812340510066,
    0.738416812340510066,0.0294808608844395667,0.232102326775050368,
    0.738416812340510066,0.232102326775050368,0.0294808608844395667,
    0.0294808608844395667,0.738416812340510066,0.232102326775050368,
    0.232102326775050368,0.738416812340510066,0.0294808608844395667
};

double cubtri7wts[] = {
    0.225000000000000000, 0.125939180544827153, 0.125939180544827153,
    0.125939180544827153, 0.132394152788506181, 0.132394152788506181,
    0.132394152788506181
};

double cubtri19wts[] = {
    0.0378610912003146833, 0.0376204254131829721, 0.0376204254131829721,
    0.0376204254131829721, 0.0783573522441173376, 0.0783573522441173376,
    0.0783573522441173376, 0.0134442673751654019, 0.0134442673751654019,
    0.0134442673751654019, 0.116271479656965896, 0.116271479656965896,
    0.116271479656965896, 0.0375097224552317488, 0.0375097224552317488,
    0.0375097224552317488, 0.0375097224552317488, 0.0375097224552317488,
    0.0375097224552317488
};

quadraturerule cubtri19 = {
    .name = "cubtri19",
    .grade = 2,
    .order = 8,
    .nnodes = 19,
    .nodes = cubtripts,
    .weights = cubtri19wts,
};

quadraturerule cubtri7 = {
    .name = "cubtri7",
    .grade = 2,
    .order = 5,
    .nnodes = 7,
    .nodes = cubtripts,
    .weights = cubtri7wts,
};

/* --------------------------------
 * Tetrahedron
 * -------------------------------- */

// Nodes and weights from Keast, Computer Methods in Applied Mechanics and Engineering,
//    Volume 55, Number 3, May 1986, pages 339-348.

double keast4pts[] = {
    0.25,0.25,0.25,0.25,
    0.78571428571428571,  0.071428571428571428, 0.071428571428571428, 0.071428571428571428,
    0.071428571428571428, 0.78571428571428571,  0.071428571428571428, 0.071428571428571428,
    0.071428571428571428, 0.071428571428571428, 0.78571428571428571,  0.071428571428571428,
    0.071428571428571428, 0.071428571428571428, 0.071428571428571428, 0.78571428571428571,
    0.39940357616679922,  0.39940357616679922,  0.10059642383320078,  0.10059642383320078,
    0.39940357616679922,  0.10059642383320078,  0.39940357616679922,  0.10059642383320078,
    0.39940357616679922,  0.10059642383320078,  0.10059642383320078,  0.39940357616679922,
    0.10059642383320078,  0.39940357616679922,  0.39940357616679922,  0.10059642383320078,
    0.10059642383320078,  0.39940357616679922,  0.10059642383320078,  0.39940357616679922,
    0.10059642383320078,  0.10059642383320078,  0.39940357616679922,  0.39940357616679922
};

double keast4wts[] = {
    -0.07893333333333333,
    0.04573333333333333333,0.04573333333333333333,
    0.04573333333333333333,0.04573333333333333333,
    0.149333333333333328,0.149333333333333328,0.149333333333333328,0.149333333333333328,0.149333333333333328,0.149333333333333328
};

quadraturerule keast4 = {
    .name = "keast4",
    .grade = 3,
    .order = 4,
    .nnodes = 11,
    .nodes = keast4pts,
    .weights = keast4wts,
    .ext = NULL
};

double keast5pts[] = {
    0.25,0.25,0.25,0.25,
    0,0.3333333333333333,0.3333333333333333,0.3333333333333333,
    0.3333333333333333,0,0.3333333333333333,0.3333333333333333,
    0.3333333333333333,0.3333333333333333,0,0.3333333333333333,
    0.3333333333333333,0.3333333333333333,0.3333333333333333,0,
    0.72727272727272727,0.090909090909090909,0.090909090909090909,0.090909090909090909,
    0.090909090909090909,0.72727272727272727,0.090909090909090909,0.090909090909090909,
    0.090909090909090909,0.090909090909090909,0.72727272727272727,0.090909090909090909,
    0.090909090909090909,0.090909090909090909,0.090909090909090909,0.72727272727272727,
    0.066550153573664281,0.066550153573664281,0.43344984642633573,0.43344984642633573,
    0.066550153573664281,0.43344984642633573,0.066550153573664281,0.43344984642633573,
    0.066550153573664281,0.43344984642633573,0.43344984642633573,0.066550153573664281,
    0.43344984642633573,0.066550153573664281,0.066550153573664281,0.43344984642633573,
    0.43344984642633573,0.066550153573664281,0.43344984642633573,0.066550153573664281,
    0.43344984642633573,0.43344984642633573,0.066550153573664281,0.066550153573664281
};

double keast5wts[] = {
    0.181702068582535114,
    0.0361607142857142958, 0.0361607142857142958, 0.0361607142857142958,
    0.0361607142857142958, 0.069871494516173845,
    
    0.069871494516173845,0.069871494516173845,0.069871494516173845,0.06569484936831872,
    0.06569484936831872,0.06569484936831872,0.06569484936831872,0.06569484936831872,
    0.06569484936831872
};

quadraturerule keast5 = {
    .name = "keast5",
    .grade = 3,
    .order = 5,
    .nnodes = 15,
    .nodes = keast5pts,
    .weights = keast5wts,
    .ext = NULL
};

// Nodes and weights from Journal of Computational and Applied Mathematics, 236, 17, 4348-4364 (2012)
double tet5pts[] = {
    0.91978967333688,0.0267367755543735,0.0267367755543735,0.0267367755543735,
    0.0267367755543735,0.91978967333688,0.0267367755543735,0.0267367755543735,
    0.0267367755543735,0.0267367755543735,0.91978967333688,0.0267367755543735,
    0.0267367755543735,0.0267367755543735,0.0267367755543735,0.91978967333688,
    0.174035630246894,0.747759888481809,0.0391022406356488,0.0391022406356488,
    0.747759888481809,0.174035630246894,0.0391022406356488,0.0391022406356488,
    0.174035630246894,0.0391022406356488,0.747759888481809,0.0391022406356488,
    0.747759888481809,0.0391022406356488,0.174035630246894,0.0391022406356488,
    0.174035630246894,0.0391022406356488,0.0391022406356488,0.747759888481809,
    0.747759888481809,0.0391022406356488,0.0391022406356488,0.174035630246894,
    0.0391022406356488,0.174035630246894,0.747759888481809,0.0391022406356488,
    0.0391022406356488,0.747759888481809,0.174035630246894,0.0391022406356488,
    0.0391022406356488,0.174035630246894,0.0391022406356488,0.747759888481809,
    0.0391022406356488,0.747759888481809,0.0391022406356488,0.174035630246894,
    0.0391022406356488,0.0391022406356488,0.174035630246894,0.747759888481809,
    0.0391022406356488,0.0391022406356488,0.747759888481809,0.174035630246894,
    0.454754599984483,0.454754599984483,0.0452454000155172,0.0452454000155172,
    0.454754599984483,0.0452454000155172,0.454754599984483,0.0452454000155172,
    0.454754599984483,0.0452454000155172,0.0452454000155172,0.454754599984483,
    0.0452454000155172,0.454754599984483,0.454754599984483,0.0452454000155172,
    0.0452454000155172,0.454754599984483,0.0452454000155172,0.454754599984483,
    0.0452454000155172,0.0452454000155172,0.454754599984483,0.454754599984483,
    0.503118645014598,0.223201037962315,0.223201037962315,0.050479279060772,
    0.223201037962315,0.503118645014598,0.223201037962315,0.050479279060772,
    0.223201037962315,0.223201037962315,0.503118645014598,0.050479279060772,
    0.503118645014598,0.223201037962315,0.050479279060772,0.223201037962315,
    0.223201037962315,0.503118645014598,0.050479279060772,0.223201037962315,
    0.223201037962315,0.223201037962315,0.050479279060772,0.503118645014598,
    0.503118645014598,0.050479279060772,0.223201037962315,0.223201037962315,
    0.223201037962315,0.050479279060772,0.503118645014598,0.223201037962315,
    0.223201037962315,0.050479279060772,0.223201037962315,0.503118645014598,
    0.050479279060772,0.503118645014598,0.223201037962315,0.223201037962315,
    0.050479279060772,0.223201037962315,0.503118645014598,0.223201037962315,
    0.050479279060772,0.223201037962315,0.223201037962315,0.503118645014598,
    0.25,0.25,0.25,0.25
};

double tet5wts[] = {
    0.0021900463965388,0.0021900463965388,0.0021900463965388,0.0021900463965388,
    0.0143395670177665,0.0143395670177665,0.0143395670177665,0.0143395670177665,
    0.0143395670177665,0.0143395670177665,0.0143395670177665,0.0143395670177665,
    0.0143395670177665,0.0143395670177665,0.0143395670177665,0.0143395670177665,
    0.0250305395686746,0.0250305395686746,0.0250305395686746,0.0250305395686746,
    0.0250305395686746,0.0250305395686746,0.0479839333057554,0.0479839333057554,
    0.0479839333057554,0.0479839333057554,0.0479839333057554,0.0479839333057554,
    0.0479839333057554,0.0479839333057554,0.0479839333057554,0.0479839333057554,
    0.0479839333057554,0.0479839333057554,0.093174573119534 };

quadraturerule tet5 = {
    .name = "tet5",
    .grade = 3,
    .order = 7,
    .nnodes = 35,
    .nodes = tet5pts,
    .weights = tet5wts,
    .ext = NULL
};

double tet6pts[] = {
    0.955143804540822,0.0149520651530592,0.0149520651530592,0.0149520651530592,
    0.0149520651530592,0.955143804540822,0.0149520651530592,0.0149520651530592,
    0.0149520651530592,0.0149520651530592,0.955143804540822,0.0149520651530592,
    0.0149520651530592,0.0149520651530592,0.0149520651530592,0.955143804540822,
    0.77997600844154,0.151831949165937,0.0340960211962615,0.0340960211962615,
    0.151831949165937,0.77997600844154,0.0340960211962615,0.0340960211962615,
    0.77997600844154,0.0340960211962615,0.151831949165937,0.0340960211962615,
    0.151831949165937,0.0340960211962615,0.77997600844154,0.0340960211962615,
    0.77997600844154,0.0340960211962615,0.0340960211962615,0.151831949165937,
    0.151831949165937,0.0340960211962615,0.0340960211962615,0.77997600844154,
    0.0340960211962615,0.77997600844154,0.151831949165937,0.0340960211962615,
    0.0340960211962615,0.151831949165937,0.77997600844154,0.0340960211962615,
    0.0340960211962615,0.77997600844154,0.0340960211962615,0.151831949165937,
    0.0340960211962615,0.151831949165937,0.0340960211962615,0.77997600844154,
    0.0340960211962615,0.0340960211962615,0.77997600844154,0.151831949165937,
    0.0340960211962615,0.0340960211962615,0.151831949165937,0.77997600844154,
    0.354934056063979,0.552655643106017,0.0462051504150017,0.0462051504150017,
    0.552655643106017,0.354934056063979,0.0462051504150017,0.0462051504150017,
    0.354934056063979,0.0462051504150017,0.552655643106017,0.0462051504150017,
    0.552655643106017,0.0462051504150017,0.354934056063979,0.0462051504150017,
    0.354934056063979,0.0462051504150017,0.0462051504150017,0.552655643106017,
    0.552655643106017,0.0462051504150017,0.0462051504150017,0.354934056063979,
    0.0462051504150017,0.354934056063979,0.552655643106017,0.0462051504150017,
    0.0462051504150017,0.552655643106017,0.354934056063979,0.0462051504150017,
    0.0462051504150017,0.354934056063979,0.0462051504150017,0.552655643106017,
    0.0462051504150017,0.552655643106017,0.0462051504150017,0.354934056063979,
    0.0462051504150017,0.0462051504150017,0.354934056063979,0.552655643106017,
    0.0462051504150017,0.0462051504150017,0.552655643106017,0.354934056063979,
    0.538104322888002,0.228190461068761,0.228190461068761,0.0055147549744775,
    0.228190461068761,0.538104322888002,0.228190461068761,0.0055147549744775,
    0.228190461068761,0.228190461068761,0.538104322888002,0.0055147549744775,
    0.538104322888002,0.228190461068761,0.0055147549744775,0.228190461068761,
    0.228190461068761,0.538104322888002,0.0055147549744775,0.228190461068761,
    0.228190461068761,0.228190461068761,0.0055147549744775,0.538104322888002,
    0.538104322888002,0.0055147549744775,0.228190461068761,0.228190461068761,
    0.228190461068761,0.0055147549744775,0.538104322888002,0.228190461068761,
    0.228190461068761,0.0055147549744775,0.228190461068761,0.538104322888002,
    0.0055147549744775,0.538104322888002,0.228190461068761,0.228190461068761,
    0.0055147549744775,0.228190461068761,0.538104322888002,0.228190461068761,
    0.0055147549744775,0.228190461068761,0.228190461068761,0.538104322888002,
    0.19618375957456,0.352305260087994,0.352305260087994,0.099205720249453,
    0.352305260087994,0.19618375957456,0.352305260087994,0.099205720249453,
    0.352305260087994,0.352305260087994,0.19618375957456,0.099205720249453,
    0.19618375957456,0.352305260087994,0.099205720249453,0.352305260087994,
    0.352305260087994,0.19618375957456,0.099205720249453,0.352305260087994,
    0.352305260087994,0.352305260087994,0.099205720249453,0.19618375957456,
    0.19618375957456,0.099205720249453,0.352305260087994,0.352305260087994,
    0.352305260087994,0.099205720249453,0.19618375957456,0.352305260087994,
    0.352305260087994,0.099205720249453,0.352305260087994,0.19618375957456,
    0.099205720249453,0.19618375957456,0.352305260087994,0.352305260087994,
    0.099205720249453,0.352305260087994,0.19618375957456,0.352305260087994,
    0.099205720249453,0.352305260087994,0.352305260087994,0.19618375957456,
    0.596564995621017,0.134478334792994,0.134478334792994,0.134478334792994,
    0.134478334792994,0.596564995621017,0.134478334792994,0.134478334792994,
    0.134478334792994,0.134478334792994,0.596564995621017,0.134478334792994,
    0.134478334792994,0.134478334792994,0.134478334792994,0.596564995621017
};

double tet6wts[] = {
    0.001037311233614,0.001037311233614,0.001037311233614,0.001037311233614,
    0.009601664539948,0.009601664539948,0.009601664539948,0.009601664539948,
    0.009601664539948,0.009601664539948,0.009601664539948,0.009601664539948,
    0.009601664539948,0.009601664539948,0.009601664539948,0.009601664539948,
    0.0164493976798232,0.0164493976798232,0.0164493976798232,0.0164493976798232,
    0.0164493976798232,0.0164493976798232,0.0164493976798232,0.0164493976798232,
    0.0164493976798232,0.0164493976798232,0.0164493976798232,0.0164493976798232,
    0.015374776651331,0.015374776651331,0.015374776651331,0.015374776651331,
    0.015374776651331,0.015374776651331,0.015374776651331,0.015374776651331,
    0.015374776651331,0.015374776651331,0.015374776651331,0.015374776651331,
    0.029352011837523,0.029352011837523,0.029352011837523,0.029352011837523,
    0.029352011837523,0.029352011837523,0.029352011837523,0.029352011837523,
    0.029352011837523,0.029352011837523,0.029352011837523,0.029352011837523,
    0.0366291366405108,0.0366291366405108,0.0366291366405108,0.0366291366405108
};

quadraturerule tet6 = {
    .name = "tet6",
    .grade = 3,
    .order = 9,
    .nnodes = 56,
    .nodes = tet6pts,
    .weights = tet6wts,
    .ext = NULL
};

// Grundmann-Möller embedded rules:
//   SIAM Journal on Numerical Analysis , Apr., 1978, Vol. 15, No. 2 (Apr., 1978), pp. 282-290
// See also a very clear example presented in
//   ACM Transactions on Mathematical Software, Volume 29, Issue 3, pp 297–308 (2003) */

double grundmannpts[] = {
    // Rule 1, order 3
    0.25,0.25,0.25,0.25,
    0.16666666666666666667,0.16666666666666666667,0.16666666666666666667,0.5,
    0.16666666666666666667,0.16666666666666666667,0.5,0.16666666666666666667,
    0.16666666666666666667,0.5,0.16666666666666666667,0.16666666666666666667,
    0.5,0.16666666666666666667,0.16666666666666666667,0.16666666666666666667,
    
    // Additional points for rule 2, order 5
    0.125,0.125,0.375,0.375,
    0.125,0.375,0.125,0.375,
    0.125,0.375,0.375,0.125,
    0.375,0.125,0.125,0.375,
    0.375,0.125,0.375,0.125,
    0.375,0.375,0.125,0.125,
    0.125,0.125,0.125,0.625,
    0.125,0.125,0.625,0.125,
    0.125,0.625,0.125,0.125,
    0.625,0.125,0.125,0.125,
    
    // Additional points for rule 3, order 7
    0.1,0.3,0.3,0.3,
    0.3,0.1,0.3,0.3,
    0.3,0.3,0.1,0.3,
    0.3,0.3,0.3,0.1,
    0.1,0.1,0.3,0.5,
    0.1,0.1,0.5,0.3,
    0.1,0.3,0.1,0.5,
    0.1,0.3,0.5,0.1,
    0.1,0.5,0.1,0.3,
    0.1,0.5,0.3,0.1,
    0.3,0.1,0.1,0.5,
    0.3,0.1,0.5,0.1,
    0.3,0.5,0.1,0.1,
    0.5,0.1,0.1,0.3,
    0.5,0.1,0.3,0.1,
    0.5,0.3,0.1,0.1,
    0.1,0.1,0.1,0.7,
    0.1,0.1,0.7,0.1,
    0.1,0.7,0.1,0.1,
    0.7,0.1,0.1,0.1,
    
    // Additional points for rule 4, order 9
    0.25,0.25,0.25,0.25,0.083333333333333333333,0.25,0.25,
       0.41666666666666666667,0.083333333333333333333,0.25,0.41666666666666666667,0.25,
       0.083333333333333333333,0.41666666666666666667,0.25,0.25,0.25,
       0.083333333333333333333,0.25,0.41666666666666666667,0.25,0.083333333333333333333,
       0.41666666666666666667,0.25,0.25,0.25,0.083333333333333333333,
       0.41666666666666666667,0.25,0.25,0.41666666666666666667,0.083333333333333333333,
       0.25,0.41666666666666666667,0.083333333333333333333,0.25,0.25,
       0.41666666666666666667,0.25,0.083333333333333333333,0.41666666666666666667,
       0.083333333333333333333,0.25,0.25,0.41666666666666666667,0.25,
       0.083333333333333333333,0.25,0.41666666666666666667,0.25,0.25,
       0.083333333333333333333,0.083333333333333333333,0.083333333333333333333,
       0.41666666666666666667,0.41666666666666666667,0.083333333333333333333,
       0.41666666666666666667,0.083333333333333333333,0.41666666666666666667,
       0.083333333333333333333,0.41666666666666666667,0.41666666666666666667,
       0.083333333333333333333,0.41666666666666666667,0.083333333333333333333,
       0.083333333333333333333,0.41666666666666666667,0.41666666666666666667,
       0.083333333333333333333,0.41666666666666666667,0.083333333333333333333,
       0.41666666666666666667,0.41666666666666666667,0.083333333333333333333,
       0.083333333333333333333,0.083333333333333333333,0.083333333333333333333,0.25,
       0.58333333333333333333,0.083333333333333333333,0.083333333333333333333,
       0.58333333333333333333,0.25,0.083333333333333333333,0.25,0.083333333333333333333,
       0.58333333333333333333,0.083333333333333333333,0.25,0.58333333333333333333,
       0.083333333333333333333,0.083333333333333333333,0.58333333333333333333,
       0.083333333333333333333,0.25,0.083333333333333333333,0.58333333333333333333,0.25,
       0.083333333333333333333,0.25,0.083333333333333333333,0.083333333333333333333,
       0.58333333333333333333,0.25,0.083333333333333333333,0.58333333333333333333,
       0.083333333333333333333,0.25,0.58333333333333333333,0.083333333333333333333,
       0.083333333333333333333,0.58333333333333333333,0.083333333333333333333,
       0.083333333333333333333,0.25,0.58333333333333333333,0.083333333333333333333,0.25,
       0.083333333333333333333,0.58333333333333333333,0.25,0.083333333333333333333,
       0.083333333333333333333,0.083333333333333333333,0.083333333333333333333,
       0.083333333333333333333,0.75,0.083333333333333333333,0.083333333333333333333,0.75,
       0.083333333333333333333,0.083333333333333333333,0.75,0.083333333333333333333,
       0.083333333333333333333,0.75,0.083333333333333333333,0.083333333333333333333,
       0.083333333333333333333,
    
    // Additional points for rule 5, order 11
       0.21428571428571428571,0.21428571428571428571,
       0.21428571428571428571,0.35714285714285714286,0.21428571428571428571,
       0.21428571428571428571,0.35714285714285714286,0.21428571428571428571,
       0.21428571428571428571,0.35714285714285714286,0.21428571428571428571,
       0.21428571428571428571,0.35714285714285714286,0.21428571428571428571,
       0.21428571428571428571,0.21428571428571428571,0.071428571428571428571,
       0.21428571428571428571,0.35714285714285714286,0.35714285714285714286,
       0.071428571428571428571,0.35714285714285714286,0.21428571428571428571,
       0.35714285714285714286,0.071428571428571428571,0.35714285714285714286,
       0.35714285714285714286,0.21428571428571428571,0.21428571428571428571,
       0.071428571428571428571,0.35714285714285714286,0.35714285714285714286,
       0.21428571428571428571,0.35714285714285714286,0.071428571428571428571,
       0.35714285714285714286,0.21428571428571428571,0.35714285714285714286,
       0.35714285714285714286,0.071428571428571428571,0.35714285714285714286,
       0.071428571428571428571,0.21428571428571428571,0.35714285714285714286,
       0.35714285714285714286,0.071428571428571428571,0.35714285714285714286,
       0.21428571428571428571,0.35714285714285714286,0.21428571428571428571,
       0.071428571428571428571,0.35714285714285714286,0.35714285714285714286,
       0.21428571428571428571,0.35714285714285714286,0.071428571428571428571,
       0.35714285714285714286,0.35714285714285714286,0.071428571428571428571,
       0.21428571428571428571,0.35714285714285714286,0.35714285714285714286,
       0.21428571428571428571,0.071428571428571428571,0.071428571428571428571,
       0.21428571428571428571,0.21428571428571428571,0.5,0.071428571428571428571,
       0.21428571428571428571,0.5,0.21428571428571428571,0.071428571428571428571,0.5,
       0.21428571428571428571,0.21428571428571428571,0.21428571428571428571,
       0.071428571428571428571,0.21428571428571428571,0.5,0.21428571428571428571,
       0.071428571428571428571,0.5,0.21428571428571428571,0.21428571428571428571,
       0.21428571428571428571,0.071428571428571428571,0.5,0.21428571428571428571,
       0.21428571428571428571,0.5,0.071428571428571428571,0.21428571428571428571,0.5,
       0.071428571428571428571,0.21428571428571428571,0.21428571428571428571,0.5,
       0.21428571428571428571,0.071428571428571428571,0.5,0.071428571428571428571,
       0.21428571428571428571,0.21428571428571428571,0.5,0.21428571428571428571,
       0.071428571428571428571,0.21428571428571428571,0.5,0.21428571428571428571,
       0.21428571428571428571,0.071428571428571428571,0.071428571428571428571,
       0.071428571428571428571,0.35714285714285714286,0.5,0.071428571428571428571,
       0.071428571428571428571,0.5,0.35714285714285714286,0.071428571428571428571,
       0.35714285714285714286,0.071428571428571428571,0.5,0.071428571428571428571,
       0.35714285714285714286,0.5,0.071428571428571428571,0.071428571428571428571,0.5,
       0.071428571428571428571,0.35714285714285714286,0.071428571428571428571,0.5,
       0.35714285714285714286,0.071428571428571428571,0.35714285714285714286,
       0.071428571428571428571,0.071428571428571428571,0.5,0.35714285714285714286,
       0.071428571428571428571,0.5,0.071428571428571428571,0.35714285714285714286,0.5,
       0.071428571428571428571,0.071428571428571428571,0.5,0.071428571428571428571,
       0.071428571428571428571,0.35714285714285714286,0.5,0.071428571428571428571,
       0.35714285714285714286,0.071428571428571428571,0.5,0.35714285714285714286,
       0.071428571428571428571,0.071428571428571428571,0.071428571428571428571,
       0.071428571428571428571,0.21428571428571428571,0.64285714285714285714,
       0.071428571428571428571,0.071428571428571428571,0.64285714285714285714,
       0.21428571428571428571,0.071428571428571428571,0.21428571428571428571,
       0.071428571428571428571,0.64285714285714285714,0.071428571428571428571,
       0.21428571428571428571,0.64285714285714285714,0.071428571428571428571,
       0.071428571428571428571,0.64285714285714285714,0.071428571428571428571,
       0.21428571428571428571,0.071428571428571428571,0.64285714285714285714,
       0.21428571428571428571,0.071428571428571428571,0.21428571428571428571,
       0.071428571428571428571,0.071428571428571428571,0.64285714285714285714,
       0.21428571428571428571,0.071428571428571428571,0.64285714285714285714,
       0.071428571428571428571,0.21428571428571428571,0.64285714285714285714,
       0.071428571428571428571,0.071428571428571428571,0.64285714285714285714,
       0.071428571428571428571,0.071428571428571428571,0.21428571428571428571,
       0.64285714285714285714,0.071428571428571428571,0.21428571428571428571,
       0.071428571428571428571,0.64285714285714285714,0.21428571428571428571,
       0.071428571428571428571,0.071428571428571428571,0.071428571428571428571,
       0.071428571428571428571,0.071428571428571428571,0.78571428571428571429,
       0.071428571428571428571,0.071428571428571428571,0.78571428571428571429,
       0.071428571428571428571,0.071428571428571428571,0.78571428571428571429,
       0.071428571428571428571,0.071428571428571428571,0.78571428571428571429,
       0.071428571428571428571,0.071428571428571428571,0.071428571428571428571
};

double grundmann1wts[] = {
    -0.8,0.45,0.45,0.45,0.45
};

double grundmann2wts[] = {
    0.26666666666666666667,-0.57857142857142857143,-0.57857142857142857143,
    -0.57857142857142857143,-0.57857142857142857143,0.3047619047619047619,
    0.3047619047619047619,0.3047619047619047619,0.3047619047619047619,
    0.3047619047619047619,0.3047619047619047619,0.3047619047619047619,
    0.3047619047619047619,0.3047619047619047619,0.3047619047619047619
};

double grundmann3wts[] = {
    -0.050793650793650793651,0.32544642857142857143,0.32544642857142857143,
       0.32544642857142857143,0.32544642857142857143,-0.54179894179894179894,
       -0.54179894179894179894,-0.54179894179894179894,-0.54179894179894179894,
       -0.54179894179894179894,-0.54179894179894179894,-0.54179894179894179894,
       -0.54179894179894179894,-0.54179894179894179894,-0.54179894179894179894,
       0.25834986772486772487,0.25834986772486772487,0.25834986772486772487,
       0.25834986772486772487,0.25834986772486772487,0.25834986772486772487,
       0.25834986772486772487,0.25834986772486772487,0.25834986772486772487,
       0.25834986772486772487,0.25834986772486772487,0.25834986772486772487,
       0.25834986772486772487,0.25834986772486772487,0.25834986772486772487,
       0.25834986772486772487,0.25834986772486772487,0.25834986772486772487,
       0.25834986772486772487,0.25834986772486772487
};

double grundmann4wts[] = {
    0.0063492063492063492063,-0.10848214285714285714,-0.10848214285714285714,
       -0.10848214285714285714,-0.10848214285714285714,0.43343915343915343915,
       0.43343915343915343915,0.43343915343915343915,0.43343915343915343915,
       0.43343915343915343915,0.43343915343915343915,0.43343915343915343915,
       0.43343915343915343915,0.43343915343915343915,0.43343915343915343915,
       -0.58715879028379028379,-0.58715879028379028379,-0.58715879028379028379,
       -0.58715879028379028379,-0.58715879028379028379,-0.58715879028379028379,
       -0.58715879028379028379,-0.58715879028379028379,-0.58715879028379028379,
       -0.58715879028379028379,-0.58715879028379028379,-0.58715879028379028379,
       -0.58715879028379028379,-0.58715879028379028379,-0.58715879028379028379,
       -0.58715879028379028379,-0.58715879028379028379,-0.58715879028379028379,
       -0.58715879028379028379,-0.58715879028379028379,0.25246753246753246753,
       0.25246753246753246753,0.25246753246753246753,0.25246753246753246753,
       0.25246753246753246753,0.25246753246753246753,0.25246753246753246753,
       0.25246753246753246753,0.25246753246753246753,0.25246753246753246753,
       0.25246753246753246753,0.25246753246753246753,0.25246753246753246753,
       0.25246753246753246753,0.25246753246753246753,0.25246753246753246753,
       0.25246753246753246753,0.25246753246753246753,0.25246753246753246753,
       0.25246753246753246753,0.25246753246753246753,0.25246753246753246753,
       0.25246753246753246753,0.25246753246753246753,0.25246753246753246753,
       0.25246753246753246753,0.25246753246753246753,0.25246753246753246753,
       0.25246753246753246753,0.25246753246753246753,0.25246753246753246753,
       0.25246753246753246753,0.25246753246753246753,0.25246753246753246753,
       0.25246753246753246753
};

double grundmann5wts[] = {
    -0.00056437389770723104056,0.024408482142857142857,0.024408482142857142857,
       0.024408482142857142857,0.024408482142857142857,-0.21015231681898348565,
       -0.21015231681898348565,-0.21015231681898348565,-0.21015231681898348565,
       -0.21015231681898348565,-0.21015231681898348565,-0.21015231681898348565,
       -0.21015231681898348565,-0.21015231681898348565,-0.21015231681898348565,
       0.61162373987894821228,0.61162373987894821228,0.61162373987894821228,
       0.61162373987894821228,0.61162373987894821228,0.61162373987894821228,
       0.61162373987894821228,0.61162373987894821228,0.61162373987894821228,
       0.61162373987894821228,0.61162373987894821228,0.61162373987894821228,
       0.61162373987894821228,0.61162373987894821228,0.61162373987894821228,
       0.61162373987894821228,0.61162373987894821228,0.61162373987894821228,
       0.61162373987894821228,0.61162373987894821228,-0.69914085914085914086,
       -0.69914085914085914086,-0.69914085914085914086,-0.69914085914085914086,
       -0.69914085914085914086,-0.69914085914085914086,-0.69914085914085914086,
       -0.69914085914085914086,-0.69914085914085914086,-0.69914085914085914086,
       -0.69914085914085914086,-0.69914085914085914086,-0.69914085914085914086,
       -0.69914085914085914086,-0.69914085914085914086,-0.69914085914085914086,
       -0.69914085914085914086,-0.69914085914085914086,-0.69914085914085914086,
       -0.69914085914085914086,-0.69914085914085914086,-0.69914085914085914086,
       -0.69914085914085914086,-0.69914085914085914086,-0.69914085914085914086,
       -0.69914085914085914086,-0.69914085914085914086,-0.69914085914085914086,
       -0.69914085914085914086,-0.69914085914085914086,-0.69914085914085914086,
       -0.69914085914085914086,-0.69914085914085914086,-0.69914085914085914086,
       -0.69914085914085914086,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715,
       0.27217694439048605715,0.27217694439048605715,0.27217694439048605715
};

quadraturerule grundmann5 = {
    .name = "grundmann5",
    .grade = 3,
    .order = 11,
    .nnodes = 126,
    .nodes = grundmannpts,
    .weights = grundmann5wts,
    .ext = NULL
};

quadraturerule grundmann4 = {
    .name = "grundmann4",
    .grade = 3,
    .order = 9,
    .nnodes = 70,
    .nodes = grundmannpts,
    .weights = grundmann4wts,
    .ext = &grundmann5
};

quadraturerule grundmann3 = {
    .name = "grundmann3",
    .grade = 3,
    .order = 7,
    .nnodes = 35,
    .nodes = grundmannpts,
    .weights = grundmann3wts,
    .ext = &grundmann4
};

quadraturerule grundmann2 = {
    .name = "grundmann2",
    .grade = 3,
    .order = 5,
    .nnodes = 15,
    .nodes = grundmannpts,
    .weights = grundmann2wts,
    .ext = &grundmann3
};

quadraturerule grundmann1 = {
    .name = "grundmann1",
    .grade = 3,
    .order = 3,
    .nnodes = 5,
    .nodes = grundmannpts,
    .weights = grundmann1wts,
    .ext = &grundmann2
};

/* --------------------------------
 * List of quadrature rules
 * -------------------------------- */

quadraturerule *quadrules[] = {
    &midpoint, &simpson,
    &gauss1, &kronrod3,
    &gauss2, &kronrod5,
    &gauss5, &kronrod11,
    &gauss7, &kronrod15,

    &tri4, &tri10, &tri20,
    &cubtri7, &cubtri19,
    
    &keast4, &keast5,
    &tet5, &tet6,

    &grundmann1, &grundmann2, &grundmann3, &grundmann4,
    NULL
};

/* **********************************************
 * Subdivision rules
 * ********************************************** */

/* -------
 *   1D
 * ------- */

/** Bisection */
double bisectionpts[] = {
    0.5, 0.5
};

double bisectionweights[] = {
    0.5, 0.5
};

int bisectionintervals[] = {
    2, 1,
    0, 2
};

subdivisionrule bisection = {
    .grade = 1,
    .npts = 1,
    .pts = bisectionpts,
    .nels = 2,
    .newels = bisectionintervals,
    .weights = bisectionweights,
    .alt = NULL
};

/** Trisection */
double trisectionpts[] = {
    0.666666666666666667, 0.333333333333333333,
    0.333333333333333333, 0.666666666666666667
};

double trisectionweights[] = {
    0.333333333333333333, 0.333333333333333333, 0.333333333333333333
};

int trisectionintervals[] = {
    0, 2,
    3, 1,
    2, 3
};

subdivisionrule trisection = {
    .grade = 1,
    .npts = 2,
    .pts = trisectionpts,
    .nels = 3,
    .newels = trisectionintervals,
    .weights = trisectionweights,
    .alt = NULL
};

/* -------
 *   2D
 * ------- */

/*
 *       2
 *      / \
 *     / | \
 *    /  |  \
 *   0 - 3 - 1
 */

/** Bisection of 2D triangle */
double tribisectionpts[] = {
    0.5, 0.5, 0.0
};

double tribisectionweights[] = {
    0.5, 0.5
};

int tribisectiontris[] = {
    0, 3, 2,
    3, 1, 2
};

subdivisionrule trianglebisection = {
    .grade = 2,
    .npts = 1,
    .pts = tribisectionpts,
    .nels = 2,
    .newels = tribisectiontris,
    .weights = tribisectionweights,
    .alt = NULL
};

/** Quadrasection of 2D triangle */

/*
 *       2
 *      / \
 *     5 - 4
 *    / \ / \
 *   0 - 3 - 1
 */

double triquadrasectionpts[] = {
    0.5, 0.5, 0.0,
    0.0, 0.5, 0.5,
    0.5, 0.0, 0.5
};

double triquadrasectionweights[] = {
    0.25, 0.25, 0.25, 0.25
};

int triquadrasectiontris[] = {
    0, 3, 5,
    3, 1, 4,
    3, 4, 5,
    5, 4, 2
};

subdivisionrule trianglequadrasection = {
    .grade = 2,
    .npts = 3,
    .pts = triquadrasectionpts,
    .nels = 4,
    .newels = triquadrasectiontris,
    .weights = triquadrasectionweights,
    .alt = &trianglebisection
};

/* -------
 *   3D
 * ------- */

/** Splitting of tetrahedra */
double tetsubdivpts[] = {
    0.5, 0.5, 0.0, 0.0,
    0.5, 0.0, 0.5, 0.0,
    0.5, 0.0, 0.0, 0.5,
    0.0, 0.5, 0.5, 0.0,
    0.0, 0.5, 0.0, 0.5,
    0.0, 0.0, 0.5, 0.5
};

double tetsubdivwts[] = {
    0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125
};

int tetsubdivtets[] =  {
    1, 4, 7, 8,
    0, 4, 7, 9,
    0, 4, 8, 9,
    4, 7, 8, 9,
    0, 5, 7, 9,
    0, 6, 8, 9,
    2, 5, 7, 9,
    3, 6, 8, 9 };

subdivisionrule tetsection = {
    .grade = 3,
    .npts = 6,
    .pts = tetsubdivpts,
    .nels = 8,
    .newels = tetsubdivtets,
    .weights = tetsubdivwts,
    .alt = NULL
};

subdivisionrule *subdivisionrules[] = {
    &bisection,
    &trianglequadrasection,
    &tetsection,
    NULL
};


/* **********************************************
 * Integrator data structure and operations
 * ********************************************** */

DEFINE_VARRAY(quadratureworkitem, quadratureworkitem)

/** Initialize an integrator structure */
void integrator_init(integrator *integrate) {
    integrate->integrand=NULL;
    
    integrate->dim=0;
    integrate->nbary=0;
    integrate->nquantity=0;
    integrate->nqdof=0;
    integrate->ndof=0;
    
    integrate->adapt=true;
    integrate->rule = NULL;
    integrate->errrule = NULL;
    
    integrate->subdivide = NULL;
    
    integrate->workp = -1;
    integrate->freep = -1;
    varray_quadratureworkiteminit(&integrate->worklist);
    varray_doubleinit(&integrate->vertexstack);
    varray_intinit(&integrate->elementstack);
    varray_valueinit(&integrate->quantitystack);
    
    integrate->ztol = INTEGRATE_ZEROCHECK;
    integrate->tol = INTEGRATE_ACCURACYGOAL;
    integrate->maxiterations = INTEGRATE_MAXITERATIONS;
    
    integrate->niterations = 0;
    integrate->val = 0;
    integrate->err = 0;
    
    error_init(&integrate->emsg);
    
    integrate->ref = NULL;
}

/** Free data associated with an integrator */
void integrator_clear(integrator *integrate) {
    varray_quadratureworkitemclear(&integrate->worklist);
    varray_intclear(&integrate->elementstack);
    varray_doubleclear(&integrate->vertexstack);
    for (unsigned int i=0; i<integrate->quantitystack.count; i++) {
        value v=integrate->quantitystack.data[i];
        if (MORPHO_ISOBJECT(v)) morpho_freeobject(v);
    }
    varray_valueclear(&integrate->quantitystack);
}

/** Adds a vertex to the integrators vertex stack, returning the id */
int integrator_addvertex(integrator *integrate, int ndof, double *v) {
    int vid = integrate->vertexstack.count;
    varray_doubleadd(&integrate->vertexstack, v, ndof);
    return vid;
}

/** Adds an element to the element stack, returning the id. Elements consist of :
    - vertex ids
    - a number of quantity ids. */
int integrator_addelement(integrator *integrate, int *vids, int *qids) {
    int elid=integrate->elementstack.count;
    varray_intadd(&integrate->elementstack, vids, integrate->nbary);
    if (integrate->nquantity && qids) varray_intadd(&integrate->elementstack, qids, integrate->nbary);
    return elid;
}

/**
 Quantities are stored as needed on the quantity stack. The first n values are used
 to store interpolated values. As new vertices are added, n entries are added to the quantity stack

 | base:          | q list 0:      | q list 1:   |
 | q0, q1, ... qn | q0, q1, ... qn | q0, q1, ... |

 As elements are added, they include both vertex ids and references to entries on the quantity stack. Each element is 2*nbary entries long:

 nbary               nbary
 | vid0, vid1, ... : qid0, qid1, ... |

 For each element, we build an interpolation matrix,

 [ x0 y0 q0,0 q1,0 ... qn,0 ]
 [ x1 y1 q0,1 q1,1 ... qn,1 ]
 [ x2 y1 q0,2 q1,2 ... qn,2 ]

 which when multiplied by the barycentric coordinates yields the interpolated quantite

 [l0, l1, l2] . Interp -> [ x0, y0, q0, q1, ... qn ] */

/** Adds nq quantities to the quantity stack, returning the id of the first element */
int integrator_addquantity(integrator *integrate, int nq, value *quantity) {
    int qid=integrate->quantitystack.count;
    for (int i=0; i<nq; i++) {
        if (MORPHO_ISFLOAT(quantity[i])) {
            varray_valuewrite(&integrate->quantitystack, quantity[i]);
        } else if (MORPHO_ISMATRIX(quantity[i])) {
            objectmatrix *new = object_clonematrix(MORPHO_GETMATRIX(quantity[i]));
            // TODO: Raise error on fail
            varray_valuewrite(&integrate->quantitystack, MORPHO_OBJECT(new));
        } else return false;
    }
    return qid;
}

/** Count degrees of freedom */
void integrator_countquantitydof(integrator *integrate, int nq, value *quantity) {
    int ndof=0;
    for (int i=0; i<nq; i++) {
        if (MORPHO_ISFLOAT(quantity[i])) {
            ndof++;
        } else if (MORPHO_ISMATRIX(quantity[i])) {
            objectmatrix *m = MORPHO_GETMATRIX(quantity[i]);
            ndof+=matrix_countdof(m);
        } else return;
    }
    integrate->nqdof=ndof;
}

/** Retrieves the vertex pointers given an elementid.
 @warning: The pointers returned become invalid after a subsequent call to integrator_addvertex . */
void integrator_getvertices(integrator *integrate, int elementid, double **vert) {
    for (int i=0; i<integrate->nbary; i++) {
        int vid=integrate->elementstack.data[elementid+i];
        vert[i]=&(integrate->vertexstack.data[vid]);
    }
}

/** Retrieves the quantity pointers given an elementid.
 @warning: The pointers returned become invalid after a subsequent call to integrator_addvertex */
void integrator_getquantities(integrator *integrate, int elementid, value **quantities) {
    for (int i=0; i<integrate->nbary; i++) {
        int qid=integrate->elementstack.data[elementid+integrate->nbary+i]; // Note quantities stored after vertices
        quantities[i]=&(integrate->quantitystack.data[qid]);
    }
}

/** Retrieves an element with elementid */
void integrator_getelement(integrator *integrate, int elementid, int *vid, int *qid) {
    for (int i=0; i<integrate->nbary; i++) {
        vid[i]=integrate->elementstack.data[elementid+i];
        if (integrate->nquantity && qid) qid[i]=integrate->elementstack.data[elementid+integrate->nbary+i];
    }
}

/** Adds a work item to the integrator's work list.
    Uses a binary queue data structure to facilitate ln(N) push and pop - https://en.wikipedia.org/wiki/Binary_heap */
bool integrator_pushworkitem(integrator *integrate, quadratureworkitem *work) {
    varray_quadratureworkitemadd(&integrate->worklist, work, 1);
    
    for (int i=integrate->worklist.count-1, p; i>0; i=p) {
        p=floor((i-1)/2); // Parent
        if (integrate->worklist.data[i].err>integrate->worklist.data[p].err) {
            quadratureworkitem swp=integrate->worklist.data[i];
            integrate->worklist.data[i]=integrate->worklist.data[p];
            integrate->worklist.data[p]=swp;
        } else break;
    }
    
    return true;
}

/** Pops the work item with the largest error */
bool integrator_popworkitem(integrator *integrate, quadratureworkitem *work) {
    *work = integrate->worklist.data[0];
    
    // Move the last element into first place and pop
    int n=integrate->worklist.count-1;
    if (n>0) integrate->worklist.data[0]=integrate->worklist.data[n];
    integrate->worklist.count--;
    
    // Go down the heap, ensuring that the heap property is maintained
    for (int i=0, p, q; i<n; i=p) {
        p=2*i + 1; // Left - child nodes
        q=p+1;     // Right
        
        // Check if the right child element has a larger value, if it exists
        if (q<n &&
            integrate->worklist.data[q].err>integrate->worklist.data[p].err) {
            p=q;
        }
        
        // If the child element is larger, swap it up
        if (p<n && integrate->worklist.data[p].err>integrate->worklist.data[i].err) {
            quadratureworkitem swp=integrate->worklist.data[i];
            integrate->worklist.data[i]=integrate->worklist.data[p];
            integrate->worklist.data[p]=swp;
        } else break;
    }
    
    return true;
}

/** Estimate the value and error of the integrand given a worklist */
void integrator_estimate(integrator *integrate) {
    double sumval=0.0, cval=0.0, yval, tval,
           sumerr=0.0, cerr=0.0, yerr, terr;

    // Sum in reverse as smallest entries should be nearer the end
    for (int i=integrate->worklist.count-1; i>=0; i--) {
        yval=integrate->worklist.data[i].val-cval;
        yerr=integrate->worklist.data[i].err-cerr;
        tval=sumval+yval;
        terr=sumerr+yerr;
        cval=(tval-sumval)-yval;
        cerr=(terr-sumerr)-yerr;
        sumval=tval;
        sumerr=terr;
    }
    
    integrate->val = sumval;
    integrate->err = sumerr;
}

/* --------------------------------
 * Linear interpolation
 * -------------------------------- */

/*void xlinearinterpolate(int nbary, double *bary, int nels, double **v, double *x) {
    for (int j=0; j<nels; j++) x[j]=0.0;
    for (int j=0; j<nels; j++) {
        for (int k=0; k<nbary; k++) {
            x[j]+=v[k][j]*bary[k];
        }
    }
}*/

void linearinterpolate(integrator *integrate, double *bary, double *vmat, double *x) {
    // Multiply 1 x nbary (lambda) with nbary x dim (vmat)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, integrate->dim+integrate->nqdof, integrate->nbary, 1.0, bary, 1, vmat, integrate->nbary, 0.0, x, 1);
}

/** Also provide a version using BLAS to accelerate multiplication */
void preparevertices(integrator *integrate, double **v, double *vv) {
    int k=0;
    for (int j=0; j<integrate->dim; j++) {
        for (int i=0; i<integrate->nbary; i++) {
            vv[k]=v[i][j];
            k++;
        }
    }
}

/** Prepares quantities for interpolation */
void preparequantities(integrator *integrate, value **quantity, double *qmat) {
    int k=0; // DOF counter
    for (int i=0; i<integrate->nquantity; i++) {
        if (MORPHO_ISFLOAT(quantity[0][i])) {
            for (int j=0; j<integrate->nbary; j++) morpho_valuetofloat(quantity[j][i], &qmat[k*integrate->nbary+j]);
            k++;
        } else if (MORPHO_ISMATRIX(quantity[0][i])) {
            for (int j=0; j<integrate->nbary; j++) {
                objectmatrix *m = MORPHO_GETMATRIX(quantity[j][i]);
                int mdof=m->ncols*m->nrows;
                for (int l=0; l<mdof; l++) qmat[(k+l)*integrate->nbary+j]=m->elements[l];
            }
        } else return;
    }
}

/** Sets up interpolation matrix */
void prepareinterpolation(integrator *integrate, int elementid, double *vmat) {
    double *vert[integrate->nbary]; // Vertex information
    value *quantity[integrate->nbary]; // Quantities
    
    integrator_getvertices(integrate, elementid, vert);
    preparevertices(integrate, vert, vmat);
    
    if (integrate->nquantity) {
        integrator_getquantities(integrate, elementid, quantity);
        preparequantities(integrate, quantity, vmat+integrate->nbary*integrate->dim);
    }
}

/** Processes the results of interpolation */
void postprocessquantities(integrator *integrate, double *qout) {
    int k=0; // DOF counter
    for (int i=0; i<integrate->nquantity; i++) {
        value q = integrate->quantitystack.data[i];
        if (MORPHO_ISFLOAT(q)) {
            integrate->quantitystack.data[i]=MORPHO_FLOAT(qout[k]);
        } else if (MORPHO_ISMATRIX(q)) {
            objectmatrix *m = MORPHO_GETMATRIX(q);
            m->elements=&qout[k];
            k+=m->ncols*m->nrows;
        }
    }
}

/* --------------------------------
 * Function to perform quadrature
 * -------------------------------- */

/** Weighted sum of a list */
double integrate_sumlistweighted(unsigned int nel, double *list, double *wts) {
    return cblas_ddot(nel, list, 1, wts, 1);
}

/** Evaluates the integrand at specified places */
bool integrate_evalfn(integrator *integrate, quadraturerule *rule, int imin, int imax, double *vmat, double *x, double *f) {
    for (int i=imin; i<imax; i++) {
        // Interpolate the point and quantities
        linearinterpolate(integrate, &rule->nodes[integrate->nbary*i], vmat, x);
        if (integrate->nquantity) postprocessquantities(integrate, x+integrate->dim);
        
        // Evaluate function
        if (!(*integrate->integrand) (rule->grade, &rule->nodes[integrate->nbary*i], x, integrate->nquantity, integrate->quantitystack.data, integrate->ref, &f[i])) return false;
    }
    return true;
}

/** Integrates a function over an element specified in work, filling out the integral and error estimate if provided */
bool quadrature(integrator *integrate, quadraturerule *rule, quadratureworkitem *work) {
    int nmax = rule->nnodes;
    int np = 0; // Number of levels of p-refinement
    for (quadraturerule *q = rule->ext; q!=NULL; q=q->ext) {
        nmax = q->nnodes;
        np++;
    }
    
    double vmat[integrate->nbary*integrate->ndof]; // Interpolation matrix
    prepareinterpolation(integrate, work->elementid, vmat);
    
    // Evaluate function at quadrature points
    double x[integrate->ndof],f[nmax];
    if (!integrate_evalfn(integrate, rule, 0, rule->nnodes, vmat, x, f)) return false;
    
    double r[np+1];
    double eps[np+1]; eps[0]=0.0;
    
    // Obtain estimate
    r[0]=integrate_sumlistweighted(rule->nnodes, f, rule->weights);
    work->lval = work->val = work->weight*r[0];
    
    // Estimate error
    if (rule->ext!=NULL) { // Evaluate extension rule
        int nmin = rule->nnodes, ip=0;
        
        // Attempt p-refinement if available
        for (quadraturerule *q=rule->ext; q!=NULL; q=q->ext) {
            ip++;
            if (!integrate_evalfn(integrate, q, nmin, q->nnodes, vmat, x, f)) return false;
            
            r[ip]=integrate_sumlistweighted(q->nnodes, f, q->weights);
            eps[ip]=fabs(r[ip]-r[ip-1]);
            nmin = q->nnodes;
            
            if (fabs(eps[ip]/r[ip])<1e-14) break;
        }
        
        work->lval = work->weight*r[ip-1];
        work->val = work->weight*r[ip]; // Record better estimate
        work->err = work->weight*eps[ip]; // Use the difference as the error estimator
    } else if (integrate->errrule) {  // Otherwise, use the error rule to obtain the estimate
        if (rule==integrate->errrule) return true; // We already are using the error rule
        double temp = work->val; // Retain the lower order estimate
        if (!quadrature(integrate, integrate->errrule, work)) return false;
        work->lval=temp;
        work->err=fabs(work->val-temp); // Estimate error from difference of rules
    } else {
        UNREACHABLE("Integrator definition inconsistent.");
    }
    
    return true;
}

/* --------------------------------
 * Subdivision
 * -------------------------------- */

bool subdivide(integrator *integrate, quadratureworkitem *work, int *nels, quadratureworkitem *newitems) {
    subdivisionrule *rule = integrate->subdivide;
    
    // Fetch the element data
    int npts = integrate->nbary+rule->npts;
    int vid[npts], qid[npts];
    integrator_getelement(integrate, work->elementid, vid, qid);
    
    // Get ready for interpolation
    double vmat[integrate->nbary*integrate->ndof]; // Vertex information
    prepareinterpolation(integrate, work->elementid, vmat);
    
    // Interpolate vertices and quantities, and add these new vertices and quantities
    double x[integrate->ndof];
    for (int j=0; j<rule->npts; j++) {
        linearinterpolate(integrate, &rule->pts[j*integrate->nbary], vmat, x);
        vid[integrate->nbary+j]=integrator_addvertex(integrate, integrate->dim, x);
        
        if (integrate->nquantity) {
            postprocessquantities(integrate, x+integrate->dim);
            qid[integrate->nbary+j]=integrator_addquantity(integrate, integrate->nquantity, integrate->quantitystack.data);
        }
    }
    
    // Create elements
    for (int i=0; i<rule->nels; i++) {
        newitems[i].val=0.0;
        newitems[i].err=0.0;
        newitems[i].weight=work->weight*rule->weights[i];
        
        // Construct new element from the vertex ids and quantity ids
        int vids[integrate->nbary], qids[integrate->nbary];
        for (int k=0; k<integrate->nbary; k++) {
            vids[k]=vid[rule->newels[integrate->nbary*i+k]];
            if (integrate->nquantity) qids[k]=qid[rule->newels[integrate->nbary*i+k]];
        }
        
        // Define the new element
        newitems[i].elementid=integrator_addelement(integrate, vids, qids);
    }
    
    *nels = rule->nels;
    
    return true;
}

/* --------------------------------
 * Laurie's sharper error estimate
 * -------------------------------- */

/** Laurie's sharper error estimator: BIT 23 (1983), 258-261
    The norm of the difference between two rules |A-B| is usually too pessimistic;
    this attempts to extrapolate a sharper estimate if convergence looks good */
void sharpenerrorestimate(integrator *integrate, quadratureworkitem *work, int nels, quadratureworkitem *newitems) {
    double a1=work->val, b1=work->lval, a2=0, b2=0;
    for (int k=0; k<nels; k++) {
        a2+=newitems[k].val;
        b2+=newitems[k].lval;
    }
    
    // Scale errors if conditions are met
    if (fabs(a2-a1)<fabs(b2-b1) && // Laurie's second condition
        fabs(a2-b2)<fabs(a1-b1)) // Weak form of first condition (see Gonnet)
    {
        double sigma=fabs((a2-a1)/(b2-b1-a2+a1));
        for (int k=0; k<nels; k++) newitems[k].err*=sigma;
    }
}

/** Adds newitems to the work list and updates the value and error */
void update(integrator *integrate, quadratureworkitem *work, int nels, quadratureworkitem *newitems) {
    double dval=0, derr=0;
    integrate->val-=work->val;
    integrate->err-=work->err;
    for (int k=0; k<nels; k++) {
        dval+=newitems[k].val;
        derr+=newitems[k].err;
        integrator_pushworkitem(integrate, &newitems[k]);
    }
    integrate->val+=dval;
    integrate->err+=derr;
}

/* --------------------------------
 * Integrator configuration
 * -------------------------------- */

/** Finds a rule by name */
bool integrator_matchrulebyname(int grade, char *name, quadraturerule **out) {
    for (int i=0; quadrules[i]!=NULL; i++) {
        if (quadrules[i]->grade!=grade) continue;
        if (name && quadrules[i]->name &&
            (strcmp(name, quadrules[i]->name)==0)) { // Match a rule by name
            *out = quadrules[i];
            return true;
        }
    }
    return false;
}

/** Attempts to find a quadrature rule that uses rule as an extension. */
bool integrator_matchrulebyextension(quadraturerule *rule, quadraturerule **out) {
    for (int i=0; quadrules[i]!=NULL; i++) {
        if (quadrules[i]->ext==rule) {
            *out = quadrules[i];
            return true;
        }
    }
    return false;
}

/** Finds the [highest/lowest] rule with order such that minorder <= order <= maxorder */
bool integrator_matchrulebyorder(int grade, int minorder, int maxorder, bool highest, quadraturerule **out) {
    int best=-1, bestorder=(highest ? -1 : INT_MAX);
    for (int i=0; quadrules[i]!=NULL; i++) {
        if (quadrules[i]->grade!=grade) continue;
        
        if (quadrules[i]->order>=minorder &&
            quadrules[i]->order<=maxorder &&
            ( (highest && quadrules[i]->order>bestorder) ||
              (!highest && quadrules[i]->order<bestorder) )) {
            best = i;
            bestorder = quadrules[i]->order;
        }
    }
    if (best>=0) *out = quadrules[best];
    return (best>=0);
}

/** Configures an integrator based on the grade to integrate and hints for order and rule type
 * @param[in] integrate     - integrator structure to be configured
 * @param[in] adapt             - enable adaptive refinement
 * @param[in] grade              - Dimension of the vertices
 * @param[in] order              - Requested order of quadrature rule
 * @param[in] name                - Alternatively, supply the name of a known rule
 * @returns true if the configuration was successful */
bool integrator_configure(integrator *integrate, bool adapt, int grade, int order, char *name) {
    integrate->rule=NULL;
    integrate->errrule=NULL;
    integrate->adapt=adapt;
    
    integrate->nbary=grade+1; // Number of barycentric coordinates
    
    if (name) {
        if (!integrator_matchrulebyname(grade, name, &integrate->rule)) return false;
    } else if (order<0) { // If no order requested find the highest order rule available
        if (!integrator_matchrulebyorder(grade, 0, INT_MAX, true, &integrate->rule)) return false;
    } else { // Prefer a rule that integrates at least order, but otherwise find the best
        if (!(integrator_matchrulebyorder(grade, order, INT_MAX, false, &integrate->rule) ||
            integrator_matchrulebyorder(grade, 0, order, true, &integrate->rule))) return false;
    }
    
    // Check we succeeded in finding a rule
    if (!integrate->rule) return false;
    
    // Do we need to find an extension rule?
    if (adapt && integrate->rule->ext==NULL) {
        // Find if the rule obtained is an extension of another rule
        if (integrator_matchrulebyextension(integrate->rule, &integrate->rule)) {
            
        } else if (!integrator_matchrulebyorder(grade, integrate->rule->order+1, INT_MAX, false,  &integrate->errrule)) { // Otherwise attempt to find a rule of higher order
            // but if there wasn't one, find the next lowest one...
            if (!integrator_matchrulebyorder(grade, 0, integrate->rule->order-1, true,  &integrate->errrule)) return false;
        }
        
        // Ensure that the error rule is higher than the integration rule
        if (integrate->errrule && integrate->rule->order>integrate->errrule->order) {
            quadraturerule *swp=integrate->rule;
            integrate->rule=integrate->errrule;
            integrate->errrule=swp;
        }
    }
    
    // Select subdivision rule
    for (int i=0; subdivisionrules[i]!=NULL; i++) {
        if (subdivisionrules[i]->grade==grade) {
            integrate->subdivide = subdivisionrules[i];
            break;
        }
    }
    
    return true;
}

/** Configures the integrator based on the contents of a dictionary */
bool integrator_configurewithdictionary(integrator *integrate, grade g, objectdictionary *dict) {
    char *name=NULL;
    bool adapt=true;
    int order=-1;
    value val;
    
    objectstring rulelabel = MORPHO_STATICSTRING(INTEGRATE_RULELABEL);
    objectstring degreelabel = MORPHO_STATICSTRING(INTEGRATE_DEGREELABEL);
    objectstring adaptlabel = MORPHO_STATICSTRING(INTEGRATE_ADAPTLABEL);
    
    if (dictionary_get(&dict->dict, MORPHO_OBJECT(&rulelabel), &val) &&
        MORPHO_ISSTRING(val)) {
        name = MORPHO_GETCSTRING(val);
    }

    if (dictionary_get(&dict->dict, MORPHO_OBJECT(&degreelabel), &val) &&
        MORPHO_ISINTEGER(val)) {
        order = MORPHO_GETINTEGERVALUE(val);
    }
    
    if (dictionary_get(&dict->dict, MORPHO_OBJECT(&adaptlabel), &val) &&
        MORPHO_ISBOOL(val)) {
        adapt = MORPHO_GETBOOLVALUE(val);
    }
    
    return integrator_configure(integrate, adapt, g, order, name);
}

/* --------------------------------
 * Driver routine
 * -------------------------------- */

/** Integrates over a function
 * @param[in] integrate     - integrator structure, that has been configured with integrator_configure
 * @param[in] integrand     - function to integrate
 * @param[in] dim                  - Dimension of the vertices
 * @param[in] x                       - vertices of the line x[0] = {x,y,z} etc.
 * @param[in] nquantity     - number of quantities per vertex
 * @param[in] quantity       - List of quantities for each vertex.
 * @param[in] ref                  - a pointer to any data required by the function
 * @returns True on success */
bool integrator_integrate(integrator *integrate, integrandfunction *integrand, int dim, double **x, unsigned int nquantity, value **quantity, void *ref) {
    
    integrate->integrand=integrand;
    integrate->ref=ref;
    
    integrate->dim=dim; // Dimensionality of vertices
    integrate->nquantity=nquantity;
    
    integrate->worklist.count=0;    // Reset all these without deallocating
    integrate->vertexstack.count=0;
    integrate->elementstack.count=0;
    integrate->quantitystack.count=0;
    error_clear(&integrate->emsg);
    
    // Quantities used for interpolation live at the start of the quantity stack
    integrator_countquantitydof(integrate, nquantity, quantity[0]);
    integrate->ndof = integrate->dim+integrate->nqdof; // Number of degrees of freedom
    
    integrator_addquantity(integrate, nquantity, quantity[0]);
    
    // Create first element
    int vids[integrate->nbary], qids[integrate->nbary];
    for (int i=0; i<integrate->nbary; i++) {
        vids[i]=integrator_addvertex(integrate, dim, x[i]);
        if (nquantity) qids[i]=integrator_addquantity(integrate, nquantity, quantity[i]);
    }
    int elid = integrator_addelement(integrate, vids, qids);
    
    // Add it to the work list
    quadratureworkitem work;
    work.weight = 1.0;
    work.elementid = elid;
    quadrature(integrate, integrate->rule, &work); // Perform initial quadrature
    
    integrator_pushworkitem(integrate, &work);
    integrator_estimate(integrate); // Initial estimate
    
    if (integrate->adapt) for (integrate->niterations=0; integrate->niterations<=integrate->maxiterations; integrate->niterations++) {
        // Convergence check
        if (fabs(integrate->val)<integrate->ztol || fabs(integrate->err/integrate->val)<integrate->tol) break;
        
        // Get worst interval
        integrator_popworkitem(integrate, &work);
        
        // Subdivide
        int nels; // Number of elements created
        quadratureworkitem newitems[integrate->subdivide->nels];
        
        subdivide(integrate, &work, &nels, newitems);
        for (int k=0; k<nels; k++) quadrature(integrate, integrate->rule, &newitems[k]);
        
        // Error estimate
        sharpenerrorestimate(integrate, &work, nels, newitems);
        
        // Add new items to heap and update error estimates
        update(integrate, &work, nels, newitems);
    }
    
    // Final estimate by Kahan summing heap
    integrator_estimate(integrate);
    
    return true;
}

/* -------------------------------------
 * Public interface matching old version
 * ------------------------------------- */

/** Integrate over an element - public interface for one off integrals.
 * @param[in] integrand   - integrand
 * @param[in] method         - Dictionary with method selection (optional)
 * @param[in] dim                - Dimension of the vertices
 * @param[in] grade            - Grade to integrate over
 * @param[in] x                     - vertices of the triangle x[0] = {x,y,z} etc.
 * @param[in] nquantity   - number of quantities per vertex
 * @param[in] quantity     - List of quantities for each endpoint.
 * @param[in] ref                - a pointer to any data required by the function
 * @param[out] out              - value of the integral
 * @returns true on success. */
bool integrate(integrandfunction *integrand, objectdictionary *method, unsigned int dim, unsigned int grade, double **x, unsigned int nquantity, value **quantity, void *ref, double *out, double *err) {
    bool success=false;
    integrator integrate;
    integrator_init(&integrate);
    
    if (method) {
        if (!integrator_configurewithdictionary(&integrate, grade, method)) return false;
    } else if (!integrator_configure(&integrate, true, grade, -1, NULL)) return false;
    success=integrator_integrate(&integrate, integrand, dim, x, nquantity, quantity, ref);
    
    *out = integrate.val;
    *err = integrate.err;
    
    integrator_clear(&integrate);
    
    return success;
}

/* --------------------------------
 * Testing code
 * -------------------------------- */

int nevals;

bool test_integrand(unsigned int dim, double *t, double *x, unsigned int nquantity, value *quantity, void *data, double *fout) {
    //double val = pow(sin(x[0]+x[1]),-0.5); //x[0]*x[1]*x[2]; // exp(-x[0]*x[0]); //sqrt(x[0]); //exp(-x[0]*x[0]); //*x[1]*x[2];
    //if (x[0]-x[1]<0.5) val=1.0;
    double val = sin(3*x[0]+6*x[1]);
    
    *fout=val; //val*val*val*val;
    nevals++;
    return true;
}

void integrate_test1(double *out, double *err) {
    nevals = 0;

    double x0[3] = { 0, 0, 0 };
    double x1[3] = { 1, 0, 0 };
    double x2[3] = { 0, 1, 0 };
    double x3[3] = { 0, 0, 1 };
    
    double *xx[] = { x0, x1, x2, x3 };
    value *quantities[] = { NULL, NULL, NULL, NULL };
    
    integrate(test_integrand, NULL, 3, 3, xx, 0, quantities, NULL, out, err);
    
    return;
}


void integrate_test2(double *out) {
    nevals = 0;
    
    double x0[3] = { 0, 0, 0 };
    double x1[3] = { 1, 0, 0 };
    double x2[3] = { 0, 1, 0 };
    double x3[3] = { 0, 0, 1 };
    double *xx[] = { x0, x1, x2, x3 };
    value *quantities[] = { NULL, NULL, NULL, NULL };
    integrate_integrate(test_integrand, 3, 3, xx, 0, quantities, NULL, out);
}

void integrate_test(void) {
    double out=0.0, out1=0.0, err1=0.0;
    int evals1=0;
    
    nevals = 0;
    int Nmax = 1;
    for (int i=0; i<Nmax; i++) {
        evals1 = 0;
        integrate_test1(&out1, &err1);
        evals1 = nevals;
        nevals = 0;
        integrate_test2(&out);
    }
    
    printf("New integrator: %g (%g) with %i function evaluations.\n", out1, err1, evals1);
    printf("Old integrator: %g with %i function evaluations.\n", out, nevals);
    
    double trueval = 0.457142857142857142857142857143; //0.533333333333333333333333333333; //0.250000000000000000000000000000; //2.70562770562770562770562770563e-6;
    
    printf("Difference %g (relative error %g) tol: %g\n", fabs(out-out1), fabs(out-out1)/out1, INTEGRATE_ACCURACYGOAL);
    
    printf("New: %g (relative error %g) tol: %g\n", fabs(trueval-out1), fabs(trueval-out1)/trueval, INTEGRATE_ACCURACYGOAL);
    printf("Old: %g (relative error %g) tol: %g\n", fabs(trueval-out), fabs(trueval-out)/trueval, INTEGRATE_ACCURACYGOAL);
    
    exit(0);
}

#endif
