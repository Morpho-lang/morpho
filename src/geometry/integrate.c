/** @file integrate.c
 *  @author T J Atherton
 *
 *  @brief Numerical integration
*/

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include <limits.h>
#include <float.h>

#include "integrate.h"
#include "morpho.h"
#include "classes.h"

#include "linalg.h"
#include "sparse.h"
#include "geometry.h"

/* **********************************************************************
 * Old integrator
 * ********************************************************************** */

bool integrate_recognizequantities(unsigned int nquantity, value *quantity, value *out) {
    if (nquantity>0) {
        for (unsigned int i=0; i<nquantity; i++) {
            if (MORPHO_ISFLOAT(quantity[i])) {
                out[i]=MORPHO_FLOAT(0);
            } else if (MORPHO_ISMATRIX(quantity[i])) {
                out[i]=MORPHO_OBJECT(matrix_clone(MORPHO_GETMATRIX(quantity[i])));
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
                out = matrix_clone(m0);
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
                out = matrix_clone(m0);
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
                out = matrix_clone(m0);
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
 * New integrator
 * ********************************************************************** */

extern quadraturerule *quadrules[];
extern quadraturerule *defaultquadrule[];
extern subdivisionrule *subdivisionrules[];

/* **********************************************
 * Integrator data structure and operations
 * ********************************************** */

DEFINE_VARRAY(quadratureworkitem, quadratureworkitem)

/** Initialize an integrator structure */
void integrator_init(integrator *integrate) {
    integrate->integrand=NULL;
    integrate->ref=NULL;
    
    integrate->dim=0;
    integrate->x=NULL;
    integrate->nbary=0;
    integrate->nquantity=0;
    integrate->quantity=NULL;
    integrate->qval=NULL;
    integrate->qvalcapacity=0;
    
    integrate->adapt=true;
    integrate->rule=NULL;
    integrate->baserule=NULL;
    integrate->errrule=NULL;
    integrate->subdivide=NULL;
    
    varray_quadratureworkiteminit(&integrate->worklist);
    varray_doubleinit(&integrate->vertexstack);
    varray_intinit(&integrate->elementstack);
    
    integrate->ztol=INTEGRATE_ZEROCHECK;
    integrate->tol=INTEGRATE_ACCURACYGOAL;
    integrate->maxiterations=INTEGRATE_MAXITERATIONS;
    
    integrate->niterations=0;
    integrate->val=0.0;
    integrate->errest=0.0;
    integrate->err=NULL;
}

static void integrator_clearquantities(integrator *integrate);

/** Free data associated with an integrator */
void integrator_clear(integrator *integrate) {
    varray_quadratureworkitemclear(&integrate->worklist);
    varray_intclear(&integrate->elementstack);
    varray_doubleclear(&integrate->vertexstack);
    
    integrator_clearquantities(integrate);
}

/** Restore an integrator to its initial state, keeping the identity simplex. */
static void integrator_reset(integrator *integrate) {
    integrate->worklist.count=0;
    integrate->vertexstack.count=integrate->nbary*integrate->nbary;
    integrate->elementstack.count=integrate->nbary;
    
    integrate->niterations=0;
    integrate->val=0.0;
    integrate->errest=0.0;
    
    integrate->rule=integrate->baserule;
}

/** Adds a vertex to the integrators vertex stack, returning the id */
int integrator_addvertex(integrator *integrate, int ndof, double *v) {
    int vid = integrate->vertexstack.count;
    varray_doubleadd(&integrate->vertexstack, v, ndof);
    return vid;
}

/** Adds an element to the element stack, returning the id. Elements are specified by their coordinates in the reference element */
int integrator_addelement(integrator *integrate, int *vids) {
    int elid=integrate->elementstack.count;
    varray_intadd(&integrate->elementstack, vids, integrate->nbary);
    return elid;
}

/** Ensure that the integrator has nq slots for quantities */
static bool integrator_ensurequantities(integrator *integrate, int nq) {
    if (nq<=integrate->qvalcapacity) return true;
    value *nw=MORPHO_REALLOC(integrate->qval, sizeof(value)*nq);
    if (!nw) return false;
    for (int i=integrate->qvalcapacity; i<nq; i++) nw[i]=MORPHO_NIL;
    integrate->qval=nw;
    integrate->qvalcapacity=nq;
    return true;
}

/** Free persistent quantities. */
static void integrator_clearquantities(integrator *integrate) {
    if (!integrate->qval) return;
    for (int i=0; i<integrate->qvalcapacity; i++) {
        if (MORPHO_ISOBJECT(integrate->qval[i])) morpho_freeobject(integrate->qval[i]);
    }
    MORPHO_FREE(integrate->qval);
    integrate->qval=NULL;
    integrate->qvalcapacity=0;
}

/** Process the list of quantities given. Matrix qval clones persist across elements. */
bool integrator_initializequantities(integrator *integrate, int nq, quantity *quantity) {
    integrate->nquantity=nq;
    integrate->quantity=quantity;
    if (nq==0) return true;
    if (!integrator_ensurequantities(integrate, nq)) return false;
    
    for (int i=0; i<nq; i++) {
        if (!quantity[i].vals) return false;
        value q = quantity[i].vals[0]; // Take the first element from each quantity list as paradigmatic
        if (MORPHO_ISFLOAT(q)) {
            quantity[i].ndof=1;
            if (MORPHO_ISOBJECT(integrate->qval[i])) morpho_freeobject(integrate->qval[i]);
            integrate->qval[i]=q;
        } else if (MORPHO_ISMATRIX(q)) { 
            objectmatrix *m = MORPHO_GETMATRIX(q);
            quantity[i].ndof=(int) matrix_countdof(m);
            
            if (MORPHO_ISMATRIX(integrate->qval[i])) { // Try to reuse existing matrix if of the same size
                objectmatrix *old=MORPHO_GETMATRIX(integrate->qval[i]);
                if (old->nrows==m->nrows && old->ncols==m->ncols && old->nvals==m->nvals) continue;
                morpho_freeobject(integrate->qval[i]);
            } else if (MORPHO_ISOBJECT(integrate->qval[i])) morpho_freeobject(integrate->qval[i]);
            
            objectmatrix *new = matrix_clone(m);
            if (!new) return false;
            integrate->qval[i]=MORPHO_OBJECT(new);
        } else return false;
    }
    return true;
}

/** Retrieves the vertex pointers given an elementid.
 @warning: The pointers returned become invalid after a subsequent call to integrator_addvertex . */
void integrator_getvertices(integrator *integrate, int elementid, double **vert) {
    for (int i=0; i<integrate->nbary; i++) {
        int vid=integrate->elementstack.data[elementid+i];
        vert[i]=&(integrate->vertexstack.data[vid]);
    }
}

/** Retrieves an element with elementid */
void integrator_getelement(integrator *integrate, int elementid, int *vid) {
    for (int i=0; i<integrate->nbary; i++) {
        vid[i]=integrate->elementstack.data[elementid+i];
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
    integrate->errest = sumerr;
}

/* --------------------------------
 * Linear interpolation
 * -------------------------------- */

/** Construct vertex transformation matrices
 @param[in] integrate - the integrator
 @param[in] vref - vertices specified in reference element (length integrate->nbary)
 @param[out] r - matrix mapping local node coordinates to ref. el coordinates [r has nbary rows and nbary columns]
 @param[out] v - matrix mapping ref. el coordinates to physical coordinates [v has dim rows and nbary columns] */
void integrator_preparevertices(integrator *integrate, double **vref, double *r, double *v) {
    int l=0;
    if (r) for (int i=0; i<integrate->nbary; i++) { // Loop over vertices [defined rel. to ref. element]
        for (int k=0; k<integrate->nbary; k++) { // Sum over barycentric coordinates
            r[l]=vref[i][k];
            l++;
        }
    }
    
    l=0;
    if (v) for (int i=0; i<integrate->nbary; i++) { // Loop over vertices [defined rel. to ref. element]
        for (int j=0; j<integrate->dim; j++) { // Loop over dimensions
            v[l]=integrate->x[i][j];
            l++;
        }
    }
}

/** Sets up interpolation matrix */
void integrator_prepareinterpolation(integrator *integrate, int elementid, double *rmat, double *vmat) {
    double *vert[integrate->nbary]; // Vertex information
    integrator_getvertices(integrate, elementid, vert);
    integrator_preparevertices(integrate, vert, rmat, vmat);
}

/** Weighted sum of a list */
double integrator_sumlistweighted(unsigned int nel, double *list, double *wts) {
    return cblas_ddot(nel, list, 1, wts, 1);
}

/** Transforms local element coordinates to reference element coordinates */
void integrator_transformtorefelement(integrator *integrate, double *rmat, double *local, double *bary) {
    // Fast inlined multiply and add loop: Multiply nbary x nbary (rmat) with nbary x 1 (local) to get nbary x 1 (bary)
    int nbary=integrate->nbary;
    for (int j=0; j<nbary; j++) bary[j]=0;
    for (int k=0; k<nbary; k++) for (int j=0; j<nbary; j++) bary[j]+=rmat[k*nbary+j]*local[k];
}

/** Transform from reference element barycentric coordinates to physical coordinates */
void integrator_interpolatecoordinates(integrator *integrate, double *lambda, double *vmat, double *x) {
    // Fast inlined multiply and add loop: Multiply dim x nbary (vmat) with nbary x 1 (lambda) to get dim x 1 (x)
    int dim=integrate->dim, nbary=integrate->nbary;
    for (int j=0; j<dim; j++) x[j]=0;
    for (int k=0; k<nbary; k++) for (int j=0; j<dim; j++) x[j]+=vmat[k*dim+j]*lambda[k];
}

/** Physical interpolation on the root element: x = sum lambda_k vertex_k (no packed vmat). */
static void integrator_interpolatefromx(integrator *integrate, double *lambda, double *x) {
    int dim=integrate->dim, nbary=integrate->nbary;
    for (int j=0; j<dim; j++) {
        double s=0.0;
        for (int k=0; k<nbary; k++) s+=lambda[k]*integrate->x[k][j];
        x[j]=s;
    }
}

/** Sums a weighted list of quantities */
bool integrator_sumquantityweighted(int n, double *wts, value *q, value *out) {
    bool success=false;
    if (MORPHO_ISFLOAT(q[0])) {
        double s = 0.0; // Fast inlined dot product loop
        for (int j=0; j<n; j++) s += wts[j]*MORPHO_GETFLOATVALUE(q[j]);
        *out=MORPHO_FLOAT(s);
        success=true;
    } else if (MORPHO_ISMATRIX(q[0])) {
        objectmatrix *sum = MORPHO_GETMATRIX(*out);
        int ndof = (int) sum->nels; // Fast inlined axpy loop
        double *dest = sum->elements;
        for (int k=0; k<ndof; k++) dest[k] = 0.0;
        for (int j=0; j<n; j++) {
            double w = wts[j];
            double *e = MORPHO_GETMATRIX(q[j])->elements;
            for (int k=0; k<ndof; k++) dest[k] += w*e[k];
        }
        success=true;
    }
    return success;
}

/** Interpolates quantities */
void integrator_interpolatequantities(integrator *integrate, double *bary) {
    for (int i=0; i<integrate->nquantity; i++) {
        int nnodes = integrate->quantity[i].nnodes;
        double wts[nnodes];
        if (integrate->quantity[i].ifn) {
            (integrate->quantity[i].ifn) (bary, wts);
        } else {
            for (int k=0; k<nnodes; k++) wts[k]=bary[k];
        }
        
        integrator_sumquantityweighted(nnodes, wts, integrate->quantity[i].vals, &integrate->qval[i]);
    }
}

/* --------------------------------
 * Function to perform quadrature
 * -------------------------------- */

/** Evaluates the integrand at specified places.
 * @param[in] integrate - the integrator
 * @param[in] rule - the quadrature rule
 * @param[in] imin - the index of the first node to evaluate
 * @param[in] imax - the index of the last node to evaluate
 * @param[in] rmat - the transformation matrix from local to reference element coordinates (may be NULL if ridentity)
 * @param[in] vmat - the transformation matrix from reference element to physical coordinates
 * @param[in] x - the physical coordinates of the quadrature points
 * @param[out] f - the values of the integrand at the quadrature points
 * @param[in] ridentity - set true to skip the transformation from local to reference element coordinates
 * @return true if the evaluation was successful, false otherwise */
bool integrator_evalfn(integrator *integrate, quadraturerule *rule, int imin, int imax, double *rmat, double *vmat, double *x, double *f, bool ridentity) {
    int nbary=integrate->nbary;
    double nodebuf[nbary];
    
    for (int i=imin; i<imax; i++) {
        double *node;
        if (ridentity) {
            node=&rule->nodes[nbary*i];
            integrator_interpolatefromx(integrate, node, x);
        } else {
            node=nodebuf;
            integrator_transformtorefelement(integrate, rmat, &rule->nodes[nbary*i], nodebuf);
            integrator_interpolatecoordinates(integrate, node, vmat, x);
        }
        if (integrate->nquantity) integrator_interpolatequantities(integrate, node);
        
        // Evaluate function
        if (!(*integrate->integrand) (integrate->dim, node, x, integrate->nquantity, integrate->qval, integrate->ref, &f[i])) return false;
    }
    return true;
}

/** @brief Integrates a function over an element specified in work, filling out the integral and error estimate if provided 
 * @param[in] integrate - the integrator
 * @param[in] rule - the quadrature rule
 * @param[in] rmat - the transformation matrix from local to reference element coordinates (may be NULL if ridentity)
 * @param[in] vmat - the transformation matrix from reference element to physical coordinates
 * @param[in] ridentity - set true to skip the transformation from local to reference element coordinates
 * @param[out] work - the work item containing the integral and error estimate */
static bool integrator_applyrule(integrator *integrate, quadraturerule *rule, double *rmat, double *vmat, bool ridentity, quadratureworkitem *work) {
    int nmax = rule->nnodes;
    int np = 0; // Number of levels of p-refinement
    for (quadraturerule *q = rule->ext; q!=NULL; q=q->ext) { // Find maximum number of pts 
        nmax = q->nnodes;
        np++;
    }
    
    double x[integrate->dim], f[nmax]; // Evaluate function at quadrature points
    if (!integrator_evalfn(integrate, rule, 0, rule->nnodes, rmat, vmat, x, f, ridentity)) return false;
    
    double r[np+1];
    double eps[np+1]; eps[0]=0.0;
    
    // Obtain estimate
    r[0]=integrator_sumlistweighted(rule->nnodes, f, rule->weights);
    work->lval = work->val = work->weight*r[0];
    
    if (!integrate->adapt) {
        work->err=0.0;
        return true;
    }
    
    // Estimate error
    if (rule->ext!=NULL) { // Evaluate extension rule
        int nmin = rule->nnodes, ip=0;
        
        // Attempt p-refinement if available
        for (quadraturerule *q=rule->ext; q!=NULL; q=q->ext) {
            ip++;
            if (!integrator_evalfn(integrate, q, nmin, q->nnodes, rmat, vmat, x, f, ridentity)) return false;
            
            r[ip]=integrator_sumlistweighted(q->nnodes, f, q->weights);
            eps[ip]=fabs(r[ip]-r[ip-1]);
            nmin = q->nnodes;
            
            if (fabs(r[ip])<integrate->ztol ||
                fabs(eps[ip]/r[ip])<integrate->tol) break;
        }
        
        work->lval = work->weight*r[ip-1];
        work->val = work->weight*r[ip]; // Record better estimate
        work->err = work->weight*eps[ip]; // Use the difference as the error estimator
    } else if (integrate->errrule) {  // Otherwise, use the error rule to obtain the estimate
        if (rule==integrate->errrule) return true; // We already are using the error rule
        double temp = work->val; // Retain the lower order estimate
        if (!integrator_applyrule(integrate, integrate->errrule, rmat, vmat, ridentity, work)) return false;
        work->lval=temp;
        work->err=fabs(work->val-temp); // Estimate error from difference of rules
    } else {
        UNREACHABLE("Integrator definition inconsistent.");
    }
    
    return true;
}

/** Root element: identity barycentric map, physical vertices from integrate->x. No stacks. */
static bool integrator_quadrature_root(integrator *integrate, quadraturerule *rule, quadratureworkitem *work) {
    return integrator_applyrule(integrate, rule, NULL, NULL, true, work);
}

/** Integrates a function over an element specified in work, filling out the integral and error estimate if provided */
bool integrator_quadrature(integrator *integrate, quadraturerule *rule, quadratureworkitem *work) {
    double rmat[integrate->nbary*integrate->nbary];
    double vmat[integrate->nbary*integrate->dim];
    integrator_prepareinterpolation(integrate, work->elementid, rmat, vmat);
    
    return integrator_applyrule(integrate, rule, rmat, vmat, false, work);
}

/* --------------------------------
 * Subdivision
 * -------------------------------- */

/** Subdivides an element into new elements */
bool integrator_subdivide(integrator *integrate, quadratureworkitem *work, int *nels, quadratureworkitem *newitems) {
    subdivisionrule *rule = integrate->subdivide;
    
    // Fetch the element data
    int vid[integrate->nbary+rule->npts];
    integrator_getelement(integrate, work->elementid, vid);
    
    // Get ready for interpolation
    double rmat[integrate->nbary*integrate->nbary]; // Vertex information
    integrator_prepareinterpolation(integrate, work->elementid, rmat, NULL);
    
    // Interpolate vertices
    double lambda[integrate->nbary];
    for (int j=0; j<rule->npts; j++) {
        integrator_transformtorefelement(integrate, rmat, &rule->pts[j*integrate->nbary], lambda);
        vid[integrate->nbary+j]=integrator_addvertex(integrate, integrate->nbary, lambda);
    }
    
    // Create elements
    for (int i=0; i<rule->nels; i++) {
        newitems[i].val=0.0;
        newitems[i].err=0.0;
        newitems[i].weight=work->weight*rule->weights[i];
        if (!(newitems[i].weight>DBL_EPSILON)) { // Check for vanishing triangle size
            error_writewithid(integrate->err, INTEGRATE_SBDVSNS);
            return false; 
        }
        
        // Construct new element from the vertex ids
        int vids[integrate->nbary];
        for (int k=0; k<integrate->nbary; k++) {
            vids[k]=vid[rule->newels[integrate->nbary*i+k]];
        }
        
        // Define the new element
        newitems[i].elementid=integrator_addelement(integrate, vids);
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
void integrator_sharpenerrorestimate(integrator *integrate, quadratureworkitem *work, int nels, quadratureworkitem *newitems) {
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
void integrator_update(integrator *integrate, quadratureworkitem *work, int nels, quadratureworkitem *newitems) {
    double dval=0, derr=0;
    integrate->val-=work->val;
    integrate->errest-=work->err;
    for (int k=0; k<nels; k++) {
        dval+=newitems[k].val;
        derr+=newitems[k].err;
        integrator_pushworkitem(integrate, &newitems[k]);
    }
    integrate->val+=dval;
    integrate->errest+=derr;
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

/** Returns a default rule for each grade */
bool integrator_matchrulebygrade(int grade, quadraturerule **out) {
    for (int i=0; defaultquadrule[i]!=NULL; i++) {
        if (defaultquadrule[i]->grade==grade) {
            *out = defaultquadrule[i];
            return true;
        }
    }
    return false;
}

/** Builds the reference simplex in barycentric coordinates */
static void integrator_buildrefsimplex(integrator *integrate) {
    integrate->vertexstack.count=0;
    integrate->elementstack.count=0;
    
    int nbary=integrate->nbary;
    int vids[nbary];
    double xref[nbary];
    for (int i=0; i<nbary; i++) xref[i]=0.0;
    for (int i=0; i<nbary; i++) {
        xref[i]=1.0;
        vids[i]=integrator_addvertex(integrate, nbary, xref);
        xref[i]=0.0;
    }
    integrator_addelement(integrate, vids);
}

/** Configures an integrator based on the grade to integrate and hints for order and rule type
 * @param[in] integrate     - integrator structure to be configured
 * @param[in] err                  - error structure to report errors to
 * @param[in] adapt             - enable adaptive refinement
 * @param[in] grade              - Dimension of the vertices
 * @param[in] order              - Requested order of quadrature rule
 * @param[in] name                - Alternatively, supply the name of a known rule
 * @returns true if the configuration was successful */
bool integrator_configure(integrator *integrate, error *err, bool adapt, int grade, int order, char *name) {
    integrate->rule=NULL;
    integrate->baserule=NULL;
    integrate->errrule=NULL;
    integrate->adapt=adapt;
    integrate->err=err;
    integrate->nbary=grade+1; // Number of barycentric coordinates
    integrate->vertexstack.count=0;
    integrate->elementstack.count=0;
    
    if (name) {
        if (!integrator_matchrulebyname(grade, name, &integrate->rule)) {
            error_writewithid(err, INTEGRATE_RLNTFND, name);
            return false;
        }
    } else if (order>=0) {
        integrator_matchrulebyorder(grade, order, INT_MAX, false, &integrate->rule);
    } else {
        integrator_matchrulebygrade(grade, &integrate->rule);
    }
    
    // Check we succeeded in finding a rule
    if (!integrate->rule) {
        error_writewithid(err, INTEGRATE_RLUNAVLB);
        return false;
    }
    
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
    
    integrate->baserule=integrate->rule;
    
    integrator_buildrefsimplex(integrate);
    
    return true;
}

/** Configures the integrator based on the contents of a dictionary */
bool integrator_configurewithdictionary(integrator *integrate, error *err, grade g, objectdictionary *dict) {
    char *name=NULL;
    bool adapt=true;
    int order=-1;
    value val;
    
    objectstring rulelabel = MORPHO_STATICSTRING(INTEGRATE_RULELABEL);
    objectstring degreelabel = MORPHO_STATICSTRING(INTEGRATE_DEGREELABEL);
    objectstring adaptlabel = MORPHO_STATICSTRING(INTEGRATE_ADAPTLABEL);
    
    if (dictionary_get(&dict->dict, MORPHO_OBJECT(&rulelabel), &val)) {
        if (MORPHO_ISSTRING(val)) {
            name = MORPHO_GETCSTRING(val);
        } else {
            error_writewithid(err, INTEGRATE_MTHDTYP, INTEGRATE_RULELABEL, STRING_CLASSNAME);
            return false;
        }
    }

    if (dictionary_get(&dict->dict, MORPHO_OBJECT(&degreelabel), &val)) {
        if (MORPHO_ISINTEGER(val)) {
            order = MORPHO_GETINTEGERVALUE(val);
        } else {
            error_writewithid(err, INTEGRATE_MTHDTYP, INTEGRATE_DEGREELABEL, INT_CLASSNAME);
            return false;
        }
    }
    
    if (dictionary_get(&dict->dict, MORPHO_OBJECT(&adaptlabel), &val)) {
        if (MORPHO_ISBOOL(val)) {
            adapt = MORPHO_GETBOOLVALUE(val);
        } else {
            error_writewithid(err, INTEGRATE_MTHDTYP, INTEGRATE_ADAPTLABEL, BOOL_CLASSNAME);
            return false;
        }
    }
    
    return integrator_configure(integrate, err, adapt, g, order, name);
}

/* --------------------------------
 * Driver routine
 * -------------------------------- */

/** True if the current work item is already accurate enough (or adapt is off). */
static bool integrator_rootconverged(integrator *integrate, quadratureworkitem *work) {
    return !integrate->adapt ||
        fabs(work->val)<integrate->ztol ||
        fabs(work->err/work->val)<integrate->tol;
}

/** Integrates over a function
 * @param[in] integrate     - integrator structure, that has been configured with integrator_configure
 * @param[in] integrand     - function to integrate
 * @param[in] dim                  - Dimension of the vertices
 * @param[in] x                       - vertices of the line x[0] = {x,y,z} etc.
 * @param[in] nquantity     - number of quantities per vertex
 * @param[in] quantity       - List of quantities for each vertex.
 * @param[in] ref                  - a pointer to any data required by the function
 * @returns True on success */
bool integrator_integrate(integrator *integrate, integrandfunction *integrand, int dim, double **x, unsigned int nquantity, quantity *quantity, void *ref) {
    integrator_reset(integrate);
    
    integrate->integrand=integrand; // Integrand function
    integrate->ref=ref;
    
    integrate->x=x; // Vertices
    integrate->dim=dim;
    
    if (!integrator_initializequantities(integrate, nquantity, quantity)) return false;
    
    quadratureworkitem work;
    work.weight = 1.0;
    work.elementid = 0;
    if (!integrator_quadrature_root(integrate, integrate->rule, &work)) return false;
    
    // Fast path: Check if quadrature on the reference worked; if so report error estimate and return */
    if (integrator_rootconverged(integrate, &work)) {
        integrate->val=work.val;
        integrate->errest=work.err;
        return true;
    }
    
    integrator_pushworkitem(integrate, &work);
    integrator_estimate(integrate); // Initial estimate
    
    if (integrate->adapt) for (integrate->niterations=0; integrate->niterations<=integrate->maxiterations; integrate->niterations++) {
        // Convergence check
        if (fabs(integrate->val)<integrate->ztol || fabs(integrate->errest/integrate->val)<integrate->tol) break;
        
        // Get worst interval
        integrator_popworkitem(integrate, &work);
        
        // Subdivide
        int nels; // Number of elements created
        quadratureworkitem newitems[integrate->subdivide->nels];
        
        if (!integrator_subdivide(integrate, &work, &nels, newitems)) return false;
        for (int k=0; k<nels; k++) {
            if (!integrator_quadrature(integrate, integrate->rule, &newitems[k])) return false;
        }
        
        // Error estimate
        integrator_sharpenerrorestimate(integrate, &work, nels, newitems);
        
        // Add new items to heap and update error estimates
        integrator_update(integrate, &work, nels, newitems);
    }
    
    // Final estimate by Kahan summing heap
    integrator_estimate(integrate);
    
    return true;
}

/* ---------------------------------------
 * Public interface resembling old version
 * --------------------------------------- */

/** Integrate over an element - public interface for one off integrals.
 * @param[in] integrand   - integrand
 * @param[in] method         - Dictionary with method selection (optional)
 * @param[in] err                - Error structure to report errors (optional)
 * @param[in] dim                - Dimension of the vertices
 * @param[in] grade            - Grade to integrate over
 * @param[in] x                     - vertices of the triangle x[0] = {x,y,z} etc.
 * @param[in] nquantity   - number of quantities per vertex
 * @param[in] quantity     - List of quantities
 * @param[in] ref                - a pointer to any data required by the function
 * @param[out] out              - value of the integral
 * @param[out] errest        - an estimate of the error
 * @returns true on success. */
bool integrate(integrandfunction *integrand, objectdictionary *method, error *err, unsigned int dim, unsigned int grade, double **x, unsigned int nquantity, quantity *quantity, void *ref, double *out, double *errest) {
    bool success=false;
    integrator integrate;
    integrator_init(&integrate);
    
    if (method) {
        success=integrator_configurewithdictionary(&integrate, err, grade, method);
    } else {
        success=integrator_configure(&integrate, err, true, grade, -1, NULL);
    }
    
    if (success) success=integrator_integrate(&integrate, integrand, dim, x, nquantity, quantity, ref);
    
    if (success) {
        *out = integrate.val;
        if (errest) *errest = integrate.errest;
    }
    
    integrator_clear(&integrate);
    
    return success;
}

/* -------------------------------------
 * Public interface matching old version
 * ------------------------------------- */

void integrate_initialize(void) {
    morpho_defineerror(INTEGRATE_SBDVSNS, ERROR_HALT, INTEGRATE_SBDVSNS_MSG);
    morpho_defineerror(INTEGRATE_RLNTFND, ERROR_HALT, INTEGRATE_RLNTFND_MSG);
    morpho_defineerror(INTEGRATE_RLUNAVLB, ERROR_HALT, INTEGRATE_RLUNAVLB_MSG);
    morpho_defineerror(INTEGRATE_MTHDTYP, ERROR_HALT, INTEGRATE_MTHDTYP_MSG);
}

#endif
