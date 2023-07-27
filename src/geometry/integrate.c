/** @file integrate.c
 *  @author T J Atherton
 *
 *  @brief Numerical integration
*/

#include "integrate.h"
#include "morpho.h"
#include "classes.h"

#include "matrix.h"
#include "sparse.h"
#include "mesh.h"
#include "selection.h"

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
    value q01[nquantity], q12[nquantity], q20[nquantity], *qn[3];
    
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
    value q01[nquantity], q02[nquantity], q03[nquantity], q12[nquantity], q13[nquantity], q23[nquantity];
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

int nevals;

bool test_integrand(unsigned int dim, double *t, double *x, unsigned int nquantity, value *quantity, void *data, double *fout) {
    //*fout = 0.5*pow(x[0]+x[1], -0.2);
    //double val = sin(M_PI*x[0]); //1/(0.1+x[0]*x[1]);
    double val = (x[0]*x[1]*x[2]);//sqrt(x[0]*x[1]);//(x[0]*x[1]*x[2]);
    *fout=val*val*val*val; //1/sqrt(x[0]); //+ 1/sqrt(x[1]) + 1/sqrt(x[0]+x[1]);
    nevals++;
    return true;
}

/* -------------------------------------------------
 * Basic operations on the integrator data structure
 * ------------------------------------------------- */

/** Initialize an integrator structure */
void integrator_init(integrator *integrate) {
    integrate->integrand = NULL;
    integrate->rule = NULL;
    integrate->errrule = NULL;
    integrate->subdivide = NULL;
    integrate->nquantity = 0;
    integrate->tol = INTEGRATE_ACCURACYGOAL;
    integrate->ztol = INTEGRATE_ZEROCHECK;
    integrate->maxiterations = INTEGRATE_MAXITERATIONS;
    integrate->dim = 0;
    integrate->ref = NULL;
    varray_quadratureworkiteminit(&integrate->worklist);
    varray_doubleinit(&integrate->vertexstack);
    varray_intinit(&integrate->elementstack);
    error_init(&integrate->err);
}

/** Free data associated with an integrator */
void integrator_clear(integrator *integrate) {
    varray_quadratureworkitemclear(&integrate->worklist);
    varray_intclear(&integrate->elementstack);
    varray_doubleclear(&integrate->vertexstack);
}

/** Adds a vertex to the integrators vertex stack, returning the id */
int integrator_addvertex(integrator *integrate, int ndof, double *v) {
    int vid = integrate->vertexstack.count;
    varray_doubleadd(&integrate->vertexstack, v, ndof);
    return vid;
}

/** Adds an element to the element stack, returning the id */
int integrator_addelement(integrator *integrate, int nv, int *vid) {
    int elid=integrate->elementstack.count;
    varray_intadd(&integrate->elementstack, vid, nv);
    return elid;
}

/** Retrieves the vertex pointers given an elementid.
 @warning: The pointers returned should not be used after a call to integrator_addvertex */
void integrator_getvertices(integrator *integrate, int elementid, int nv, double **vert) {
    for (int i=0; i<nv; i++) {
        int vid=integrate->elementstack.data[elementid+i];
        vert[i]=&(integrate->vertexstack.data[vid]);
    }
}

/** Retrieves an element with elementid */
void integrator_getelement(integrator *integrate, int elementid, int nv, int *vid) {
    for (int i=0; i<nv; i++) vid[i]=integrate->elementstack.data[elementid+i];
}

/* --------------------------------
 * Quadrature rules
 * -------------------------------- */

/* --------------------------------
 * Simple midpoint-simpson rule
 * -------------------------------- */

double midpointnodes[] = {
    0.5, 0.5, // Midpoint
    0.0, 1.0, // } Simpsons extension
    1.0, 0.0, // }
};

double midpointweights[] = {
    1.0,
    0.66666666666666667, 0.16666666666666667, 0.16666666666666667
};

quadraturerule midpointsimpson = {
    .dim = 1,
    .order = 0,
    .nnodes = 1,
    .next = 3,
    .nodes = midpointnodes,
    .weights = midpointweights,
};

/* --------------------------------
 * Gauss-Kronrod 1-3 rule
 * -------------------------------- */

double gk13nds[] = {
    0.50000000000000000000, 0.50000000000000000000,
    0.11270166537925831148, 0.88729833462074168852,
    0.88729833462074168852, 0.11270166537925831148
};

double gk13wts[] = {
    1.0,
    0.4444444444444444444445, 0.2777777777777777777778,
    0.277777777777777777778
};

quadraturerule gk13 = {
    .dim = 1,
    .order = 0,
    .nnodes = 1,
    .next = 3,
    .nodes = gk13nds,
    .weights = gk13wts,
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

double gk25wts[] = {
    0.5, 0.5, // Gauss weights
    0.2454545454545454545455, // Kronrod extension
    0.245454545454545454546,
    0.098989898989898989899,
    0.3111111111111111111111,
    0.098989898989898989899
};

quadraturerule gk25 = {
    .dim = 1,
    .order = 0, // Incorrect
    .nnodes = 2,
    .next = 5,
    .nodes = gk25nds,
    .weights = gk25wts,
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

double gk715wts[] = {
    0.0647424830844348466355, // Gauss weights
    0.1398526957446383339505,
    0.1909150252525594724752,
    0.20897959183673469387755,
    0.1909150252525594724752,
    0.13985269574463833395075,
    0.0647424830844348466353,
    
    0.0315460463149892766454, // Kronrod extensions
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

quadraturerule gk715 = {
    .dim = 1,
    .order = 7,
    .nnodes = 7,
    .next = 15,
    .nodes = gk715nds,
    .weights = gk715wts,
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

double triwts[] = {
    -0.5625, 0.5208333333333332, 0.5208333333333332, 0.5208333333333332,
    
    0.1265625, -0.5425347222222222, -0.5425347222222222, -0.5425347222222222,
    0.4168402777777778, 0.4168402777777778, 0.4168402777777778, 0.4168402777777778,
    0.4168402777777778, 0.4168402777777778
};

quadraturerule tri410 = {
    .dim = 2,
    .order = 3, // 4pt rule is order 3,
    .nnodes = 4,
    .next = 10,
    .nodes = tripts,
    .weights = triwts,
};

double triwts1020[] = {
    0.1265625, -0.5425347222222222, -0.5425347222222222, -0.5425347222222222,
    0.4168402777777778, 0.4168402777777778, 0.4168402777777778, 0.4168402777777778,
    0.4168402777777778, 0.4168402777777778,
    
    -0.0158203125, 0.2422030009920634, 0.2422030009920634, 0.2422030009920634,
    -0.6382866753472222, -0.6382866753472222, -0.6382866753472222, -0.6382866753472222,
    -0.6382866753472222, -0.6382866753472222,
    
    0.4118931361607142, 0.4118931361607142, 0.4118931361607142, 0.4118931361607142,
    0.4118931361607142, 0.4118931361607142, 0.4118931361607142, 0.4118931361607142,
    0.4118931361607142, 0.4118931361607142
};

quadraturerule tri1020 = {
    .dim = 2,
    .order = 4, // Wrong
    .nnodes = 10,
    .next = 20,
    .nodes = tripts,
    .weights = triwts1020,
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

double cubtriwts[] ={
    0.225000000000000000, 0.125939180544827153, 0.125939180544827153,
    0.125939180544827153, 0.132394152788506181, 0.132394152788506181,
    0.132394152788506181,
    
    0.0378610912003146833, 0.0376204254131829721, 0.0376204254131829721,
    0.0376204254131829721, 0.0783573522441173376, 0.0783573522441173376,
    0.0783573522441173376, 0.0134442673751654019, 0.0134442673751654019,
    0.0134442673751654019, 0.116271479656965896, 0.116271479656965896,
    0.116271479656965896, 0.0375097224552317488, 0.0375097224552317488,
    0.0375097224552317488, 0.0375097224552317488, 0.0375097224552317488,
    0.0375097224552317488
};

quadraturerule cubtri = {
    .dim = 2,
    .order = 8,
    .nnodes = 7,
    .next = 19,
    .nodes = cubtripts,
    .weights = cubtriwts,
};

/* --------------------------------
 * Tetrahedron
 * -------------------------------- */

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
    .dim = 3,
    .order = 4,
    .nnodes = 35,
    .next = INTEGRATE_NOEXT,
    .nodes = tet5pts,
    .weights = tet5wts,
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
    .dim = 3,
    .order = 4,
    .nnodes = 56,
    .next = INTEGRATE_NOEXT,
    .nodes = tet6pts,
    .weights = tet6wts,
};

quadraturerule *quadrules[] = {
    &tri410,
    &tri1020,
    &tet5,
    &tet6
};

/* --------------------------------
 * Linear interpolation rule
 * -------------------------------- */

bool linearinterpolate(int nbary, double *bary, int dim, double **v, double *x) {
    for (int j=0; j<dim; j++) x[j]=0.0;
    for (int k=0; k<nbary; k++) {
        for (int j=0; j<dim; j++) {
            x[j]+=bary[k]*v[k][j];
        }
    }
}

/* --------------------------------
 * Function to perform quadrature
 * -------------------------------- */

/** Weighted sum of a list using Kahan summation */
double integrate_sumlistweighted(unsigned int nel, double *list, double *wts) {
    double sum=0.0, c=0.0, y,t;

    for (unsigned int i=0; i<nel; i++) {
        y=(list[i]*wts[i])-c;
        t=sum+y;
        c=(t-sum)-y;
        sum=t;
    }

    return sum;
}

/** Integrates a function over an element specified in work, filling out the integral and error estimate if provided */
bool quadrature(integrator *integrate, quadraturerule *rule, quadratureworkitem *work) {
    int n = (rule->next!=INTEGRATE_NOEXT ? rule->next : rule->nnodes);
    int nbary = rule->dim+1; // Number of barycentric points
    
    double *vert[nbary];
    integrator_getvertices(integrate, work->elementid, nbary, vert);
    
    double x[integrate->dim];
    double f[n];
    
    for (unsigned int i=0; i<n; i++) {
        // Interpolate the point
        linearinterpolate(nbary, &rule->nodes[nbary*i], integrate->dim, vert, x);
        
        // Evaluate function
        if (!(*integrate->integrand) (rule->dim, &rule->nodes[nbary*i], x, integrate->nquantity, NULL, integrate->ref, &f[i])) return false;
    }
    
    double r1 = integrate_sumlistweighted(rule->nnodes, f, rule->weights);
    work->lval = work->val = work->weight*r1;
    work->err = -1;
    
    // Estimate error
    if (rule->next!=INTEGRATE_NOEXT) { // Evaluate extension rule
        double r2 = integrate_sumlistweighted(rule->next, f, &rule->weights[rule->nnodes]);
        work->val = work->weight*r2; // Record better estimate
        work->err = work->weight*fabs(r2-r1); // Use the difference as the error estimator
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
 * Subdivision rules
 * -------------------------------- */

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
    .dim = 1,
    .npts = 1,
    .pts = bisectionpts,
    .nels = 2,
    .newels = bisectionintervals,
    .weights = bisectionweights
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
    .dim = 1,
    .npts = 2,
    .pts = trisectionpts,
    .nels = 3,
    .newels = trisectionintervals,
    .weights = trisectionweights
};

/* -------
 *   2D
 * ------- */

/** Quadrasection of 2D triangle */
double triquadrasectionpts[] = {
    0.5, 0.5, 0.0,
    0.0, 0.5, 0.5,
    0.5, 0.0, 0.5
};

double triquadrasectionweights[] = {
    0.25, 0.25, 0.25, 0.25
};

/*
 *       2
 *      / \
 *     5 - 4
 *    / \ / \
 *   0 - 3 - 1
 */

int triquadrasectiontris[] = {
    0, 3, 5,
    3, 1, 4,
    3, 4, 5,
    5, 4, 2
};

subdivisionrule trianglequadrasection = {
    .dim = 2,
    .npts = 3,
    .pts = triquadrasectionpts,
    .nels = 4,
    .newels = triquadrasectiontris,
    .weights = triquadrasectionweights
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
    .dim = 3,
    .npts = 6,
    .pts = tetsubdivpts,
    .nels = 8,
    .newels = tetsubdivtets,
    .weights = tetsubdivwts
};

/* --------------------------------
 * Subdivision
 * -------------------------------- */

bool subdivide(integrator *integrate, quadratureworkitem *work, varray_quadratureworkitem *worklist, int *nels) {
    subdivisionrule *rule = integrate->subdivide;
    quadratureworkitem newitems[rule->nels];
    int nindx = rule->dim+1;
    double *vert[nindx]; // Old vertices
    double x[rule->npts][integrate->dim]; // Interpolated vertices
    
    integrator_getvertices(integrate, work->elementid, nindx, vert);
    
    // Copy across vertex ids from old element
    int vid[nindx+rule->npts];
    integrator_getelement(integrate, work->elementid, nindx, vid);
    
    // Interpolate vertices
    for (int j=0; j<rule->npts; j++) {
        linearinterpolate(nindx, &rule->pts[j*nindx], integrate->dim, vert, x[j]);
    }
    
    // Add vertices
    for (int j=0; j<rule->npts; j++) {
        vid[nindx+j]=integrator_addvertex(integrate, integrate->dim, x[j]);
    }
    
    // Add elements
    for (int i=0; i<rule->nels; i++) {
        newitems[i].val=0.0;
        newitems[i].err=0.0;
        newitems[i].weight=work->weight*rule->weights[i];
        
        // Find vertex ids for the element
        int element[nindx];
        for (int k=0; k<nindx; k++) element[k]=vid[rule->newels[nindx*i+k]];
        
        // Define the new element
        newitems[i].elementid=integrator_addelement(integrate, nindx, element);
    }
    
    // Add these to the work list
    varray_quadratureworkitemadd(worklist, newitems, rule->nels);
    
    *nels = rule->nels;
}

/* --------------------------------
 * Work items
 * -------------------------------- */

DEFINE_VARRAY(quadratureworkitem, quadratureworkitem)

/** Compares two work items in terms of their error */
int _quadratureworkitemcmp(const void *l, const void *r) {
    quadratureworkitem *a = (quadratureworkitem *) l;
    quadratureworkitem *b = (quadratureworkitem *) r;
    if (b->err < a->err) return 1;
    else if (b->err > a->err) return -1;
    return 0;
}

/** Estimate the value of the integrand given a worklist */
double integrate_estimate(varray_quadratureworkitem *worklist) {
    double sum=0.0, c=0.0, y,t;

    for (unsigned int i=0; i<worklist->count; i++) {
        y=worklist->data[i].val-c;
        t=sum+y;
        c=(t-sum)-y;
        sum=t;
    }

    return sum;
}

/** Estimate the error on the integrand given a worklist */
double integrate_error(varray_quadratureworkitem *worklist) {
    double sum=0.0, c=0.0, y,t;

    for (unsigned int i=0; i<worklist->count; i++) {
        y=worklist->data[i].err-c;
        t=sum+y;
        c=(t-sum)-y;
        sum=t;
    }

    return sum;
}

/* --------------------------------
 * Driver routine
 * -------------------------------- */

void integrate(integrandfunction *integrand, unsigned int dim, unsigned int grade, double **x, unsigned int nquantity, value **quantity, void *ref, double *out) {
    
}

void integrate_test(void) {
    nevals = 0;
    
    integrator integ;
    integrator_init(&integ);
    integ.integrand = test_integrand;
    integ.rule = &tet5; //&cubtri; //&tri1020; //&cubtri;//&tri1020;// &tet5; //&tri1020; // &tri410; // = gk715; ;  //;
    integ.errrule = &tet6;
    integ.subdivide = &tetsection; //&trianglequadrasection;
    integ.dim = 3;
    
    double err, est;
    
    double x0[3] = { 0, 0, 0 };
    double x1[3] = { 1, 0, 0 };
    double x2[3] = { 0, 1, 0 };
    double x3[3] = { 0, 0, 1 };
    int v0 = integrator_addvertex(&integ, 3, x0);
    int v1 = integrator_addvertex(&integ, 3, x1);
    int v2 = integrator_addvertex(&integ, 3, x2);
    int v3 = integrator_addvertex(&integ, 3, x3);
    int el0[] = { v0, v1, v2, v3 };
    int elid = integrator_addelement(&integ, 4, el0);

    quadratureworkitem work;
    work.weight = 1.0;
    work.elementid = elid;
    quadrature(&integ, integ.rule, &work); // Perform initial quadrature
    
    varray_quadratureworkitemwrite(&integ.worklist, work);
    int iter;
    
    for (iter=0; iter<integ.maxiterations; iter++) {
        // Check error
        err = integrate_error(&integ.worklist);
        est = integrate_estimate(&integ.worklist);
        
        if (fabs(est)<integ.ztol || fabs(err/est)<integ.tol) break;
        
        // Ensure quadrature list remains sorted
        qsort(integ.worklist.data, integ.worklist.count, sizeof(quadratureworkitem), _quadratureworkitemcmp);
        
        // Pick worst element
        varray_quadratureworkitempop(&integ.worklist, &work);
        
        // Subdivide
        int nels; // Number of elements created
        subdivide(&integ, &work, &integ.worklist, &nels);
        
        // Perform quadrature on each new element
        for (int k=0; k<nels; k++) quadrature(&integ, integ.rule, &integ.worklist.data[integ.worklist.count-k-1]);
        
        // Laurie's sharper error estimator: BIT 23 (1983), 258-261
        // The norm of the difference between two rules |A-B| is
        // usually too pessimistic; this attempts to extrapolate a sharper estimate
        double a1=work.val, b1=work.lval, a2=0, b2=0;
        for (int k=0; k<nels; k++) {
            a2+=integ.worklist.data[integ.worklist.count-k-1].val;
            b2+=integ.worklist.data[integ.worklist.count-k-1].lval;
        }
        
        if (fabs(a2-a1)<fabs(b2-b1) && // Laurie's second consition
            fabs(a2-b2)<fabs(a1-b1)) // Weak form of first condition (see Gonnet)
        {
            double sigma=fabs((a2-a1)/(b2-b1-a2+a1));
            for (int k=0; k<nels; k++) {
                integ.worklist.data[integ.worklist.count-k-1].err*=sigma;
            }
        }
    }
    
    printf("New integrator: %g with %i iterations and %i function evaluations.\n", est, iter, nevals);
    
    nevals = 0;
    double out;
    double *xx[] = { x0, x1, x2, x3 };
    value *quantities[] = { NULL, NULL, NULL, NULL };
    integrate_integrate(test_integrand, 3, 3, xx, 0, quantities, NULL, &out);
    
    printf("Old integrator: %g with %i function evaluations.\n", out, nevals);
    
    double trueval = 0.0000118928690357261785833214404643; //0.261799387799149436538553615273; //0.457142857142857142857142857143; //6.34286348572062857777143491429e-8;//2.70562770562770562770562770563e-6;
    
    printf("Difference %g (relative error %g) tol: %g\n", fabs(out-est), fabs(out-est)/est, integ.tol);
    
    printf("New: %g (relative error %g) tol: %g\n", fabs(trueval-est), fabs(trueval-est)/trueval, integ.tol);
    printf("Old: %g (relative error %g) tol: %g\n", fabs(trueval-out), fabs(trueval-out)/trueval, integ.tol);
    
    integrator_clear(&integ);
    
    exit(0);
}
