/** @file integrate.c
 *  @author T J Atherton
 *
 *  @brief Numerical integration
*/

#include "integrate.h"
#include "morpho.h"
#include "object.h"
#include "builtin.h"
#include "veneer.h"
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
        if ((*function) (dim, lambda, xx, nquantity, q, ref, &fout)){
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
