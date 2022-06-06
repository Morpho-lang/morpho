/**
 * @file kernals.h
 * @author Danny Goldstein 
 * @brief This file contains the macroized kernals for CUDA and openCL
 * to be included or complied respectivly
 * @version 0.1
 * @date 2022-04-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef MORPHO_EPS
#define MORPHO_EPS 1e-16
#endif
#ifdef CUDA_ACC
#define KERNAL_PREFIX __global__
#define DEVICE_PREFIX __device__
#define GLOBAL_PREFIX
#define CUDASHARE(declaration) __shared__ declaration
#define OPENCLSHARED(declaration)
#define GETID blockDim.x * blockIdx.x + threadIdx.x
#define BLOCKID blockDim.x * blockIdx.x
#define THREADIDX threadIdx.x
#define THREADIDY threadIdx.y
#define BLOCKIDX blockIdx.x
#define BLOCKIDY blockIdx.y
#define BLOCKSIZEX blockDim.x
#define BLOCKSIZEY blockDim.y
#define GRIDDIMX gridDim.x
#define GRIDDIMY gridDim.y
#define SYNCGROUP __syncthreads()


DEVICE_PREFIX double atomic_Add(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#else
#define KERNAL_PREFIX __kernel
#define DEVICE_PREFIX
#define GLOBAL_PREFIX __global
#define CUDASHARE(declaration) 
#define OPENCLSHARED(declaration) , __local declaration
#define GETID get_global_id(0)
#define THREADIDX get_local_id(0)
#define THREADIDY get_local_id(1)
#define BLOCKIDX get_group_id(0)
#define BLOCKIDY get_group_id(1)
#define BLOCKSIZEX get_local_size(0)
#define BLOCKSIZEY get_local_size(1)
#define GRIDDIMX  get_num_groups(0)
#define GRIDDIMY  get_num_groups(1)
#define TILE_DIM 32
#define BLOCK_COLS 8
#define SYNCGROUP barrier(CLK_LOCAL_MEM_FENCE);

typedef struct {
    int nentries;
    int nrows;
    int ncols;
    int *cptr; // Pointers to column entries
    int *rix; // Row indices
    double *values; // Values
} objectgpusparse_light;


#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

double __attribute__((overloadable)) atomic_Add(__global double *valq,double delta) {
   union {
     double f;
     unsigned long  i;
   } old;
   union {
     double f;
     unsigned long  i;
   } new1;
  do {
     old.f = *valq;
     new1.f = old.f + delta;
   } while (atom_cmpxchg((volatile __global unsigned long *)valq, old.i, new1.i) != old.i);
   return old.f;
}  
#endif

// DEVICE_PREFIX bool gpu_area_integrand(double *vert, int dim, int id,int nv, int *vid, double *out);
// DEVICE_PREFIX bool gpu_area_gradient(double *vert, const int dim, int id,const int nv, int *vid, double *frc);
// DEVICE_PREFIX bool gpu_volumeenclosed_integrand(double *vert, int dim, int id,int nv, int *vid, double *out);
// DEVICE_PREFIX bool gpu_volumeenclosed_gradient(double *vert, const int dim, int id,const int nv, int *vid, double *frc);

KERNAL_PREFIX void ScalarAddition(GLOBAL_PREFIX double *matrix ,double scalar,GLOBAL_PREFIX double *out, unsigned int size) {
    // a += b * lambda
    int i = GETID;
    if (i<size){
        *(out+i) = *(matrix + i) + scalar;
    }
}

KERNAL_PREFIX void transposeDiagonal(GLOBAL_PREFIX double *odata, GLOBAL_PREFIX double *idata, int width, int height OPENCLSHARED(double tile[TILE_DIM][TILE_DIM+1])) {
    CUDASHARE(double tile[TILE_DIM][TILE_DIM+1]);
    // printf("I'm in workgroup (%d,%d) and I am work item (%d,%d) my global number is (%d,%d)\n",(int)BLOCKIDX,(int)BLOCKIDY,(int)THREADIDX,(int)THREADIDY,get_global_id(0),get_global_id(1));

    
    int blockIdx_x, blockIdx_y;
    // diagonal reordering
    if (width == height) {
        blockIdx_y = BLOCKIDX;
        blockIdx_x = (BLOCKIDX+BLOCKIDY)%GRIDDIMX;
    } else {
        int bid = BLOCKIDX + GRIDDIMX*BLOCKIDY;
        blockIdx_y = bid%GRIDDIMY;
        blockIdx_x = ((bid/GRIDDIMY)+blockIdx_y)%GRIDDIMX;
    }
    int xIndex = blockIdx_x*TILE_DIM + THREADIDX;
    int yIndex = blockIdx_y*TILE_DIM + THREADIDY;
    int index_in = yIndex + xIndex*height;
    bool outOfBounds = yIndex<height && xIndex<width;
    if (outOfBounds) {
        for (int i=0; i<TILE_DIM && (i+blockIdx_x*TILE_DIM) < width; i+=BLOCK_COLS) {
            tile[THREADIDY][THREADIDX+i] = idata[index_in+i*height];
            // printf("Picking up element %d which is %f and putting in [%d,%d]\n",index_in+i*height,tile[THREADIDY][THREADIDX+i],THREADIDY,THREADIDX+i);

        }
    }
    xIndex = blockIdx_y*TILE_DIM + THREADIDX;
    yIndex = blockIdx_x*TILE_DIM + THREADIDY;
    int index_out = yIndex + (xIndex)*width;
    outOfBounds = xIndex<height && yIndex<width;
    SYNCGROUP;
    if (outOfBounds) {
        for (int i=0; i<TILE_DIM  && i+blockIdx_y*TILE_DIM < height; i+=BLOCK_COLS) {
            odata[index_out+i*width] = tile[THREADIDX+i][THREADIDY];
            // printf("Landing element %d which is %f and was in [%d,%d]\n",index_out+i*width, tile[THREADIDX+i][THREADIDY],THREADIDX+i,THREADIDY);

        }
    }
}

DEVICE_PREFIX bool gpusparse_getrowindices(objectgpusparse_light *gpusparse, int col, int* nentries, int **entries){
    if (col>=gpusparse->ncols) return false;
    *nentries=gpusparse->cptr[col+1]-gpusparse->cptr[col];
    *entries=gpusparse->rix+gpusparse->cptr[col];
    return true;

}



/* **********************************************************************
 * GPU Common library functions for functionals
 * ********************************************************************** */

/** Calculate the difference of two vectors */
DEVICE_PREFIX void gpu_functional_vecadd(unsigned int n, double *a, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]+b[i];
}

/** Add with scale */
DEVICE_PREFIX void gpu_functional_vecaddscale(unsigned int n, double *a, double lambda, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]+lambda*b[i];
}

/** Calculate the difference of two vectors */
DEVICE_PREFIX void gpu_functional_vecsub(unsigned int n, double *a, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]-b[i];
}

/** Scale a vector */
DEVICE_PREFIX void gpu_functional_vecscale(unsigned int n, double lambda, double *a, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=lambda*a[i];
}

/** Calculate the norm of a vector */
DEVICE_PREFIX double gpu_functional_vecnorm(unsigned int n, double *a) {
    double sum = 0;
    double val;
    for (int i = 0; i<n; i++){ 
        val = a[i];
        sum+= val*val;
    }
    return sqrt(sum);
}

/** Dot product of two vectors */
DEVICE_PREFIX double gpu_functional_vecdot(unsigned int n, double *a, double *b) {
    double sum = 0;
    for (int i = 0; i<n; i++){
        sum +=a[i]*b[i];
    }
    return sum;
}


/** 3D cross product  */
DEVICE_PREFIX void gpu_functional_veccross(double *a, double *b, double *out) {
    out[0]=a[1]*b[2]-a[2]*b[1];
    out[1]=a[2]*b[0]-a[0]*b[2];
    out[2]=a[0]*b[1]-a[1]*b[0];
}
DEVICE_PREFIX void gpu_matrix_addtocolumn(GLOBAL_PREFIX double *elements,int nrows, int col, double scale,  double *v) {
    for (int i = 0; i < nrows; i++){
        atomic_Add(&elements[col*nrows+i],scale*v[i]);
    }
}
DEVICE_PREFIX void gpu_matrix_getcolumn(double *elements, int nrows, int col,double **out){
    *out = elements + col*nrows;
}




// bool length_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out);
// bool area_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out);
// bool volume_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out);

// /** Calculate element size */
// bool functional_elementsize(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, double *out) {
//     switch (g) {
//         case 1: return length_integrand(v, mesh, id, nv, vid, NULL, out);
//         case 2: return area_integrand(v, mesh, id, nv, vid, NULL, out);
//         case 3: return volume_integrand(v, mesh, id, nv, vid, NULL, out);
//     }
//     return false;
// }

// TODO move to shared memory for speedup
DEVICE_PREFIX bool gpu_area_integrand(GLOBAL_PREFIX double *vert, int dim, int id, int nv, int *vid,GLOBAL_PREFIX double *out){
    double *x[3], s0[3], s1[3], cx[3];
    for (int j=0; j<nv; j++) gpu_matrix_getcolumn(vert,dim, vid[j], &x[j]);
    gpu_functional_vecsub(dim, x[1], x[0], s0);
    gpu_functional_vecsub(dim, x[2], x[1], s1);

    gpu_functional_veccross(s0, s1, cx);

    *out=0.5*gpu_functional_vecnorm(dim, cx);
    return true;

}
DEVICE_PREFIX bool gpu_area_gradient(GLOBAL_PREFIX double *vert, int dim, int id,int nv, int *vid,GLOBAL_PREFIX double *frc) {
    double *x[3], s0[3], s1[3], s01[3], s010[3], s011[3];
    double norm;
    for (int j=0; j<3; j++) gpu_matrix_getcolumn(vert, dim, vid[j], &x[j]);

    gpu_functional_vecsub(dim, x[1], x[0], s0);
    gpu_functional_vecsub(dim, x[2], x[1], s1);

    gpu_functional_veccross(s0, s1, s01);
    norm=gpu_functional_vecnorm(dim, s01);
    if (norm<MORPHO_EPS) return false;

    gpu_functional_veccross(s01, s0, s010);
    gpu_functional_veccross(s01, s1, s011);
    gpu_matrix_addtocolumn(frc,dim, vid[0], 0.5/norm, s011);
    gpu_matrix_addtocolumn(frc,dim, vid[2], 0.5/norm, s010);

    gpu_functional_vecadd(dim, s010, s011, s0);
    gpu_matrix_addtocolumn(frc,dim, vid[1], -0.5/norm, s0);

    return true;
}
/** Calculate enclosed volume */
DEVICE_PREFIX bool gpu_volumeenclosed_integrand(GLOBAL_PREFIX double *vert, int dim, int id,int nv, int *vid,GLOBAL_PREFIX double *out) {
    double *x[3], cx[3];
    for (int j=0; j<nv; j++) gpu_matrix_getcolumn(vert,dim, vid[j], &x[j]);

    gpu_functional_veccross(x[0], x[1], cx);

    *out=fabs(gpu_functional_vecdot(dim, cx, x[2]))/6.0;
    return true;
}

/** Calculate gradient */
DEVICE_PREFIX bool gpu_volumeenclosed_gradient(GLOBAL_PREFIX double *vert, const int dim, int id,const int nv, int *vid,GLOBAL_PREFIX double *frc) {
    double *x[3], cx[3], dot;
    for (int j=0; j<nv; j++) gpu_matrix_getcolumn(vert, dim, vid[j], &x[j]);

    gpu_functional_veccross(x[0], x[1], cx);
    dot=gpu_functional_vecdot(dim, cx, x[2]);
    dot/=fabs(dot);

    gpu_matrix_addtocolumn(frc,dim, vid[2], dot/6.0, cx);

    gpu_functional_veccross(x[1], x[2], cx);
    gpu_matrix_addtocolumn(frc,dim, vid[0], dot/6.0, cx);

    gpu_functional_veccross(x[2], x[0], cx);
    gpu_matrix_addtocolumn(frc,dim, vid[1], dot/6.0, cx);

    return true;
}
// DEVICE_PREFIX functional_integrand_gpu p_gpu_area_integrand = gpu_area_integrand;
// DEVICE_PREFIX functional_gradient_gpu p_gpu_area_gradient = gpu_area_gradient;
// DEVICE_PREFIX functional_integrand_gpu p_gpu_volumeenclosed_integrand = gpu_volumeenclosed_integrand;
// DEVICE_PREFIX functional_gradient_gpu p_gpu_volumeenclosed_gradient = gpu_volumeenclosed_gradient;


/********************************
 * Kernels to call functionsals *
 * *****************************/
#ifdef CUDA_ACC
KERNAL_PREFIX void functionalIntegrandEval(double* verts, int dim, objectgpusparse_light *s,\
                      int nelements,int integrandNo, functional_integrand_gpu *integrand,double* out){
    // calculate contribution from element i
    int i = GETID;
    if (i<nelements){
        int *vid = NULL; //vertex id
        int nv; // number of vertices in this grade element
        if (s) gpusparse_getrowindices(s,i,&nv,&vid);
        else vid = &i;
        if (vid && nv>0) { 
            (integrand[integrandNo]) (verts,dim,i,nv,vid,&out[i]);

        }
    }
}
KERNAL_PREFIX void functionalGradEval(double* verts, int dim, objectgpusparse_light* s,\
                      int nelements,int gradNo, functional_gradient_gpu *grad,double* out){
    // calculate contribution from element i
    int i = GETID;
    if (i<nelements){
        int *vid = NULL; //vertex id
        int nv; // number of vertices in this grade element
        if (s) gpusparse_getrowindices(s,i,&nv,&vid);
        else vid = &i;
        if (vid && nv>0) { 
            (grad[gradNo]) (verts,dim,i,nv,vid,out);

        }
    }
}
#else
KERNAL_PREFIX void functionalIntegrandEval(GLOBAL_PREFIX double* verts, int dim,GLOBAL_PREFIX objectgpusparse_light *s,\
                      int nelements,int integrandNo,GLOBAL_PREFIX double* out){
    // calculate contribution from element i
    int i = GETID;
    if (i<nelements){
        int *vid = NULL; //vertex id
        int nv; // number of vertices in this grade element
        if (s) gpusparse_getrowindices(s,i,&nv,&vid);
        else vid = &i;
        if (vid && nv>0) { 
            switch (integrandNo) {
                case 0:
                    gpu_area_integrand(verts,dim,i,nv,vid,&out[i]);
                    break;
                case 1:
                    gpu_volumeenclosed_integrand(verts,dim,i,nv,vid,&out[i]);
                    break;
            }
        }
    }
}
KERNAL_PREFIX void functionalGradEval(GLOBAL_PREFIX double* verts, int dim, GLOBAL_PREFIX objectgpusparse_light* s,\
                      int nelements,int gradNo, GLOBAL_PREFIX double* out){
    // calculate contribution from element i
    int i = GETID;
    if (i<nelements){
        int *vid = NULL; //vertex id
        int nv; // number of vertices in this grade element
        if (s) gpusparse_getrowindices(s,i,&nv,&vid);
        else vid = &i;
        if (vid && nv>0) {
            switch (gradNo) {
                case 0:
                    gpu_area_gradient(verts,dim,i,nv,vid,out);
                    break;
                case 1:
                    gpu_volumeenclosed_gradient(verts,dim,i,nv,vid,out);
                    break;
            }
        }
    }
}
#endif