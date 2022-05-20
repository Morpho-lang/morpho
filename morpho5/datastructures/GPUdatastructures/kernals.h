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

__global__ void ScalarAddition(double *matrix ,double scalar, double *out, unsigned int size){
    // a += b * lambda
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<size){
        *(out+i) = *(matrix + i) + scalar;
    }
}

__global__ void transposeDiagonal(double *odata, double *idata, int width, int height) {
    __shared__ double tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    // diagonal reordering
    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }
    int xIndex = blockIdx_x*TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y*TILE_DIM + threadIdx.y;
    int index_in = yIndex + xIndex*height;
    bool outOfBounds = yIndex<height && xIndex<width;
    if (outOfBounds) {
        for (int i=0; i<TILE_DIM && (i+blockIdx_x*TILE_DIM) < width; i+=BLOCK_COLS) {
            tile[threadIdx.y][threadIdx.x+i] = idata[index_in+i*height];
        }
    }
    xIndex = blockIdx_y*TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x*TILE_DIM + threadIdx.y;
    int index_out = yIndex + (xIndex)*width;
    outOfBounds = xIndex<height && yIndex<width;
    __syncthreads();
    if (outOfBounds) {
        for (int i=0; i<TILE_DIM  && i+blockIdx_y*TILE_DIM < height; i+=BLOCK_COLS) {
            odata[index_out+i*width] = tile[threadIdx.x+i][threadIdx.y];
        }
    }
}

__device__ bool gpusparse_getrowindices(objectgpusparse_light *gpusparse, int col, int* nentries, int **entries){
    if (col>=gpusparse->ncols) return false;
    *nentries=gpusparse->cptr[col+1]-gpusparse->cptr[col];
    *entries=gpusparse->rix+gpusparse->cptr[col];
    return true;

}
__global__ void functionalIntegrandEval(double* verts, int dim, objectgpusparse_light *s,\
                      int nelements,int integrandNo, functional_integrand_gpu *integrand,double* out){
    // calculate contribution from element i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
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
__global__ void functionalGradEval(double* verts, int dim, objectgpusparse_light* s,\
                      int nelements,int gradNo, functional_gradient_gpu *grad,double* out){
    // calculate contribution from element i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
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
/* **********************************************************************
 * GPU Common library functions for functionals
 * ********************************************************************** */

/** Calculate the difference of two vectors */
__device__ void gpu_functional_vecadd(unsigned int n, double *a, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]+b[i];
}

/** Add with scale */
__device__ void gpu_functional_vecaddscale(unsigned int n, double *a, double lambda, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]+lambda*b[i];
}

/** Calculate the difference of two vectors */
__device__ void gpu_functional_vecsub(unsigned int n, double *a, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]-b[i];
}

/** Scale a vector */
__device__ void gpu_functional_vecscale(unsigned int n, double lambda, double *a, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=lambda*a[i];
}

/** Calculate the norm of a vector */
__device__ double gpu_functional_vecnorm(unsigned int n, double *a) {
    double sum = 0;
    double val;
    for (int i = 0; i<n; i++){ 
        val = a[i];
        sum+= val*val;
    }
    return sqrt(sum);
}

/** Dot product of two vectors */
__device__ double gpu_functional_vecdot(unsigned int n, double *a, double *b) {
    double sum = 0;
    for (int i = 0; i<n; i++){
        sum +=a[i]*b[i];
    }
    return sum;
}
__device__ double atomic_Add(double* address, double val)
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

/** 3D cross product  */
__device__ void gpu_functional_veccross(double *a, double *b, double *out) {
    out[0]=a[1]*b[2]-a[2]*b[1];
    out[1]=a[2]*b[0]-a[0]*b[2];
    out[2]=a[0]*b[1]-a[1]*b[0];
}
__device__ void gpu_matrix_addtocolumn(double *elements,int nrows, int col, double scale,  double *v) {
    for (int i = 0; i < nrows; i++){
        atomic_Add(&elements[col*nrows+i],scale*v[i]);
    }
}
__device__ void gpu_matrix_getcolumn(double *elements, int nrows, int col,double **out){
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
__device__ bool gpu_area_integrand(double *vert, int dim, int id, int nv, int *vid, double *out){
    double *x[3], s0[3], s1[3], cx[3];
    for (int j=0; j<nv; j++) gpu_matrix_getcolumn(vert,dim, vid[j], &x[j]);
    gpu_functional_vecsub(dim, x[1], x[0], s0);
    gpu_functional_vecsub(dim, x[2], x[1], s1);

    gpu_functional_veccross(s0, s1, cx);

    *out=0.5*gpu_functional_vecnorm(dim, cx);
    return true;

}
__device__ bool gpu_area_gradient(double *vert, int dim, int id,int nv, int *vid, double *frc) {
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
__device__ bool gpu_volumeenclosed_integrand(double *vert, int dim, int id,int nv, int *vid, double *out) {
    double *x[3], cx[3];
    for (int j=0; j<nv; j++) gpu_matrix_getcolumn(vert,dim, vid[j], &x[j]);

    gpu_functional_veccross(x[0], x[1], cx);

    *out=fabs(gpu_functional_vecdot(dim, cx, x[2]))/6.0;
    return true;
}

/** Calculate gradient */
__device__ bool gpu_volumeenclosed_gradient(double *vert, const int dim, int id,const int nv, int *vid, double *frc) {
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
// __device__ functional_integrand_gpu p_gpu_area_integrand = gpu_area_integrand;
// __device__ functional_gradient_gpu p_gpu_area_gradient = gpu_area_gradient;
// __device__ functional_integrand_gpu p_gpu_volumeenclosed_integrand = gpu_volumeenclosed_integrand;
// __device__ functional_gradient_gpu p_gpu_volumeenclosed_gradient = gpu_volumeenclosed_gradient;


