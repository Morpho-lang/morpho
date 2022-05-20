/* @file gpuinterface.c
* @author Danny Goldstein
* @breif a GPU interface for morpho desniged to work for openCL and CUDA
* This is the cuda interface
*/ 
#ifndef cudainterface_h
#define cudainterface_h
#include "cuda.h"
#include <stdio.h>
#include <stdbool.h>
//#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusolver_common.h"
#include "cusolverDn.h"

//#include "vm.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef bool (*functional_integrand_gpu) (double *vert, int dim, int id, int nv, int *vid, double *out);
typedef bool (*functional_gradient_gpu) (double *vert, int dim, int id, int nv, int *vid, double *out);

// we put a ligher version of spare here to avoid having to include object here (nvcc doens't like it)
typedef struct {
    int nentries;
    int nrows;
    int ncols;
    int *cptr; // Pointers to column entries
    int *rix; // Row indices
    double *values; // Values
} objectgpusparse_light;


// typedef struct {
//     int nentries;
//     int nrows;
//     int ncols;
//     int *cptr; // Pointers to column entries
//     int *rix; // Row indices
//     double *values; // Values
// } objectgpumatrix_light;


typedef struct {
    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;
	cublasHandle_t cublasHandle;
    cusolverStatus_t cusolverStatus;
    cusolverDnHandle_t cusolverHandle;

    functional_integrand_gpu *d_integrands;
    functional_gradient_gpu *d_gradients;
    
    //vm * v;
    bool init;
} GPUStatus;

extern GPUStatus myGPUstatus;


void GPUsetup(GPUStatus* cudaInterface);//,vm* v);
void GPUallocate(GPUStatus* cudaInterface,void** ptr, unsigned int size);
void GPUdeallocate(GPUStatus* cudaInterface,void* dPointer);
void GPUreallocate(GPUStatus* cudaInterface,void** ptr, unsigned int newsize,unsigned int oldsize);
void GPUmemset(GPUStatus* cudaInterface,void* devicePointer, int bytepattern, unsigned int size);
void GPUcopy_to_host(GPUStatus* cudaInterface,void * hostPointer, void * devicepointer, unsigned int size);
void GPUcopy_to_device(GPUStatus* cudaInterface,void* devicePointer, void* hostPointer, unsigned int size);
void GPUcopy_device_to_device(GPUStatus* cudaInterface,void* devicePointersrc, void* devicePointerdest, unsigned int size);
void GPUcopy_symbol(void* dst, void* symbol, size_t size);

void GPUScalarAddition(GPUStatus* cudaInterface, double* Matrix, double scalar, double *out, int size);
void GPUTranspose(GPUStatus* cudaInterface, double* in, double* out, int ncols, int nrows);
int GPUSparseTranspose(GPUStatus* cudaInterface, objectgpusparse_light *in,objectgpusparse_light *out);
/** cuBLAS functions*/
void dotProduct(GPUStatus* cudaInterface, double* v1, double * v2, int size, double * out);
void GPUdot(GPUStatus* cudaInterface,int size, double *x,int incx, double *y, int incy,double* out);
void GPUaxpy(GPUStatus* cudaInterface, int n, double* alpha, double * x, int incx, double * y, int incy);
void GPUcopy(GPUStatus* cudaInterface,int n, double * x, int incx, double *y, int incy);
void GPUScale(GPUStatus* cudaInterface, int n, const double *alpha, double *x, int incx);
void GPUgemm(GPUStatus* cudaInterface, int m, int n, int k, const double *alpha,\
             const double *A, int lda, const double *B, int ldb,\
             const double *beta, double *C, int ldc);
void GPUnrm2(GPUStatus* cudaInterface,int n, const double *x, int incx, double *result);

/** cuSolve function*/
void GPUgesv(GPUStatus* cudaInterface, int n, int nrhs, double* dA, int ldda, int* dipiv, double* dB,\
 int lddb, double* dX, int lddx, int * niter, int *dinfo);
void GPUgetrf(GPUStatus* cudaInterface,int nrows, int ncols,double* elements,int lda,int* pivot,int* devInfo);
void GPUgetrs(GPUStatus* cudaInterface, int nrows, int ncolsB, double * A, int lda, int *devIpiv, double* B, int ldb, int *devInfo);
void GPUcall_functional(GPUStatus* cudaInterface,double* verts, int dim, objectgpusparse_light* conn, int grade, int nelements,int integrandNo, double* out);
void GPUcall_functionalgrad(GPUStatus* cudaInterface,double* verts, int dim, objectgpusparse_light* conn, int grade, int nelements,int gradientNo, double* out);

__device__ void gpu_functional_vecadd(unsigned int n, double *a, double *b, double *out);
__device__ void gpu_functional_vecaddscale(unsigned int n, double *a, double lambda, double *b, double *out);
__device__ double gpu_functional_vecdot(unsigned int n, double *a, double *b);
__device__ double gpu_functional_vecnorm(unsigned int n, double *a);
__device__ void gpu_functional_vecscale(unsigned int n, double lambda, double *a, double *out);
__device__ void gpu_functional_vecsub(unsigned int n, double *a, double *b, double *out);
__device__ void gpu_functional_veccross(double *a, double *b, double *out);
__device__ void gpu_matrix_addtocolumn(double *elements, int nrows, int col, double scale, double *v);
__device__ void gpu_matrix_getcolumn(double *elements, int nrows, int col,double **out);
__device__ bool gpu_area_integrand(double *vert, int dim, int id, int nv, int *vid, double *out);
__device__ bool gpu_area_gradient(double *vert, int dim, int id,int nv, int *vid, double *frc); 
__device__ bool gpu_volumeenclosed_integrand(double *vert, int dim, int id,int nv, int *vid, double *out);
__device__ bool gpu_volumeenclosed_gradient(double *vert, int dim, int id,int nv, int *vid, double *frc);
// extern __device__ functional_integrand_gpu p_gpu_area_integrand;
// extern __device__ functional_gradient_gpu p_gpu_area_gradient;
// extern __device__ functional_integrand_gpu p_gpu_volumeenclosed_integrand;
// extern __device__ functional_gradient_gpu p_gpu_volumeenclosed_gradient;
// __device__ functional_integrand_gpu p_gpu_area_integrand;
// __device__ functional_gradient_gpu p_gpu_area_gradient;
// __device__ functional_integrand_gpu p_gpu_volumeenclosed_integrand;
// __device__ functional_gradient_gpu p_gpu_volumeenclosed_gradient;
// __device__ functional_integrand_gpu p_gpu_area_integrand = gpu_area_integrand;
// __device__ functional_gradient_gpu p_gpu_area_gradient = gpu_area_gradient;
// __device__ functional_integrand_gpu p_gpu_volumeenclosed_integrand = gpu_volumeenclosed_integrand;
// __device__ functional_gradient_gpu p_gpu_volumeenclosed_gradient = gpu_volumeenclosed_gradient;


#ifdef __cplusplus
}
#endif

#endif