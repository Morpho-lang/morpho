/* @file gpuinterface.c
* @author Danny Goldstein
* @breif a GPU interface for morpho desniged to work for openCL and CUDA
* This is the cuda interface
*/ 
#ifndef cudainterface_h
#define cudainterface_h
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
//#include "vm.h"



typedef struct {
    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;
	cublasHandle_t cublasHandle;
    //vm * v;
    bool init;
} GPUStatus;
void GPUsetup(GPUStatus* cudaInterface);//,vm* v);
void GPUallocate(GPUStatus* cudaInterface,void** ptr, unsigned int size);
void GPUdeallocate(GPUStatus* cudaInterface,void* dPointer);
void GPUmemset(GPUStatus* cudaInterface,void* devicePointer, void* hostPointer, unsigned int size);
void GPUcopy_to_host(GPUStatus* cudaInterface,void * hostPointer, void * devicepointer, unsigned int size);
void GPUcopy_to_device(GPUStatus* cudaInterface,void* devicePointer, void* hostPointer, unsigned int size);

/** cuBLAS functions*/
void dotProduct(GPUStatus* cudaInterface, double* v1, double * v2, int size, double * out);
void GPUaxpy(GPUStatus* cudaInterface, int n, double* alpha, double * x, double * incx, double * y, double * incy);
void GPUcopy(GPUStatus* cudaInterface,int n, double * x, int incx, double *y, int incy);
void GPUScale(GPUStatus* cudaInterface, int n, const double *alpha, double *x, int incx);
void GPUgemm(GPUStatus* cudaInterface, int m, int n, int k, const double *alpha,\
             const double *A, int lda, const double *B, int ldb,\
             const double *beta, double *C, int ldc);

#endif