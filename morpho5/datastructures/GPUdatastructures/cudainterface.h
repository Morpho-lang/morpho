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
typedef struct {
    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;
	cublasHandle_t cublasHandle;
    cusolverStatus_t cusolverStatus;
    cusolverDnHandle_t cusolverHandle;
    //vm * v;
    bool init;
} GPUStatus;
void GPUsetup(GPUStatus* cudaInterface);//,vm* v);
void GPUallocate(GPUStatus* cudaInterface,void** ptr, unsigned int size);
void GPUdeallocate(GPUStatus* cudaInterface,void* dPointer);
void GPUmemset(GPUStatus* cudaInterface,void* devicePointer, int bytepattern, unsigned int size);
void GPUcopy_to_host(GPUStatus* cudaInterface,void * hostPointer, void * devicepointer, unsigned int size);
void GPUcopy_to_device(GPUStatus* cudaInterface,void* devicePointer, void* hostPointer, unsigned int size);

void GPUScalarAddition(GPUStatus* cudaInterface, double* Matrix, double scalar, double *out, int size);
void GPUTranspose(GPUStatus* cudaInterface, double* in, double* out, int ncols, int nrows);

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

#ifdef __cplusplus
}
#endif

#endif