/* @file gpuinterface.c
* @author Danny Goldstein
* @breif a GPU interface for morpho desniged to work for openCL and CUDA
* This is the cuda interface
*/ 
#ifndef openCLInterface_h
#define openCLInterface_h

#define CL_TARGET_OPENCL_VERSION 220
#include <build.h>
#include "gpuinterface.h"
#include <stdio.h>
#include <stdbool.h>
#include "CL/cl.h"
#include <clblast_c.h>
#include <stdio.h> 
#include <math.h>

#define NUMBER_OF_PROGRAMS 6
#define MORPHO_OPENCL_KERNALS {"ScalarAddition", "transposeDiagonal","functionalIntegrandEval","functionalIntegrandEvalConn","functionalGradEval","functionalGradEvalConn"}
typedef enum {GPU_MATRIXSCALARADDITION, GPU_MATRIXTRANSPOSE,GPU_INTEGRANDEVAL,GPU_INTEGRANDEVALCONN,GPU_GRADEVAL,GPU_GRADEVALCONN} kernalNum;


#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int openCLStatus;
	cl_device_id device_id;             		// compute device id 
    cl_context context;                 		// compute context
    cl_command_queue commands;          		// compute command queue
    cl_program program;// compute program
	cl_kernel kernel_list[NUMBER_OF_PROGRAMS];	// compute kernel
    CLBlastStatusCode CLBlastStatus;
    //vm * v;
    bool init;
} GPUStatus;

extern GPUStatus myGPUstatus;

void GPUsetup(GPUStatus* openCLInterface);//,vm* v);
void GPUallocate(GPUStatus* openCLInterface,void** ptr, unsigned int size);
void GPUdeallocate(GPUStatus* openCLInterface,void* dPointer);
void GPUmemset(GPUStatus* openCLInterface,void* devicePointer, int bytepattern, unsigned int size);
void GPUcopy_to_host(GPUStatus* openCLInterface,void * hostPointer, void * devicepointer,size_t offset, unsigned int size);
void GPUcopy_to_device(GPUStatus* openCLInterface,void* devicePointer, size_t offset, void* hostPointer, unsigned int size);
void GPUcopy_device_to_device(GPUStatus* cudaInterface,void* devicePointersrc, void* devicePointerdest, unsigned int size);

void GPUScalarAddition(GPUStatus* openCLInterface, double* Matrix, double scalar, double *out, int size);
void GPUTranspose(GPUStatus* openCLInterface, double* in, double* out, int ncols, int nrows);
int GPUSparseTranspose(GPUStatus* cudaInterface, objectgpusparse_light *in,objectgpusparse_light *out);

/** cuBLAS functions*/
void dotProduct(GPUStatus* openCLInterface, double* v1, double * v2, int size, double * out);
void GPUdot(GPUStatus* openCLInterface,int size, double *x,int incx, double *y, int incy,double* out);
void GPUSum(GPUStatus* openCLInterface,int size, double *x, int inc, double *out); 
void GPUaxpy(GPUStatus* openCLInterface, int n, double* alpha, double * x, int incx, double * y, int incy);
void GPUcopy(GPUStatus* openCLInterface,int n, double * x, int incx, double *y, int incy);
void GPUScale(GPUStatus* openCLInterface, int n, const double *alpha, double *x, int incx);
void GPUgemm(GPUStatus* openCLInterface, int m, int n, int k, const double *alpha,\
             const double *A, int lda, const double *B, int ldb,\
             const double *beta, double *C, int ldc);
void GPUnrm2(GPUStatus* openCLInterface,int n, const double *x, int incx, double *result);

/** cuSolve function*/
void GPUgesv(GPUStatus* openCLInterface, int n, int nrhs, double* dA, int ldda, int* dipiv, double* dB,\
 int lddb, double* dX, int lddx, int * niter, int *dinfo);
void GPUgetrf(GPUStatus* openCLInterface,int nrows, int ncols,double* elements,int lda,int* pivot,int* devInfo);
void GPUgetrs(GPUStatus* openCLInterface, int nrows, int ncolsB, double * A, int lda, int *devIpiv, double* B, int ldb, int *devInfo);
void GPUcall_functional(GPUStatus* cudaInterface,double* verts, int dim, objectgpusparse_light* conn, int grade, int nelements,int integrandNo, double* out);
void GPUcall_functionalgrad(GPUStatus* cudaInterface,double* verts, int dim, objectgpusparse_light* conn, int grade, int nelements,int gradientNo, double* out);

const char* getErrorString(cl_int openCLStatus);
const char* getCLBlastErrorString(CLBlastStatusCode code);
#ifdef __cplusplus
}
#endif

#endif