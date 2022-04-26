/* @file gpuinterface.c
* @author Danny Goldstein
* @breif a GPU interface for morpho desniged to work for openCL and CUDA
* This is the cuda interface
*/ 
#include "cudainterface.h"

#define TILE_DIM 32
#define BLOCK_COLS 8

#include "kernals.h"

#define GPUALLOCATE "cudaMalloc"
#define GPUDEALLOCATE "cudaFree"
#define GPUMEMSET "cudaMemset"
#define GPUMEMCPY "cudaMemCopy"
#define GPUSCALARADD "Scalar Addition"
#define GPUTRANSPOSE "Transpose"
#define GPUDOT "cublasDotProduct"
#define GPUAXPY "cublasAXPY"
#define GPUSCALE "cublasScale"
#define GPUCOPY "cublasCopy"
#define GPUNRM2 "MatrixNorm"
#define GPUMULT "cublasGEMM"
#define GPUGETRF "cusolveLU"
#define GPUGETRS "cusolveAXB"


extern "C" {

void GPUStatusCheck(GPUStatus* cudaInterface, const char * errid){
    cudaInterface->cudaStatus = cudaDeviceSynchronize();

    if (cudaInterface->cudaStatus != cudaSuccess) {
        // morpho_runtimeerror(cudaInterface->v, errid);
        printf("GPU error: %s from %s\n",cudaGetErrorName(cudaInterface->cudaStatus),errid);
    }
    if (cudaInterface->cublasStatus != CUBLAS_STATUS_SUCCESS) {
        // morpho_runtimeerror(cudaInterface->v, errid);
        printf("cublas error: %d from %s\n",cudaInterface->cublasStatus,errid);
    }
    if (cudaInterface->cusolverStatus!=CUSOLVER_STATUS_SUCCESS){
        printf("cusolve error: %d from %s\n",cudaInterface->cusolverStatus,errid);
    }
}

void GPUsetup(GPUStatus* cudaInterface) { //, vm* v) {
    if (!cudaInterface->init) {
        cudaInterface->cudaStatus = cudaSetDevice(0);
        cudaInterface->cublasStatus = cublasCreate(&cudaInterface->cublasHandle);
        cudaInterface->cusolverStatus = cusolverDnCreate(&cudaInterface -> cusolverHandle);
        //cudaInterface->v = NULL;
        cudaInterface->init = true;
    }
}
void GPUallocate(GPUStatus* cudaInterface,void** ptr, unsigned int size){
    cudaInterface->cudaStatus = cudaMalloc((void**)ptr,(int) size);
    GPUStatusCheck(cudaInterface,GPUALLOCATE);
}

void GPUdeallocate(GPUStatus* cudaInterface,void* dPointer) {
    if (dPointer) {
        cudaInterface->cudaStatus = cudaFree(dPointer);
    }
    GPUStatusCheck(cudaInterface,GPUDEALLOCATE);
}

void GPUmemset(GPUStatus* cudaInterface,void* devicePointer, int bytepattern, unsigned int size){
    cudaInterface->cudaStatus = cudaMemset(devicePointer,bytepattern,size);
    GPUStatusCheck(cudaInterface,GPUMEMSET);

}
void GPUcopy_to_host(GPUStatus* cudaInterface,void * hostPointer, void * devicepointer, unsigned int size) {
    cudaInterface->cudaStatus = cudaMemcpy(hostPointer, devicepointer, size, cudaMemcpyDeviceToHost);
    GPUStatusCheck(cudaInterface,GPUMEMCPY);

}
void GPUcopy_to_device(GPUStatus* cudaInterface,void* devicePointer, void* hostPointer, unsigned int size) {
    cudaInterface->cudaStatus = cudaMemcpy(devicePointer, hostPointer, size, cudaMemcpyHostToDevice);
    GPUStatusCheck(cudaInterface,GPUMEMCPY);

}

void GPUScalarAddition(GPUStatus* cudaInterface, double* Matrix, double scalar, double *out, int size){
    unsigned int blockSize = 64;
    unsigned int numberOfBlocks = ceil(size / (float) blockSize);

    ScalarAddition<<<numberOfBlocks, blockSize>>>(Matrix,scalar,out,size);
    GPUStatusCheck(cudaInterface,GPUSCALARADD);
}

void GPUTranspose(GPUStatus* cudaInterface, double* in, double* out, int ncols, int nrows) {

    dim3 grid(ceil(ncols/(double)TILE_DIM), ceil(nrows/(double)TILE_DIM)), threads(BLOCK_COLS,TILE_DIM);
    transposeDiagonal<<<grid,threads>>>(out, in, ncols, nrows);
    GPUStatusCheck(cudaInterface,GPUTRANSPOSE);

}







void dotProduct(GPUStatus* cudaInterface, double* v1, double * v2, int size, double * out){
    cudaInterface->cublasStatus = cublasDdot(cudaInterface->cublasHandle, size, v1, 1, v2 , 1, out);
    GPUStatusCheck(cudaInterface,GPUDOT);
}

void GPUdot(GPUStatus* cudaInterface,int size, double *x,int incx, double *y, int incy,double* out){
    cudaInterface->cublasStatus = cublasDdot(cudaInterface->cublasHandle, size, x, incx, y , incy, out);
    GPUStatusCheck(cudaInterface,GPUDOT);
}

/**
 * @brief axpy perfoms y[j] = alpha * x[k] + y[j]
 * with ranges i = 1...n
 * k = (i-1)*incx
 * j = (i-1)*incy
 * @param n number of elements to perfom operation on
 * @param alpha scalar multiple
 * @param x vector
 * @param incx incremnt for x
 * @param y vector
 * @param incy increment for y
 */
void GPUaxpy(GPUStatus* cudaInterface, int n, double* alpha, double * x, int incx, double * y, int incy){
    cudaInterface->cublasStatus =  cublasDaxpy(cudaInterface->cublasHandle, n,alpha, x, incx, y, incy);
    GPUStatusCheck(cudaInterface,GPUAXPY);
}

/**
 * @brief copy perfoms y[j] = x[k] with ranges i = 1...n
 * k = (i-1)*incx
 * j = (i-1)*incy
 * @param n number of elements to perfom operation on
 * @param x source data
 * @param incx increment for x
 * @param y target location
 * @param incy increment for y
 */
void GPUcopy(GPUStatus* cudaInterface,int n, double * x, int incx, double *y, int incy){
    cudaInterface->cublasStatus = cublasDcopy(cudaInterface->cublasHandle, n, x, incx, y, incy);
    GPUStatusCheck(cudaInterface,GPUCOPY);
}

/**
 * @brief Scales a marix by a value alpha in place
 *  performs x[j] = alpha x[j] for i = 1,...,n and j = (i-1)*incx
 * @param n number of elements to scale
 * @param alpha number to scale the elements by
 * @param x poitner to start of matrix
 * @param incx space between elements in x to increment
 */
void GPUScale(GPUStatus* cudaInterface, int n, const double *alpha, double *x, int incx) {
    cudaInterface->cublasStatus = cublasDscal(cudaInterface->cublasHandle, n, alpha, x, incx);
    GPUStatusCheck(cudaInterface,GPUSCALE);


}
void GPUnrm2(GPUStatus* cudaInterface,int n, const double *x, int incx, double *result) {
    cudaInterface->cublasStatus = cublasDnrm2(cudaInterface->cublasHandle, n,  x, incx, result);
    GPUStatusCheck(cudaInterface,GPUNRM2);

}


/**
 * @brief Performs C = alpha A*B + beta C
 * With the option to transpose A or B before multiplying
 * A 
 * B a
 * C 
 * @param m number of rows in A
 * @param n number of columns of B
 * @param k number of columns of A and rows of B
 * @param alpha scalar used in multiplication
 * @param A array of dimensions lda x k
 * @param lda leading dimension of two-dimensional array used to store the matrix A. 
 * @param B rray of dimension ldb x n
 * @param beta <type> scalar used for multiplication. If beta==0, C does not have to be a valid input. 
 * @param ldb leading dimension of two-dimensional array used to store matrix B. 
 * @param C array of dimensions ldc x n
 * @param ldc leading dimension of a two-dimensional array used to store the matrix C. 
 */
void GPUgemm(GPUStatus* cudaInterface,\
                           int m, int n, int k,\
                           const double          *alpha,\
                           const double          *A, int lda,\
                           const double          *B, int ldb,\
                           const double          *beta,\
                           double          *C, int ldc) {
    cudaInterface->cublasStatus = cublasDgemm(cudaInterface->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,\
                                                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    GPUStatusCheck(cudaInterface,GPUMULT);
}

/************************************************
*       LAPACK-like cuSolver interface          *
************************************************/
// /**
 
// handle      host 	input 	Handle to the cusolverDN library context.
// n           host 	input 	Number of rows and columns of square matrix A. Should be non-negative.
// nrhs 	    host 	input 	Number of right hand sides to solve. Should be non-negative.
// dA 	        device 	None 	Matrix A with size n-by-n. Can be NULL.
// ldda 	    host 	input 	Leading dimension of two-dimensional array used to store matrix A. lda >= n.
// dipiv 	    device 	None 	Pivoting sequence. Not used and can be NULL.
// dB 	        device 	None 	Set of right hand sides B of size n-by-nrhs. Can be NULL.
// lddb 	    host 	input 	Leading dimension of two-dimensional array used to store matrix of right hand sides B. ldb >= n.
// dX 	        device 	None 	Set of soultion vectors X of size n-by-nrhs. Can be NULL.
// lddx 	    host 	input 	Leading dimension of two-dimensional array used to store matrix of solution vectors X. ldx >= n.
// dwork 	    device 	none 	Pointer to device workspace. Not used and can be NULL.
// lwork_bytes host 	output 	Pointer to a variable where required size of temporary workspace in bytes will be stored. Can't be NULL. 
// */
// void GPUgesv(GPUStatus* cudaInterface, int n, int nrhs, double* dA, int ldda, double* dB,\
//  int lddb, double* dX, int lddx, int * niter, int *dinfo) {
//     size_t lwork_bytes;
//     int * dipiv = NULL;
//     //cusolverDnIRSParams_t params = 
//     //cudaInterface->cusolverStatus = cusolverDnIRSXgesv_bufferSize( cudaInterface->cusolverHandle,\
//             params, n, nrhs, &lwork_bytes);

//     GPUallocate(cudaInterface,(void**)&dipiv,sizeof(int)*n);

//     cudaInterface->cusolverStatus = cusolverDnDDgesv_bufferSize(\
//     cudaInterface->cusolverHandle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, NULL, &lwork_bytes);

//     void * dWorkspace = NULL;
//     //cusolverDnIRSInfos_t info;
//     GPUallocate(cudaInterface,(void**)&dWorkspace,lwork_bytes);
// /*cudaInterface->cusolverStatus cusolverDnIRSXgesv( cudaInterface->cusolverHandle,\
//         params,
//         info,
//         int                         n,
//         int                         nrhs,
//         void                    *   dA,
//         int                         ldda,
//         void                    *   dB,
//         int                         lddb,
//         void                    *   dX,
//         int                         lddx,
//         void                    *   dWorkspace,
//         size_t                      lwork_bytes,
//         int                     *   dinfo);
// */
// //USE GETRS AND GETRF
//     cudaInterface->cusolverStatus = cusolverDnDDgesv(cudaInterface->cusolverHandle,\
//                         n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,\
//                         lwork_bytes, niter, dinfo);

//     GPUdeallocate(cudaInterface,dWorkspace);
//     GPUdeallocate(cudaInterface,dipiv);
// }

void GPUgetrf(GPUStatus* cudaInterface,int nrows, int ncols,double* elements,int lda,int* pivot,int* devInfo){
    int Lwork;
    cudaInterface->cusolverStatus = cusolverDnDgetrf_bufferSize(cudaInterface->cusolverHandle, nrows, ncols, elements, lda, &Lwork);
    double *Workspace;
    GPUallocate(cudaInterface, (void**)&Workspace,sizeof(double)*Lwork);
    cudaInterface->cusolverStatus = cusolverDnDgetrf(cudaInterface->cusolverHandle, nrows, ncols, elements, lda,\
           Workspace, pivot, devInfo );

    GPUStatusCheck(cudaInterface,GPUGETRF);
    GPUdeallocate(cudaInterface,Workspace);
}

void GPUgetrs(GPUStatus* cudaInterface, int nrows, int ncolsB, double * A, int lda, int *devIpiv, double* B, int ldb, int *devInfo) {

    cudaInterface->cusolverStatus = cusolverDnDgetrs(cudaInterface->cusolverHandle,\
           CUBLAS_OP_N, nrows, ncolsB, A, lda, devIpiv, B, ldb, devInfo);
    GPUStatusCheck(cudaInterface,GPUGETRS);
}
}
#undef TILE_DIM
#undef BLOCK_COLS