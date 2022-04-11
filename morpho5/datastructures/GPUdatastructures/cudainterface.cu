/* @file gpuinterface.c
* @author Danny Goldstein
* @breif a GPU interface for morpho desniged to work for openCL and CUDA
* This is the cuda interface
*/ 
#include "cudainterface.h"
#include "kernals.h"

#define GPUALLOCATE "cudaMalloc"
#define GPUDEALLOCATE "cudaFree"
#define GPUMEMSET "cudaMemset"
#define GPUMEMCPY "cudaMemCopy"
#define GPUSCALARADD "scalarAddition"
#define GPUDOT "cublasDotProduct"
#define GPUAXPY "cublasAXPY"
#define GPUSCALE "cublasScale"
#define GPUCOPY "cublasCopy"
#define GPUMULT "cublasGEMM"




void GPUStatusCheck(GPUStatus* cudaInterface, const char * errid){
    if (cudaInterface->cudaStatus != cudaSuccess) {
        // morpho_runtimeerror(cudaInterface->v, errid);
        printf("GPU error");
    }
    if (cudaInterface->cublasStatus != cudaSuccess) {
        // morpho_runtimeerror(cudaInterface->v, errid);
        printf("cublas error");
    }
}

void GPUsetup(GPUStatus* cudaInterface) { //, vm* v) {
    if (!cudaInterface->init) {
        cudaInterface->cudaStatus = cudaSetDevice(0);
        cudaInterface->cublasStatus = cublasCreate(&cudaInterface->cublasHandle);
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







void dotProduct(GPUStatus* cudaInterface, double* v1, double * v2, int size, double * out){
    cudaInterface->cublasStatus = cublasDdot(cudaInterface->cublasHandle, size, v1, 1, v2 , 1, out);
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
