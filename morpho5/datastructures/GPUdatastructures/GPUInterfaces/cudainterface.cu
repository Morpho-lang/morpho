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
GPUStatus myGPUstatus = {.cudaStatus = cudaSuccess, .cublasStatus = CUBLAS_STATUS_NOT_INITIALIZED, .cublasHandle = NULL, .init = NULL};
// p_gpu_area_integrand = gpu_area_integrand;
// p_gpu_area_gradient = gpu_area_gradient;
// p_gpu_volumeenclosed_integrand = gpu_volumeenclosed_integrand;
// p_gpu_volumeenclosed_gradient = gpu_volumeenclosed_gradient;
__device__ functional_integrand_gpu p_gpu_area_integrand = gpu_area_integrand;
__device__ functional_gradient_gpu p_gpu_area_gradient = gpu_area_gradient;
__device__ functional_integrand_gpu p_gpu_volumeenclosed_integrand = gpu_volumeenclosed_integrand;
__device__ functional_gradient_gpu p_gpu_volumeenclosed_gradient = gpu_volumeenclosed_gradient;


void GPUStatusCheck(GPUStatus* cudaInterface, const char * errid){
    if (cudaInterface->cudaStatus != cudaSuccess) {
        printf("GPU error: %s from %s\n",cudaGetErrorName(cudaInterface->cudaStatus),errid);
        exit(-1);
    }

    cudaInterface->cudaStatus = cudaDeviceSynchronize();

    if (cudaInterface->cudaStatus != cudaSuccess) {
        // morpho_runtimeerror(cudaInterface->v, errid);
        printf("GPU error: %s from %s\n",cudaGetErrorName(cudaInterface->cudaStatus),errid);
        exit(-1);
    }
    if (cudaInterface->cublasStatus != CUBLAS_STATUS_SUCCESS) {
        // morpho_runtimeerror(cudaInterface->v, errid);
        printf("cublas error: %d from %s\n",cudaInterface->cublasStatus,errid);
        exit(-1);
    }
    if (cudaInterface->cusolverStatus!=CUSOLVER_STATUS_SUCCESS){
        printf("cusolve error: %d from %s\n",cudaInterface->cusolverStatus,errid);
        exit(-1);
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
    functional_integrand_gpu h_area_integrand;
    functional_integrand_gpu h_volumeenclosed_integrand;
    cudaMemcpyFromSymbol(&h_area_integrand,p_gpu_area_integrand,sizeof(functional_integrand_gpu));
    cudaMemcpyFromSymbol(&h_volumeenclosed_integrand,p_gpu_volumeenclosed_integrand,sizeof(functional_integrand_gpu));
    functional_integrand_gpu h_integrands[] = {h_area_integrand, h_volumeenclosed_integrand};
    functional_integrand_gpu *d_integrands;
    cudaMalloc(&d_integrands,2*sizeof(functional_integrand_gpu));
    cudaMemcpy(d_integrands,h_integrands,2*sizeof(functional_integrand_gpu),cudaMemcpyHostToDevice);
    cudaInterface->d_integrands = d_integrands;


    functional_gradient_gpu h_area_gradient;
    functional_gradient_gpu h_volumeenclosed_gradient;
    cudaMemcpyFromSymbol(&h_area_gradient,p_gpu_area_gradient,sizeof(functional_gradient_gpu));
    cudaMemcpyFromSymbol(&h_volumeenclosed_gradient,p_gpu_volumeenclosed_gradient,sizeof(functional_gradient_gpu));
    functional_gradient_gpu h_gradient[] = {h_area_gradient, h_volumeenclosed_gradient};
    functional_gradient_gpu *d_gradient;
    cudaMalloc(&d_gradient,2*sizeof(functional_gradient_gpu));
    cudaMemcpy(d_gradient,h_gradient,2*sizeof(functional_gradient_gpu),cudaMemcpyHostToDevice);
    cudaInterface->d_gradients = d_gradient;
    



}
void GPUallocate(GPUStatus* cudaInterface,void** ptr, unsigned int size){
    // size_t free;
    // size_t free_new;
    // size_t total;
    // cuMemGetInfo(&free,&total);
    cudaInterface->cudaStatus = cudaMalloc((void**)ptr,(int) size);
    // cuMemGetInfo(&free_new,&total);
    // printf("requested %u bytes actually allocated %lu, currently %lu bytes free\n",size,free-free_new,free_new); 

    GPUStatusCheck(cudaInterface,GPUALLOCATE);
}

void GPUdeallocate(GPUStatus* cudaInterface,void* dPointer) {
    // size_t free;
    // size_t total;
    // cuMemGetInfo(&free,&total);

    if (dPointer) {
        cudaInterface->cudaStatus = cudaFree(dPointer);
    }
    GPUStatusCheck(cudaInterface,GPUDEALLOCATE);
    // size_t free_new;
    // cuMemGetInfo(&free_new,&total);
    // printf("Deallocacated %lu bytes\n",free_new-free);

}

void GPUreallocate(GPUStatus* cudaInterface,void** ptr, unsigned int newsize,unsigned int oldsize){
    void* newptr = NULL;
    // allocate a new space
    GPUallocate(cudaInterface,&newptr,newsize);
    GPUStatusCheck(cudaInterface,GPUALLOCATE);
    if (*ptr) {
        // copy things over room here for improvment
        unsigned int size = (newsize<oldsize? newsize : oldsize); // take the smaller of the two
        cudaMemcpy(newptr,*ptr,size,cudaMemcpyDeviceToDevice);
        // delete the old one
        GPUStatusCheck(cudaInterface,GPUMEMCPY);
        GPUdeallocate(cudaInterface,*ptr);

    }
    *ptr = newptr;
}

void GPUmemset(GPUStatus* cudaInterface,void* devicePointer, int bytepattern, unsigned int size){
    cudaInterface->cudaStatus = cudaMemset(devicePointer,bytepattern,size);
    GPUStatusCheck(cudaInterface,GPUMEMSET);

}
void GPUcopy_to_host(GPUStatus* cudaInterface,void * hostPointer, void * devicePointer, size_t offset, unsigned int size) {
    // we cast to char* because a charater is size one
    // we expect offset to be the size in bytes
    void* d_pos = (void*)((char*)devicePointer+offset);
    cudaInterface->cudaStatus = cudaMemcpy(hostPointer, d_pos, size, cudaMemcpyDeviceToHost);
    GPUStatusCheck(cudaInterface,GPUMEMCPY);

}
void GPUcopy_to_device(GPUStatus* cudaInterface,void* devicePointer, size_t offset, void* hostPointer, unsigned int size) {
    void* d_pos = (void*)((char*)devicePointer+offset);
    cudaInterface->cudaStatus = cudaMemcpy(d_pos, hostPointer, size, cudaMemcpyHostToDevice);
    GPUStatusCheck(cudaInterface,GPUMEMCPY);

}
void GPUcopy_device_to_device(GPUStatus* cudaInterface,void* devicePointersrc, void* devicePointerdest, unsigned int size) {
    cudaInterface->cudaStatus = cudaMemcpy(devicePointerdest, devicePointersrc, size, cudaMemcpyDeviceToDevice);
    GPUStatusCheck(cudaInterface,GPUMEMCPY);

}
void GPUcopy_symbol(void* dst, void* symbol, size_t size){
    cudaMemcpyFromSymbol(dst,symbol,size);
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

int GPUSparseTranspose(GPUStatus* cudaInterface, objectgpusparse_light *in,objectgpusparse_light *out){
    printf("Not Implmeneted Yet");
    return 1;
}

void dotProduct(GPUStatus* cudaInterface, double* v1, double * v2, int size, double * out){
    cudaInterface->cublasStatus = cublasDdot(cudaInterface->cublasHandle, size, v1, 1, v2 , 1, out);
    GPUStatusCheck(cudaInterface,GPUDOT);
}

void GPUdot(GPUStatus* cudaInterface,int size, double *x,int incx, double *y, int incy,double* out){
    cudaInterface->cublasStatus = cublasDdot(cudaInterface->cublasHandle, size, x, incx, y , incy, out);
    GPUStatusCheck(cudaInterface,GPUDOT);
}
void GPUSum(GPUStatus* cudaInterface,int size, double *x, int inc, double *out) {
    double *GPUone;
    double one = 1.0;
    GPUallocate(cudaInterface,(void **)&GPUone,sizeof(double));
    GPUcopy_to_device(cudaInterface,GPUone,0,&one,sizeof(double));
    GPUdot(cudaInterface,size, x ,inc, GPUone, 0, out);
    GPUdeallocate(cudaInterface,GPUone);
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

/************************************************
*               Functional interface            *
************************************************/
void GPUcall_functional(GPUStatus* cudaInterface,double* verts, int dim, objectgpusparse_light* conn,\
                      int grade, int nelements,int integrandNo, double* out){
    int nblocks = ceil(nelements/32.0);

    functionalIntegrandEval<<<nblocks,32>>>(verts,dim,conn,nelements,integrandNo,cudaInterface->d_integrands,out);
    GPUStatusCheck(cudaInterface,"Call Functional integrand");

}
void GPUcall_functionalgrad(GPUStatus* cudaInterface,double* verts, int dim, objectgpusparse_light* conn,\
                            int grade, int nelements,int gradientNo, double* out) {
    int nblocks = ceil(nelements/32.0);
    functionalGradEval<<<nblocks,32>>>(verts,dim,conn,nelements,gradientNo,cudaInterface->d_gradients,out);
        GPUStatusCheck(cudaInterface,"Call Functional gradient");

}

  
/***************************************
 * Sparse Arithmatic                   *
 * ************************************/
// cusparseSpMatDescr_t makecuSparse(objectgpusparse *in){
//     cusparseSpMatDescr_t out;
//     cusparseCreateCsc(&out,in->nrows,in->ncols,in->nentries, in->cptr, in->rix, in->values,
//                   CUSPARSE_INDEX_32I,
//                   CUSPARSE_INDEX_32I,
//                   CUSPARSE_INDEX_32I,
//                   CUDA_R_64F);
//     return out;


// }

// cusparseDnMatDescr_t makecuDence(objectgpusparse *in) {
//     cusparseSpMatDescr_t incusparse = makecuSparse(in);
//     cusparseDnMatDescr_t out;
//     size_t bufferSize;
//     cusparseSparseToDense_bufferSize(in->status->cusparseHandle, incusparse,out,CUSPARSE_SPARSETODENSE_ALG_DEFAULT,&bufferSize);
//     void* buffer;
//     GPUallocate(in->status,&buffer,bufferSize);

//     cusparseSparseToDense(in->status->cusparseHandle, incusparse,out,CUSPARSE_SPARSETODENSE_ALG_DEFAULT,buffer);
//     return out;


// }
// cusparseSpMatDescr_t convertcuDenseToSparse(GPUStatus *status, cusparseDnMatDescr_t in) {
    
//     cusparseSpMatDescr_t out;
//     size_t bufferSize;
//     status->cusparseStatus = cusparseDenseToSparse_bufferSize(status->cusparseHandle,in , out,CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,&bufferSize);

//     void* buffer = NULL;
//     GPUallocate(status,&buffer,bufferSize);

//     status->cusparseStatus = cusparseDenseToSparse_analysis(status->cusparseHandle, in, out, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,buffer);

//     status->cusparseStatus = cusparseDenseToSparse_convert(status->cusparseHandle, in, out, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,buffer);
//     GPUdeallocate(status,buffer);
// }

// void convertcuSparseToSparse(cusparseSpMatDescr_t in, objectgpusparse *out) {
//     int64_t nrows;
//     int64_t ncols;
//     int64_t nvalues;
//     double *values;
//     cusparseSpMatGetSize(in,&nrows, &ncols, &nvalues);
//     cusparseSpMatGetValues(in,&values);

                     

    
// }


}
#undef TILE_DIM
#undef BLOCK_COLS