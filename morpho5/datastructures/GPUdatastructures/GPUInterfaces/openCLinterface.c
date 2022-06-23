/* @file gpuinterface.c
* @author Danny Goldstein
* @breif a GPU interface for morpho desniged to work for openCL and openCL
* This is the openCL interface
*/ 
#include "openCLinterface.h"

#define TILE_DIM 32
#define BLOCK_COLS 8
#define MAX_SOURCE_SIZE (0x100000)



#define GPUALLOCATE "openCLMalloc"
#define GPUDEALLOCATE "openCLFree"
#define GPUMEMSET "openCLMemset"
#define GPUMEMCPY "openCLMemCopy"
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

GPUStatus myGPUstatus = {.openCLStatus = 0, .device_id = NULL, .context = NULL, .commands = NULL,.program=NULL,.kernel_list=NULL,.CLBlastStatus = 0, .init=false};

void GPUStatusCheck(GPUStatus* openCLInterface, const char * caller){
    if (openCLInterface->openCLStatus != CL_SUCCESS) {
		printf("%s from %s\n",getErrorString(openCLInterface->openCLStatus),caller);
    }
    if (openCLInterface->CLBlastStatus != CLBlastSuccess){
		printf("%s from %s\n",getCLBlastErrorString(openCLInterface->CLBlastStatus),caller);
    }
    // printf("Just did %s, commands are here %p, context is here %p\n",caller,openCLInterface->commands,openCLInterface->context);

}

void GPUsetup(GPUStatus* openCLInterface) { //, vm* v) {
    if (openCLInterface->init) return;
	cl_uint num_entries = 2;
	cl_platform_id *platforms;
	cl_uint *num_platforms;

	char retvalue[50];
    char *programNames[NUMBER_OF_PROGRAMS] = MORPHO_OPENCL_KERNALS;



	size_t kernel_code_size;
	cl_uint ret_num_platforms;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	char buildLog[1024];
	int i;
	cl_device_id OPENCL_DEVICE_ID;

	openCLInterface->openCLStatus = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	GPUStatusCheck(openCLInterface,"clGetPlatformIDs");
	openCLInterface->openCLStatus = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &openCLInterface->device_id, &ret_num_devices);
	GPUStatusCheck(openCLInterface,"clGetDeviceIDs");
    
  
	// Create a compute context 
	//
	openCLInterface->context = clCreateContext(0, 1, &openCLInterface->device_id, NULL, NULL, &openCLInterface->openCLStatus);
	GPUStatusCheck(openCLInterface,"create context");
	if (!openCLInterface->context)
	{
		printf("openCLStatus: Failed to create a compute context!\n");
	}

	// Create a command que
	//
	openCLInterface->commands = clCreateCommandQueueWithProperties(openCLInterface->context, openCLInterface->device_id, 0, &openCLInterface->openCLStatus);
	if (!openCLInterface->commands)
	{
		printf("openCLStatus: Failed to create a command que!\n");
	}
	GPUStatusCheck(openCLInterface,"create command queue");


    FILE *fp = fopen(MORPHO_KERNAL_DIRECTORY, "r");
    if (!fp) {
	fprintf(stderr, "Failed to open kernel file '%s'\n", MORPHO_KERNAL_DIRECTORY);
    }
	
    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size= fread( source_str, 1, MAX_SOURCE_SIZE, fp);


	openCLInterface->program = clCreateProgramWithSource(openCLInterface->context,1,(const char**)&source_str,&source_size,&openCLInterface->openCLStatus);
	if (!openCLInterface->program||openCLInterface->openCLStatus != CL_SUCCESS){
		printf("failed to create program!\n");
		GPUStatusCheck(openCLInterface,"create program");
	}
	openCLInterface->openCLStatus = clBuildProgram(openCLInterface->program, 0, NULL, NULL, NULL, NULL);
	if (openCLInterface->openCLStatus != CL_SUCCESS)
	{
		size_t len;
		char buffer[20480];

		printf("openCLStatus: Failed to build program executable! %s\n",getErrorString(openCLInterface->openCLStatus));

		clGetProgramBuildInfo(openCLInterface->program, openCLInterface->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}



	// register compile and register the list of kernals
	for (int i = 0; i<NUMBER_OF_PROGRAMS; i++){	
		openCLInterface->kernel_list[i] = clCreateKernel(openCLInterface->program,programNames[i], &openCLInterface->openCLStatus);
		if(!openCLInterface->kernel_list[i] || openCLInterface->openCLStatus != CL_SUCCESS){
			printf("openCLStatus: Failed to create compute kernal %s!\n",programNames[i]);
		}
	}
    openCLInterface->init = true;
    return;

}
void GPUallocate(GPUStatus* openCLInterface,void** ptr, unsigned int size){
	cl_mem var; // need to malloc this first
    *ptr = malloc(sizeof(cl_mem));
    *(cl_mem*)*ptr = clCreateBuffer(openCLInterface->context,CL_MEM_READ_WRITE,size,NULL,&openCLInterface->openCLStatus);
    GPUStatusCheck(openCLInterface,GPUALLOCATE);
}

void GPUdeallocate(GPUStatus* openCLInterface,void* dPointer) {
    if (dPointer) {
        openCLInterface->openCLStatus = clReleaseMemObject(*(cl_mem*)dPointer);
        free(dPointer);
    }
    GPUStatusCheck(openCLInterface,GPUDEALLOCATE);
}

void GPUmemset(GPUStatus* openCLInterface,void* devicePointer, int bytepattern, unsigned int size){
    // convert to double
    double pattern = bytepattern;
    openCLInterface->openCLStatus = clEnqueueFillBuffer(openCLInterface->commands,*(cl_mem*)devicePointer,(void*)&pattern,sizeof(double),0,size,0,NULL,NULL);
    GPUStatusCheck(openCLInterface,GPUMEMSET);

}
void GPUcopy_to_host(GPUStatus* openCLInterface,void * hostPointer, void * devicepointer, size_t offset, unsigned int size) {
    openCLInterface->openCLStatus = clEnqueueReadBuffer(openCLInterface->commands,*(cl_mem*) devicepointer,CL_TRUE,offset,size,hostPointer,0,NULL,NULL);
    GPUStatusCheck(openCLInterface,GPUMEMCPY);

}
void GPUcopy_to_device(GPUStatus* openCLInterface,void* devicePointer, size_t offset, void* hostPointer, unsigned int size) {
    openCLInterface->openCLStatus = clEnqueueWriteBuffer(openCLInterface->commands, *(cl_mem*)devicePointer, CL_TRUE,offset,size,hostPointer,0,NULL,NULL);
    GPUStatusCheck(openCLInterface,GPUMEMCPY);

}
void GPUcopy_device_to_device(GPUStatus* openCLInterface,void* devicePointersrc, void* devicePointerdest, unsigned int size) {
 openCLInterface->openCLStatus = clEnqueueCopyBuffer(openCLInterface->commands, *(cl_mem*) devicePointersrc, *(cl_mem*) devicePointerdest,\
  	0, 0, size, 0, NULL, NULL);
    GPUStatusCheck(openCLInterface,GPUMEMCPY);

}
// typedef struct _cl_buffer_region {
//     size_t origin;
//     size_t size;
// } cl_buffer_region;

// void GPUAdvanceBuffer(GPUStatus* openCLInterface,void *buffer,int size,int advance,void **out) {
//     cl_buffer_region info ={.origin = advance, .size = size};

//     (cl_mem*)out = clCreateSubBuffer ( 	cl_mem buffer,
//   	cl_mem_flags flags,
//   	cl_buffer_create_type buffer_create_type,
//   	const void *buffer_create_info,
//   	cl_int *errcode_ret)
// }
 



void GPUScalarAddition(GPUStatus* openCLInterface, double* Matrix, double scalar, double *out, int size){
    // unsigned int blockSize = 64;
    // unsigned int numberOfBlocks = ceil(size / (float) blockSize);

    openCLInterface->openCLStatus =  clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXSCALARADDITION], 0,sizeof(cl_mem),(cl_mem*)(Matrix));
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXSCALARADDITION], 1,sizeof(double),&scalar);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXSCALARADDITION], 2,sizeof(cl_mem),(cl_mem*)(out));
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXSCALARADDITION], 3,sizeof(unsigned int),&size);
    GPUStatusCheck(openCLInterface,GPUSCALARADD);

    // size_t local = blockSize;
	size_t global = size;
	// if (size<blockSize){
		// global = blockSize;
	// } else {global = size;}

	openCLInterface->openCLStatus = clEnqueueNDRangeKernel(openCLInterface->commands, openCLInterface->kernel_list[GPU_MATRIXSCALARADDITION], 1, NULL, &global, NULL, 0, NULL, NULL);
	clFinish(openCLInterface->commands);
    GPUStatusCheck(openCLInterface,GPUSCALARADD);
}

void GPUTranspose(GPUStatus* openCLInterface, double* in, double* out, int ncols, int nrows) {
    unsigned int blockSize = 64;
    unsigned int numberOfBlocks = ceil(ncols / (float) blockSize);

    openCLInterface->openCLStatus =  clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXTRANSPOSE], 0,sizeof(cl_mem),(cl_mem*)(out));
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXTRANSPOSE], 1,sizeof(cl_mem),(cl_mem*)in);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXTRANSPOSE], 2,sizeof(int),&ncols);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXTRANSPOSE], 3,sizeof(int),&nrows);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXTRANSPOSE], 4,sizeof(double)*TILE_DIM*(TILE_DIM+1),NULL);
    GPUStatusCheck(openCLInterface,GPUTRANSPOSE);

    size_t local[2] = {BLOCK_COLS,TILE_DIM};
	size_t global[2] = {BLOCK_COLS*ceil(ncols/(double)TILE_DIM), TILE_DIM*ceil(ncols/(double)TILE_DIM)};

	openCLInterface->openCLStatus = clEnqueueNDRangeKernel(openCLInterface->commands, openCLInterface->kernel_list[GPU_MATRIXTRANSPOSE], 2, NULL, global, local, 0, NULL, NULL);
	clFinish(openCLInterface->commands);
    //dim3 grid(ceil(ncols/(double)TILE_DIM), ceil(nrows/(double)TILE_DIM)), threads(BLOCK_COLS,TILE_DIM);
    //transposeDiagonal<<<grid,threads>>>(out, in, ncols, nrows);
    GPUStatusCheck(openCLInterface,GPUTRANSPOSE);

}

int GPUSparseTranspose(GPUStatus* cudaInterface, objectgpusparse_light *in,objectgpusparse_light *out){
    printf("Sparse Transpose not implemented");
}



void dotProduct(GPUStatus* openCLInterface, double* v1, double * v2, int size, double * out){
    void* dot;
    GPUallocate(openCLInterface,&dot,sizeof(double)*size);

    openCLInterface->CLBlastStatus = CLBlastDdot(size, *(cl_mem*)dot, 0, *(cl_mem*)v1, 0, 1,*(cl_mem*)v2, 0, 1,&openCLInterface->commands,NULL);

    GPUStatusCheck(openCLInterface,GPUDOT);
    GPUcopy_to_host(openCLInterface,out,dot,0,sizeof(double));
    GPUdeallocate(openCLInterface,dot);
}
void GPUdot(GPUStatus* openCLInterface,int size, double *x,int incx, double *y, int incy,double* out){
    void* dot;
    GPUallocate(openCLInterface,&dot,sizeof(double)*size);
    openCLInterface->CLBlastStatus = CLBlastDdot(size, *(cl_mem*)dot, 0, *(cl_mem*)x, 0, incx,*(cl_mem*)y, 0, incy,&openCLInterface->commands,NULL);
    GPUStatusCheck(openCLInterface,GPUDOT);
    GPUcopy_to_host(openCLInterface,out,dot,0,sizeof(double));
    GPUdeallocate(openCLInterface,dot);

}
void GPUSum(GPUStatus* openCLInterface,int size, double *x, int inc, double *out) {
    void* ones;
    GPUallocate(openCLInterface,&ones,sizeof(double)*size);
    GPUmemset(openCLInterface,ones,1,size*sizeof(double));
    GPUdot(openCLInterface,size,(double*)ones,1,x,inc,out);
    GPUdeallocate(openCLInterface,ones);
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
void GPUaxpy(GPUStatus* openCLInterface, int n, double* alpha, double * x, int incx, double * y, int incy){
    openCLInterface->CLBlastStatus = CLBlastDaxpy(n, *alpha, *(cl_mem*) x, 0, 1, *(cl_mem*) y, 0, 1, &openCLInterface->commands,NULL);
    GPUStatusCheck(openCLInterface,GPUAXPY);
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
void GPUcopy(GPUStatus* openCLInterface,int n, double * x, int incx, double *y, int incy){
    openCLInterface->CLBlastStatus = CLBlastDcopy(n,*(cl_mem*)x,0,incx,*(cl_mem*)y,0,incy,&openCLInterface->commands,NULL);
    // openCLInterface->cublasStatus = cublasDcopy(openCLInterface->cublasHandle, n, x, incx, y, incy);
    GPUStatusCheck(openCLInterface,GPUCOPY);
}

/**
 * @brief Scales a marix by a value alpha in place
 *  performs x[j] = alpha x[j] for i = 1,...,n and j = (i-1)*incx
 * @param n number of elements to scale
 * @param alpha number to scale the elements by
 * @param x poitner to start of matrix
 * @param incx space between elements in x to increment
 */
void GPUScale(GPUStatus* openCLInterface, int n, const double *alpha, double *x, int incx) {
    openCLInterface->CLBlastStatus = CLBlastDscal(n, *alpha, *(cl_mem*) x, 0, 1, &openCLInterface->commands, NULL);
    GPUStatusCheck(openCLInterface,GPUSCALE);
}

void GPUnrm2(GPUStatus* openCLInterface,int n, const double *x, int incx, double *result) {

    void* buffer;

    GPUallocate(openCLInterface,&buffer,sizeof(double));
    openCLInterface->CLBlastStatus = CLBlastDnrm2(n, *(cl_mem*) buffer, 0, *(cl_mem*) x, 0, incx, &openCLInterface->commands, NULL);
    GPUStatusCheck(openCLInterface,GPUNRM2);

    GPUcopy_to_host(openCLInterface,result,buffer,0,sizeof(double));
    GPUdeallocate(openCLInterface,buffer);


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
void GPUgemm(GPUStatus* openCLInterface,\
                           int m, int n, int k,\
                           const double          *alpha,\
                           const double          *A, int lda,\
                           const double          *B, int ldb,\
                           const double          *beta,\
                           double          *C, int ldc) {
    openCLInterface -> CLBlastStatus = CLBlastDgemm(CLBlastLayoutColMajor, CLBlastTransposeNo, CLBlastTransposeNo,\
                                m,  n,  k, *alpha, *(cl_mem*) A, 0, lda, *(cl_mem*) B, 0, ldb, *beta,\
                               *(cl_mem*) C, 0, ldc,&openCLInterface->commands,NULL);
    GPUStatusCheck(openCLInterface,GPUMULT);
}

/************************************************
*       LAPACK-like viennaCL interface          *
*       http://viennacl.sourceforge.net         *
************************************************/


void GPUgetrf(GPUStatus* openCLInterface,int nrows, int ncols,double* elements,int lda,int* pivot,int* devInfo){
    /*int Lwork;
    openCLInterface->cusolverStatus = cusolverDnDgetrf_bufferSize(openCLInterface->cusolverHandle, nrows, ncols, elements, lda, &Lwork);
    double *Workspace;
    GPUallocate(openCLInterface, (void**)&Workspace,sizeof(double)*Lwork);
    openCLInterface->cusolverStatus = cusolverDnDgetrf(openCLInterface->cusolverHandle, nrows, ncols, elements, lda,\
           Workspace, pivot, devInfo );

    GPUStatusCheck(openCLInterface,GPUGETRF);*/
    printf("%s not supported under openCL",GPUGETRF);
    // GPUdeallocate(openCLInterface,Workspace);
}

void GPUgetrs(GPUStatus* openCLInterface, int nrows, int ncolsB, double * A, int lda, int *devIpiv, double* B, int ldb, int *devInfo) {

    // openCLInterface->cusolverStatus = cusolverDnDgetrs(openCLInterface->cusolverHandle,\
    //        CUBLAS_OP_N, nrows, ncolsB, A, lda, devIpiv, B, ldb, devInfo);
    // GPUStatusCheck(openCLInterface,GPUGETRS);
    printf("%s not supported under openCL",GPUGETRS);
}

/************************************************
*               Functional interface            *
************************************************/
void GPUcall_functional(GPUStatus* openCLInterface,double* verts, int dim, objectgpusparse_light* conn,\
                      int grade, int nelements,int integrandNo, double* out){

	size_t global = nelements;

    if (conn) {
        int *nentries = &conn->nentries;
        int *nrows = &conn->nrows;
        int *ncols = &conn->ncols;
        cl_mem *cptr = (cl_mem*)conn->cptr;
        cl_mem *rix = (cl_mem*)conn->rix;
        openCLInterface->openCLStatus =  clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVALCONN], 0,sizeof(cl_mem),(cl_mem*)(verts));
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVALCONN], 1,sizeof(int),&dim);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVALCONN], 2,sizeof(int),&nelements);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVALCONN], 3,sizeof(int),nentries);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVALCONN], 4,sizeof(int),nrows);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVALCONN], 5,sizeof(int),ncols);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVALCONN], 6,sizeof(cl_mem),cptr);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVALCONN], 7,sizeof(cl_mem),rix);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVALCONN], 8,sizeof(int),&integrandNo);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVALCONN], 9,sizeof(cl_mem),(cl_mem*)out);
        GPUStatusCheck(openCLInterface,"Call Functional integrand");
        openCLInterface->openCLStatus = clEnqueueNDRangeKernel(openCLInterface->commands, openCLInterface->kernel_list[GPU_INTEGRANDEVALCONN], 1, NULL, &global, NULL, 0, NULL, NULL);
	
    } else {
        openCLInterface->openCLStatus =  clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 0,sizeof(cl_mem),(cl_mem*)(verts));
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 1,sizeof(int),&dim);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 2,sizeof(int),&nelements);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 3,sizeof(int),&integrandNo);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 4,sizeof(cl_mem),(cl_mem*)out);
        GPUStatusCheck(openCLInterface,"Call Functional integrand");
        openCLInterface->openCLStatus = clEnqueueNDRangeKernel(openCLInterface->commands, openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 1, NULL, &global, NULL, 0, NULL, NULL);


    }
    clFinish(openCLInterface->commands);
    GPUStatusCheck(openCLInterface,"Call Functional integrand");
}
void GPUcall_functionalgrad(GPUStatus* openCLInterface,double* verts, int dim, objectgpusparse_light* conn,\
                            int grade, int nelements,int gradientNo, double* out) {
	size_t global = nelements;
    // switch between using a connectivity matrix or not
    if (conn) {

        openCLInterface->openCLStatus =  clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVALCONN], 0,sizeof(cl_mem),(cl_mem*)(verts));
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVALCONN], 1,sizeof(int),&dim);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVALCONN], 2,sizeof(int),&nelements);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVALCONN], 3,sizeof(int),&conn->nentries);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVALCONN], 4,sizeof(int),&conn->nrows);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVALCONN], 5,sizeof(int),&conn->ncols);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVALCONN], 6,sizeof(cl_mem),(cl_mem*)conn->cptr);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVALCONN], 7,sizeof(cl_mem),(cl_mem*)conn->rix);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVALCONN], 8,sizeof(int),&gradientNo);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVALCONN], 9,sizeof(cl_mem),(cl_mem*)out);
        openCLInterface->openCLStatus = clEnqueueNDRangeKernel(openCLInterface->commands, openCLInterface->kernel_list[GPU_GRADEVALCONN], 1, NULL, &global, NULL, 0, NULL, NULL);
    } else {
        openCLInterface->openCLStatus =  clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVAL], 0,sizeof(cl_mem),(cl_mem*)(verts));
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVAL], 1,sizeof(int),&dim);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVAL], 2,sizeof(int),&nelements);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVAL], 3,sizeof(int),&gradientNo);
        openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVAL], 4,sizeof(cl_mem),(cl_mem*)out);
    	openCLInterface->openCLStatus = clEnqueueNDRangeKernel(openCLInterface->commands, openCLInterface->kernel_list[GPU_GRADEVAL], 1, NULL, &global, NULL, 0, NULL, NULL);
    }
	clFinish(openCLInterface->commands);
    GPUStatusCheck(openCLInterface,"Call Functional gradient");

}

/** openCLStatus lookup */
const char* getErrorString(cl_int openCLStatus) {
    switch(openCLStatus){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL openCLStatus";
    }
}

const char* getCLBlastErrorString(CLBlastStatusCode code) {
    switch(code){ 
        case CLBlastSuccess: return "CL_SUCCESS";
        case CLBlastOpenCLCompilerNotAvailable: return "CL_COMPILER_NOT_AVAILABLE";
        case CLBlastTempBufferAllocFailure    : return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CLBlastOpenCLOutOfResources      : return "CL_OUT_OF_RESOURCES";
        case CLBlastOpenCLOutOfHostMemory     : return "CL_OUT_OF_HOST_MEMORY";
        case CLBlastOpenCLBuildProgramFailure : return "CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error";
        case CLBlastInvalidValue              : return "CL_INVALID_VALUE";
        case CLBlastInvalidCommandQueue       : return "CL_INVALID_COMMAND_QUEUE";
        case CLBlastInvalidMemObject          : return "CL_INVALID_MEM_OBJECT";
        case CLBlastInvalidBinary             : return "CL_INVALID_BINARY";
        case CLBlastInvalidBuildOptions       : return "CL_INVALID_BUILD_OPTIONS";
        case CLBlastInvalidProgram            : return "CL_INVALID_PROGRAM";
        case CLBlastInvalidProgramExecutable  : return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CLBlastInvalidKernelName         : return "CL_INVALID_KERNEL_NAME";
        case CLBlastInvalidKernelDefinition   : return "CL_INVALID_KERNEL_DEFINITION";
        case CLBlastInvalidKernel             : return "CL_INVALID_KERNEL";
        case CLBlastInvalidArgIndex           : return "CL_INVALID_ARG_INDEX";
        case CLBlastInvalidArgValue           : return "CL_INVALID_ARG_VALUE";
        case CLBlastInvalidArgSize            : return "CL_INVALID_ARG_SIZE";
        case CLBlastInvalidKernelArgs         : return "CL_INVALID_KERNEL_ARGS";
        case CLBlastInvalidLocalNumDimensions : return "CL_INVALID_WORK_DIMENSION: Too many thread dimensions";
        case CLBlastInvalidLocalThreadsTotal  : return "CL_INVALID_WORK_GROUP_SIZE: Too many threads in total";
        case CLBlastInvalidLocalThreadsDim    : return "CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension";
        case CLBlastInvalidGlobalOffset       : return "CL_INVALID_GLOBAL_OFFSET";
        case CLBlastInvalidEventWaitList      : return "CL_INVALID_EVENT_WAIT_LIST";
        case CLBlastInvalidEvent              : return "CL_INVALID_EVENT";
        case CLBlastInvalidOperation          : return "CL_INVALID_OPERATION";
        case CLBlastInvalidBufferSize         : return "CL_INVALID_BUFFER_SIZE";
        case CLBlastInvalidGlobalWorkSize     : return "CL_INVALID_GLOBAL_WORK_SIZE";

        // Status codes in common with the clBLAS library
        case CLBlastNotImplemented            : return "Routine or functionality not implemented yet";
        case CLBlastInvalidMatrixA            : return "Matrix A is not a valid OpenCL buffer";
        case CLBlastInvalidMatrixB            : return "Matrix B is not a valid OpenCL buffer";
        case CLBlastInvalidMatrixC            : return "Matrix C is not a valid OpenCL buffer";
        case CLBlastInvalidVectorX            : return "Vector X is not a valid OpenCL buffer";
        case CLBlastInvalidVectorY            : return "Vector Y is not a valid OpenCL buffer";
        case CLBlastInvalidDimension          : return "Dimensions M, N, and K have to be larger than zero";
        case CLBlastInvalidLeadDimA           : return "LD of A is smaller than the matrix's first dimension";
        case CLBlastInvalidLeadDimB           : return "LD of B is smaller than the matrix's first dimension";
        case CLBlastInvalidLeadDimC           : return "LD of C is smaller than the matrix's first dimension";
        case CLBlastInvalidIncrementX         : return "Increment of vector X cannot be zero";
        case CLBlastInvalidIncrementY         : return "Increment of vector Y cannot be zero";
        case CLBlastInsufficientMemoryA       : return "Matrix A's OpenCL buffer is too small";
        case CLBlastInsufficientMemoryB       : return "Matrix B's OpenCL buffer is too small";
        case CLBlastInsufficientMemoryC       : return "Matrix C's OpenCL buffer is too small";
        case CLBlastInsufficientMemoryX       : return "Vector X's OpenCL buffer is too small";
        case CLBlastInsufficientMemoryY       : return "Vector Y's OpenCL buffer is too small";

        // Custom additional status codes for CLBlast
        case CLBlastInsufficientMemoryTemp    : return "Temporary buffer provided to GEMM routine is too small";
        case CLBlastInvalidBatchCount         : return "The batch count needs to be positive";
        case CLBlastInvalidOverrideKernel     : return "Trying to override parameters for an invalid kernel";
        case CLBlastMissingOverrideParameter  : return "Missing override parameter(s) for the target kernel";
        case CLBlastInvalidLocalMemUsage      : return "Not enough local memory available on this device";
        case CLBlastNoHalfPrecision           : return "Half precision (16-bits) not supported by the device";
        case CLBlastNoDoublePrecision         : return "Double precision (64-bits) not supported by the device";
        case CLBlastInvalidVectorScalar       : return "The unit-sized vector is not a valid OpenCL buffer";
        case CLBlastInsufficientMemoryScalar  : return "The unit-sized vector's OpenCL buffer is too small";
        case CLBlastDatabaseError             : return "Entry for the device was not found in the database";
        case CLBlastUnknownError              : return "A catch-all error code representing an unspecified error";
        case CLBlastUnexpectedError           : return "A catch-all error code representing an unexpected exception";

        default: return "Unknown CLBLAST openCLStatus";
    }
}

#undef TILE_DIM
#undef BLOCK_COLS