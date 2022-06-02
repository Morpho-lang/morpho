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
        printf("openCLStatus: Failed to %s!\n",caller);
    }

}

void GPUsetup(GPUStatus* openCLInterface) { //, vm* v) {
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
	printf("--->available number of platforms = %d\n", ret_num_platforms);
	openCLInterface->openCLStatus = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &openCLInterface->device_id, &ret_num_devices);
	GPUStatusCheck(openCLInterface,"clGetDeviceIDs");
	printf("--->available number of devices = %d\n", ret_num_devices);
    
  
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
		printf("openCLStatus: Failed to create a command commands!\n");
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

}
void GPUallocate(GPUStatus* openCLInterface,void** ptr, unsigned int size){
	cl_mem* var = NULL;
    *var = clCreateBuffer(openCLInterface->context,CL_MEM_READ_WRITE,size,NULL,&openCLInterface->openCLStatus);
    GPUStatusCheck(openCLInterface,GPUALLOCATE);
}

void GPUdeallocate(GPUStatus* openCLInterface,void* dPointer) {
	openCLInterface->openCLStatus = clReleaseMemObject(*(cl_mem*)dPointer);
    GPUStatusCheck(openCLInterface,GPUDEALLOCATE);
}

void GPUmemset(GPUStatus* openCLInterface,void* devicePointer, int bytepattern, unsigned int size){
    openCLInterface->openCLStatus = clEnqueueFillBuffer(openCLInterface->commands,devicePointer,(void*)&bytepattern,sizeof(double),0,size,0,NULL,NULL);
    GPUStatusCheck(openCLInterface,GPUMEMSET);

}
void GPUcopy_to_host(GPUStatus* openCLInterface,void * hostPointer, void * devicepointer, unsigned int size) {
    openCLInterface->openCLStatus = clEnqueueReadBuffer(openCLInterface->commands,*(cl_mem*) devicepointer,CL_TRUE,0,size,hostPointer,0,NULL,NULL);
    GPUStatusCheck(openCLInterface,GPUMEMCPY);

}
void GPUcopy_to_device(GPUStatus* openCLInterface,void* devicePointer, void* hostPointer, unsigned int size) {
    openCLInterface->openCLStatus = clEnqueueWriteBuffer(openCLInterface->commands, *(cl_mem*)devicePointer, CL_TRUE,0,size,hostPointer,0,NULL,NULL);
    GPUStatusCheck(openCLInterface,GPUMEMCPY);

}
void GPUcopy_device_to_device(GPUStatus* openCLInterface,void* devicePointersrc, void* devicePointerdest, unsigned int size) {
 openCLInterface->openCLStatus = clEnqueueCopyBuffer(openCLInterface->commands, *(cl_mem*) devicePointersrc, *(cl_mem*) devicePointerdest,\
  	0, 0, size, 0, NULL, NULL);
    GPUStatusCheck(openCLInterface,GPUMEMCPY);

}


void GPUScalarAddition(GPUStatus* openCLInterface, double* Matrix, double scalar, double *out, int size){
    unsigned int blockSize = 64;
    unsigned int numberOfBlocks = ceil(size / (float) blockSize);

    openCLInterface->openCLStatus =  clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXSCALARADDITION], 0,sizeof(cl_mem),(cl_mem*)(Matrix));
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXSCALARADDITION], 1,sizeof(double),&scalar);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXSCALARADDITION], 2,sizeof(cl_mem),(cl_mem*)(out));
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_MATRIXSCALARADDITION], 3,sizeof(unsigned int),&size);
    GPUStatusCheck(openCLInterface,GPUSCALARADD);

    size_t local = blockSize;
	size_t global;
	if (size<blockSize){
		global = blockSize;
	} else {global = size;}

	openCLInterface->openCLStatus = clEnqueueNDRangeKernel(openCLInterface->commands, openCLInterface->kernel_list[GPU_MATRIXSCALARADDITION], 1, NULL, &local, &global, 0, NULL, NULL);
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
    GPUStatusCheck(openCLInterface,GPUSCALARADD);

    size_t local[2] = {BLOCK_COLS,TILE_DIM};
	size_t global[2] = {BLOCK_COLS*ceil(ncols/(double)TILE_DIM), TILE_DIM*ceil(ncols/(double)TILE_DIM)};

	openCLInterface->openCLStatus = clEnqueueNDRangeKernel(openCLInterface->commands, openCLInterface->kernel_list[GPU_MATRIXTRANSPOSE], 2, NULL, local, global, 0, NULL, NULL);
	clFinish(openCLInterface->commands);
    //dim3 grid(ceil(ncols/(double)TILE_DIM), ceil(nrows/(double)TILE_DIM)), threads(BLOCK_COLS,TILE_DIM);
    //transposeDiagonal<<<grid,threads>>>(out, in, ncols, nrows);
    GPUStatusCheck(openCLInterface,GPUTRANSPOSE);

}

int GPUSparseTranspose(GPUStatus* cudaInterface, objectgpusparse_light *in,objectgpusparse_light *out){
    printf("Sparse Transpose not implemented");
}



void dotProduct(GPUStatus* openCLInterface, double* v1, double * v2, int size, double * out){
    openCLInterface->CLBlastStatus = CLBlastDdot(size, *(cl_mem*)out, 0, *(cl_mem*)v1, 0, 1,*(cl_mem*)v2, 0, 1,&openCLInterface->commands,NULL);
    GPUStatusCheck(openCLInterface,GPUDOT);
}
void GPUdot(GPUStatus* openCLInterface,int size, double *x,int incx, double *y, int incy,double* out){
    openCLInterface->CLBlastStatus = CLBlastDdot(size, *(cl_mem*)out, 0, *(cl_mem*)x, 0, incx,*(cl_mem*)y, 0, incy,&openCLInterface->commands,NULL);
    GPUStatusCheck(openCLInterface,GPUDOT);
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
    CLBlastDcopy(n,*(cl_mem*)x,0,incx,*(cl_mem*)y,0,incy,&openCLInterface->commands,NULL);
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
    openCLInterface->CLBlastStatus = CLBlastDnrm2(n, *(cl_mem*) result, 0, *(cl_mem*) x, 0, incx, &openCLInterface->commands, NULL);
    GPUStatusCheck(openCLInterface,GPUNRM2);

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
    openCLInterface->openCLStatus =  clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 0,sizeof(cl_mem),(cl_mem*)(verts));
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 1,sizeof(int),&dim);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 2,sizeof(cl_mem),(cl_mem*)conn);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 3,sizeof(int),&grade);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 4,sizeof(int),&nelements);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 5,sizeof(int),&integrandNo);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 6,sizeof(cl_mem),(cl_mem*)out);

    size_t local = 32;
	size_t global = nelements;
    if (nelements<local) { global = local;}

	openCLInterface->openCLStatus = clEnqueueNDRangeKernel(openCLInterface->commands, openCLInterface->kernel_list[GPU_INTEGRANDEVAL], 1, NULL, &local, &global, 0, NULL, NULL);
	clFinish(openCLInterface->commands);
    GPUStatusCheck(openCLInterface,"Call Functional integrand");
}
void GPUcall_functionalgrad(GPUStatus* openCLInterface,double* verts, int dim, objectgpusparse_light* conn,\
                            int grade, int nelements,int gradientNo, double* out) {

    openCLInterface->openCLStatus =  clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVAL], 0,sizeof(cl_mem),(cl_mem*)(verts));
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVAL], 1,sizeof(int),&dim);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVAL], 2,sizeof(cl_mem),(cl_mem*)conn);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVAL], 3,sizeof(int),&grade);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVAL], 4,sizeof(int),&nelements);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVAL], 5,sizeof(int),&gradientNo);
	openCLInterface->openCLStatus |= clSetKernelArg(openCLInterface->kernel_list[GPU_GRADEVAL], 6,sizeof(cl_mem),(cl_mem*)out);

    size_t local = 32;
	size_t global = nelements;
    if (nelements<local) { global = local;}

	openCLInterface->openCLStatus = clEnqueueNDRangeKernel(openCLInterface->commands, openCLInterface->kernel_list[GPU_GRADEVAL], 1, NULL, &local, &global, 0, NULL, NULL);
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


#undef TILE_DIM
#undef BLOCK_COLS