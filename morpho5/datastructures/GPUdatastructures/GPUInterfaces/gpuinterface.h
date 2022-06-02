
#ifndef GPUINTERFACE_H
#define GPUINTERFACE_H

typedef enum { AREA_KERNAL, VOLUMEENCLOSED_KERNAL, NO_KERNAL } kernal_enum;
// we put a ligher version of spare here to avoid having to include object here (nvcc doens't like it)
typedef struct {
    int nentries;
    int nrows;
    int ncols;
    int *cptr; // Pointers to column entries
    int *rix; // Row indices
    double *values; // Values
} objectgpusparse_light;




#ifdef CUDA_ACC
    #include "cudainterface.h"
#endif
#ifdef OPENCL_ACC
    #include "openCLinterface.h"
#endif
#endif
