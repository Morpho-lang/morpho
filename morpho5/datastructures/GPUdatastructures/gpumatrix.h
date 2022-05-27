/** @file gpumatrix.h
 *  @author T J Atherton
 *
 *  @brief Veneer class over the gpuobjectgpumatrix type that interfaces with blas and lapack
 */

#ifndef gpumatrix_h
#define gpumatrix_h

#include <stdio.h>
#include "veneer.h"
#include "cudainterface.h"

// #define MORPHO_USE_CUDA
// /** Use CUDA or openCL for GPU Accleartion */
// #ifdef MORPHO_USE_CUDA
// #elif MORPHO_USE_OPENCL
//     #include "openclinterface.h"
// #else
//     #error "Please choose a GPU option"

// #endif



/* -------------------------------------------------------
 * Matrix objects
 * ------------------------------------------------------- */

extern objecttype objectgpumatrixtype;
#define OBJECT_GPUMATRIX objectgpumatrixtype

/** Matrices are a purely numerical collection type oriented toward linear algebra.
    Elements are stored in column-major format, i.e.
        [ 1 2 ]
        [ 3 4 ]
    is stored ( 1, 3, 2, 4 ) in memory. This is for compatibility with standard linear algebra packages */

typedef struct {
    object obj;
    unsigned int nrows;
    unsigned int ncols;
    double *elements;
    GPUStatus *status;
} objectgpumatrix;

/** Tests whether an object is a gpumatrix */
#define MORPHO_ISGPUMATRIX(val) object_istype(val, OBJECT_GPUMATRIX)

/** Gets the object as an gpumatrix */
#define MORPHO_GETGPUMATRIX(val)   ((objectgpumatrix *) MORPHO_GETOBJECT(val))

/** Creates a gpumatrix object */
objectgpumatrix *object_newgpumatrix(unsigned int nrows, unsigned int ncols, bool zero);

/** Creates a new gpumatrix from an array */
objectgpumatrix *object_gpumatrixfromarray(objectarray *array);

/** Creates a new gpumatrix from an existing gpumatrix */
objectgpumatrix *object_clonegpumatrix(objectgpumatrix *array);

/** Creates a new matrix from a GPU matrix */
objectmatrix *object_matrixfromgpumatrix(objectgpumatrix* in);

/** Creates a new GPU matrix from a matrix */
objectgpumatrix *object_gpumatrixfrommatrix(objectmatrix* in);

/** Deletes a gpumatrix from the GPU */
void objectgpumatrix_freefn(object *obj);

/** Macro to decide if a gpumatrix is 'small' or 'large' and hence static or dynamic allocation should be used. */
#define GPUMATRIX_ISSMALL(m) (m->nrows*m->ncols<MORPHO_MAXIMUMSTACKALLOC)

/* -------------------------------------------------------
 * Matrix class
 * ------------------------------------------------------- */

#define GPUMATRIX_CLASSNAME "GPUMatrix"

#define GPUMATRIX_TRANSPOSE_METHOD "transpose"
#define GPUMATRIX_TRACE_METHOD "trace"
#define GPUMATRIX_INNER_METHOD "inner"
#define GPUMATRIX_DET_METHOD "det"
#define GPUMATRIX_EIGENVALUES_METHOD "eigenvalues"
#define GPUMATRIX_EIGENSYSTEM_METHOD "eigensystem"
#define GPUMATRIX_NORM_METHOD "norm"
#define GPUMATRIX_GETCOLUMN_METHOD "column"
#define GPUMATRIX_SETCOLUMN_METHOD "setcolumn"

#define GPUMATRIX_DIMENSIONS_METHOD "dimensions"

#define GPUMATRIX_INDICESOUTSIDEBOUNDS       "GPUMtrxBnds"
#define GPUMATRIX_INDICESOUTSIDEBOUNDS_MSG   "GPUMatrix index out of bounds."

#define GPUMATRIX_INVLDINDICES               "GPUMtrxInvldIndx"
#define GPUMATRIX_INVLDINDICES_MSG           "GPUMatrix indices must be integers."

#define GPUMATRIX_INVLDNUMINDICES            "GPUMtrxInvldNumIndx"
#define GPUMATRIX_INVLDNUMINDICES_MSG        "Matrix expects two arguments for indexing."

#define GPUMATRIX_CONSTRUCTOR                "GPUMtrxCns"
#define GPUMATRIX_CONSTRUCTOR_MSG            "Matrix() constructor should be called either with dimensions or an array, list or gpumatrix initializer."

#define GPUMATRIX_INVLDARRAYINIT             "GPUMtrxInvldInit"
#define GPUMATRIX_INVLDARRAYINIT_MSG         "Invalid initializer passed to Matrix()."

#define GPUMATRIX_ARITHARGS                  "GPUMtrxInvldArg"
#define GPUMATRIX_ARITHARGS_MSG              "Matrix arithmetic methods expect a gpumatrix or number as their argument."

#define GPUMATRIX_INCOMPATIBLEMATRICES       "GPUMtrxIncmptbl"
#define GPUMATRIX_INCOMPATIBLEMATRICES_MSG   "Matrices have incompatible shape."

#define GPUMATRIX_SINGULAR                   "GPUMtrxSnglr"
#define GPUMATRIX_SINGULAR_MSG               "Matrix is singular."

#define GPUMATRIX_NOTSQ                      "GPUMtrxNtSq"
#define GPUMATRIX_NOTSQ_MSG                  "Matrix is not square."

#define GPUMATRIX_SETCOLARGS                 "GPUMtrxStClArgs"
#define GPUMATRIX_SETCOLARGS_MSG             "Method setcolumn expects an integer column index and a column gpumatrix as arguments."

/* -------------------------------------------------------
 * Matrix errors
 * ------------------------------------------------------- */

typedef enum { GPUMATRIX_OK, GPUMATRIX_INCMPTBLDIM, GPUMATRIX_SING, GPUMATRIX_INVLD, GPUMATRIX_BNDS, GPUMATRIX_NSQ, GPUMATRIX_ALLOC } objectgpumatrixerror;

/* -------------------------------------------------------
 * Matrix interface
 * ------------------------------------------------------- */

bool gpumatrix_getarraydimensions(objectarray *array, unsigned int dim[], unsigned int maxdim, unsigned int *ndim);
value gpumatrix_getarrayelement(objectarray *array, unsigned int ndim, unsigned int *indx);

bool gpumatrix_getlistdimensions(objectlist *list, unsigned int dim[], unsigned int maxdim, unsigned int *ndim);
bool gpumatrix_getlistelement(objectlist *list, unsigned int ndim, unsigned int *indx, value *val);

bool gpumatrix_setelement(objectgpumatrix *gpumatrix, unsigned int row, unsigned int col, double value);
bool gpumatrix_setelementfromval(objectgpumatrix *a, int ind, value val);
bool gpumatrix_getelement(objectgpumatrix *gpumatrix, unsigned int row, unsigned int col, double *value);

bool gpumatrix_getcolumn(objectgpumatrix *gpumatrix, unsigned int col, double **v);
bool gpumatrix_setcolumn(objectgpumatrix *gpumatrix, unsigned int col, double *v);
bool gpumatrix_addtocolumn(objectgpumatrix *m, unsigned int col, double alpha, double *v);

objectgpumatrixerror gpumatrix_copy(objectgpumatrix *a, objectgpumatrix *out);
objectgpumatrixerror gpumatrix_add(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out);
objectgpumatrixerror gpumatrix_accumulate(objectgpumatrix *a, double lambda, objectgpumatrix *b);
objectgpumatrixerror gpumatrix_sub(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out);
objectgpumatrixerror gpumatrix_mul(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out);
objectgpumatrixerror gpumatrix_inner(objectgpumatrix *a, objectgpumatrix *b, double *out);
objectgpumatrixerror gpumatrix_divs(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out);
objectgpumatrixerror gpumatrix_divl(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out);
objectgpumatrixerror gpumatrix_inverse(objectgpumatrix *a, objectgpumatrix *out);
objectgpumatrixerror gpumatrix_transpose(objectgpumatrix *a, objectgpumatrix *out);
objectgpumatrixerror gpumatrix_trace(objectgpumatrix *a, double *out);
objectgpumatrixerror gpumatrix_scale(objectgpumatrix *a, double scale);
objectgpumatrixerror gpumatrix_identity(objectgpumatrix *a);
double gpumatrix_sum(objectgpumatrix *a);
//objectgpumatrixerror gpumatrix_det(objectgpumatrix *a, double *out);
//objectgpumatrixerror gpumatrix_eigensystem(objectgpumatrix *a, double *val, objectgpumatrix *vec);

void gpumatrix_print(objectgpumatrix *m);

void gpumatrix_initialize(void);

#endif