/** @file sparse.h
 *  @author T J Atherton
 *
 *  @brief Veneer class over the objectsparse type that provides sparse matrices
 */

#ifndef gpusparse_h
#define gpusparse_h

#include <stdio.h>
#include "object.h"
#include "morpho.h"
#include "cudainterface.h"
#include "sparse.h"


/* ***************************************
 * Sparse objects
 * *************************************** */


typedef struct {
    object obj;

    int nentries;
    int nrows;
    int ncols;
    int *cptr; // Pointers to column entries
    int *rix; // Row indices
    double *values; // Values
    GPUStatus *status;

} objectgpusparse;

/** Tests whether an object is a sparse matrix */
#define MORPHO_ISGPUSPARSE(val) object_istype(val, OBJECT_GPUSPARSE)

/** Gets the object as a sparse matrix */
#define MORPHO_GETGPUSPARSE(val)   ((objectgpusparse *) MORPHO_GETOBJECT(val))

extern objecttype objectgpusparsetype;
#define OBJECT_GPUSPARSE objectgpusparsetype

/* ***************************************
 * The Sparse class
 * *************************************** */

#define GPUSPARSE_CLASSNAME "GPUSparse"

#define GPUSPARSE_COPYTOHOST "CopyToHost"

#define GPUSPARSE_CONSTRUCTOR                "GPUSprsCns"
#define GPUSPARSE_CONSTRUCTOR_MSG            "GPUSparse() should be with a Sparse Matrix"

/* ***************************************
 * Object sparse interface
 * *************************************** */

typedef enum { GPUSPARSE_OK, GPUSPARSE_INCMPTBLDIM, GPUSPARSE_CONVFAILED, GPUSPARSE_FAILED } objectgpusparseerror;

#define MAKEGPUSPARSE_LIGHT(s) (objectgpusparse_light*)((void*)s+sizeof(object))

objectgpusparse *gpusparse_clone(objectgpusparse *s);

void gpusparse_clear(objectgpusparse *a);
size_t gpusparse_size(objectgpusparse *a);

objectgpusparse *object_newgpusparse();
void gpusparse_copyfromcpu(objectgpusparse *gpuccs, sparseccs *ccs);

/* ***************************************
 * Initialization
 * *************************************** */

void gpusparse_initialize(void);

#endif /* sparse_h */
