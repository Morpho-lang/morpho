/** @file sparse.c
 *  @author T J Atherton
 *
 *  @brief Veneer class over the objectsparse type that provides sparse matrices
 */

#include "build.h"
#include "morpho.h"
#include "dictionary.h"
#include "common.h"
#include "matrix.h"
#include "builtin.h"
#include "veneer.h"
#include "cudainterface.h"
#ifdef GPU_ACC
    #include "gpumatrix.h"
    #include "gpusparse.h"
#endif
#include <limits.h>
#include <stdlib.h>

/* ***************************************
 * Compressed Column Storage Format
 * *************************************** */

/** Initializes an empty objectgpusparse */
void gpusparseccs_init(objectgpusparse *ccs) {
    ccs->nentries=0;
    ccs->nrows=0;
    ccs->ncols=0;
    ccs->cptr=NULL;
    ccs->rix=NULL;
    ccs->values=NULL;
    ccs->status = &myGPUstatus;
    GPUsetup(&myGPUstatus);
}

/** Clears all data structures associated with a objectgpusparse */
void gpusparse_copyfromcpu(objectgpusparse *gpuccs, sparseccs *ccs) {
    gpuccs->nentries=ccs->nentries;
    gpuccs->nrows=ccs->nrows;
    gpuccs->ncols=ccs->ncols;
    GPUallocate(gpuccs->status,(void**)&gpuccs->cptr,sizeof(int)*(ccs->ncols+1));
    GPUallocate(gpuccs->status,(void**)&gpuccs->rix,sizeof(int)*ccs->nentries);
    GPUcopy_to_device(gpuccs->status,(void*)gpuccs->cptr,(void*)ccs->cptr,sizeof(int)*(ccs->ncols+1));
    GPUcopy_to_device(gpuccs->status,(void*)gpuccs->rix,(void*)ccs->rix,sizeof(int)*(ccs->nentries));
    if (ccs->values) {
        GPUallocate(gpuccs->status,(void**)&gpuccs->values,sizeof(double)*ccs->nentries);
        GPUcopy_to_device(gpuccs->status,(void*)gpuccs->values,(void*)ccs->values,sizeof(double)*(ccs->nentries));
    }
}
void gpusparse_copyfromgpu(sparseccs *ccs, objectgpusparse *gpuccs) {
    ccs->nentries=gpuccs->nentries;
    ccs->nrows=gpuccs->nrows;
    ccs->ncols=gpuccs->ncols;
    GPUallocate(gpuccs->status,(void**)&gpuccs->cptr,sizeof(int)*(ccs->ncols+1));
    GPUallocate(gpuccs->status,(void**)&gpuccs->rix,sizeof(int)*ccs->nentries);
    GPUallocate(gpuccs->status,(void**)&gpuccs->values,sizeof(double)*ccs->nentries);
    GPUcopy_to_host(gpuccs->status,(void*)ccs->cptr,(void*)gpuccs->cptr,sizeof(int)*(ccs->ncols+1));
    GPUcopy_to_host(gpuccs->status,(void*)ccs->rix,(void*)gpuccs->rix,sizeof(int)*(ccs->nentries));
    GPUcopy_to_host(gpuccs->status,(void*)ccs->values,(void*)gpuccs->values,sizeof(double)*(ccs->nentries));

}

/** Clears any data attached to a sparse matrix */
void gpusparse_clear(objectgpusparse *ccs) {
    GPUdeallocate(ccs->status,ccs->cptr);
    GPUdeallocate(ccs->status,ccs->rix);
    GPUdeallocate(ccs->status,ccs->values);
}

/** Number of entries in a objectgpusparse */
unsigned int gpusparse_count(objectgpusparse *ccs) {
    return ccs->nentries;
}

/** Resizes a dest matrix ot match the size of source */
bool gpusparse_resize(objectgpusparse *dest,int nrows,int ncols,int nentries){
    gpusparse_clear(dest);

    dest->nentries=nentries;
    dest->nrows=nrows;
    dest->ncols=ncols;

    GPUallocate(dest->status,(void**)&dest->cptr,sizeof(int)*(ncols+1));
    GPUallocate(dest->status,(void**)&dest->rix,sizeof(int)*nentries);
    GPUallocate(dest->status,(void**)&dest->values,sizeof(double)*nentries);
}

/** Copies one objectgpusparse matrix to another, reallocating as necessary */
bool gpusparse_copy(objectgpusparse *src, objectgpusparse *dest) {
    bool success=false;
    gpusparse_resize(dest, src->nrows, src->ncols, src->nentries);

    GPUcopy_device_to_device(dest->status,dest->cptr, src->cptr, sizeof(int)*(src->ncols+1));
    GPUcopy_device_to_device(dest->status,dest->rix, src->rix, sizeof(int)*(src->nentries));
    if (src->values) GPUcopy_device_to_device(dest->status,dest->values, src->values, sizeof(double)*src->nentries);
    success=true;

    return success;
}

/* ***************************************
 * objectsparse definition
 * *************************************** */
objecttype objectgpusparsetype;
/** GPUSparse object definitions */
void objectgpusparse_printfn(object *obj) {
    printf("<GPUSparse>");
}

void objectgpusparse_freefn(object *obj) {
    objectgpusparse *s = (objectgpusparse *) obj;
    gpusparse_clear(s);
}

size_t objectgpusparse_sizefn(object *obj) {
    return sizeof(objectgpusparse);
}

objecttypedefn objectgpusparsedefn = {
    .printfn=objectgpusparse_printfn,
    .markfn=NULL,
    .freefn=objectgpusparse_freefn,
    .sizefn=objectgpusparse_sizefn
};

/* ***************************************
 * objectsparse objects
 * *************************************** */

/** Creates a sparse matrix object
 * @param[in] nrows } Optional number of rows and columns
 * @param[in] ncols } */
objectgpusparse *object_newgpusparse() {
    objectgpusparse *new = (objectgpusparse *) object_new(sizeof(objectgpusparse), OBJECT_GPUSPARSE);
    if (new) {
        gpusparseccs_init(new);
    }    
    return new;
}

/** Clones a sparse matrix */
objectgpusparse *gpusparse_clone(objectgpusparse *s) {
    objectgpusparse *new = object_newgpusparse();
 
    if (new) {
        gpusparse_copy(s, new);
    }
    
    return new;
}

/* ***************************************
 * GPUSparse builtin class
 * *************************************** */

void gpusparse_raiseerror(vm *v, objectgpusparseerror err) {
    switch(err) {
        case GPUSPARSE_OK: break;
        case GPUSPARSE_INCMPTBLDIM: morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES); break;
        case GPUSPARSE_CONVFAILED: morpho_runtimeerror(v, SPARSE_CONVFAILEDERR); break;
        case GPUSPARSE_FAILED: morpho_runtimeerror(v, SPARSE_OPFAILEDERR); break;
    }
}

/** Constructs a Matrix object */
value gpusparse_constructor(vm *v, int nargs, value *args) {
    int nrows, ncols;
    objectgpusparse *new=NULL;
    value out=MORPHO_NIL; 
    
     if (nargs == 1 &&
               MORPHO_ISSPARSE(MORPHO_GETARG(args, 0))) {
        objectsparse *sparse_in = MORPHO_GETSPARSE(MORPHO_GETARG(args,0));
        if (sparse_checkformat(sparse_in, SPARSE_CCS, true, true)){
                new = object_newgpusparse();
                gpusparse_copyfromcpu(new,&sparse_in->ccs);
            }        
    } else {
        morpho_runtimeerror(v, GPUSPARSE_CONSTRUCTOR);
    }
    
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}
/** Arithmatic functions */



/** add function */
// objectsparseerror gpusparse_add(objectgpusparse *x,objectgpusparse *y,objectgpusparse *out) {
//     //use cusparseAxpby() to perform Y = aX +bY were Y is a full matrix and X is sparse
//     //first create a cusparse matrix from  
//     GPUSparseAdd(MAKEGPUSPARSE_LIGHT(x),MAKEGPUSPARSE_LIGHT(y),MAKEGPUSPARSE_LIGHT(out));
//     // cusparseAxpby(x->status->cusparseHandle,&mult,xcusparse,&mult,ycusparseD);

// }

// /** Enumerate protocol */
// value GPUSparse_enumerate(vm *v, int nargs, value *args) {
//     objectsparse *s=MORPHO_GETSPARSE(MORPHO_SELF(args));
//     value out=MORPHO_NIL;
    
//     if (nargs==1) {
//         if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
//             int i=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
            
//             sparse_enumerate(s, i, &out);
//         }
//     }
    
//     return out;
// }

/** Print a sparse matrix */
value GPUSparse_print(vm *v, int nargs, value *args) {
    objectgpusparse *s=MORPHO_GETGPUSPARSE(MORPHO_SELF(args));
    printf("<GPUSpare>");
    // gpusparse_print(s);
    
    return MORPHO_NIL;
}

/** Sparse Transpose */
value GPUSparse_transpose(vm *v, int nargs, value *args) {
    objectgpusparse *a=MORPHO_GETGPUSPARSE(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    objectgpusparse *new = object_newgpusparse();
    if (new) {
        objectgpusparseerror err = GPUSparseTranspose(a->status,(objectgpusparse_light*)a, (objectgpusparse_light*)new);
        if (err==SPARSE_OK) {
            out=MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        } else {
            gpusparse_raiseerror(v, err);
        }
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

/** Clone a sparse matrix */
value GPUSparse_clone(vm *v, int nargs, value *args) {
    objectgpusparse *s=MORPHO_GETGPUSPARSE(MORPHO_SELF(args));
    value out = MORPHO_NIL;
    objectgpusparse *new=gpusparse_clone(s);
    
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

/** Count number of elements */
value GPUSparse_count(vm *v, int nargs, value *args) {
    objectgpusparse *s=MORPHO_GETGPUSPARSE(MORPHO_SELF(args));
    value out = MORPHO_INTEGER(0);    
    out=MORPHO_INTEGER(s->nentries);
    
    return out;
}

/** GPUSparse dimensions */
value GPUSparse_dimensions(vm *v, int nargs, value *args) {
    objectgpusparse *s=MORPHO_GETGPUSPARSE(MORPHO_SELF(args));
    value dim[2];
    value out=MORPHO_NIL;
    
    if (s->ncols>0) {
        dim[0]=MORPHO_INTEGER(s->nrows);
        dim[1]=MORPHO_INTEGER(s->ncols);
    }
    
    objectlist *new=object_newlist(2, dim);
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

value GPUSparse_copytohost(vm *v, int nargs, value *args) {
    objectgpusparse *s=MORPHO_GETGPUSPARSE(MORPHO_SELF(args));
    objectsparse *new = object_newsparse(NULL,NULL);
    value out;
    
    if (new) {
        gpusparse_copyfromgpu(&new->ccs, s);
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

// /** Arithmatic */

// value GPUSparse_add(vm *v, int nargs, value *args) {
//     objectgpusparse *a=MORPHO_GETGPUSPARSE(MORPHO_SELF(args));
//     value out=MORPHO_NIL;
 
//     if (nargs==1 && MORPHO_ISGPUSPARSE(MORPHO_GETARG(args, 0))) {
//         objectgpusparse *b=MORPHO_GETGPUSPARSE(MORPHO_GETARG(args, 0));
        
//         objectgpusparse *new = object_newgpusparse(NULL, NULL);
//         if (new) {
//             objectsparseerror err =gpusparse_add(a, b, new);
//             if (err==SPARSE_OK) {
//                 out=MORPHO_OBJECT(new);
//                 morpho_bindobjects(v, 1, &out);
//             } else {
//                 sparse_raiseerror(v, err);
//             }
//         } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
//     }
    
//     return out;
// }

// /** Subtract sparse matrices */
// value GPUSparse_sub(vm *v, int nargs, value *args) {
//     objectgpusparse *a=MORPHO_GETGPUSPARSE(MORPHO_SELF(args));
//     value out=MORPHO_NIL;
 
//     if (nargs==1 && MORPHO_ISSPARSE(MORPHO_GETARG(args, 0))) {
//         objectgpusparse *b=MORPHO_GETGPUSPARSE(MORPHO_GETARG(args, 0));
        
//         objectgpusparse *new = object_newsparse(NULL, NULL);
//         if (new) {
//             objectgpusparseerror err =sparse_add(a, b, 1.0, -1.0, new);
//             if (err==SPARSE_OK) {
//                 out=MORPHO_OBJECT(new);
//                 morpho_bindobjects(v, 1, &out);
//             } else {
//                 sparse_raiseerror(v, err);
//             }
//         } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
//     }
    
//     return out;
// }

// /** Multiply sparse matrices */
// value GPUSparse_mul(vm *v, int nargs, value *args) {
//     objectgpusparse *a=MORPHO_GETGPUSPARSE(MORPHO_SELF(args));
//     value out=MORPHO_NIL;
 
//     if (nargs==1 && MORPHO_ISSPARSE(MORPHO_GETARG(args, 0))) {
//         objectgpusparse *b=MORPHO_GETGPUSPARSE(MORPHO_GETARG(args, 0));
        
//         objectgpusparse *new = object_newsparse(NULL, NULL);
//         if (new) {
//             objectgpusparseerror err =sparse_mul(a, b, new);
//             if (err==SPARSE_OK) {
//                 out=MORPHO_OBJECT(new);
//                 morpho_bindobjects(v, 1, &out);
//             } else {
//                 sparse_raiseerror(v, err);
//             }
//         } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
//     }
    
//     return out;
// }

// /** Sparse rhs not implemented */
// value GPUSparse_div(vm *v, int nargs, value *args) {
//     return MORPHO_NIL;
// }

// /** Solve a linear system b/ A where A is sparse */
// value GPUSparse_divr(vm *v, int nargs, value *args) {
//     objectgpusparse *a=MORPHO_GETGPUSPARSE(MORPHO_SELF(args));
//     value out=MORPHO_NIL;
 
//     if (nargs==1 && MORPHO_ISMATRIX(MORPHO_GETARG(args, 0))) {
//         objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
        
//         objectmatrix *new = object_newmatrix(b->nrows, b->ncols, false);
//         if (new) {
//             objectgpusparseerror err =sparse_div(a, b, new);
//             if (err==SPARSE_OK) {
//                 out=MORPHO_OBJECT(new);
//                 morpho_bindobjects(v, 1, &out);
//             } else {
//                 sparse_raiseerror(v, err);
//             }
//         } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
//     }
    
//     return out;
// }




MORPHO_BEGINCLASS(GPUSparse)
MORPHO_METHOD(MORPHO_PRINT_METHOD, GPUSparse_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_TRANSPOSE_METHOD, GPUSparse_transpose, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, GPUSparse_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_DIMENSIONS_METHOD, GPUSparse_dimensions, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUSPARSE_COPYTOHOST,GPUSparse_copytohost,BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, GPUSparse_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* ***************************************
 * Initialization
 * *************************************** */

void gpusparse_initialize(void) {    
    objectgpusparsetype = object_addtype(&objectgpusparsedefn);

    builtin_addfunction(GPUSPARSE_CLASSNAME, gpusparse_constructor, BUILTIN_FLAGSEMPTY);
    
    value gpusparseclass=builtin_addclass(GPUSPARSE_CLASSNAME, MORPHO_GETCLASSDEFINITION(GPUSparse), MORPHO_NIL);
    object_setveneerclass(OBJECT_GPUSPARSE, gpusparseclass);
    
    morpho_defineerror(GPUSPARSE_CONSTRUCTOR, ERROR_HALT, GPUSPARSE_CONSTRUCTOR_MSG);
    //sparse_test();
}
