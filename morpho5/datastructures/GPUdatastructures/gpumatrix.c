/** @file gpumatrix.c
 *  @author T J Atherton
 *
 *  @brief Veneer class over the objectgpumatrix type that interfaces with blas and lapack
 */

#include <string.h>
#include "object.h"
#include "gpumatrix.h"
#include "sparse.h"
#include "morpho.h"
#include "builtin.h"
#include "veneer.h"
#include "common.h"

/* **********************************************************************
 * gpuMatrix objects
 * ********************************************************************** */

objecttype objectgpumatrixtype;
GPUStatus myGPUstatus = {.cudaStatus = cudaSuccess, .cublasStatus = CUBLAS_STATUS_NOT_INITIALIZED, .cublasHandle = NULL, .init = NULL};
/** Function object definitions */
size_t objectgpumatrix_sizefn(object *obj) {
    return sizeof(objectgpumatrix);
}

void objectgpumatrix_printfn(object *obj) {
    printf("<gpuMatrix>");
}
void objectgpufree(object *obj) {
    objectgpumatrix * gpumat = (objectgpumatrix *) obj;
    GPUdeallocate(gpumat->status,gpumat->elements);
}


objecttypedefn objectgpumatrixdefn = {
    .printfn=objectgpumatrix_printfn,
    .markfn=NULL,
    .freefn=objectgpufree,
    .sizefn=objectgpumatrix_sizefn
};

/** Creates a gpumatrix object */
objectgpumatrix *object_newgpumatrix(unsigned int nrows, unsigned int ncols, bool zero) {
    unsigned int nel = nrows*ncols;
    objectgpumatrix *new = (objectgpumatrix *) object_new(sizeof(objectgpumatrix), OBJECT_GPUMATRIX);
    GPUsetup(&myGPUstatus);
    if (new) {
        new->status=&myGPUstatus;
        new->ncols=ncols;
        new->nrows=nrows;
        GPUallocate(new->status,(void **)&new->elements,nel*sizeof(double));
        if (zero) {
            GPUmemset(new->status,new->elements, 0, sizeof(double)*nel);
        }
    }
    
    return new;
}

/* **********************************************************************
 * Other constructors
 * ********************************************************************** */

/*
 * Create matrices from array objects
 */

/** Recurses into an objectarray to find the dimensions of the array and all child arrays
 * @param[in] array - to search
 * @param[out] dim - array of dimensions to be filled out (must be zero'd before initial call)
 * @param[in] maxdim - maximum number of dimensions
 * @param[out] ndim - number of dimensions of the array */
bool gpumatrix_getarraydimensions(objectarray *array, unsigned int dim[], unsigned int maxdim, unsigned int *ndim) {
    unsigned int n=0, m=0;
    for (n=0; n<maxdim && n<array->ndim; n++) {
        int k=MORPHO_GETINTEGERVALUE(array->data[n]);
        if (k>dim[n]) dim[n]=k;
    }
    
    if (maxdim<array->ndim) return false;
    
    for (unsigned int i=array->ndim; i<array->ndim+array->nelements; i++) {
        if (MORPHO_ISARRAY(array->data[i])) {
            if (!gpumatrix_getarraydimensions(MORPHO_GETARRAY(array->data[i]), dim+n, maxdim-n, &m)) return false;
        }
    }
    *ndim=n+m;
    
    return true;
}

/** Looks up an array element recursively if necessary */
value gpumatrix_getarrayelement(objectarray *array, unsigned int ndim, unsigned int *indx) {
    unsigned int na=array->ndim;
    value out;
    
    if (array_getelement(array, na, indx, &out)==ARRAY_OK) {
        if (ndim==na) return out;
        if (MORPHO_ISARRAY(out)) {
            return gpumatrix_getarrayelement(MORPHO_GETARRAY(out), ndim-na, indx+na);
        }
    }
    
    return MORPHO_NIL;
}

/** Creates a new array from a list of values */
objectgpumatrix *object_gpumatrixfromarray(objectarray *array) {
    unsigned int dim[2]={0,1}; // The 1 is to allow for vector arrays.
    unsigned int ndim=0;
    objectgpumatrix *ret=NULL;
    bool success;
    if (gpumatrix_getarraydimensions(array, dim, 2, &ndim)) {
        ret=object_newgpumatrix(dim[0], dim[1], true);
    }
    
    unsigned int indx[2];
    if (ret) for (unsigned int i=0; i<dim[0]; i++) {
        for (unsigned int j=0; j<dim[1]; j++) {
            indx[0]=i; indx[1]=j;
            value f = gpumatrix_getarrayelement(array, ndim, indx);
            success = gpumatrix_setelementfromval(ret,j*dim[0]+i,f);
            if (!success) {
                object_free((object *) ret); return NULL;
            }
        }
    }
    
    return ret;
}
bool gpumatrix_setelementfromval(objectgpumatrix *a, int ind, value val) {
    if (morpho_isnumber(val)) {
        double v;
        morpho_valuetofloat(val, &v);
        GPUcopy_to_device(a->status,&a->elements[ind],&v,sizeof(double));
    } else if (!MORPHO_ISNIL(val)) {
        return false;
    }
    return true;

}
/*
 * Create matrices from lists
 */

/** Recurses into an objectlist to find the dimensions of the array and all child arrays
 * @param[in] list - to search
 * @param[out] dim - array of dimensions to be filled out (must be zero'd before initial call)
 * @param[in] maxdim - maximum number of dimensions
 * @param[out] ndim - number of dimensions of the array */
bool gpumatrix_getlistdimensions(objectlist *list, unsigned int dim[], unsigned int maxdim, unsigned int *ndim) {
    unsigned int m=0;
    /* Store the length */
    if (list->val.count>dim[0]) dim[0]=list->val.count;
    
    for (unsigned int i=0; i<list->val.count; i++) {
        if (MORPHO_ISLIST(list->val.data[i]) && maxdim>0) {
            gpumatrix_getlistdimensions(MORPHO_GETLIST(list->val.data[i]), dim+1, maxdim-1, &m);
        }
    }
    *ndim=m+1;
    
    return true;
}

/** Gets a gpumatrix element from a (potentially nested) list. */
bool gpumatrix_getlistelement(objectlist *list, unsigned int ndim, unsigned int *indx, value *val) {
    value out=MORPHO_NIL;
    objectlist *l=list;
    for (unsigned int i=0; i<ndim; i++) {
        if (indx[i]<l->val.count) {
            out=l->val.data[indx[i]];
            if (i<ndim-1 && MORPHO_ISLIST(out)) l=MORPHO_GETLIST(out);
        } else return false;
    }
    *val=out;
    return true;
}

/* Creates a gpumatrix from a list */
objectgpumatrix *object_gpumatrixfromlist(objectlist *list) {
    unsigned int dim[2]={0,1}; // The 1 is to allow for vector arrays.
    unsigned int ndim=0;
    objectgpumatrix *ret=NULL;
    
    if (gpumatrix_getlistdimensions(list, dim, 2, &ndim)) {
        ret=object_newgpumatrix(dim[0], dim[1], true);
    }
    
    unsigned int indx[2];
    if (ret) for (unsigned int i=0; i<dim[0]; i++) {
        for (unsigned int j=0; j<dim[1]; j++) {
            indx[0]=i; indx[1]=j;
            value f;
            if (gpumatrix_getlistelement(list, ndim, indx, &f) &&
                morpho_isnumber(f)) {
                gpumatrix_setelementfromval(ret,j*dim[0]+i,f);
            } else {
                object_free((object *) ret);
                return NULL;
            }
        }
    }
    
    return ret;
}

/** Creates a gpumatrix from a list of floats */
objectgpumatrix *object_gpumatrixfromfloats(unsigned int nrows, unsigned int ncols, double *list) {
    objectgpumatrix *ret=NULL;
    
    ret=object_newgpumatrix(nrows, ncols, true);
    if (ret) GPUcopy_to_device(ret->status,ret->elements,list,nrows*ncols*sizeof(double));
    
    return ret;
}

/*
 * Clone matrices
 */

/** Clone a gpumatrix */
objectgpumatrix *object_clonegpumatrix(objectgpumatrix *in) {
    objectgpumatrix *new = object_newgpumatrix(in->nrows, in->ncols, false);
    
    if (new) {
        GPUcopy(in->status,in->ncols*in->nrows,in->elements,1,new->elements,1);
    }
    
    return new;
}

/* **********************************************************************
 * gpuMatrix operations
 * ********************************************************************* */

/** @brief Sets a gpumatrix element.
    @returns true if the element is in the range of the gpumatrix, false otherwise */
bool gpumatrix_setelement(objectgpumatrix *gpumatrix, unsigned int row, unsigned int col, double value) {
    if (col<gpumatrix->ncols && row<gpumatrix->nrows) {
        
        GPUcopy_to_device(gpumatrix->status,&gpumatrix->elements[col*gpumatrix->nrows+row],&value,sizeof(double));
        return true;
    }
    return false;
}

/** @brief Gets a gpumatrix element
 *  @returns true if the element is in the range of the gpumatrix, false otherwise */
bool gpumatrix_getelement(objectgpumatrix *gpumatrix, unsigned int row, unsigned int col, double *value) {
    if (col<gpumatrix->ncols && row<gpumatrix->nrows) {
        if (value) GPUcopy_to_host(gpumatrix->status,value,&gpumatrix->elements[col*gpumatrix->nrows+row],sizeof(double));
        return true;
    }
    return false;
}

/** @brief Gets a column's entries
//  *  @param[in] gpumatrix - the gpumatrix
//  *  @param[in] col - column number
//  *  @param[out] v - column entries (gpumatrix->nrows in number)
//  *  @returns true if the element is in the range of the gpumatrix, false otherwise */
// bool gpumatrix_getcolumn(objectgpumatrix *gpumatrix, unsigned int col, double **v) {
//     if (col<gpumatrix->ncols) {
//         *v=&gpumatrix->elements[col*gpumatrix->nrows];
//         return true;
//     }
//     return false;
// }

// /** @brief Sets a column's entries
//  *  @param[in] gpumatrix - the gpumatrix
//  *  @param[in] col - column number
//  *  @param[in] v - column entries (gpumatrix->nrows in number)
//  *  @returns true if the element is in the range of the gpumatrix, false otherwise */
// bool gpumatrix_setcolumn(objectgpumatrix *gpumatrix, unsigned int col, double *v) {
//     if (col<gpumatrix->ncols) {
//         cblas_dcopy(gpumatrix->nrows, v, 1, &gpumatrix->elements[col*gpumatrix->nrows], 1);
//         return true;
//     }
//     return false;
// }

/** @brief Add a vector to a column in a gpumatrix
 *  @param[in] m - the gpumatrix
 *  @param[in] col - column number
 *  @param[in] alpha - scale
 *  @param[out] v - column entries (gpumatrix->nrows in number) [should have m->nrows entries]
 *  @returns true on success */
// bool gpumatrix_addtocolumn(objectgpumatrix *m, unsigned int col, double alpha, double *v) {
//     if (col<m->ncols) {
//         cblas_daxpy(m->nrows, alpha, v, 1, &m->elements[col*m->nrows], 1);
//         return true;
//     }
//     return false;
// }

/* **********************************************************************
 * gpuMatrix arithmetic
 * ********************************************************************* */

objectgpumatrixerror gpumatrix_copy(objectgpumatrix *a, objectgpumatrix *out) {
    if (a->ncols==out->ncols && a->nrows==out->nrows) {
        GPUcopy(a->status, a->ncols*a->nrows, a->elements,1, out->elements, 1);
        return GPUMATRIX_OK;
    }
    return GPUMATRIX_INCMPTBLDIM;
}

/** Performs a + b -> out. */
objectgpumatrixerror gpumatrix_add(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out) {
    if (a->ncols==b->ncols && a->ncols==out->ncols &&
        a->nrows==b->nrows && a->nrows==out->nrows) {
        double scale = 1;
        if (a!=out) GPUcopy(a->status, a->ncols * a->nrows, a->elements, 1, out->elements, 1);
        GPUaxpy(a->status, a->ncols * a->nrows, &scale, b->elements, 1, out->elements, 1);
        return GPUMATRIX_OK;
    }
    return GPUMATRIX_INCMPTBLDIM;
}

/** Performs lambda*a + beta -> out. */
objectgpumatrixerror gpumatrix_addscalar(objectgpumatrix *a, double lambda, double beta, objectgpumatrix *out) {
    if (a->ncols==out->ncols && a->nrows==out->nrows) {
        GPUScalarAddition(a->status,a->elements,beta,out->elements,a->ncols*a->nrows);

        return GPUMATRIX_OK;
    }

    return GPUMATRIX_INCMPTBLDIM;
}

/** Performs a + lambda*b -> a. */
objectgpumatrixerror gpumatrix_accumulate(objectgpumatrix *a, double lambda, objectgpumatrix *b) {
    if (a->ncols==b->ncols && a->nrows==b->nrows ) {
        GPUaxpy(a->status,a->ncols * a->nrows, &lambda, b->elements, 1, a->elements, 1);
        return GPUMATRIX_OK;
    }
    return GPUMATRIX_INCMPTBLDIM;
}

/** Performs a - b -> out */
objectgpumatrixerror gpumatrix_sub(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out) {
    if (a->ncols==b->ncols && a->ncols==out->ncols &&
        a->nrows==b->nrows && a->nrows==out->nrows) {
        double scale = -1;
        if (a!=out) GPUcopy(a->status, a->ncols * a->nrows, a->elements, 1, out->elements, 1);
        GPUaxpy(a->status,a->ncols * a->nrows, &scale, b->elements, 1, out->elements, 1);
        return GPUMATRIX_OK;
    }
    return GPUMATRIX_INCMPTBLDIM;
}

/** Performs a * b -> out */
objectgpumatrixerror gpumatrix_mul(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out) {
    if (a->ncols==b->nrows && a->nrows==out->nrows && b->ncols==out->ncols) {
        double alpha = 1;
        double beta = 0;
        GPUgemm(a->status,a->nrows,b->ncols,a->ncols,&alpha,a->elements,a->nrows, b->elements, b->nrows, &beta, out->elements, out->nrows);
        return GPUMATRIX_OK;
    }
    return GPUMATRIX_INCMPTBLDIM;
}

/** Finds the Frobenius inner product of two matrices  */
objectgpumatrixerror gpumatrix_inner(objectgpumatrix *a, objectgpumatrix *b, double *out) {
    if (a->ncols==b->ncols && a->nrows==b->nrows) {
        *out=cblas_ddot(a->ncols*a->nrows, a->elements, 1, b->elements, 1);
        return GPUMATRIX_OK;
    }
    return GPUMATRIX_INCMPTBLDIM;
}


/** Solves the system a.x = b
 * @param[in] a  lhs
 * @param[in] b  rhs
 * @param[in] out - the solution x
 * @param[out] lu - LU decomposition of a; you must provide an array the same size as a.
 * @param[out] pivot - you must provide an array with the same number of rows as a.
 * @returns objectgpumatrixerror indicating the status; GPUMATRIX_OK indicates success.
 * */
static objectgpumatrixerror gpumatrix_div(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out, double *lu, int *pivot) {
    int n=a->nrows, nrhs = b->ncols, info;
    
    cblas_dcopy(a->ncols * a->nrows, a->elements, 1, lu, 1);
    if (b!=out) cblas_dcopy(b->ncols * b->nrows, b->elements, 1, out->elements, 1);
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, lu, n, pivot, out->elements, n);
#else
    dgesv_(&n, &nrhs, lu, &n, pivot, out->elements, &n, &info);
#endif
    
    return (info==0 ? GPUMATRIX_OK : (info>0 ? GPUMATRIX_SING : GPUMATRIX_INVLD));
}

/** Solves the system a.x = b for small matrices (test with GPUMATRIX_ISSMALL)
 * @warning Uses the C stack for storage, which avoids malloc but can cause stack overflow */
objectgpumatrixerror gpumatrix_divs(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out) {
    if (a->ncols==b->nrows && a->ncols == out->nrows) {
        int pivot[a->nrows];
        double lu[a->nrows*a->ncols];
        
        return gpumatrix_div(a, b, out, lu, pivot);
    }
    return GPUMATRIX_INCMPTBLDIM;
}

/** Solves the system a.x = b for large matrices (test with GPUMATRIX_ISSMALL)  */
objectgpumatrixerror gpumatrix_divl(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out) {
    objectgpumatrixerror ret = GPUMATRIX_ALLOC; // Returned if allocation fails
    if (!(a->ncols==b->nrows && a->ncols == out->nrows)) return GPUMATRIX_INCMPTBLDIM;
    
    int *pivot=MORPHO_MALLOC(sizeof(int)*a->nrows);
    double *lu=MORPHO_MALLOC(sizeof(double)*a->nrows*a->ncols);
    
    if (pivot && lu) ret=gpumatrix_div(a, b, out, lu, pivot);
    
    if (pivot) MORPHO_FREE(pivot);
    if (lu) MORPHO_FREE(lu);
    
    return ret;
}

/** Inverts the gpumatrix a
 * @param[in] a  lhs
 * @param[in] out - the solution x
 * @returns objectgpumatrixerror indicating the status; GPUMATRIX_OK indicates success.
 * */
objectgpumatrixerror gpumatrix_inverse(objectgpumatrix *a, objectgpumatrix *out) {
    int nrows=a->nrows, ncols=a->ncols, info;
    if (!(a->ncols==out->nrows && a->ncols == out->nrows)) return GPUMATRIX_INCMPTBLDIM;
    
    int pivot[nrows];
    
    cblas_dcopy(a->ncols * a->nrows, a->elements, 1, out->elements, 1);
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgetrf(LAPACK_COL_MAJOR, nrows, ncols, out->elements, nrows, pivot);
#else
    dgetrf_(&nrows, &ncols, out->elements, &nrows, pivot, &info);
#endif

    if (info!=0) return (info>0 ? GPUMATRIX_SING : GPUMATRIX_INVLD);
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgetri(LAPACK_COL_MAJOR, nrows, out->elements, nrows, pivot);
#else
    int lwork=nrows*ncols; double work[nrows*ncols];
    dgetri_(&nrows, out->elements, &nrows, pivot, work, &lwork, &info);
#endif
    
    return (info==0 ? GPUMATRIX_OK : (info>0 ? GPUMATRIX_SING : GPUMATRIX_INVLD));
}

/** Sums all elements of a gpumatrix using Kahan summation */
double gpumatrix_sum(objectgpumatrix *a) {
    unsigned int nel=a->ncols*a->nrows;
    double sum=0.0, c=0.0, y,t;
    
    for (unsigned int i=0; i<nel; i++) {
        y=a->elements[i]-c;
        t=sum+y;
        c=(t-sum)-y;
        sum=t;
    }
    return sum;
}

/** Computes the Frobenius norm of a gpumatrix */
double gpumatrix_norm(objectgpumatrix *a) {
    double nrm2=cblas_dnrm2(a->ncols*a->nrows, a->elements, 1);
    return nrm2;
}

/** Transpose a gpumatrix */
objectgpumatrixerror gpumatrix_transpose(objectgpumatrix *a, objectgpumatrix *out) {
    if (!(a->ncols==out->nrows && a->nrows == out->ncols)) return GPUMATRIX_INCMPTBLDIM;

    /* Copy elements a column at a time */
    for (unsigned int i=0; i<a->ncols; i++) {
        cblas_dcopy(a->nrows, a->elements+(i*a->nrows), 1, out->elements+i, a->ncols);
    }
    return GPUMATRIX_OK;
}

/** Calculate the trace of a gpumatrix */
objectgpumatrixerror gpumatrix_trace(objectgpumatrix *a, double *out) {
    if (a->nrows!=a->ncols) return GPUMATRIX_NSQ;
    *out=1.0;
    *out=cblas_ddot(a->nrows, a->elements, a->ncols+1, out, 0);
    
    return GPUMATRIX_OK;
}

/** Scale a gpumatrix */
objectgpumatrixerror gpumatrix_scale(objectgpumatrix *a, double scale) {
    cblas_dscal(a->ncols*a->nrows, scale, a->elements, 1);
    
    return GPUMATRIX_OK;
}

/** Load the indentity gpumatrix*/
objectgpumatrixerror gpumatrix_identity(objectgpumatrix *a) {
    if (a->ncols!=a->nrows) return GPUMATRIX_NSQ;
    for (int i=0; i<a->nrows; i++) for (int j=0; j<a->ncols; j++) a->elements[i+a->nrows*j]=(i==j ? 1.0 : 0.0);
    
    return GPUMATRIX_OK;
}

/** Prints a gpumatrix */
void gpumatrix_print(objectgpumatrix *m) {
    double elements[m->nrows*m->ncols];
    GPUcopy_to_host(m->status,elements,m->elements,sizeof(double)*m->nrows*m->ncols);
    for (int i=0; i<m->nrows; i++) { // Rows run from 0...m
        printf("[ ");
        for (int j=0; j<m->ncols; j++) { // Columns run from 0...k
            printf("%g ", (fabs(elements[j+i*m->ncols])<MORPHO_EPS ? 0 : elements[i+j*m->nrows]));
        }
        printf("]%s", (i<m->nrows-1 ? "\n" : ""));
    }
}

/* **********************************************************************
 * gpuMatrix veneer class
 * ********************************************************************* */


#undef MORPHO_GETMATRIX
#undef MORPHO_ISMATRIX
#define MORPHO_GETMATRIX MORPHO_GETGPUMATRIX
#define MORPHO_ISMATRIX MORPHO_ISGPUMATRIX
#define objectmatrix objectgpumatrix

#define matrix_constructor gpumatrix_constructor

#define object_newmatrix object_newgpumatrix
#define object_matrixfromarray object_gpumatrixfromarray
#define object_matrixfromlist object_gpumatrixfromlist
#define object_matrixfromfloats object_gpumatrixfromfloats
#define object_clonematrix object_clonegpumatrix

#define matrix_slicedim gpumatrix_slicedim
#define matrix_sliceconstructor gpumatrix_sliceconstructor
#define matrix_slicecopy gpumatrix_slicecopy
#define matrix_getelement gpumatrix_getelement
#define matrix_setelement gpumatrix_setelement
#define matrix_getcolumn gpumatrix_getcolumn
#define matrix_setcolumn gpumatrix_setcolumn


#define matrix_add gpumatrix_add
#define matrix_addscalar gpumatrix_addscalar
#define matrix_sub gpumatrix_sub
#define matrix_mul gpumatrix_mul
#define matrix_scale gpumatrix_scale
#define matrix_divs gpumatrix_divs
#define matrix_divl gpumatrix_divl
#define matrix_accumulate gpumatrix_accumulate
#define matrix_inner gpumatrix_inner
#define matrix_sum gpumatrix_sum
#define matrix_norm gpumatrix_norm
#define matrix_transpose gpumatrix_transpose
#define matrix_trace gpumatrix_trace
#define matrix_print gpumatrix_print

#define Matrix_getindex GPUMatrix_getindex
#define Matrix_setindex GPUMatrix_setindex
#define Matrix_setcolumn GPUMatrix_setcolumn
#define Matrix_getcolumn GPUMatrix_getcolumn
#define Matrix_print GPUMatrix_print
#define Matrix_add GPUMatrix_add
#define Matrix_addr GPUMatrix_addr
#define Matrix_sub GPUMatrix_sub
#define Matrix_subr GPUMatrix_subr
#define Matrix_mul GPUMatrix_mul
#define Matrix_mulr GPUMatrix_mulr
#define Matrix_div GPUMatrix_div
#define Matrix_acc GPUMatrix_acc
#define Matrix_inner GPUMatrix_inner
#define Matrix_sum GPUMatrix_sum
#define Matrix_norm GPUMatrix_norm
#define Matrix_transpose GPUMatrix_transpose
#define Matrix_trace GPUMatrix_trace
#define Matrix_enumerate GPUMatrix_enumerate
#define Matrix_count GPUMatrix_count
#define Matrix_dimensions GPUMatrix_dimensions
#define Matrix_clone GPUMatrix_clone


#include "matrixveneer2.h"

#undef MORPHO_GETMATRIX
#undef MORPHO_ISMATRIX


#undef objectmatrix

#undef matrix_constructor

#undef object_newmatrix
#undef object_matrixfromarray
#undef object_matrixfromlist
#undef object_matrixfromfloats
#undef object_clonematrix

#undef matrix_slicedim
#undef matrix_sliceconstructor
#undef matrix_slicecopy
#undef matrix_getelement
#undef matrix_setelement
#undef matrix_getcolumn
#undef matrix_setcolumn

#undef matrix_add
#undef matrix_addscalar
#undef matrix_sub
#undef matrix_mul
#undef matrix_scale
#undef matrix_divs
#undef matrix_divl
#undef matrix_accumulate
#undef matrix_inner
#undef matrix_sum
#undef matrix_norm
#undef matrix_transpose
#undef matrix_trace
#undef matrix_print


#undef Matrix_setindex
#undef Matrix_getindex
#undef Matrix_setcolumn
#undef Matrix_getcolumn
#undef Matrix_print
#undef Matrix_add
#undef Matrix_addr
#undef Matrix_sub
#undef Matrix_subr
#undef Matrix_mul
#undef Matrix_mulr
#undef Matrix_div
#undef Matrix_acc
#undef Matrix_inner
#undef Matrix_sum
#undef Matrix_norm
#undef Matrix_transpose
#undef Matrix_trace
#undef Matrix_enumerate
#undef Matrix_count
#undef Matrix_dimensions
#undef Matrix_clone

#define MORPHO_ISMATRIX(val) object_istype(val, OBJECT_MATRIX)
#define MORPHO_GETMATRIX(val)   ((objectmatrix *) MORPHO_GETOBJECT(val))


MORPHO_BEGINCLASS(GPUMatrix)
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, GPUMatrix_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, GPUMatrix_setindex, BUILTIN_FLAGSEMPTY),
//MORPHO_METHOD(GPUMATRIX_GETCOLUMN_METHOD, GPUMatrix_getcolumn, BUILTIN_FLAGSEMPTY),
//MORPHO_METHOD(GPUMATRIX_SETCOLUMN_METHOD, GPUMatrix_setcolumn, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, GPUMatrix_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADD_METHOD, GPUMatrix_add, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADDR_METHOD, GPUMatrix_addr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUB_METHOD, GPUMatrix_sub, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUBR_METHOD, GPUMatrix_subr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_MUL_METHOD, GPUMatrix_mul, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_MULR_METHOD, GPUMatrix_mulr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_DIV_METHOD, GPUMatrix_div, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ACC_METHOD, GPUMatrix_acc, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUMATRIX_INNER_METHOD, GPUMatrix_inner, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUM_METHOD, GPUMatrix_sum, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUMATRIX_NORM_METHOD, GPUMatrix_norm, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUMATRIX_TRANSPOSE_METHOD, GPUMatrix_transpose, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUMATRIX_TRACE_METHOD, GPUMatrix_trace, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, GPUMatrix_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, GPUMatrix_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUMATRIX_DIMENSIONS_METHOD, GPUMatrix_dimensions, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, GPUMatrix_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************* */

void gpumatrix_initialize(void) {
    objectgpumatrixtype=object_addtype(&objectgpumatrixdefn);
    
    builtin_addfunction(GPUMATRIX_CLASSNAME, gpumatrix_constructor, BUILTIN_FLAGSEMPTY);
    
    value gpumatrixclass=builtin_addclass(GPUMATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(GPUMatrix), MORPHO_NIL);
    object_setveneerclass(OBJECT_GPUMATRIX, gpumatrixclass);
    
    morpho_defineerror(GPUMATRIX_INDICESOUTSIDEBOUNDS, ERROR_HALT, GPUMATRIX_INDICESOUTSIDEBOUNDS_MSG);
    morpho_defineerror(GPUMATRIX_INVLDINDICES, ERROR_HALT, GPUMATRIX_INVLDINDICES_MSG);
    morpho_defineerror(GPUMATRIX_INVLDNUMINDICES, ERROR_HALT, GPUMATRIX_INVLDNUMINDICES_MSG);
    morpho_defineerror(GPUMATRIX_CONSTRUCTOR, ERROR_HALT, GPUMATRIX_CONSTRUCTOR_MSG);
    morpho_defineerror(GPUMATRIX_INVLDARRAYINIT, ERROR_HALT, GPUMATRIX_INVLDARRAYINIT_MSG);
    morpho_defineerror(GPUMATRIX_ARITHARGS, ERROR_HALT, GPUMATRIX_ARITHARGS_MSG);
    morpho_defineerror(GPUMATRIX_INCOMPATIBLEMATRICES, ERROR_HALT, GPUMATRIX_INCOMPATIBLEMATRICES_MSG);
    morpho_defineerror(GPUMATRIX_SINGULAR, ERROR_HALT, GPUMATRIX_SINGULAR_MSG);
    morpho_defineerror(GPUMATRIX_NOTSQ, ERROR_HALT, GPUMATRIX_NOTSQ_MSG);
    morpho_defineerror(GPUMATRIX_SETCOLARGS, ERROR_HALT, GPUMATRIX_SETCOLARGS_MSG);
}
