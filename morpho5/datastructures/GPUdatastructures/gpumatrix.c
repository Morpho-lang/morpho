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

/** Function object definitions */
size_t objectgpumatrix_sizefn(object *obj) {
    return sizeof(objectgpumatrix)+sizeof(double) *
            ((objectgpumatrix *) obj)->ncols *
            ((objectgpumatrix *) obj)->nrows;
}

void objectgpumatrix_printfn(object *obj) {
    printf("<gpuMatrix>");
}

objecttypedefn objectgpumatrixdefn = {
    .printfn=objectgpumatrix_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectgpumatrix_sizefn
};

/** Creates a gpumatrix object */
objectgpumatrix *object_newgpumatrix(unsigned int nrows, unsigned int ncols, bool zero) {
    unsigned int nel = nrows*ncols;
    objectgpumatrix *new = (objectgpumatrix *) object_new(sizeof(objectgpumatrix)+nel*sizeof(double), OBJECT_GPUMATRIX);
    
    if (new) {
        new->ncols=ncols;
        new->nrows=nrows;
        new->elements=new->gpumatrixdata;
        if (zero) {
            memset(new->elements, 0, sizeof(double)*nel);
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
    
    if (gpumatrix_getarraydimensions(array, dim, 2, &ndim)) {
        ret=object_newgpumatrix(dim[0], dim[1], true);
    }
    
    unsigned int indx[2];
    if (ret) for (unsigned int i=0; i<dim[0]; i++) {
        for (unsigned int j=0; j<dim[1]; j++) {
            indx[0]=i; indx[1]=j;
            value f = gpumatrix_getarrayelement(array, ndim, indx);
            if (morpho_isnumber(f)) {
                morpho_valuetofloat(f, &ret->elements[j*dim[0]+i]);
            } else if (!MORPHO_ISNIL(f)) {
                object_free((object *) ret); return NULL;
            }
        }
    }
    
    return ret;
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
                morpho_valuetofloat(f, &ret->elements[j*dim[0]+i]);
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
    if (ret) cblas_dcopy(ncols*nrows, list, 1, ret->elements, 1);
    
    return ret;
}

/*
 * Clone matrices
 */

/** Clone a gpumatrix */
objectgpumatrix *object_clonegpumatrix(objectgpumatrix *in) {
    objectgpumatrix *new = object_newgpumatrix(in->nrows, in->ncols, false);
    
    if (new) {
        cblas_dcopy(in->ncols * in->nrows, in->elements, 1, new->elements, 1);
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
        gpumatrix->elements[col*gpumatrix->nrows+row]=value;
        return true;
    }
    return false;
}

/** @brief Gets a gpumatrix element
 *  @returns true if the element is in the range of the gpumatrix, false otherwise */
bool gpumatrix_getelement(objectgpumatrix *gpumatrix, unsigned int row, unsigned int col, double *value) {
    if (col<gpumatrix->ncols && row<gpumatrix->nrows) {
        if (value) *value=gpumatrix->elements[col*gpumatrix->nrows+row];
        return true;
    }
    return false;
}

/** @brief Gets a column's entries
 *  @param[in] gpumatrix - the gpumatrix
 *  @param[in] col - column number
 *  @param[out] v - column entries (gpumatrix->nrows in number)
 *  @returns true if the element is in the range of the gpumatrix, false otherwise */
bool gpumatrix_getcolumn(objectgpumatrix *gpumatrix, unsigned int col, double **v) {
    if (col<gpumatrix->ncols) {
        *v=&gpumatrix->elements[col*gpumatrix->nrows];
        return true;
    }
    return false;
}

/** @brief Sets a column's entries
 *  @param[in] gpumatrix - the gpumatrix
 *  @param[in] col - column number
 *  @param[in] v - column entries (gpumatrix->nrows in number)
 *  @returns true if the element is in the range of the gpumatrix, false otherwise */
bool gpumatrix_setcolumn(objectgpumatrix *gpumatrix, unsigned int col, double *v) {
    if (col<gpumatrix->ncols) {
        cblas_dcopy(gpumatrix->nrows, v, 1, &gpumatrix->elements[col*gpumatrix->nrows], 1);
        return true;
    }
    return false;
}

/** @brief Add a vector to a column in a gpumatrix
 *  @param[in] m - the gpumatrix
 *  @param[in] col - column number
 *  @param[in] alpha - scale
 *  @param[out] v - column entries (gpumatrix->nrows in number) [should have m->nrows entries]
 *  @returns true on success */
bool gpumatrix_addtocolumn(objectgpumatrix *m, unsigned int col, double alpha, double *v) {
    if (col<m->ncols) {
        cblas_daxpy(m->nrows, alpha, v, 1, &m->elements[col*m->nrows], 1);
        return true;
    }
    return false;
}

/* **********************************************************************
 * gpuMatrix arithmetic
 * ********************************************************************* */

objectgpumatrixerror gpumatrix_copy(objectgpumatrix *a, objectgpumatrix *out) {
    if (a->ncols==out->ncols && a->nrows==out->nrows) {
        cblas_dcopy(a->ncols * a->nrows, a->elements, 1, out->elements, 1);
        return GPUMATRIX_OK;
    }
    return GPUMATRIX_INCMPTBLDIM;
}

/** Performs a + b -> out. */
objectgpumatrixerror gpumatrix_add(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out) {
    if (a->ncols==b->ncols && a->ncols==out->ncols &&
        a->nrows==b->nrows && a->nrows==out->nrows) {
        if (a!=out) cblas_dcopy(a->ncols * a->nrows, a->elements, 1, out->elements, 1);
        cblas_daxpy(a->ncols * a->nrows, 1.0, b->elements, 1, out->elements, 1);
        return GPUMATRIX_OK;
    }
    return GPUMATRIX_INCMPTBLDIM;
}

/** Performs lambda*a + beta -> out. */
objectgpumatrixerror gpumatrix_addscalar(objectgpumatrix *a, double lambda, double beta, objectgpumatrix *out) {
    if (a->ncols==out->ncols && a->nrows==out->nrows) {
        for (unsigned int i=0; i<out->nrows*out->ncols; i++) {
            out->elements[i]=lambda*a->elements[i]+beta;
        }
        return GPUMATRIX_OK;
    }

    return GPUMATRIX_INCMPTBLDIM;
}

/** Performs a + lambda*b -> a. */
objectgpumatrixerror gpumatrix_accumulate(objectgpumatrix *a, double lambda, objectgpumatrix *b) {
    if (a->ncols==b->ncols && a->nrows==b->nrows ) {
        cblas_daxpy(a->ncols * a->nrows, lambda, b->elements, 1, a->elements, 1);
        return GPUMATRIX_OK;
    }
    return GPUMATRIX_INCMPTBLDIM;
}

/** Performs a - b -> out */
objectgpumatrixerror gpumatrix_sub(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out) {
    if (a->ncols==b->ncols && a->ncols==out->ncols &&
        a->nrows==b->nrows && a->nrows==out->nrows) {
        if (a!=out) cblas_dcopy(a->ncols * a->nrows, a->elements, 1, out->elements, 1);
        cblas_daxpy(a->ncols * a->nrows, -1.0, b->elements, 1, out->elements, 1);
        return GPUMATRIX_OK;
    }
    return GPUMATRIX_INCMPTBLDIM;
}

/** Performs a * b -> out */
objectgpumatrixerror gpumatrix_mul(objectgpumatrix *a, objectgpumatrix *b, objectgpumatrix *out) {
    if (a->ncols==b->nrows && a->nrows==out->nrows && b->ncols==out->ncols) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, a->nrows, b->ncols, a->ncols, 1.0, a->elements, a->nrows, b->elements, b->nrows, 0.0, out->elements, out->nrows);
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
    for (int i=0; i<m->nrows; i++) { // Rows run from 0...m
        printf("[ ");
        for (int j=0; j<m->ncols; j++) { // Columns run from 0...k
            double v;
            gpumatrix_getelement(m, i, j, &v);
            printf("%g ", (fabs(v)<MORPHO_EPS ? 0 : v));
        }
        printf("]%s", (i<m->nrows-1 ? "\n" : ""));
    }
}

/* **********************************************************************
 * gpuMatrix veneer class
 * ********************************************************************* */
#define MORPHO_GPUMATRIX_TYPE gpu
#define MORPHO_GPUMATRIX_TYPE_CAP GPU

#include "matrixveneer.h"

#undef MORPHO_GPUMATRIX_TYPE
#undef MORPHO_GPUMATRIX_TYPE_CAP


MORPHO_BEGINCLASS(gpuMatrix)
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, gpuMatrix_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, gpuMatrix_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUMATRIX_GETCOLUMN_METHOD, gpuMatrix_getcolumn, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUMATRIX_SETCOLUMN_METHOD, gpuMatrix_setcolumn, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, gpuMatrix_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADD_METHOD, gpuMatrix_add, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADDR_METHOD, gpuMatrix_addr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUB_METHOD, gpuMatrix_sub, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUBR_METHOD, gpuMatrix_subr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_MUL_METHOD, gpuMatrix_mul, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_MULR_METHOD, gpuMatrix_mulr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_DIV_METHOD, gpuMatrix_div, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ACC_METHOD, gpuMatrix_acc, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUMATRIX_INNER_METHOD, gpuMatrix_inner, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUM_METHOD, gpuMatrix_sum, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUMATRIX_NORM_METHOD, gpuMatrix_norm, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUMATRIX_TRANSPOSE_METHOD, gpuMatrix_transpose, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUMATRIX_TRACE_METHOD, gpuMatrix_trace, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, gpuMatrix_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, gpuMatrix_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(GPUMATRIX_DIMENSIONS_METHOD, gpuMatrix_dimensions, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, gpuMatrix_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************* */

void gpugpumatrix_initialize(void) {
    objectgpumatrixtype=object_addtype(&objectgpumatrixdefn);
    
    builtin_addfunction(GPUMATRIX_CLASSNAME, gpumatrix_constructor, BUILTIN_FLAGSEMPTY);
    
    value gpumatrixclass=builtin_addclass(GPUMATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(gpuMatrix), MORPHO_NIL);
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
