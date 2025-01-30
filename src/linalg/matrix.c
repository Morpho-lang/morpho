/** @file matrix.c
 *  @author T J Atherton
 *
 *  @brief Veneer class over the objectmatrix type that interfaces with blas and lapack
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_LINALG

#include <string.h>
#include "morpho.h"
#include "classes.h"

#include "matrix.h"
#include "sparse.h"
#include "format.h"

/* **********************************************************************
 * Matrix objects
 * ********************************************************************** */

objecttype objectmatrixtype;

/** Function object definitions */
size_t objectmatrix_sizefn(object *obj) {
    return sizeof(objectmatrix)+sizeof(double) *
            ((objectmatrix *) obj)->ncols *
            ((objectmatrix *) obj)->nrows;
}

void objectmatrix_printfn(object *obj, void *v) {
    morpho_printf(v, "<Matrix>");
}

objecttypedefn objectmatrixdefn = {
    .printfn=objectmatrix_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectmatrix_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/** Creates a matrix object */
objectmatrix *object_newmatrix(unsigned int nrows, unsigned int ncols, bool zero) {
    unsigned int nel = nrows*ncols;
    objectmatrix *new = (objectmatrix *) object_new(sizeof(objectmatrix)+nel*sizeof(double), OBJECT_MATRIX);
    
    if (new) {
        new->ncols=ncols;
        new->nrows=nrows;
        new->elements=new->matrixdata;
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

void matrix_raiseerror(vm *v, objectmatrixerror err) {
    switch(err) {
        case MATRIX_OK: break;
        case MATRIX_INCMPTBLDIM: morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES); break;
        case MATRIX_SING: morpho_runtimeerror(v, MATRIX_SINGULAR); break;
        case MATRIX_INVLD: morpho_runtimeerror(v, MATRIX_INVLDARRAYINIT); break;
        case MATRIX_BNDS: morpho_runtimeerror(v, MATRIX_INDICESOUTSIDEBOUNDS); break;
        case MATRIX_NSQ: morpho_runtimeerror(v, MATRIX_NOTSQ); break;
        case MATRIX_FAILED: morpho_runtimeerror(v, MATRIX_OPFAILED); break;
        case MATRIX_ALLOC: morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); break;
    }
}

/** Recurses into an objectarray to find the dimensions of the array and all child arrays
 * @param[in] array - to search
 * @param[out] dim - array of dimensions to be filled out (must be zero'd before initial call)
 * @param[in] maxdim - maximum number of dimensions
 * @param[out] ndim - number of dimensions of the array */
bool matrix_getarraydimensions(objectarray *array, unsigned int dim[], unsigned int maxdim, unsigned int *ndim) {
    unsigned int n=0, m=0;
    for (n=0; n<maxdim && n<array->ndim; n++) {
        int k=MORPHO_GETINTEGERVALUE(array->data[n]);
        if (k>dim[n]) dim[n]=k;
    }
    
    if (maxdim<array->ndim) return false;
    
    for (unsigned int i=array->ndim; i<array->ndim+array->nelements; i++) {
        if (MORPHO_ISARRAY(array->data[i])) {
            if (!matrix_getarraydimensions(MORPHO_GETARRAY(array->data[i]), dim+n, maxdim-n, &m)) return false;
        }
    }
    *ndim=n+m;
    
    return true;
}

/** Looks up an array element recursively if necessary */
value matrix_getarrayelement(objectarray *array, unsigned int ndim, unsigned int *indx) {
    unsigned int na=array->ndim;
    value out;
    
    if (array_getelement(array, na, indx, &out)==ARRAY_OK) {
        if (ndim==na) return out;
        if (MORPHO_ISARRAY(out)) {
            return matrix_getarrayelement(MORPHO_GETARRAY(out), ndim-na, indx+na);
        }
    }
    
    return MORPHO_NIL;
}

/** Creates a new array from a list of values */
objectmatrix *object_matrixfromarray(objectarray *array) {
    unsigned int dim[2]={0,1}; // The 1 is to allow for vector arrays.
    unsigned int ndim=0;
    objectmatrix *ret=NULL;
    
    if (matrix_getarraydimensions(array, dim, 2, &ndim)) {
        ret=object_newmatrix(dim[0], dim[1], true);
    }
    
    unsigned int indx[2];
    if (ret) for (unsigned int i=0; i<dim[0]; i++) {
        for (unsigned int j=0; j<dim[1]; j++) {
            indx[0]=i; indx[1]=j;
            value f = matrix_getarrayelement(array, ndim, indx);
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
bool matrix_getlistdimensions(objectlist *list, unsigned int dim[], unsigned int maxdim, unsigned int *ndim) {
    unsigned int m=0;
    
    if (maxdim==0) return false;
    
    /* Store the length */
    if (list->val.count>dim[0]) dim[0]=list->val.count;
    
    for (unsigned int i=0; i<list->val.count; i++) {
        if (MORPHO_ISLIST(list->val.data[i]) && maxdim>0) {
            matrix_getlistdimensions(MORPHO_GETLIST(list->val.data[i]), dim+1, maxdim-1, &m);
        }
    }
    *ndim=m+1;
    
    return true;
}

/** Gets a matrix element from a (potentially nested) list. */
bool matrix_getlistelement(objectlist *list, unsigned int ndim, unsigned int *indx, value *val) {
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

/* Creates a matrix from a list */
objectmatrix *object_matrixfromlist(objectlist *list) {
    unsigned int dim[2]={0,1}; // The 1 is to allow for vector arrays.
    unsigned int ndim=0;
    objectmatrix *ret=NULL;
    
    if (matrix_getlistdimensions(list, dim, 2, &ndim)) {
        ret=object_newmatrix(dim[0], dim[1], true);
    }
    
    if (ndim>2) return false;
    
    unsigned int indx[2];
    if (ret) for (unsigned int i=0; i<dim[0]; i++) {
        for (unsigned int j=0; j<dim[1]; j++) {
            indx[0]=i; indx[1]=j;
            value f;
            if (matrix_getlistelement(list, ndim, indx, &f) &&
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

/** Creates a matrix from a list of floats */
objectmatrix *object_matrixfromfloats(unsigned int nrows, unsigned int ncols, double *list) {
    objectmatrix *ret=NULL;
    
    ret=object_newmatrix(nrows, ncols, true);
    if (ret) cblas_dcopy(ncols*nrows, list, 1, ret->elements, 1);
    
    return ret;
}

/*
 * Clone matrices
 */

/** Clone a matrix */
objectmatrix *object_clonematrix(objectmatrix *in) {
    objectmatrix *new = object_newmatrix(in->nrows, in->ncols, false);
    
    if (new) {
        cblas_dcopy(in->ncols * in->nrows, in->elements, 1, new->elements, 1);
    }
    
    return new;
}

/* **********************************************************************
 * Matrix operations
 * ********************************************************************* */

/** @brief Sets a matrix element.
    @returns true if the element is in the range of the matrix, false otherwise */
bool matrix_setelement(objectmatrix *matrix, unsigned int row, unsigned int col, double value) {
    if (col<matrix->ncols && row<matrix->nrows) {
        matrix->elements[col*matrix->nrows+row]=value;
        return true;
    }
    return false;
}

/** @brief Gets a matrix element
 *  @returns true if the element is in the range of the matrix, false otherwise */
bool matrix_getelement(objectmatrix *matrix, unsigned int row, unsigned int col, double *value) {
    if (col<matrix->ncols && row<matrix->nrows) {
        if (value) *value=matrix->elements[col*matrix->nrows+row];
        return true;
    }
    return false;
}

/** @brief Gets a column's entries
 *  @param[in] matrix - the matrix
 *  @param[in] col - column number
 *  @param[out] v - column entries (matrix->nrows in number)
 *  @returns true if the element is in the range of the matrix, false otherwise */
bool matrix_getcolumn(objectmatrix *matrix, unsigned int col, double **v) {
    if (col<matrix->ncols) {
        *v=&matrix->elements[col*matrix->nrows];
        return true;
    }
    return false;
}

/** @brief Sets a column's entries
 *  @param[in] matrix - the matrix
 *  @param[in] col - column number
 *  @param[in] v - column entries (matrix->nrows in number)
 *  @returns true if the element is in the range of the matrix, false otherwise */
bool matrix_setcolumn(objectmatrix *matrix, unsigned int col, double *v) {
    if (col<matrix->ncols) {
        cblas_dcopy(matrix->nrows, v, 1, &matrix->elements[col*matrix->nrows], 1);
        return true;
    }
    return false;
}

/** @brief Add a vector to a column in a matrix
 *  @param[in] m - the matrix
 *  @param[in] col - column number
 *  @param[in] alpha - scale
 *  @param[out] v - column entries (matrix->nrows in number) [should have m->nrows entries]
 *  @returns true on success */
bool matrix_addtocolumn(objectmatrix *m, unsigned int col, double alpha, double *v) {
    if (col<m->ncols) {
        cblas_daxpy(m->nrows, alpha, v, 1, &m->elements[col*m->nrows], 1);
        return true;
    }
    return false;
}

/* **********************************************************************
 * Matrix arithmetic
 * ********************************************************************* */

/** Copies one matrix to another */
unsigned int matrix_countdof(objectmatrix *a) {
    return a->ncols*a->nrows;
}

/** Copies one matrix to another */
objectmatrixerror matrix_copy(objectmatrix *a, objectmatrix *out) {
    if (a->ncols==out->ncols && a->nrows==out->nrows) {
        cblas_dcopy(a->ncols * a->nrows, a->elements, 1, out->elements, 1);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Copies a matrix to another at an arbitrary point */
objectmatrixerror matrix_copyat(objectmatrix *a, objectmatrix *out, int row0, int col0) {
    if (col0+a->ncols<=out->ncols && row0+a->nrows<=out->nrows) {
        for (int j=0; j<a->ncols; j++) {
            for (int i=0; i<a->nrows; i++) {
                double value;
                if (!matrix_getelement(a, i, j, &value)) return MATRIX_BNDS;
                if (!matrix_setelement(out, row0+i, col0+j, value)) return MATRIX_BNDS;
            }
        }
        
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Performs a + b -> out. */
objectmatrixerror matrix_add(objectmatrix *a, objectmatrix *b, objectmatrix *out) {
    if (a->ncols==b->ncols && a->ncols==out->ncols &&
        a->nrows==b->nrows && a->nrows==out->nrows) {
        if (a!=out) cblas_dcopy(a->ncols * a->nrows, a->elements, 1, out->elements, 1);
        cblas_daxpy(a->ncols * a->nrows, 1.0, b->elements, 1, out->elements, 1);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Performs lambda*a + beta -> out. */
objectmatrixerror matrix_addscalar(objectmatrix *a, double lambda, double beta, objectmatrix *out) {
    if (a->ncols==out->ncols && a->nrows==out->nrows) {
        for (unsigned int i=0; i<out->nrows*out->ncols; i++) {
            out->elements[i]=lambda*a->elements[i]+beta;
        }
        return MATRIX_OK;
    }

    return MATRIX_INCMPTBLDIM;
}

/** Performs a + lambda*b -> a. */
objectmatrixerror matrix_accumulate(objectmatrix *a, double lambda, objectmatrix *b) {
    if (a->ncols==b->ncols && a->nrows==b->nrows ) {
        cblas_daxpy(a->ncols * a->nrows, lambda, b->elements, 1, a->elements, 1);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Performs a - b -> out */
objectmatrixerror matrix_sub(objectmatrix *a, objectmatrix *b, objectmatrix *out) {
    if (a->ncols==b->ncols && a->ncols==out->ncols &&
        a->nrows==b->nrows && a->nrows==out->nrows) {
        if (a!=out) cblas_dcopy(a->ncols * a->nrows, a->elements, 1, out->elements, 1);
        cblas_daxpy(a->ncols * a->nrows, -1.0, b->elements, 1, out->elements, 1);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Performs a * b -> out */
objectmatrixerror matrix_mul(objectmatrix *a, objectmatrix *b, objectmatrix *out) {
    if (a->ncols==b->nrows && a->nrows==out->nrows && b->ncols==out->ncols) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, a->nrows, b->ncols, a->ncols, 1.0, a->elements, a->nrows, b->elements, b->nrows, 0.0, out->elements, out->nrows);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Finds the Frobenius inner product of two matrices  */
objectmatrixerror matrix_inner(objectmatrix *a, objectmatrix *b, double *out) {
    if (a->ncols==b->ncols && a->nrows==b->nrows) {
        *out=cblas_ddot(a->ncols*a->nrows, a->elements, 1, b->elements, 1);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Computes the outer product of two matrices  */
objectmatrixerror matrix_outer(objectmatrix *a, objectmatrix *b, objectmatrix *out) {
    int m=a->nrows*a->ncols, n=b->nrows*b->ncols;
    if (m==out->nrows && n==out->ncols) {
        cblas_dger(CblasColMajor, m, n, 1, a->elements, 1, b->elements, 1, out->elements, out->nrows);
        return MATRIX_OK;
    }
    return MATRIX_INCMPTBLDIM;
}

/** Solves the system a.x = b
 * @param[in] a  lhs
 * @param[in] b  rhs
 * @param[in] out - the solution x
 * @param[out] lu - LU decomposition of a; you must provide an array the same size as a.
 * @param[out] pivot - you must provide an array with the same number of rows as a.
 * @returns objectmatrixerror indicating the status; MATRIX_OK indicates success.
 * */
static objectmatrixerror matrix_div(objectmatrix *a, objectmatrix *b, objectmatrix *out, double *lu, int *pivot) {
    int n=a->nrows, nrhs = b->ncols, info;
    
    cblas_dcopy(a->ncols * a->nrows, a->elements, 1, lu, 1);
    if (b!=out) cblas_dcopy(b->ncols * b->nrows, b->elements, 1, out->elements, 1);
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, lu, n, pivot, out->elements, n);
#else
    dgesv_(&n, &nrhs, lu, &n, pivot, out->elements, &n, &info);
#endif
    
    return (info==0 ? MATRIX_OK : (info>0 ? MATRIX_SING : MATRIX_INVLD));
}

/** Solves the system a.x = b for small matrices (test with MATRIX_ISSMALL)
 * @warning Uses the C stack for storage, which avoids malloc but can cause stack overflow */
objectmatrixerror matrix_divs(objectmatrix *a, objectmatrix *b, objectmatrix *out) {
    if (a->ncols==b->nrows && a->ncols == out->nrows) {
        int pivot[a->nrows];
        double lu[a->nrows*a->ncols];
        
        return matrix_div(a, b, out, lu, pivot);
    }
    return MATRIX_INCMPTBLDIM;
}

/** Solves the system a.x = b for large matrices (test with MATRIX_ISSMALL)  */
objectmatrixerror matrix_divl(objectmatrix *a, objectmatrix *b, objectmatrix *out) {
    objectmatrixerror ret = MATRIX_ALLOC; // Returned if allocation fails
    if (!(a->ncols==b->nrows && a->ncols == out->nrows)) return MATRIX_INCMPTBLDIM;
    
    int *pivot=MORPHO_MALLOC(sizeof(int)*a->nrows);
    double *lu=MORPHO_MALLOC(sizeof(double)*a->nrows*a->ncols);
    
    if (pivot && lu) ret=matrix_div(a, b, out, lu, pivot);
    
    if (pivot) MORPHO_FREE(pivot);
    if (lu) MORPHO_FREE(lu);
    
    return ret;
}

/** Inverts the matrix a
 * @param[in] a  lhs
 * @param[in] out - the solution x
 * @returns objectmatrixerror indicating the status; MATRIX_OK indicates success.
 * */
objectmatrixerror matrix_inverse(objectmatrix *a, objectmatrix *out) {
    int nrows=a->nrows, ncols=a->ncols, info;
    if (!(a->ncols==out->nrows && a->ncols == out->nrows)) return MATRIX_INCMPTBLDIM;
    
    int pivot[nrows];
    
    cblas_dcopy(a->ncols * a->nrows, a->elements, 1, out->elements, 1);
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgetrf(LAPACK_COL_MAJOR, nrows, ncols, out->elements, nrows, pivot);
#else
    dgetrf_(&nrows, &ncols, out->elements, &nrows, pivot, &info);
#endif

    if (info!=0) return (info>0 ? MATRIX_SING : MATRIX_INVLD);
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgetri(LAPACK_COL_MAJOR, nrows, out->elements, nrows, pivot);
#else
    int lwork=nrows*ncols; double work[nrows*ncols];
    dgetri_(&nrows, out->elements, &nrows, pivot, work, &lwork, &info);
#endif
    
    return (info==0 ? MATRIX_OK : (info>0 ? MATRIX_SING : MATRIX_INVLD));
}

/** Compute eigenvalues and eigenvectors of a matrix
 * @param[in] a - an objectmatrix to diagonalize of size n
 * @param[out] wr - a buffer of size n will hold the real part of the eigenvalues on exit
 * @param[out] wi - a buffer of size n will hold the imag part of the eigenvalues on exit
 * @param[out] vec - (optional) will be filled out with eigenvectors as columns (should be of size n)
 * @returns an error code or MATRIX_OK on success */
objectmatrixerror matrix_eigensystem(objectmatrix *a, double *wr, double *wi, objectmatrix *vec) {
    int info, n=a->nrows;
    if (a->nrows!=a->ncols) return MATRIX_NSQ;
    if (vec && ((a->nrows!=vec->nrows) || (a->nrows!=vec->ncols))) return MATRIX_INCMPTBLDIM;
    
    // Copy a to prevent destruction
    size_t size = ((size_t) n) * ((size_t) n) * sizeof(double);
    double *acopy=MORPHO_MALLOC(size);
    if (!acopy) return MATRIX_ALLOC;
    cblas_dcopy(n*n, a->elements, 1, acopy, 1);
    
#ifdef MORPHO_LINALG_USE_LAPACKE
    info=LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', (vec ? 'V' : 'N'), n, acopy, n, wr, wi, NULL, n, (vec ? vec->elements : NULL), n);
#else
    int lwork=4*n; double work[4*n];
    dgeev_("N", (vec ? "V" : "N"), &n, acopy, &n, wr, wi, NULL, &n, (vec ? vec->elements : NULL), &n, work, &lwork, &info);
#endif
    
    if (acopy) MORPHO_FREE(acopy); // Free up buffer
        
    if (info!=0) return (info>0 ? MATRIX_FAILED : MATRIX_INVLD);
    
    return MATRIX_OK;
}


/** Sums all elements of a matrix using Kahan summation */
double matrix_sum(objectmatrix *a) {
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

/** Norms */

/** Computes the Frobenius norm of a matrix */
double matrix_norm(objectmatrix *a) {
    double nrm2=cblas_dnrm2(a->ncols*a->nrows, a->elements, 1);
    return nrm2;
}

/** Computes the L1 norm of a matrix */
double matrix_L1norm(objectmatrix *a) {
    unsigned int nel=a->ncols*a->nrows;
    double sum=0.0, c=0.0, y,t;
    
    for (unsigned int i=0; i<nel; i++) {
        y=fabs(a->elements[i])-c;
        t=sum+y;
        c=(t-sum)-y;
        sum=t;
    }
    return sum;
}

/** Computes the Ln norm of a matrix */
double matrix_Lnnorm(objectmatrix *a, double n) {
    unsigned int nel=a->ncols*a->nrows;
    double sum=0.0, c=0.0, y,t;
    
    for (unsigned int i=0; i<nel; i++) {
        y=pow(a->elements[i],n)-c;
        t=sum+y;
        c=(t-sum)-y;
        sum=t;
    }
    return pow(sum,1.0/n);
}

/** Computes the Linf norm of a matrix */
double matrix_Linfnorm(objectmatrix *a) {
    unsigned int nel=a->ncols*a->nrows;
    double max=0.0;
    
    for (unsigned int i=0; i<nel; i++) {
        double y=fabs(a->elements[i]);
        if (y>max) max=y;
    }
    return max;
}

/** Transpose a matrix */
objectmatrixerror matrix_transpose(objectmatrix *a, objectmatrix *out) {
    if (!(a->ncols==out->nrows && a->nrows == out->ncols)) return MATRIX_INCMPTBLDIM;

    /* Copy elements a column at a time */
    for (unsigned int i=0; i<a->ncols; i++) {
        cblas_dcopy(a->nrows, a->elements+(i*a->nrows), 1, out->elements+i, a->ncols);
    }
    return MATRIX_OK;
}

/** Calculate the trace of a matrix */
objectmatrixerror matrix_trace(objectmatrix *a, double *out) {
    if (a->nrows!=a->ncols) return MATRIX_NSQ;
    *out=1.0;
    *out=cblas_ddot(a->nrows, a->elements, a->ncols+1, out, 0);
    
    return MATRIX_OK;
}

/** Scale a matrix */
objectmatrixerror matrix_scale(objectmatrix *a, double scale) {
    cblas_dscal(a->ncols*a->nrows, scale, a->elements, 1);
    
    return MATRIX_OK;
}

/** Load the indentity matrix*/
objectmatrixerror matrix_identity(objectmatrix *a) {
    if (a->ncols!=a->nrows) return MATRIX_NSQ;
    memset(a->elements, 0, sizeof(double)*a->nrows*a->ncols);
    for (int i=0; i<a->nrows; i++) a->elements[i+a->nrows*i]=1.0;
    return MATRIX_OK;
}

/** Prints a matrix */
void matrix_print(vm *v, objectmatrix *m) {
    for (int i=0; i<m->nrows; i++) { // Rows run from 0...m
        morpho_printf(v, "[ ");
        for (int j=0; j<m->ncols; j++) { // Columns run from 0...k
            double val;
            matrix_getelement(m, i, j, &val);
            morpho_printf(v, "%g ", (fabs(val)<MORPHO_EPS ? 0 : val));
        }
        morpho_printf(v, "]%s", (i<m->nrows-1 ? "\n" : ""));
    }
}

/** Prints a matrix to a buffer */
bool matrix_printtobuffer(objectmatrix *m, char *format, varray_char *out) {
    for (int i=0; i<m->nrows; i++) { // Rows run from 0...m
        varray_charadd(out, "[ ", 2);
        
        for (int j=0; j<m->ncols; j++) { // Columns run from 0...k
            double val;
            matrix_getelement(m, i, j, &val);
            if (!format_printtobuffer(MORPHO_FLOAT(val), format, out)) return false; 
            varray_charadd(out, " ", 1);
        }
        varray_charadd(out, "]", 1);
        if (i<m->nrows-1) varray_charadd(out, "\n", 1);
    }
    return true;
}

/** Matrix eigensystem */
bool matrix_eigen(vm *v, objectmatrix *a, value *evals, value *evecs) {
    double *ev = MORPHO_MALLOC(sizeof(double)*a->nrows*2); // Allocate temporary memory for eigenvalues
    double *er=ev, *ei=ev+a->nrows;
    
    objectmatrix *vecs=NULL; // A new matrix for eigenvectors
    objectlist *vallist = object_newlist(0, NULL); // List to hold eigenvalues
    bool success=false;
    
    if (evecs) vecs=object_clonematrix(a); // Clones a to hold eigenvectors
    
    // Check that everything was allocated correctly
    if (!(ev && vallist && (!evecs || vecs))) {
        morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto matrix_eigen_cleanup; };
    
    objectmatrixerror err=matrix_eigensystem(a, er, ei, vecs);
    
    if (err!=MATRIX_OK) {
        matrix_raiseerror(v, err);
        goto matrix_eigen_cleanup;
    }
        
    // Now process the eigenvalues
    for (int i=0; i<a->nrows; i++) {
        if (fabs(ei[i])<MORPHO_EPS) {
            list_append(vallist, MORPHO_FLOAT(er[i]));
        } else {
            objectcomplex *c = object_newcomplex(er[i], ei[i]);
            if (c) {
                list_append(vallist, MORPHO_OBJECT(c));
            } else {
                morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
                goto matrix_eigen_cleanup;
            }
        }
    }
        
    if (evals) *evals = MORPHO_OBJECT(vallist);
    if (evecs) *evecs = MORPHO_OBJECT(vecs);
    
    success=true;
    
matrix_eigen_cleanup:
    if (ev) MORPHO_FREE(ev);
    
    if (!success) {
        if (vallist) {
            for (unsigned int i=0; i<vallist->val.count; i++) {
                if (MORPHO_ISOBJECT(vallist->val.data[i])) object_free(MORPHO_GETOBJECT(vallist->val.data[i]));
            }
            object_free((object *) vallist);
        }
        if (vecs) object_free((object *) vecs);
    }
    
    return success;
}

/* **********************************************************************
 * Matrix veneer class
 * ********************************************************************* */

/** Constructs a Matrix object */
value matrix_constructor(vm *v, int nargs, value *args) {
    unsigned int nrows, ncols;
    objectmatrix *new=NULL;
    value out=MORPHO_NIL;
    
    if (nargs==2 &&
         MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)) &&
         MORPHO_ISINTEGER(MORPHO_GETARG(args, 1)) ) {
        nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        ncols = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
        new=object_newmatrix(nrows, ncols, true);
    } else if (nargs==1 &&
               MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        ncols = 1;
        new=object_newmatrix(nrows, ncols, true);
    } else if (nargs==1 &&
               MORPHO_ISARRAY(MORPHO_GETARG(args, 0))) {
        new=object_matrixfromarray(MORPHO_GETARRAY(MORPHO_GETARG(args, 0)));
        if (!new) morpho_runtimeerror(v, MATRIX_INVLDARRAYINIT);
#ifdef MORPHO_INCLUDE_SPARSE
    } else if (nargs==1 &&
               MORPHO_ISLIST(MORPHO_GETARG(args, 0))) {
        new=object_matrixfromlist(MORPHO_GETLIST(MORPHO_GETARG(args, 0)));
        if (!new) {
            /** Could this be a concatenation operation? */
            objectsparseerror err = sparse_catmatrix(MORPHO_GETLIST(MORPHO_GETARG(args, 0)), &new);
            if (err==SPARSE_INVLDINIT) {
                morpho_runtimeerror(v, MATRIX_INVLDARRAYINIT);
            } else if (err!=SPARSE_OK) sparse_raiseerror(v, err);
        }
#endif
    } else if (nargs==1 &&
               MORPHO_ISMATRIX(MORPHO_GETARG(args, 0))) {
        new=object_clonematrix(MORPHO_GETMATRIX(MORPHO_GETARG(args, 0)));
        if (!new) morpho_runtimeerror(v, MATRIX_INVLDARRAYINIT);
#ifdef MORPHO_INCLUDE_SPARSE
    } else if (nargs==1 &&
               MORPHO_ISSPARSE(MORPHO_GETARG(args, 0))) {
        objectsparseerror err=sparse_tomatrix(MORPHO_GETSPARSE(MORPHO_GETARG(args, 0)), &new);
        if (err!=SPARSE_OK) morpho_runtimeerror(v, MATRIX_INVLDARRAYINIT);
#endif
    } else morpho_runtimeerror(v, MATRIX_CONSTRUCTOR);
    
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

/** Creates an identity matrix */
value matrix_identityconstructor(vm *v, int nargs, value *args) {
    int n;
    objectmatrix *new=NULL;
    value out = MORPHO_NIL;
    
    if (nargs==1 &&
               MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        n = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        new=object_newmatrix(n, n, false);
        if (new) {
            matrix_identity(new);
        } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    } else morpho_runtimeerror(v, MATRIX_IDENTCONSTRUCTOR);
    
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

/** Checks that a matrix is indexed with 2 indices with a generic interface */
bool matrix_slicedim(value * a, unsigned int ndim){
	if (ndim>2||ndim<0) return false;
	return true;
}

/** Constucts a new matrix with a generic interface */
void matrix_sliceconstructor(unsigned int *slicesize,unsigned int ndim,value* out){
	unsigned int numcol = 1;
	if (ndim == 2) {
		numcol = slicesize[1];
	}
	*out = MORPHO_OBJECT(object_newmatrix(slicesize[0],numcol,false));
}
/** Copies data from a at indx to out at newindx with a generic interface */
objectarrayerror matrix_slicecopy(value * a,value * out, unsigned int ndim, unsigned int *indx,unsigned int *newindx){
	double num; // matrices store doubles;
	unsigned int colindx = 0;
	unsigned int colnewindx = 0;	
	
	if (ndim == 2) {
		colindx = indx[1];
		colnewindx = newindx[1];
	}

	if (!(matrix_getelement(MORPHO_GETMATRIX(*a),indx[0],colindx,&num)&&
		matrix_setelement(MORPHO_GETMATRIX(*out),newindx[0],colnewindx,num))){
		return ARRAY_OUTOFBOUNDS;
	}
	return ARRAY_OK;
}

/** Rolls the matrix list */
void matrix_rollflat(objectmatrix *a, objectmatrix *b, int nplaces) {
    unsigned int N = a->nrows*a->ncols;
    int n = abs(nplaces);
    if (n>N) n = n % N;
    unsigned int Np = N - n; // Number of elements to roll
    
    if (nplaces<0) {
        memcpy(b->matrixdata, a->matrixdata+n, sizeof(double)*Np);
        memcpy(b->matrixdata+Np, a->matrixdata, sizeof(double)*n);
    } else {
        memcpy(b->matrixdata+n, a->matrixdata, sizeof(double)*Np);
        if (n>0) memcpy(b->matrixdata, a->matrixdata+Np, sizeof(double)*n);
    }
}

/** Copies arow from matrix a into brow for matrix b */
void matrix_copyrow(objectmatrix *a, int arow, objectmatrix *b, int brow) {
    cblas_dcopy(a->ncols, a->elements+arow, a->nrows, b->elements+brow, a->nrows);
}

/** Rolls a list by a number of elements */
objectmatrix *matrix_roll(objectmatrix *a, int nplaces, int axis) {
    objectmatrix *new=object_newmatrix(a->nrows, a->ncols, false);
    
    if (new) {
        switch(axis) {
            case 0: { // TODO: Could probably be faster
                for (int i=0; i<a->nrows; i++) {
                    int j = (i+nplaces);
                    if (j<0) j+=a->nrows;
                    matrix_copyrow(a, i, new, j % a->nrows);
                }
            }
                break;
            case 1: matrix_rollflat(a, new, nplaces*a->nrows); break;
        }
    }

    return new;
}

/** Gets the matrix element with given indices */
value Matrix_getindex(vm *v, int nargs, value *args) {
    objectmatrix *m=MORPHO_GETMATRIX(MORPHO_SELF(args));
    unsigned int indx[2]={0,0};
    value out = MORPHO_NIL;
	if (nargs>2){
		morpho_runtimeerror(v, MATRIX_INVLDNUMINDICES);
		return out;
	}
    
    if (array_valuelisttoindices(nargs, args+1, indx)) {
        double outval;
        if (!matrix_getelement(m, indx[0], indx[1], &outval)) {
            morpho_runtimeerror(v, MATRIX_INDICESOUTSIDEBOUNDS);
        } else {
            out = MORPHO_FLOAT(outval);
        }
    } else { // now try to get a slice
		objectarrayerror err = getslice(&MORPHO_SELF(args), &matrix_slicedim, &matrix_sliceconstructor, &matrix_slicecopy, nargs, &MORPHO_GETARG(args,0), &out);
		if (err!=ARRAY_OK) MORPHO_RAISE(v, array_to_matrix_error(err) );
		if (MORPHO_ISOBJECT(out)){
			morpho_bindobjects(v,1,&out);
		} else morpho_runtimeerror(v, MATRIX_INVLDINDICES);
	}
    return out;
}

/** Sets the matrix element with given indices */
value Matrix_setindex(vm *v, int nargs, value *args) {
    objectmatrix *m=MORPHO_GETMATRIX(MORPHO_SELF(args));
    unsigned int indx[2]={0,0};
    
    if (array_valuelisttoindices(nargs-1, args+1, indx)) {
        double value=0.0;
        if (MORPHO_ISFLOAT(args[nargs])) value=MORPHO_GETFLOATVALUE(args[nargs]);
        if (MORPHO_ISINTEGER(args[nargs])) value=(double) MORPHO_GETINTEGERVALUE(args[nargs]);

        if (!matrix_setelement(m, indx[0], indx[1], value)) {
            morpho_runtimeerror(v, MATRIX_INDICESOUTSIDEBOUNDS);
        }
    } else morpho_runtimeerror(v, MATRIX_INVLDINDICES);
    
    return MORPHO_NIL;
}

/** Sets the column of a matrix */
value Matrix_setcolumn(vm *v, int nargs, value *args) {
    objectmatrix *m=MORPHO_GETMATRIX(MORPHO_SELF(args));
    
    if (nargs==2 &&
        MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISMATRIX(MORPHO_GETARG(args, 1))) {
        unsigned int col = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        objectmatrix *src = MORPHO_GETMATRIX(MORPHO_GETARG(args, 1));
        
        if (col<m->ncols) {
            if (src && src->ncols*src->nrows==m->nrows) {
                matrix_setcolumn(m, col, src->elements);
            } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
        } else morpho_runtimeerror(v, MATRIX_INDICESOUTSIDEBOUNDS);
    } else morpho_runtimeerror(v, MATRIX_SETCOLARGS);
    
    return MORPHO_NIL;
}

/** Gets a column of a matrix */
value Matrix_getcolumn(vm *v, int nargs, value *args) {
    objectmatrix *m=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (nargs==1 &&
        MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        unsigned int col = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        
        if (col<m->ncols) {
            double *vals;
            if (matrix_getcolumn(m, col, &vals)) {
                objectmatrix *new=object_matrixfromfloats(m->nrows, 1, vals);
                if (new) {
                    out=MORPHO_OBJECT(new);
                    morpho_bindobjects(v, 1, &out);
                }
            }
        } else morpho_runtimeerror(v, MATRIX_INDICESOUTSIDEBOUNDS);
    } else morpho_runtimeerror(v, MATRIX_SETCOLARGS);
    
    return out;
}

/** Prints a matrix */
value Matrix_print(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    if (!MORPHO_ISMATRIX(self)) return Object_print(v, nargs, args);
    
    objectmatrix *m=MORPHO_GETMATRIX(MORPHO_SELF(args));
    matrix_print(v, m);
    return MORPHO_NIL;
}

/** Formatted conversion to a string */
value Matrix_format(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    
    if (nargs==1 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        varray_char str;
        varray_charinit(&str);
        
        if (matrix_printtobuffer(MORPHO_GETMATRIX(MORPHO_SELF(args)),
                                 MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)),
                                 &str)) {
            out = object_stringfromvarraychar(&str);
            if (MORPHO_ISOBJECT(out)) morpho_bindobjects(v, 1, &out);
        } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        
        varray_charclear(&str);
    } else {
        morpho_runtimeerror(v, VALUE_FRMTARG);
    }
    
    return out;
}

/** Matrix add */
value Matrix_assign(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
 
    if (nargs==1 && MORPHO_ISMATRIX(MORPHO_GETARG(args, 0))) {
        objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
        
        if (a->ncols==b->ncols && a->nrows==b->nrows) {
            matrix_copy(b, a);
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    }
    
    return MORPHO_NIL;
}

/** Matrix add */
value Matrix_add(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISMATRIX(MORPHO_GETARG(args, 0))) {
        objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
        
        if (a->ncols==b->ncols && a->nrows==b->nrows) {
            objectmatrix *new = object_newmatrix(a->nrows, a->ncols, false);
            if (new) {
                out=MORPHO_OBJECT(new);
                matrix_add(a, b, new);
            }
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double val;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
            objectmatrix *new = object_newmatrix(a->nrows, a->ncols, false);
            if (new) {
                out=MORPHO_OBJECT(new);
                matrix_addscalar(a, 1.0, val, new);
            }
        }
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    
    return out;
}

/** Right add */
value Matrix_addr(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && (MORPHO_ISNIL(MORPHO_GETARG(args, 0)) ||
                     MORPHO_ISNUMBER(MORPHO_GETARG(args, 0)))) {
        int i=0;
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) i=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        if (MORPHO_ISFLOAT(MORPHO_GETARG(args, 0))) i=(fabs(MORPHO_GETFLOATVALUE(MORPHO_GETARG(args, 0)))<MORPHO_EPS ? 0 : 1);
        
        if (i==0) {
            objectmatrix *new = object_clonematrix(a);
            if (new) {
                out=MORPHO_OBJECT(new);
                morpho_bindobjects(v, 1, &out);
            }
        } else { // If there is a value we're trying to add to just use Matrix_add for that 
            out = Matrix_add(v, nargs, args);
        }
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}

/** Matrix subtract */
value Matrix_sub(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISMATRIX(MORPHO_GETARG(args, 0))) {
        objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
        
        if (a->ncols==b->ncols && a->nrows==b->nrows) {
            objectmatrix *new = object_newmatrix(a->nrows, a->ncols, false);
            if (new) {
                out=MORPHO_OBJECT(new);
                matrix_sub(a, b, new);
            }
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double val;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
            objectmatrix *new = object_newmatrix(a->nrows, a->ncols, false);
            if (new) {
                out=MORPHO_OBJECT(new);
                matrix_addscalar(a, 1.0, -val, new);
            }
        }
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    
    return out;
}

/** Right subtract */
value Matrix_subr(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && (MORPHO_ISNIL(MORPHO_GETARG(args, 0)) ||
                     MORPHO_ISNUMBER(MORPHO_GETARG(args, 0)))) {
        int i=(MORPHO_ISNIL(MORPHO_GETARG(args, 0)) ? 0 : MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)));

        if (MORPHO_ISFLOAT(MORPHO_GETARG(args, 0))) i=(fabs(MORPHO_GETFLOATVALUE(MORPHO_GETARG(args, 0)))<MORPHO_EPS ? 0 : 1);

        
        if (i==0) {
            objectmatrix *new = object_clonematrix(a);
            if (new) {
                out=MORPHO_OBJECT(new);
                matrix_scale(new, -1.0);
                morpho_bindobjects(v, 1, &out);
            }
        } else if (MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
            // try and subtract like normal
            double val;
            if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
                objectmatrix *new = object_newmatrix(a->nrows, a->ncols, false);
                if (new) {
                    matrix_addscalar(a, 1.0, -val, new);
                    // now that did self - arg[0] and we want arg[0] - self so scale the whole thing by -1
                    matrix_scale(new, -1.0);
                    out=MORPHO_OBJECT(new);
                    morpho_bindobjects(v, 1, &out);
                }
            }

        } else morpho_runtimeerror(v, VM_INVALIDARGS);
    } else morpho_runtimeerror(v, VM_INVALIDARGS);
    
    return out;
}

/** Matrix multiply */
value Matrix_mul(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISMATRIX(MORPHO_GETARG(args, 0))) {
        objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
        
        if (a->ncols==b->nrows) {
            objectmatrix *new = object_newmatrix(a->nrows, b->ncols, false);
            if (new) {
                out=MORPHO_OBJECT(new);
                matrix_mul(a, b, new);
                morpho_bindobjects(v, 1, &out);
            }
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double scale=1.0;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale)) {
            objectmatrix *new = object_clonematrix(a);
            if (new) {
                out=MORPHO_OBJECT(new);
                matrix_scale(new, scale);
                morpho_bindobjects(v, 1, &out);
            }
        }
#ifdef MORPHO_INCLUDE_SPARSE
    } else if (nargs==1 && MORPHO_ISSPARSE(MORPHO_GETARG(args, 0))) {
        // Returns nil to ensure it gets passed to mulr on Sparse
#endif
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}

/** Called when multiplying on the right */
value Matrix_mulr(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double scale=1.0;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale)) {
            objectmatrix *new = object_clonematrix(a);
            if (new) {
                out=MORPHO_OBJECT(new);
                matrix_scale(new, scale);
                morpho_bindobjects(v, 1, &out);
            }
        }
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}

/** Solution of linear system a.x = b (i.e. x = b/a) */
value Matrix_div(vm *v, int nargs, value *args) {
    objectmatrix *b=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISMATRIX(MORPHO_GETARG(args, 0))) {
        objectmatrix *a=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
        
        if (a->ncols==b->nrows) {
            objectmatrix *new = object_newmatrix(b->nrows, b->ncols, false);
            if (new) {
                objectmatrixerror err;
                if (MATRIX_ISSMALL(a)) {
                    err=matrix_divs(a, b, new);
                } else {
                    err=matrix_divl(a, b, new);
                }
                if (err==MATRIX_SING) {
                    morpho_runtimeerror(v, MATRIX_SINGULAR);
                    object_free((object *) new);
                } else {
                    out=MORPHO_OBJECT(new);
                    morpho_bindobjects(v, 1, &out);
                }
            }
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
#ifdef MORPHO_INCLUDE_SPARSE
    } else if (nargs==1 && MORPHO_ISSPARSE(MORPHO_GETARG(args, 0))) {
        /* Division by a sparse matrix: redirect to the divr selector of Sparse. */
        value vargs[2]={args[1],args[0]};
        return Sparse_divr(v, nargs, vargs);
#endif
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        /* Division by a scalar */
        double scale=1.0;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale)) {
            if (fabs(scale)<MORPHO_EPS) MORPHO_RAISE(v, VM_DVZR);
            
            objectmatrix *new = object_clonematrix(b);
            if (new) {
                out=MORPHO_OBJECT(new);
                matrix_scale(new, 1.0/scale);
                morpho_bindobjects(v, 1, &out);
            }
        }
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}

/** Matrix accumulate */
value Matrix_acc(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==2 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISMATRIX(MORPHO_GETARG(args, 1))) {
        objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 1));
        
        if (a->ncols==b->ncols && a->nrows==b->nrows) {
            out=MORPHO_SELF(args);
            double lambda=1.0;
            morpho_valuetofloat(MORPHO_GETARG(args, 0), &lambda);
            matrix_accumulate(a, lambda, b);
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return MORPHO_NIL;
}

/** Frobenius inner product */
value Matrix_inner(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISMATRIX(MORPHO_GETARG(args, 0))) {
        objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
        
        double prod=0.0;
        if (matrix_inner(a, b, &prod)==MATRIX_OK) {
            out = MORPHO_FLOAT(prod);
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}

/** Outer product */
value Matrix_outer(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISMATRIX(MORPHO_GETARG(args, 0))) {
        objectmatrix *b=MORPHO_GETMATRIX(MORPHO_GETARG(args, 0));
        objectmatrix *new=object_newmatrix(a->nrows*a->ncols, b->nrows*b->ncols, true);
        
        if (new &&
            matrix_outer(a, b, new)==MATRIX_OK) {
            out=MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}

/** Matrix sum */
value Matrix_sum(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    return MORPHO_FLOAT(matrix_sum(a));
}

/** Roll a matrix */
value Matrix_roll(vm *v, int nargs, value *args) {
    objectmatrix *slf = MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out = MORPHO_NIL;
    int roll, axis=0;

    if (nargs>0 &&
        morpho_valuetoint(MORPHO_GETARG(args, 0), &roll)) {
        
        if (nargs==2 && !morpho_valuetoint(MORPHO_GETARG(args, 1), &axis)) return out;
        
        objectmatrix *new = matrix_roll(slf, roll, axis);

        if (new) {
            out = MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        }

    } else morpho_runtimeerror(v, LIST_ADDARGS);

    return out;
}


/** Matrix norm */
value Matrix_norm(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out = MORPHO_NIL;
    
    if (nargs==1) {
        value arg = MORPHO_GETARG(args, 0);
        
        if (MORPHO_ISNUMBER(arg)) {
            double n;
            
            if (morpho_valuetofloat(arg, &n)) {
                if (fabs(n-1.0)<MORPHO_EPS) {
                    out=MORPHO_FLOAT(matrix_L1norm(a));
                } else if (fabs(n-2.0)<MORPHO_EPS) {
                    out=MORPHO_FLOAT(matrix_norm(a));
                } else if (isinf(n)) {
                    out=MORPHO_FLOAT(matrix_Linfnorm(a));
                } else {
                    out=MORPHO_FLOAT(matrix_Lnnorm(a, n));
                }
            } else morpho_runtimeerror(v, MATRIX_NORMARGS);
        } else morpho_runtimeerror(v, MATRIX_NORMARGS);
    } else if (nargs==0) {
        out=MORPHO_FLOAT(matrix_norm(a));
    } else morpho_runtimeerror(v, MATRIX_NORMARGS);
    
    return out;
}

/** Matrix eigenvalues */
value Matrix_eigenvalues(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value evals=MORPHO_NIL;
    
    if (matrix_eigen(v, a, &evals, NULL)) {
        objectlist *new = MORPHO_GETLIST(evals);
        list_append(new, evals); // Ensure we retain the List object
        morpho_bindobjects(v, new->val.count, new->val.data);
        new->val.count--; // And pop it back off
    }
    
    return evals;
}

/** Matrix eigensystem */
value Matrix_eigensystem(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value evals=MORPHO_NIL, evecs=MORPHO_NIL, out=MORPHO_NIL;
    objectlist *resultlist = object_newlist(0, NULL);
    if (!resultlist) {
        morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        return MORPHO_NIL;
    }
    
    if (matrix_eigen(v, a, &evals, &evecs)) {
        objectlist *evallist = MORPHO_GETLIST(evals);
        
        list_append(resultlist, evals); // Create the output list
        list_append(resultlist, evecs);
        out=MORPHO_OBJECT(resultlist);
        
        list_append(evallist, evals); // Ensure we bind all objects at once
        list_append(evallist, evecs); // by popping them onto the evallist.
        list_append(evallist, out);   //
        morpho_bindobjects(v, evallist->val.count, evallist->val.data);
        evallist->val.count-=3; // and then popping them back off.
    }
    
    return out;
}

/** Inverts a matrix */
value Matrix_inverse(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    // The inverse will have the number of rows and number of columns
    // swapped. 
    objectmatrix *new = object_newmatrix(a->ncols, a->nrows, false);
    if (new) {
        objectmatrixerror mi = matrix_inverse(a, new);
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
        
        if (mi!=MATRIX_OK) matrix_raiseerror(v, mi);
    }
    
    return out;
}

/** Transpose of a matrix */
value Matrix_transpose(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    objectmatrix *new = object_newmatrix(a->ncols, a->nrows, false);
    if (new) {
        matrix_transpose(a, new);
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

/** Reshape a matrix */
value Matrix_reshape(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    
    if (nargs==2 &&
        MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISINTEGER(MORPHO_GETARG(args, 1))) {
        int nrows = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        int ncols = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
        
        if (nrows*ncols==a->nrows*a->ncols) {
            a->nrows=nrows;
            a->ncols=ncols;
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);   
    } else morpho_runtimeerror(v, MATRIX_RESHAPEARGS);
    
    return MORPHO_NIL;
}

/** Trace of a matrix */
value Matrix_trace(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (a->nrows==a->ncols) {
        double tr;
        if (matrix_trace(a, &tr)==MATRIX_OK) out=MORPHO_FLOAT(tr);
    } else {
        morpho_runtimeerror(v, MATRIX_NOTSQ);
    }
    
    return out;
}

/** Enumerate protocol */
value Matrix_enumerate(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (nargs==1) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            int i=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
            
            if (i<0) out=MORPHO_INTEGER(a->ncols*a->nrows);
            else if (i<a->ncols*a->nrows) out=MORPHO_FLOAT(a->elements[i]);
        }
    }
    
    return out;
}


/** Number of matrix elements */
value Matrix_count(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    
    return MORPHO_INTEGER(a->ncols*a->nrows);
}

/** Matrix dimensions */
value Matrix_dimensions(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    value dim[2];
    value out=MORPHO_NIL;
    
    dim[0]=MORPHO_INTEGER(a->nrows);
    dim[1]=MORPHO_INTEGER(a->ncols);
    
    objectlist *new=object_newlist(2, dim);
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

/** Clones a matrix */
value Matrix_clone(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    objectmatrix *new=object_clonematrix(a);
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    return out;
}

MORPHO_BEGINCLASS(Matrix)
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, Matrix_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, Matrix_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_GETCOLUMN_METHOD, Matrix_getcolumn, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_SETCOLUMN_METHOD, Matrix_setcolumn, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Matrix_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_FORMAT_METHOD, Matrix_format, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ASSIGN_METHOD, Matrix_assign, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADD_METHOD, Matrix_add, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADDR_METHOD, Matrix_addr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUB_METHOD, Matrix_sub, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUBR_METHOD, Matrix_subr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_MUL_METHOD, Matrix_mul, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_MULR_METHOD, Matrix_mulr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_DIV_METHOD, Matrix_div, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ACC_METHOD, Matrix_acc, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_INNER_METHOD, Matrix_inner, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_OUTER_METHOD, Matrix_outer, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUM_METHOD, Matrix_sum, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_NORM_METHOD, Matrix_norm, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_INVERSE_METHOD, Matrix_inverse, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_TRANSPOSE_METHOD, Matrix_transpose, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_RESHAPE_METHOD, Matrix_reshape, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_EIGENVALUES_METHOD, Matrix_eigenvalues, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_EIGENSYSTEM_METHOD, Matrix_eigensystem, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_TRACE_METHOD, Matrix_trace, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, Matrix_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Matrix_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_DIMENSIONS_METHOD, Matrix_dimensions, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ROLL_METHOD, Matrix_roll, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Matrix_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************* */

void matrix_initialize(void) {
    objectmatrixtype=object_addtype(&objectmatrixdefn);
    
    builtin_addfunction(MATRIX_CLASSNAME, matrix_constructor, MORPHO_FN_CONSTRUCTOR);
    builtin_addfunction(MATRIX_IDENTITYCONSTRUCTOR, matrix_identityconstructor, BUILTIN_FLAGSEMPTY);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    value matrixclass=builtin_addclass(MATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(Matrix), objclass);
    object_setveneerclass(OBJECT_MATRIX, matrixclass);
    
    morpho_defineerror(MATRIX_INDICESOUTSIDEBOUNDS, ERROR_HALT, MATRIX_INDICESOUTSIDEBOUNDS_MSG);
    morpho_defineerror(MATRIX_INVLDINDICES, ERROR_HALT, MATRIX_INVLDINDICES_MSG);
    morpho_defineerror(MATRIX_INVLDNUMINDICES, ERROR_HALT, MATRIX_INVLDNUMINDICES_MSG);
    morpho_defineerror(MATRIX_CONSTRUCTOR, ERROR_HALT, MATRIX_CONSTRUCTOR_MSG);
    morpho_defineerror(MATRIX_INVLDARRAYINIT, ERROR_HALT, MATRIX_INVLDARRAYINIT_MSG);
    morpho_defineerror(MATRIX_ARITHARGS, ERROR_HALT, MATRIX_ARITHARGS_MSG);
    morpho_defineerror(MATRIX_RESHAPEARGS, ERROR_HALT, MATRIX_RESHAPEARGS_MSG);
    morpho_defineerror(MATRIX_INCOMPATIBLEMATRICES, ERROR_HALT, MATRIX_INCOMPATIBLEMATRICES_MSG);
    morpho_defineerror(MATRIX_SINGULAR, ERROR_HALT, MATRIX_SINGULAR_MSG);
    morpho_defineerror(MATRIX_NOTSQ, ERROR_HALT, MATRIX_NOTSQ_MSG);
    morpho_defineerror(MATRIX_OPFAILED, ERROR_HALT, MATRIX_OPFAILED_MSG);
    morpho_defineerror(MATRIX_SETCOLARGS, ERROR_HALT, MATRIX_SETCOLARGS_MSG);
    morpho_defineerror(MATRIX_NORMARGS, ERROR_HALT, MATRIX_NORMARGS_MSG);
    morpho_defineerror(MATRIX_IDENTCONSTRUCTOR, ERROR_HALT, MATRIX_IDENTCONSTRUCTOR_MSG);
}

#endif
