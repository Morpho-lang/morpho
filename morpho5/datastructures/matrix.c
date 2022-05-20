/** @file matrix.c
 *  @author T J Atherton
 *
 *  @brief Veneer class over the objectmatrix type that interfaces with blas and lapack
 */

#include <string.h>
#include "object.h"
#include "matrix.h"
#include "gpumatrix.h"
#include "sparse.h"
#include "morpho.h"
#include "builtin.h"
#include "veneer.h"
#include "common.h"

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

void objectmatrix_printfn(object *obj) {
    printf("<Matrix>");
}

objecttypedefn objectmatrixdefn = {
    .printfn=objectmatrix_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectmatrix_sizefn
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

objectmatrixerror matrix_copy(objectmatrix *a, objectmatrix *out) {
    if (a->ncols==out->ncols && a->nrows==out->nrows) {
        cblas_dcopy(a->ncols * a->nrows, a->elements, 1, out->elements, 1);
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

/** Computes the Frobenius norm of a matrix */
double matrix_norm(objectmatrix *a) {
    double nrm2=cblas_dnrm2(a->ncols*a->nrows, a->elements, 1);
    return nrm2;
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
    for (int i=0; i<a->nrows; i++) for (int j=0; j<a->ncols; j++) a->elements[i+a->nrows*j]=(i==j ? 1.0 : 0.0);
    
    return MATRIX_OK;
}

/** Prints a matrix */
void matrix_print(objectmatrix *m) {
    for (int i=0; i<m->nrows; i++) { // Rows run from 0...m
        printf("[ ");
        for (int j=0; j<m->ncols; j++) { // Columns run from 0...k
            double v;
            matrix_getelement(m, i, j, &v);
            printf("%g ", (fabs(v)<MORPHO_EPS ? 0 : v));
        }
        printf("]%s", (i<m->nrows-1 ? "\n" : ""));
    }
}

/* **********************************************************************
 * Matrix veneer class
 * ********************************************************************* */

#include "matrixveneer2.h"

value matrix_constructor_wrapper(vm *v, int nargs, value *args){
    value out = MORPHO_NIL;
    if (nargs==1 && MORPHO_ISGPUMATRIX(MORPHO_GETARG(args, 0))) {
        objectmatrix *new=NULL;
        objectgpumatrix *in = MORPHO_GETGPUMATRIX(MORPHO_GETARG(args, 0));
        new=object_matrixfromgpumatrix(in);
        if (new) {
            out = MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        }
    } else {
        out = matrix_constructor(v, nargs, args);
    }
    return out;
}

MORPHO_BEGINCLASS(Matrix)
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, Matrix_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, Matrix_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_GETCOLUMN_METHOD, Matrix_getcolumn, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_SETCOLUMN_METHOD, Matrix_setcolumn, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Matrix_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADD_METHOD, Matrix_add, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADDR_METHOD, Matrix_addr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUB_METHOD, Matrix_sub, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUBR_METHOD, Matrix_subr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_MUL_METHOD, Matrix_mul, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_MULR_METHOD, Matrix_mulr, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_DIV_METHOD, Matrix_div, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ACC_METHOD, Matrix_acc, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_INNER_METHOD, Matrix_inner, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUM_METHOD, Matrix_sum, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_NORM_METHOD, Matrix_norm, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_TRANSPOSE_METHOD, Matrix_transpose, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_TRACE_METHOD, Matrix_trace, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, Matrix_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Matrix_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MATRIX_DIMENSIONS_METHOD, Matrix_dimensions, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Matrix_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************* */

void matrix_initialize(void) {
    objectmatrixtype=object_addtype(&objectmatrixdefn);
    
    builtin_addfunction(MATRIX_CLASSNAME, matrix_constructor_wrapper, BUILTIN_FLAGSEMPTY);
    
    value matrixclass=builtin_addclass(MATRIX_CLASSNAME, MORPHO_GETCLASSDEFINITION(Matrix), MORPHO_NIL);
    object_setveneerclass(OBJECT_MATRIX, matrixclass);
    
    morpho_defineerror(MATRIX_INDICESOUTSIDEBOUNDS, ERROR_HALT, MATRIX_INDICESOUTSIDEBOUNDS_MSG);
    morpho_defineerror(MATRIX_INVLDINDICES, ERROR_HALT, MATRIX_INVLDINDICES_MSG);
    morpho_defineerror(MATRIX_INVLDNUMINDICES, ERROR_HALT, MATRIX_INVLDNUMINDICES_MSG);
    morpho_defineerror(MATRIX_CONSTRUCTOR, ERROR_HALT, MATRIX_CONSTRUCTOR_MSG);
    morpho_defineerror(MATRIX_INVLDARRAYINIT, ERROR_HALT, MATRIX_INVLDARRAYINIT_MSG);
    morpho_defineerror(MATRIX_ARITHARGS, ERROR_HALT, MATRIX_ARITHARGS_MSG);
    morpho_defineerror(MATRIX_INCOMPATIBLEMATRICES, ERROR_HALT, MATRIX_INCOMPATIBLEMATRICES_MSG);
    morpho_defineerror(MATRIX_SINGULAR, ERROR_HALT, MATRIX_SINGULAR_MSG);
    morpho_defineerror(MATRIX_NOTSQ, ERROR_HALT, MATRIX_NOTSQ_MSG);
    morpho_defineerror(MATRIX_SETCOLARGS, ERROR_HALT, MATRIX_SETCOLARGS_MSG);
}
