// matrix veneer super class 

#ifndef MORPHO_MATRIX_TYPE
    #error matrix vanner used without type
#endif
#ifndef MORPHO_MATRIX_TYPE_CAP
    #error matrix vanner used without cap type
#endif


#define glue1(a,b) a ## b
#define glue(a,b) glue1(a,b)

#define MATRIXOBJTYPE1(type) object ## type ## matrix
#define MATRIXOBJTYPE2(type) MATRIXOBJTYPE1(type)
#define MATRIXOBJTYPE MATRIXOBJTYPE2(MORPHO_MATRIX_TYPE)


#define IS_MATRIX1(type) MORPHO_IS ## type ## MATRIX 
#define IS_MATRIX2(type) IS_MATRIX1(type)
#define IS_MATRIX IS_MATRIX2(MORPHO_MATRIX_TYPE_CAP)


#define GET_MATRIX1(type) MORPHO_GET ## type ## MATRIX
#define GET_MATRIX2(type) GET_MATRIX1(type)
#define GET_MATRIX GET_MATRIX2(MORPHO_MATRIX_TYPE_CAP)

/** Constructs a Matrix object */
value glue(MORPHO_MATRIX_TYPE,matrix_constructor)(vm *v, int nargs, value *args) {
    unsigned int nrows, ncols;
    MATRIXOBJTYPE *new=NULL;
    value out=MORPHO_NIL;
    
    if ( nargs==2 &&
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
    } else if (nargs==1 &&
               MORPHO_ISLIST(MORPHO_GETARG(args, 0))) {
        new=object_matrixfromlist(MORPHO_GETLIST(MORPHO_GETARG(args, 0)));
        if (!new) morpho_runtimeerror(v, MATRIX_INVLDARRAYINIT);
    } else if (nargs==1 &&
               IS_MATRIX(MORPHO_GETARG(args, 0))) {
        new=object_clonematrix(GET_MATRIX(MORPHO_GETARG(args, 0)));
        if (!new) morpho_runtimeerror(v, MATRIX_INVLDARRAYINIT);
    } else morpho_runtimeerror(v, MATRIX_CONSTRUCTOR);
    
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

/** Checks that a matrix is indexed with 2 indices with a generic interface */
bool glue(MORPHO_MATRIX_TYPE,matrix_slicedim)(value * a, unsigned int ndim){
	if (ndim>2||ndim<0) return false;
	return true;
}

/** Constucts a new matrix with a generic interface */
void glue(MORPHO_MATRIX_TYPE,matrix_sliceconstructor)(unsigned int *slicesize,unsigned int ndim,value* out){
	unsigned int numcol = 1;
	if (ndim == 2) {
		numcol = slicesize[1];
	}
	*out = MORPHO_OBJECT(object_newmatrix(slicesize[0],numcol,false));
}
/** Copies data from a at indx to out at newindx with a generic interface */
objectarrayerror glue(MORPHO_MATRIX_TYPE,matrix_slicecopy)(value * a,value * out, unsigned int ndim, unsigned int *indx,unsigned int *newindx){
	double num; // matrices store doubles;
	unsigned int colindx = 0;
	unsigned int colnewindx = 0;	
	
	if (ndim == 2) {
		colindx = indx[1];
		colnewindx = newindx[1];
	}

	if (!(glue(MORPHO_MATRIX_TYPE,matrix_getelement)(GET_MATRIX(*a),indx[0],colindx,&num)&&
		glue(MORPHO_MATRIX_TYPE,matrix_setelement)(GET_MATRIX(*out),newindx[0],colnewindx,num))){
		return ARRAY_OUTOFBOUNDS;
	}
	return ARRAY_OK;
}

/** Gets the matrix element with given indices */
value glue(MORPHO_MATRIX_TYPE,Matrix_getindex)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *m=GET_MATRIX(MORPHO_SELF(args));
    unsigned int indx[2]={0,0};
    value out = MORPHO_NIL;
	if (nargs>2){
		morpho_runtimeerror(v, MATRIX_INVLDNUMINDICES);
		return out;
	}
    
    if (array_valuelisttoindices(nargs, args+1, indx)) {
        double outval;
        if (!glue(MORPHO_MATRIX_TYPE,matrix_getelement)(m, indx[0], indx[1], &outval)) {
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
value glue(MORPHO_MATRIX_TYPE,Matrix_setindex)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *m=GET_MATRIX(MORPHO_SELF(args));
    unsigned int indx[2]={0,0};
    
    if (array_valuelisttoindices(nargs-1, args+1, indx)) {
        double value=0.0;
        if (MORPHO_ISFLOAT(args[nargs])) value=MORPHO_GETFLOATVALUE(args[nargs]);
        if (MORPHO_ISINTEGER(args[nargs])) value=(double) MORPHO_GETINTEGERVALUE(args[nargs]);

        if (!glue(MORPHO_MATRIX_TYPE,matrix_setelement)(m, indx[0], indx[1], value)) {
            morpho_runtimeerror(v, MATRIX_INDICESOUTSIDEBOUNDS);
        }
    } else morpho_runtimeerror(v, MATRIX_INVLDINDICES);
    
    return MORPHO_NIL;
}

/** Sets the column of a matrix */
value glue(MORPHO_MATRIX_TYPE,Matrix_setcolumn)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *m=GET_MATRIX(MORPHO_SELF(args));
    
    if (nargs==2 &&
        MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)) &&
        IS_MATRIX(MORPHO_GETARG(args, 1))) {
        unsigned int col = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        MATRIXOBJTYPE *src = GET_MATRIX(MORPHO_GETARG(args, 1));
        
        if (col<m->ncols) {
            if (src && src->ncols*src->nrows==m->nrows) {
                glue(MORPHO_MATRIX_TYPE,matrix_setcolumn)(m, col, src->elements);
            } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
        } else morpho_runtimeerror(v, MATRIX_INDICESOUTSIDEBOUNDS);
    } else morpho_runtimeerror(v, MATRIX_SETCOLARGS);
    
    return MORPHO_NIL;
}

/** Gets a column of a matrix */
value glue(MORPHO_MATRIX_TYPE,Matrix_getcolumn)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *m=GET_MATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (nargs==1 &&
        MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        unsigned int col = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        
        if (col<m->ncols) {
            double *vals;
            if (glue(MORPHO_MATRIX_TYPE,matrix_getcolumn)(m, col, &vals)) {
                MATRIXOBJTYPE *new=object_matrixfromfloats(m->nrows, 1, vals);
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
value glue(MORPHO_MATRIX_TYPE,Matrix_print)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *m=GET_MATRIX(MORPHO_SELF(args));
    glue(MORPHO_MATRIX_TYPE,matrix_print)(m);
    return MORPHO_NIL;
}

/** Matrix add */
value glue(MORPHO_MATRIX_TYPE,Matrix_add)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && IS_MATRIX(MORPHO_GETARG(args, 0))) {
        MATRIXOBJTYPE *b=GET_MATRIX(MORPHO_GETARG(args, 0));
        
        if (a->ncols==b->ncols && a->nrows==b->nrows) {
            MATRIXOBJTYPE *new = object_newmatrix(a->nrows, a->ncols, false);
            if (new) {
                out=MORPHO_OBJECT(new);
                glue(MORPHO_MATRIX_TYPE,matrix_add)(a, b, new);
            }
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double val;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
            MATRIXOBJTYPE *new = object_newmatrix(a->nrows, a->ncols, false);
            if (new) {
                out=MORPHO_OBJECT(new);
                glue(MORPHO_MATRIX_TYPE,matrix_addscalar)(a, 1.0, val, new);
            }
        }
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    
    return out;
}

/** Right add */
value glue(MORPHO_MATRIX_TYPE,Matrix_addr)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && (MORPHO_ISNIL(MORPHO_GETARG(args, 0)) ||
                     MORPHO_ISNUMBER(MORPHO_GETARG(args, 0)))) {
        int i=0;
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) i=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        if (MORPHO_ISFLOAT(MORPHO_GETARG(args, 0))) i=(fabs(MORPHO_GETFLOATVALUE(MORPHO_GETARG(args, 0)))<MORPHO_EPS ? 0 : 1);
        
        if (i==0) {
            MATRIXOBJTYPE *new = object_clonematrix(a);
            if (new) {
                out=MORPHO_OBJECT(new);
                morpho_bindobjects(v, 1, &out);
            }
        } else UNREACHABLE("Right addition to non-zero value.");
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}

/** Matrix subtract */
value glue(MORPHO_MATRIX_TYPE,Matrix_sub)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && IS_MATRIX(MORPHO_GETARG(args, 0))) {
        MATRIXOBJTYPE *b=GET_MATRIX(MORPHO_GETARG(args, 0));
        
        if (a->ncols==b->ncols && a->nrows==b->nrows) {
            MATRIXOBJTYPE *new = object_newmatrix(a->nrows, a->ncols, false);
            if (new) {
                out=MORPHO_OBJECT(new);
                glue(MORPHO_MATRIX_TYPE,matrix_sub)(a, b, new);
            }
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double val;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &val)) {
            MATRIXOBJTYPE *new = object_newmatrix(a->nrows, a->ncols, false);
            if (new) {
                out=MORPHO_OBJECT(new);
                glue(MORPHO_MATRIX_TYPE,matrix_addscalar)(a, 1.0, -val, new);
            }
        }
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    
    return out;
}

/** Right subtract */
value glue(MORPHO_MATRIX_TYPE,Matrix_subr)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && (MORPHO_ISNIL(MORPHO_GETARG(args, 0)) ||
                     MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)))) {
        int i=(MORPHO_ISNIL(MORPHO_GETARG(args, 0)) ? 0 : MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)));
        
        if (i==0) {
            MATRIXOBJTYPE *new = object_clonematrix(a);
            if (new) {
                out=MORPHO_OBJECT(new);
                glue(MORPHO_MATRIX_TYPE,matrix_scale)(new, -1.0);
                morpho_bindobjects(v, 1, &out);
            }
        } else morpho_runtimeerror(v, VM_INVALIDARGS);
    } else morpho_runtimeerror(v, VM_INVALIDARGS);
    
    return out;
}

/** Matrix multiply */
value glue(MORPHO_MATRIX_TYPE,Matrix_mul)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && IS_MATRIX(MORPHO_GETARG(args, 0))) {
        MATRIXOBJTYPE *b=GET_MATRIX(MORPHO_GETARG(args, 0));
        
        if (a->ncols==b->nrows) {
            MATRIXOBJTYPE *new = object_newmatrix(a->nrows, b->ncols, false);
            if (new) {
                out=MORPHO_OBJECT(new);
                glue(MORPHO_MATRIX_TYPE,matrix_mul)(a, b, new);
                morpho_bindobjects(v, 1, &out);
            }
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double scale=1.0;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale)) {
            MATRIXOBJTYPE *new = object_clonematrix(a);
            if (new) {
                out=MORPHO_OBJECT(new);
                glue(MORPHO_MATRIX_TYPE,matrix_scale)(new, scale);
                morpho_bindobjects(v, 1, &out);
            }
        }
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}

/** Called when multiplying on the right */
value glue(MORPHO_MATRIX_TYPE,Matrix_mulr)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double scale=1.0;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale)) {
            MATRIXOBJTYPE *new = object_clonematrix(a);
            if (new) {
                out=MORPHO_OBJECT(new);
                glue(MORPHO_MATRIX_TYPE,matrix_scale)(new, scale);
                morpho_bindobjects(v, 1, &out);
            }
        }
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}

/** Solution of linear system a.x = b (i.e. x = b/a) */
value glue(MORPHO_MATRIX_TYPE,Matrix_div)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *b=GET_MATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && IS_MATRIX(MORPHO_GETARG(args, 0))) {
        MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_GETARG(args, 0));
        
        if (a->ncols==b->nrows) {
            MATRIXOBJTYPE *new = object_newmatrix(b->nrows, b->ncols, false);
            if (new) {
                objectmatrixerror err;
                if (MATRIX_ISSMALL(a)) {
                    err=glue(MORPHO_MATRIX_TYPE,matrix_divs)(a, b, new);
                } else {
                    err=glue(MORPHO_MATRIX_TYPE,matrix_divl)(a, b, new);
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
    } else if (nargs==1 && MORPHO_ISSPARSE(MORPHO_GETARG(args, 0))) {
        /* Division by a sparse matrix: redirect to the divr selector of Sparse. */
        value vargs[2]={args[1],args[0]};
        return Sparse_divr(v, nargs, vargs);
    } else if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        /* Division by a scalar */
        double scale=1.0;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &scale)) {
            if (fabs(scale)<MORPHO_EPS) MORPHO_RAISE(v, VM_DVZR);
            
            MATRIXOBJTYPE *new = object_clonematrix(b);
            if (new) {
                out=MORPHO_OBJECT(new);
                glue(MORPHO_MATRIX_TYPE,matrix_scale)(new, 1.0/scale);
                morpho_bindobjects(v, 1, &out);
            }
        }
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}

/** Matrix accumulate */
value glue(MORPHO_MATRIX_TYPE,Matrix_acc)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==2 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0)) &&
        IS_MATRIX(MORPHO_GETARG(args, 1))) {
        MATRIXOBJTYPE *b=GET_MATRIX(MORPHO_GETARG(args, 1));
        
        if (a->ncols==b->ncols && a->nrows==b->nrows) {
            out=MORPHO_SELF(args);
            double lambda=1.0;
            morpho_valuetofloat(MORPHO_GETARG(args, 0), &lambda);
            glue(MORPHO_MATRIX_TYPE,matrix_accumulate)(a, lambda, b);
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return MORPHO_NIL;
}

/** Frobenius inner product */
value glue(MORPHO_MATRIX_TYPE,Matrix_inner)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    if (nargs==1 && IS_MATRIX(MORPHO_GETARG(args, 0))) {
        MATRIXOBJTYPE *b=GET_MATRIX(MORPHO_GETARG(args, 0));
        
        double prod=0.0;
        if (glue(MORPHO_MATRIX_TYPE,matrix_inner)(a, b, &prod)==MATRIX_OK) {
            out = MORPHO_FLOAT(prod);
        } else morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
    } else morpho_runtimeerror(v, MATRIX_ARITHARGS);
    
    return out;
}


/** Matrix sum */
value glue(MORPHO_MATRIX_TYPE,Matrix_sum)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    return MORPHO_FLOAT(glue(MORPHO_MATRIX_TYPE,matrix_sum)(a));
}

/** Matrix norm */
value glue(MORPHO_MATRIX_TYPE,Matrix_norm)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    return MORPHO_FLOAT(glue(MORPHO_MATRIX_TYPE,matrix_norm)(a));
}

/** Transpose of a matrix */
value glue(MORPHO_MATRIX_TYPE,Matrix_transpose)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;
 
    MATRIXOBJTYPE *new = object_newmatrix(a->ncols, a->nrows, false);
    if (new) {
        glue(MORPHO_MATRIX_TYPE,matrix_transpose)(a, new);
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }
    
    return out;
}

/** Trace of a matrix */
value glue(MORPHO_MATRIX_TYPE,Matrix_trace)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (a->nrows==a->ncols) {
        double tr;
        if (glue(MORPHO_MATRIX_TYPE,matrix_trace)(a, &tr)==MATRIX_OK) out=MORPHO_FLOAT(tr);
    } else {
        morpho_runtimeerror(v, MATRIX_NOTSQ);
    }
    
    return out;
}

/** Enumerate protocol */
value glue(MORPHO_MATRIX_TYPE,Matrix_enumerate)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
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
value glue(MORPHO_MATRIX_TYPE,Matrix_count)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    
    return MORPHO_INTEGER(a->ncols*a->nrows);
}

/** Matrix dimensions */
value glue(MORPHO_MATRIX_TYPE,Matrix_dimensions)(vm *v, int nargs, value *args) {
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
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
value glue(MORPHO_MATRIX_TYPE,Matrix_clone)(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    MATRIXOBJTYPE *a=GET_MATRIX(MORPHO_SELF(args));
    MATRIXOBJTYPE *new=object_clonematrix(a);
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    return out;
}

#undef glue1(a,b)
#undef glue(a,b)

#undef MATRIXOBJTYPE1(type)
#undef MATRIXOBJTYPE2(type)
#undef MATRIXOBJTYPE

#undef IS_MATRIX1(type)
#undef IS_MATRIX2(type)
#undef IS_MATRIX

#undef GET_MATRIX1(type)
#undef GET_MATRIX2(type)
#undef GET_MATRIX