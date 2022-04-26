/** Constructs a Matrix object */
value matrix_constructor(vm *v, int nargs, value *args) {
    unsigned int nrows, ncols;
    objectmatrix *new=NULL;
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
               MORPHO_ISMATRIX(MORPHO_GETARG(args, 0))) {
        new=object_clonematrix(MORPHO_GETMATRIX(MORPHO_GETARG(args, 0)));
        if (!new) morpho_runtimeerror(v, MATRIX_INVLDARRAYINIT);
    } else morpho_runtimeerror(v, MATRIX_CONSTRUCTOR);
    
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

/** Prints a matrix */
value Matrix_print(vm *v, int nargs, value *args) {
    objectmatrix *m=MORPHO_GETMATRIX(MORPHO_SELF(args));
    matrix_print(m);
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
        int i = 0;
        if (MORPHO_ISNIL(MORPHO_GETARG(args, 0))) i = 0;
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
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
                    out=MORPHO_OBJECT(new);
                    matrix_addscalar(a, 1.0, -val, new);
                    // now that did self - arg[0] and we want arg[0] - self so scale the whole thing by -1
                    matrix_scale(new, -1.0);
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
    } else if (nargs==1 && MORPHO_ISSPARSE(MORPHO_GETARG(args, 0))) {
        /* Division by a sparse matrix: redirect to the divr selector of Sparse. */
        value vargs[2]={args[1],args[0]};
        return Sparse_divr(v, nargs, vargs);
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


/** Matrix sum */
value Matrix_sum(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    return MORPHO_FLOAT(matrix_sum(a));
}

/** Matrix norm */
value Matrix_norm(vm *v, int nargs, value *args) {
    objectmatrix *a=MORPHO_GETMATRIX(MORPHO_SELF(args));
    return MORPHO_FLOAT(matrix_norm(a));
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
/**/