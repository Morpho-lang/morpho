/** @file array.c
 *  @author T J Atherton
 *
 *  @brief Defines array object type and Array class
 */

#include "morpho.h"
#include "classes.h"
#include "common.h"

/* **********************************************************************
 * Array objects
 * ********************************************************************** */

/** Array object definitions */
void objectarray_printfn(object *obj, void *v) {
    morpho_printf(v, "<Array>");
}

void objectarray_markfn(object *obj, void *v) {
    objectarray *c = (objectarray *) obj;
    for (unsigned int i=0; i<c->nelements; i++) {
        morpho_markvalue(v, c->values[i]);
    }
}

size_t objectarray_sizefn(object *obj) {
    return sizeof(objectarray) +
        sizeof(value) * ( ((objectarray *) obj)->nelements+2*((objectarray *) obj)->ndim );
}

objecttypedefn objectarraydefn = {
    .printfn=objectarray_printfn,
    .markfn=objectarray_markfn,
    .freefn=NULL,
    .sizefn=objectarray_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/** Initializes an array given the size */
void object_arrayinit(objectarray *array, unsigned int ndim, unsigned int *dim) {
    object_init((object *) array, OBJECT_ARRAY);
    unsigned int nel = (ndim==0 ? 0 : 1);

    /* Store pointers into the data array */
    array->dimensions=array->data;
    array->multipliers=array->data+ndim;
    array->values=array->data+2*ndim;

    /* Store the description of array dimensions */
    array->ndim=ndim;
    for (unsigned int i=0; i<ndim; i++) {
        array->dimensions[i]=MORPHO_INTEGER(dim[i]);
        array->multipliers[i]=MORPHO_INTEGER(nel);
        nel*=dim[i];
    }

    /* Store the size of the object for convenient access */
    array->nelements=nel;

    /* Arrays are initialized to nil. */
#ifdef MORPHO_NAN_BOXING
    memset(array->values, 0, sizeof(value)*nel);
#else
    for (unsigned int i=0; i<nel; i++) array->values[i]=MORPHO_FLOAT(0.0);
#endif
}

/** @brief Creates an array object
 * @details Arrays are stored in memory as follows:
 *          objectarray structure with flexible array member value
 *          value [0..dim-1] the dimensions of the array
 *          value [dim..2*dim-1] stores multipliers for each dimension to translate to the index
 *          value [2*dim..] array elements in column major order, i.e. the matrix
 *          [ [ 1, 2],
 *           [ 3, 4] ] is stored as:
 *          <structure> // the structure
 *          2, 2, // the dimensions
 *          1, 2, // multipliers for each index to access elements
 *          1, 3, 2, 4 // the elements in column major order */
objectarray *object_newarray(unsigned int ndim, unsigned int *dim) {
    /* Calculate the number of elements */
    unsigned int nel=(ndim==0 ? 0 : dim[0]);
    for (unsigned int i=1; i<ndim; i++) nel*=dim[i];

    size_t size = sizeof(objectarray)+sizeof(value)*(2*ndim + nel);

    objectarray *new = (objectarray *) object_new(size, OBJECT_ARRAY);
    if (new) object_arrayinit(new, ndim, dim);

    return new;
}

/* **********************************************************************
 * Array utility functions
 * ********************************************************************** */

/** Converts a list of values to a list of integers */
bool array_valuelisttoindices(unsigned int ndim, value *in, unsigned int *out) {

    for (unsigned int i=0; i<ndim; i++) {
        if (MORPHO_ISINTEGER(in[i])) out[i]=MORPHO_GETINTEGERVALUE(in[i]);
        else if(MORPHO_ISFLOAT(in[i])) out[i]=round(MORPHO_GETFLOATVALUE(in[i]));
        else return false;
    }

    return true;
}

/** Creates a new 1D array from a list of values */
objectarray *object_arrayfromvaluelist(unsigned int n, value *v) {
    objectarray *new = object_newarray(1, &n);

    if (new) memcpy(new->values, v, sizeof(value)*n);

    return new;
}

/** Creates a new 1D array from a list of varray_value */
objectarray *object_arrayfromvarrayvalue(varray_value *v) {
    return object_arrayfromvaluelist(v->count, v->data);
}

/** Creates a new array object with the dimensions given as a list of values */
objectarray *object_arrayfromvalueindices(unsigned int ndim, value *dim) {
    unsigned int indx[ndim];
    if (array_valuelisttoindices(ndim, dim, indx)) {
        return object_newarray(ndim, indx);
    }
    return NULL;
}

/** Clones an array. Does *not* clone the contents. */
objectarray *object_clonearray(objectarray *array) {
    objectarray *new = object_arrayfromvalueindices(array->ndim, array->data);

    if (new) memcpy(new->data, array->data, sizeof(value)*(array->nelements+2*array->ndim));

    return new;
}

/** Recursively print a slice of an array */
bool array_print_recurse(vm *v, objectarray *a, unsigned int *indx, unsigned int dim, varray_char *out) {
    unsigned int bnd = MORPHO_GETINTEGERVALUE(a->dimensions[dim]);
    value val=MORPHO_NIL;

    varray_charadd(out, "[ ", 2);
    for (indx[dim]=0; indx[dim]<bnd; indx[dim]++) {
        if (dim==a->ndim-1) { // Print if innermost element
            if (array_getelement(a, a->ndim, indx, &val)==ARRAY_OK) {
                morpho_printtobuffer(v, val, out);
            } else return false;
        } else if (!array_print_recurse(v, a, indx, dim+1, out)) return false; // Otherwise recurse

        if (indx[dim]<bnd-1) { // Separators between items
            varray_charadd(out, ", ", 2);
        }
    }
    varray_charadd(out, " ]", 2);

    return true;
}

/* Print the contents of an array */
void array_print(vm *v, objectarray *a) {
    varray_char out;
    varray_charinit(&out);

    unsigned int indx[a->ndim];
    if (array_print_recurse(v, a, indx, 0, &out)) {
        varray_charwrite(&out, '\0'); // Ensure zero terminated
        morpho_printf(v, "%s", out.data);
    }

    varray_charclear(&out);
}

/** Converts an array error into an error code */
errorid array_error(objectarrayerror err) {
    switch (err) {
        case ARRAY_OUTOFBOUNDS: return VM_OUTOFBOUNDS;
        case ARRAY_WRONGDIM: return VM_ARRAYWRONGDIM;
        case ARRAY_NONINTINDX: return VM_NONNUMINDX;
        case ARRAY_ALLOC_FAILED: return ERROR_ALLOCATIONFAILED;
        case ARRAY_OK: UNREACHABLE("array_error called incorrectly.");
    }
    UNREACHABLE("Unhandled array error.");
    return VM_OUTOFBOUNDS;
}

/** Converts an array error into an matrix error code for use in slices*/
errorid array_to_matrix_error(objectarrayerror err) {
#ifdef MORPHO_INCLUDE_LINALG
    switch (err) {
        case ARRAY_OUTOFBOUNDS: return MATRIX_INDICESOUTSIDEBOUNDS;
        case ARRAY_WRONGDIM: return MATRIX_INVLDNUMINDICES;
        case ARRAY_NONINTINDX: return MATRIX_INVLDINDICES;
        case ARRAY_ALLOC_FAILED: return ERROR_ALLOCATIONFAILED;
        case ARRAY_OK: UNREACHABLE("array_to_matrix_error called incorrectly.");
    }
    UNREACHABLE("Unhandled array error.");
    return VM_OUTOFBOUNDS;
#else
    return array_error(err);
#endif
}

/** Converts an array error into an list error code for use in slices*/
errorid array_to_list_error(objectarrayerror err) {
    switch (err) {
        case ARRAY_OUTOFBOUNDS: return VM_OUTOFBOUNDS;
        case ARRAY_WRONGDIM: return LIST_NUMARGS;
        case ARRAY_NONINTINDX: return LIST_ARGS;
        case ARRAY_ALLOC_FAILED: return ERROR_ALLOCATIONFAILED;
        case ARRAY_OK: UNREACHABLE("array_to_list_error called incorrectly.");
    }
    UNREACHABLE("Unhandled array error.");
    return VM_OUTOFBOUNDS;
}

/** Gets an array element */
objectarrayerror array_getelement(objectarray *a, unsigned int ndim, unsigned int *indx, value *out) {
    unsigned int k=0;

    if (ndim!=a->ndim) return ARRAY_WRONGDIM;

    for (unsigned int i=0; i<ndim; i++) {
        if (indx[i]>=MORPHO_GETINTEGERVALUE(a->dimensions[i])) return ARRAY_OUTOFBOUNDS;
        k+=indx[i]*MORPHO_GETINTEGERVALUE(a->multipliers[i]);
    }

    *out = a->values[k];
    return ARRAY_OK;
}

/** Creates a slice from a slicable object A.
 * @param[in] a - the sliceable object (array, list, matrix, etc..).
 * @param[in] dimFcn - a function that checks if the number of indecies is compatabile with the slicable object.
 * @param[in] constuctor - a function that create the a new object of the type of a.
 * @param[in] copy - a function that can copy information from a to out.
 * @param[in] ndim - the number of dimensions being indexed.
 * @param[in] slices - a set of indices that can be lists ranges or ints.
 * @param[out] out - returns the requested slice of a.
*/
objectarrayerror getslice(value *a, bool dimFcn(value *, unsigned int),
                          void constructor(unsigned int *, unsigned int,value *),
                          objectarrayerror copy(value * ,value *, unsigned int, unsigned int *, unsigned int *),
                          unsigned int ndim, value *slices, value *out) {
    //dimension checking
    if (!(*dimFcn) (a,ndim)) return ARRAY_WRONGDIM;

    unsigned int slicesize[ndim];
    for (unsigned int i=0; i<ndim; i++) {
        if (MORPHO_ISINTEGER(slices[i])||MORPHO_ISFLOAT(slices[i])) {// if this is an number
            slicesize[i] = 1; // it only has one element
        } else if (MORPHO_ISLIST(slices[i])) { // if this is a list
            objectlist * s = MORPHO_GETLIST(slices[i]);
            slicesize[i] = s->val.count; // get the number of elements
        } else if (MORPHO_ISRANGE(slices[i])) { //if its a range
            objectrange * s = MORPHO_GETRANGE(slices[i]);
            slicesize[i] = range_count(s);
        } else return ARRAY_NONINTINDX; // by returning array a VM_NONNUMIDX will be thrown
    }

    // initialize out with the right size
    (constructor) (slicesize, ndim, out);
    
    if (!MORPHO_ISOBJECT(*out)) return ARRAY_ALLOC_FAILED;

    // fill it out recurivly
    unsigned int indx[ndim];
    unsigned int newindx[ndim];
    objectarrayerror err = setslicerecursive(a, out, copy, ndim, 0, indx, newindx, slices);
    
    if (err!=ARRAY_OK) { // Free allocated object if an error has occurred
        morpho_freeobject(*out);
        *out = MORPHO_NIL;
    }
    
    return err;
}

/** Iterates though the a ndim number of provided slices recursivly and copies the data from a to out.
 * @param[in] a - the sliceable object (array, list, matrix, etc..).
 * @param[out] out - returns the requeted slice of a.
 * @param[in] copy - a function that can copy information from a to out.
 * @param[in] ndim - the total number of dimentions being indexed.
 * @param[in] curdim - the current dimention being indexed.
 * @param[in] indx - an ndim list of indices that builds up to a locataion in a to copy data from.
 * @param[in] newindx - the place in out to put the data copied from a
 * @param[in] slices - a set of indices that can be lists ranges or ints.
*/
objectarrayerror setslicerecursive(value* a, value* out,objectarrayerror copy(value * ,value *, unsigned int, unsigned int *,unsigned int *),\
                                   unsigned int ndim, unsigned int curdim, unsigned int *indx,unsigned int *newindx, value *slices){
    // this gets given an array and out and a list of slices,
    // we resolve the top slice to a number and add it to a list
    objectarrayerror arrayerr;

    if (curdim == ndim) { // we've resolved all the indices we can now use the list
        arrayerr = (*copy)(a,out,ndim,indx,newindx);
        if (arrayerr!=ARRAY_OK) return arrayerr;
    } else { // we need to iterate though the current object
        if (MORPHO_ISINTEGER(slices[curdim])) {
            indx[curdim] = MORPHO_GETINTEGERVALUE(slices[curdim]);
            newindx[curdim] = 0;

            arrayerr = setslicerecursive(a, out, copy, ndim, curdim+1, indx, newindx, slices);
            if (arrayerr!=ARRAY_OK) return arrayerr;

        } else if (MORPHO_ISLIST(slices[curdim])) { // if this is a list

            objectlist * s = MORPHO_GETLIST(slices[curdim]);
            for (unsigned int  i = 0; i<s->val.count; i++ ){ // iterate through the list
                if (MORPHO_ISINTEGER(s->val.data[i])) {
                    indx[curdim] = MORPHO_GETINTEGERVALUE(s->val.data[i]);
                    newindx[curdim] = i;
                } else return ARRAY_NONINTINDX;

                arrayerr = setslicerecursive(a, out, copy, ndim, curdim+1, indx, newindx, slices);
                if (arrayerr!=ARRAY_OK) return arrayerr;

            }
        } else if (MORPHO_ISRANGE(slices[curdim])) { //if its a range
            objectrange * s = MORPHO_GETRANGE(slices[curdim]);
            value rangeValue;
            for (unsigned int  i = 0; i<range_count(s); i++) { // iterate though the range
                rangeValue=range_iterate(s,i);
                if (MORPHO_ISINTEGER(rangeValue)) {
                    indx[curdim] = MORPHO_GETINTEGERVALUE(rangeValue);
                    newindx[curdim] = i;
                } else return ARRAY_NONINTINDX;
                arrayerr = setslicerecursive(a, out, copy, ndim, curdim+1, indx, newindx, slices);
                if (arrayerr!=ARRAY_OK) return arrayerr;
            }
        } else return ARRAY_NONINTINDX;
            //if (!(*dimFcn)(a,ndim)) return ARRAY_WRONGDIM;

    }
    return ARRAY_OK;
}

/** Sets an array element */
objectarrayerror array_setelement(objectarray *a, unsigned int ndim, unsigned int *indx, value in) {
    unsigned int k=0;

    if (ndim!=a->ndim) return ARRAY_WRONGDIM;

    for (unsigned int i=0; i<ndim; i++) {
        if (indx[i]>=MORPHO_GETINTEGERVALUE(a->dimensions[i])) return ARRAY_OUTOFBOUNDS;
        k+=indx[i]*MORPHO_GETINTEGERVALUE(a->multipliers[i]);
    }

    a->values[k]=in;
    return ARRAY_OK;
}

/* ---------------------------
 * Array constructor functions
 * --------------------------- */

/** Returns the maximum nesting depth in a list, including this one.
 * @param[in] list - the list to examine
 * @param[out] out - optionally return the dimensions of the nested lists.
 * To get dimension information:
 * Call list_nestingdepth with out set to NULL; this returns the size of the array needed.
 * Initialize the dimension array to zero.
 * Call list_nestingdepth again with out set to an output array */
unsigned int list_nestingdepth(objectlist *list, unsigned int *out) {
    unsigned int dim=0;
    for (unsigned int i=0; i<list->val.count; i++) {
        if (MORPHO_ISLIST(list->val.data[i])) {
            unsigned int sdim=list_nestingdepth(MORPHO_GETLIST(list->val.data[i]), ( out ? out+1 : NULL));
            if (sdim>dim) dim=sdim;
        }
    }
    if (out && list->val.count>*out) *out=list->val.count;
    return dim+1;
}

/* Internal function that recursively copied a nested list into an array.
   Use public interface array_copyfromnestedlist */
static void array_copyfromnestedlistrecurse(objectlist *list, unsigned int ndim, unsigned int *indx, unsigned int depth, objectarray *out) {
    for (unsigned int i=0; i<list->val.count; i++) {
        indx[depth] = i;
        value val = list->val.data[i];
        if (MORPHO_ISLIST(val)) array_copyfromnestedlistrecurse(MORPHO_GETLIST(val), ndim, indx, depth+1, out);
        else array_setelement(out, ndim, indx, val);
    }
}

/** Copies a nested list into an array.*/
void array_copyfromnestedlist(objectlist *in, objectarray *out) {
    unsigned int indx[out->ndim];
    for (unsigned int i=0; i<out->ndim; i++) indx[i]=0;
    array_copyfromnestedlistrecurse(in, out->ndim, indx, 0, out);
}

/** Constructs an array from a list initializer or returns NULL if the initializer isn't compatible with the requested array */
objectarray *array_constructfromlist(unsigned int ndim, unsigned int *dim, objectlist *initializer) {
    // Establish the dimensions of the nested list
    unsigned int nldim = list_nestingdepth(initializer, NULL);
    unsigned int ldim[nldim];
    for (unsigned int i=0; i<nldim; i++) ldim[i]=0;
    list_nestingdepth(initializer, ldim);

    if (ndim>0) { // Check compatibility
        if (ndim!=nldim) return NULL;
        for (unsigned int i=0; i<ndim; i++) if (ldim[i]!=dim[i]) return NULL;
    }

    objectarray *new = object_newarray(nldim, ldim);
    array_copyfromnestedlist(initializer, new);

    return new;
}

/** Constructs an array from an initializer or returns NULL if the initializer isn't compatible with the requested array */
objectarray *array_constructfromarray(unsigned int ndim, unsigned int *dim, objectarray *initializer) {
    if (ndim>0) { // Check compatibility
        if (ndim!=initializer->ndim) return NULL;
        for (unsigned int i=0; i<ndim; i++) {
            if (dim[i]!=MORPHO_GETINTEGERVALUE(initializer->dimensions[i])) return NULL;
        }
    }

    return object_clonearray(initializer);
}

/** Array constructor function */
value array_constructor(vm *v, int nargs, value *args) {
    unsigned int ndim; // Number of dimensions
    unsigned int dim[nargs+1]; // Size of each dimension
    value initializer=MORPHO_NIL; // An initializer if provided

    // Check that args are present
    if (nargs==0) { morpho_runtimeerror(v, ARRAY_ARGS); return MORPHO_NIL; }

    for (ndim=0; ndim<nargs; ndim++) { // Loop over arguments
        if (!MORPHO_ISNUMBER(MORPHO_GETARG(args, ndim))) break; // Stop once a non-numerical argument is encountered
    }

    // Get dimensions
    if (ndim>0) array_valuelisttoindices(ndim, &MORPHO_GETARG(args, 0), dim);
    // Initializer is the first non-numerical argument; anything after is ignored
    if (ndim<nargs) initializer=MORPHO_GETARG(args, ndim);

    objectarray *new=NULL;

    // Now construct the array
    if (MORPHO_ISNIL(initializer)) {
        new = object_newarray(ndim, dim);
    } else if (MORPHO_ISARRAY(initializer)) {
        new = array_constructfromarray(ndim, dim, MORPHO_GETARRAY(initializer));
        if (!new) morpho_runtimeerror(v, ARRAY_CMPT);
    } else if (MORPHO_ISLIST(initializer)) {
        new = array_constructfromlist(ndim, dim, MORPHO_GETLIST(initializer));
        if (!new) morpho_runtimeerror(v, ARRAY_CMPT);
    } else {
        morpho_runtimeerror(v, ARRAY_ARGS);
    }

    // Bind the new array to the VM
    value out=MORPHO_NIL;
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }

    return out;
}

/** Checks that an array is being indexed with the correct number of indices with a generic interface */
bool array_slicedim(value * a, unsigned int ndim){
    objectarray * array= MORPHO_GETARRAY(*a);
    if (ndim>array->ndim) return false;
    return true;
}

/** Constructsan array is with a generic interface */
void array_sliceconstructor(unsigned int *slicesize,unsigned int ndim,value* out){
    *out = MORPHO_OBJECT(object_newarray(ndim,slicesize));
}

/** Copies data from array a to array out with a generic interface */
objectarrayerror array_slicecopy(value * a,value * out, unsigned int ndim, unsigned int *indx,unsigned int *newindx){
    value data;
    objectarrayerror arrayerr;
    arrayerr = array_getelement(MORPHO_GETARRAY(*a),ndim,indx,&data); // read the data
    if (arrayerr!=ARRAY_OK) return arrayerr;

    arrayerr=array_setelement(MORPHO_GETARRAY(*out), ndim, newindx, data); // write the data
    return arrayerr;

}

/* **********************************************************************
 * Array class
 * ********************************************************************** */

/** Gets the array element with given indices */
value Array_getindex(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectarray *array=MORPHO_GETARRAY(MORPHO_SELF(args));
    unsigned int indx[nargs];

    if (array_valuelisttoindices(nargs, &MORPHO_GETARG(args, 0), indx)) {
        objectarrayerror err=array_getelement(array, nargs, indx, &out);
        if (err!=ARRAY_OK) MORPHO_RAISE(v, array_error(err) );

    } else {
        // these aren't simple indices, lets try to make a slice
        objectarrayerror err = getslice(&MORPHO_SELF(args),&array_slicedim,&array_sliceconstructor,&array_slicecopy,nargs,&MORPHO_GETARG(args, 0),&out);
        if (err!=ARRAY_OK) MORPHO_RAISE(v, array_error(err) );
        if (!MORPHO_ISNIL(out)){
            morpho_bindobjects(v,1,&out);
        } else MORPHO_RAISE(v, VM_NONNUMINDX);
    }

    return out;
}

/** Sets the matrix element with given indices */
value Array_setindex(vm *v, int nargs, value *args) {
    objectarray *array=MORPHO_GETARRAY(MORPHO_SELF(args));
    unsigned int indx[nargs-1];

    if (array_valuelisttoindices(nargs-1, &MORPHO_GETARG(args, 0), indx)) {
        objectarrayerror err=array_setelement(array, nargs-1, indx, MORPHO_GETARG(args, nargs-1));
        if (err!=ARRAY_OK) MORPHO_RAISE(v, array_error(err) );
    } else MORPHO_RAISE(v, VM_NONNUMINDX);

    return MORPHO_NIL;
}

/** Print an array */
value Array_print(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    if (!MORPHO_ISARRAY(self)) return Object_print(v, nargs, args);
    
    array_print(v, MORPHO_GETARRAY(self));

    return MORPHO_NIL;
}

/** Find an array's size */
value Array_count(vm *v, int nargs, value *args) {
    objectarray *slf = MORPHO_GETARRAY(MORPHO_SELF(args));

    return MORPHO_INTEGER(slf->nelements);
}

/** Array dimensions */
value Array_dimensions(vm *v, int nargs, value *args) {
    objectarray *a=MORPHO_GETARRAY(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    objectlist *new=object_newlist(a->ndim, a->data);

    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);

    return out;
}

/** Enumerate members of an array */
value Array_enumerate(vm *v, int nargs, value *args) {
    objectarray *slf = MORPHO_GETARRAY(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

        if (n<0) {
            out=MORPHO_INTEGER(slf->nelements);
        } else if (n<slf->nelements) {
            out=slf->values[n];
        } else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
    } else MORPHO_RAISE(v, ENUMERATE_ARGS);

    return out;
}

/** Clone an array */
value Array_clone(vm *v, int nargs, value *args) {
    objectarray *slf = MORPHO_GETARRAY(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    objectarray *new = object_clonearray(slf);
    if (new) {
        out = MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    }

    return out;
}

MORPHO_BEGINCLASS(Array)
MORPHO_METHOD(MORPHO_PRINT_METHOD, Array_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Array_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(ARRAY_DIMENSIONS_METHOD, Array_dimensions, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, Array_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, Array_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, Array_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Array_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

objecttype objectarraytype;

void array_initialize(void) {
    // Create array object type
    objectarraytype=object_addtype(&objectarraydefn);
    
    // Locate the Object class to use as the parent class of Array
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // Array constructor function
    morpho_addfunction(ARRAY_CLASSNAME, ARRAY_CLASSNAME " (...)", array_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
    
    // Create Array veneer class
    value arrayclass=builtin_addclass(ARRAY_CLASSNAME, MORPHO_GETCLASSDEFINITION(Array), objclass);
    object_setveneerclass(OBJECT_ARRAY, arrayclass);
    
    // Array error messages
    morpho_defineerror(ARRAY_ARGS, ERROR_HALT, ARRAY_ARGS_MSG);
    morpho_defineerror(ARRAY_INIT, ERROR_HALT, ARRAY_INIT_MSG);
    morpho_defineerror(ARRAY_CMPT, ERROR_HALT, ARRAY_CMPT_MSG);
}
