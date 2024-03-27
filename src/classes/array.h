/** @file array.h
 *  @author T J Atherton
 *
 *  @brief Defines array object type and Array class
 */

#ifndef array_h
#define array_h

#include "object.h"

/* -------------------------------------------------------
 * Array object type
 * ------------------------------------------------------- */

extern objecttype objectarraytype;
#define OBJECT_ARRAY objectarraytype

typedef struct {
    object obj;
    unsigned int ndim;
    unsigned int nelements;
    value *values;
    value *dimensions;
    value *multipliers;
    value data[];
} objectarray;

/** Tests whether an object is an array */
#define MORPHO_ISARRAY(val) object_istype(val, OBJECT_ARRAY)

/** Gets the object as an array */
#define MORPHO_GETARRAY(val)   ((objectarray *) MORPHO_GETOBJECT(val))

/** Creates an array object */
objectarray *object_newarray(unsigned int dimension, unsigned int *dim);

/** Creates a new array from a list of values */
objectarray *object_arrayfromvaluelist(unsigned int n, value *v);

/** Creates a new 1D array from a list of varray_value */
objectarray *object_arrayfromvarrayvalue(varray_value *v);

/** Creates a new array object with the dimensions given as a list of values */
objectarray *object_arrayfromvalueindices(unsigned int ndim, value *dim);

/* -------------------------------------------------------
 * Array veneer class
 * ------------------------------------------------------- */

#define ARRAY_CLASSNAME                   "Array"

#define ARRAY_DIMENSIONS_METHOD           "dimensions"

/* -------------------------------------------------------
 * Array error messages
 * ------------------------------------------------------- */

#define ARRAY_ARGS                        "ArrayArgs"
#define ARRAY_ARGS_MSG                    "Array must be called with integer dimensions as arguments."

#define ARRAY_INIT                        "ArrayInit"
#define ARRAY_INIT_MSG                    "Array initializer must be another array or a list."

#define ARRAY_CMPT                        "ArrayCmpt"
#define ARRAY_CMPT_MSG                    "Array initializer is not compatible with the requested dimensions."

/* -------------------------------------------------------
 * Array interface
 * ------------------------------------------------------- */

/* Public interfaces to various data structures */
typedef enum { ARRAY_OK, ARRAY_WRONGDIM, ARRAY_OUTOFBOUNDS,ARRAY_NONINTINDX } objectarrayerror;

errorid array_error(objectarrayerror err);
errorid array_to_matrix_error(objectarrayerror err);
errorid array_to_list_error(objectarrayerror err);

bool array_valuelisttoindices(unsigned int ndim, value *in, unsigned int *out);
objectarrayerror array_getelement(objectarray *a, unsigned int ndim, unsigned int *indx, value *out);
objectarrayerror array_setelement(objectarray *a, unsigned int ndim, unsigned int *indx, value in);
void array_print(vm *v, objectarray *a);
objectarrayerror setslicerecursive(value* a, value* out,objectarrayerror copy(value * ,value *,\
                                    unsigned int, unsigned int *,unsigned int *),unsigned int ndim,\
                                    unsigned int curdim, unsigned int *indx,unsigned int *newindx, value *slices);
objectarrayerror getslice(value *a, bool dimFcn(value *,unsigned int),\
                          void constuctor(unsigned int *,unsigned int,value *),\
                          objectarrayerror copy(value * ,value *, unsigned int, unsigned int *,unsigned int *),\
                          unsigned int ndim, value *slices, value *out);
objectarrayerror array_slicecopy(value * a,value * out, unsigned int ndim, unsigned int *indx,unsigned int *newindx);
void array_sliceconstructor(unsigned int *slicesize,unsigned int ndim,value* out);
bool array_slicedim(value * a, unsigned int ndim);

void array_initialize(void);

#endif
