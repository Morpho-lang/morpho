/** @file veneer.h
 *  @author T J Atherton
 *
 *  @brief Veneer classes over built in objects
 */

#ifndef veneer_h
#define veneer_h

#include "builtin.h"
#include "matrix.h"
#include "list.h"
#include "range.h"
#include "strng.h"

/* ---------------------------
 * Veneer classes
 * --------------------------- */

#define OBJECT_CLASSNAME "Object"
#define ARRAY_CLASSNAME "Array"
#define DICTIONARY_CLASSNAME "Dictionary"
#define FUNCTION_CLASSNAME "Function"
#define CLOSURE_CLASSNAME "Closure"
#define INVOCATION_CLASSNAME "Invocation"
#define ERROR_CLASSNAME "Error"

#define ERROR_TAG_PROPERTY "tag"
#define ERROR_MESSAGE_PROPERTY "message"

#define STRING_SPLIT_METHOD "split"
#define STRING_ISNUMBER_METHOD "isnumber"
#define ARRAY_DIMENSIONS_METHOD "dimensions"

#define DICTIONARY_KEYS_METHOD "keys"
#define DICTIONARY_CONTAINS_METHOD "contains"
#define DICTIONARY_REMOVE_METHOD "remove"
#define DICTIONARY_CLEAR_METHOD "clear"

#define SETINDEX_ARGS                     "SetIndxArgs"
#define SETINDEX_ARGS_MSG                 "Setindex method expects an index and a value as arguments."

#define ENUMERATE_ARGS                    "EnmrtArgs"
#define ENUMERATE_ARGS_MSG                "Enumerate method expects a single integer argument."

#define DICT_DCTKYNTFND                   "DctKyNtFnd"
#define DICT_DCTKYNTFND_MSG               "Key not found in dictionary."

#define RESPONDSTO_ARG                    "RspndsToArg"
#define RESPONDSTO_ARG_MSG                "Method respondsto expects a single string argument or no argrument."

#define HAS_ARG                    		  "HasArg"
#define HAS_ARG_MSG                		  "Method has expects a single string argument or no argument."

#define ISMEMBER_ARG                      "IsMmbrArg"
#define ISMEMBER_ARG_MSG                  "Method ismember expects a single argument."

#define DICT_DCTSTARG                     "DctStArg"
#define DICT_DCTSTARG_MSG                 "Dictionary set methods (union, intersection, difference) expect a dictionary as the argument."

#define CLASS_INVK                        "ClssInvk"
#define CLASS_INVK_MSG                    "Cannot invoke method '%s' on a class."

#define ARRAY_ARGS                        "ArrayArgs"
#define ARRAY_ARGS_MSG                    "Array must be called with integer dimensions as arguments."

#define ARRAY_INIT                        "ArrayInit"
#define ARRAY_INIT_MSG                    "Array initializer must be another array or a list."

#define ARRAY_CMPT                        "ArrayCmpt"
#define ARRAY_CMPT_MSG                    "Array initializer is not compatible with the requested dimensions."

#define INVOCATION_ARGS                   "InvocationArgs"
#define INVOCATION_ARGS_MSG               "Invocation must be called with an object and a method name as arguments."

#define INVOCATION_METHOD                 "InvocationMethod"
#define INVOCATION_METHOD_MSG             "Method not found."

#define ERROR_ARGS                        "ErrorArgs"
#define ERROR_ARGS_MSG                    "Error much be called with a tag and a default message as arguments."

#define OBJECT_CANTCLONE                  "ObjCantClone"
#define OBJECT_CANTCLONE_MSG              "Cannot clone this object."

#define OBJECT_IMMUTABLE                  "ObjImmutable"
#define OBJECT_IMMUTABLE_MSG              "Cannot modify this object."

/* Object methods */
value Object_print(vm *v, int nargs, value *args);

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

void veneer_initialize(void);

#endif /* veneer_h */
