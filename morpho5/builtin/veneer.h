/** @file veneer.h
 *  @author T J Atherton
 *
 *  @brief Veneer classes over built in objects
 */

#ifndef veneer_h
#define veneer_h

#include "builtin.h"

/* ---------------------------
 * Veneer classes
 * --------------------------- */

#define OBJECT_CLASSNAME "Object"
#define STRING_CLASSNAME "String"
#define ARRAY_CLASSNAME "Array"
#define LIST_CLASSNAME "List"
#define DICTIONARY_CLASSNAME "Dictionary"
#define RANGE_CLASSNAME "Range"

#define STRING_SPLIT_METHOD "split"
#define ARRAY_DIMENSIONS_METHOD "dimensions"

#define LIST_ISMEMBER_METHOD "ismember"
#define LIST_SORT_METHOD "sort"
#define LIST_ORDER_METHOD "order"
#define LIST_POP_METHOD "pop"
#define LIST_REMOVE_METHOD "remove"

#define DICTIONARY_KEYS_METHOD "keys"

#define RANGE_ARGS                        "RngArgs"
#define RANGE_ARGS_MSG                    "Range expects numerical arguments: a start, an end and an optional stepsize."

#define SETINDEX_ARGS                     "SetIndxArgs"
#define SETINDEX_ARGS_MSG                 "Setindex method expects an index and a value as arguments."

#define ENUMERATE_ARGS                    "EnmrtArgs"
#define ENUMERATE_ARGS_MSG                "Enumerate method expects a single integer argument."

#define DICT_DCTKYNTFND                   "DctKyNtFnd"
#define DICT_DCTKYNTFND_MSG               "Key not found in dictionary."

#define RESPONDSTO_ARG                    "RspndsToArg"
#define RESPONDSTO_ARG_MSG                "Method respondsto expects a single string argument."

#define ISMEMBER_ARG                      "IsMmbrArg"
#define ISMEMBER_ARG_MSG                  "Method ismember expects a single argument."

#define DICT_DCTSTARG                     "DctStArg"
#define DICT_DCTSTARG_MSG                 "Dictionary set methods (union, intersection, difference) expect a dictionary as the argument."

#define CLASS_INVK                        "ClssInvk"
#define CLASS_INVK_MSG                    "Cannot invoke method '%s' on a class."

#define LIST_ENTRYNTFND                   "EntryNtFnd"
#define LIST_ENTRYNTFND_MSG               "Entry not found."

#define LIST_ADDARGS                      "LstAddArgs"
#define LIST_ADDARGS_MSG                  "Add method requires a list."

#define STRING_IMMTBL                     "StrngImmtbl"
#define STRING_IMMTBL_MSG                 "Strings are immutable."

#define ARRAY_ARGS                        "ArrayArgs"
#define ARRAY_ARGS_MSG                    "Array much be called with integer arguments."

/* Public interfaces to various data structures */
typedef enum { ARRAY_OK, ARRAY_WRONGDIM, ARRAY_NONNUMERICALINDX, ARRAY_OUTOFBOUNDS } objectarrayerror;

int string_countchars(objectstring *s);
char *string_index(objectstring *s, int i);

errorid array_error(objectarrayerror err);

bool array_valuestoindices(unsigned int ndim, value *indx, unsigned int *iout);
bool array_indicestoelement(objectarray *array, unsigned int ndim, unsigned int *indx, unsigned int *ixout);
objectarrayerror array_getelement(objectarray *array, unsigned int ndim, value *indx, value *out);
objectarrayerror array_setelement(objectarray *array, unsigned int ndim, value *indx, value set);

bool list_resize(objectlist *list, int size);
void list_append(objectlist *list, value v);
unsigned int list_length(objectlist *list);
bool list_getelement(objectlist *list, int i, value *out);
void list_sort(objectlist *list);
objectlist *list_clone(objectlist *list);

void veneer_initialize(void);

#endif /* veneer_h */
