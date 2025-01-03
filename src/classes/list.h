/** @file list.h
 *  @author T J Atherton
 *
 *  @brief Defines list object type and List class
 */

#ifndef list_h
#define list_h

#include "object.h"

/* -------------------------------------------------------
 * List object type
 * ------------------------------------------------------- */

extern objecttype objectlisttype;
#define OBJECT_LIST objectlisttype

typedef struct {
    object obj;
    varray_value val;
} objectlist;

/** Tests whether an object is a list */
#define MORPHO_ISLIST(val) object_istype(val, OBJECT_LIST)

/** Gets the object as a list */
#define MORPHO_GETLIST(val)   ((objectlist *) MORPHO_GETOBJECT(val))

/** Create a static list - you must initialize the list separately */
#define MORPHO_STATICLIST      { .obj.type=OBJECT_LIST, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .val.count=0, .val.capacity=0, .val.data=NULL }

objectlist *object_newlist(unsigned int nval, value *val);

/* -------------------------------------------------------
 * List veneer class
 * ------------------------------------------------------- */

#define LIST_CLASSNAME                    "List"

#define LIST_ISMEMBER_METHOD              "ismember"
#define LIST_SORT_METHOD                  "sort"
#define LIST_ORDER_METHOD                 "order"
#define LIST_POP_METHOD                   "pop"
#define LIST_INSERT_METHOD                "insert"
#define LIST_REMOVE_METHOD                "remove"
#define LIST_TUPLES_METHOD                "tuples"
#define LIST_SETS_METHOD                  "sets"
#define LIST_REVERSE_METHOD               "reverse"

/* -------------------------------------------------------
 * List error messages
 * ------------------------------------------------------- */

#define LIST_ENTRYNTFND                   "EntryNtFnd"
#define LIST_ENTRYNTFND_MSG               "Entry not found."

#define LIST_ADDARGS                      "LstAddArgs"
#define LIST_ADDARGS_MSG                  "Add method requires a list."

#define LIST_SRTFN                        "LstSrtFn"
#define LIST_SRTFN_MSG                    "List sort function must return an integer."

#define LIST_ARGS                         "LstArgs"
#define LIST_ARGS_MSG                     "Lists must be called with integer dimensions as arguments."

#define LIST_NUMARGS                      "LstNumArgs"
#define LIST_NUMARGS_MSG                  "Lists can only be indexed with one argument."

/* -------------------------------------------------------
 * List interface
 * ------------------------------------------------------- */

value List_getindex(vm *v, int nargs, value *args);

bool list_resize(objectlist *list, int size);
void list_append(objectlist *list, value v);
unsigned int list_length(objectlist *list);
bool list_getelement(objectlist *list, int i, value *out);
void list_sort(objectlist *list);
objectlist *list_clone(objectlist *list);

void list_initialize(void);

#endif
