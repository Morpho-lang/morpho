/** @file list.c
 *  @author T J Atherton
 *
 *  @brief Implements the List class
 */

#include "morpho.h"
#include "classes.h"
#include "common.h"

/* **********************************************************************
 * objectlist definitions
 * ********************************************************************** */

void objectlist_printfn(object *obj, void *v) {
    morpho_printf(v, "<List>");
}

void objectlist_freefn(object *obj) {
    objectlist *list = (objectlist *) obj;
    varray_valueclear(&list->val);
}

void objectlist_markfn(object *obj, void *v) {
    objectlist *c = (objectlist *) obj;
    morpho_markvarrayvalue(v, &c->val);
}

size_t objectlist_sizefn(object *obj) {
    return sizeof(objectlist)+sizeof(value) *
            ((objectlist *) obj)->val.capacity;
}

objecttypedefn objectlistdefn = {
    .printfn=objectlist_printfn,
    .markfn=objectlist_markfn,
    .freefn=objectlist_freefn,
    .sizefn=objectlist_sizefn
};

/** Creates a new list */
objectlist *object_newlist(unsigned int nval, value *val) {
    objectlist *new = (objectlist *) object_new(sizeof(objectlist), OBJECT_LIST);

    if (new) {
        varray_valueinit(&new->val);
        if (val) varray_valueadd(&new->val, val, nval);
        else varray_valueresize(&new->val, nval);
    }

    return new;
}

/* **********************************************************************
 * objectlist utility functions
 * ********************************************************************** */

/** Resizes a list */
bool list_resize(objectlist *list, int size) {
    return varray_valueresize(&list->val, size);
}

/** Appends an item to a list */
void list_append(objectlist *list, value v) {
    varray_valuewrite(&list->val, v);
}

/** Appends an item to a list */
unsigned int list_length(objectlist *list) {
    return list->val.count;
}

/** Removes an element from a list
 * @param[in] list a list object
 * @param[in] indx position to insert
 * @param[in] nval number of values to insert
 * @param[in] vals the entries to insert
 * @returns true on success */
bool list_insert(objectlist *list, int indx, int nval, value *vals) {
    int i = indx;
    while (i<0) i+=list->val.count+1;
    if (i>list->val.count) return false;
    if (nval>list->val.capacity-list->val.count) if (!list_resize(list, list->val.count+nval)) return false;

    memmove(list->val.data+i+nval, list->val.data+i, sizeof(value)*(list->val.count-i));
    memcpy(list->val.data+i, vals, sizeof(value)*nval);

    list->val.count+=nval;

    return true;
}

/** Removes an element from a list
 * @param[in] list a list object
 * @param[in] val the entry to remove
 * @returns true on success */
bool list_remove(objectlist *list, value val) {
    /* Find the element */
    for (unsigned int i=0; i<list->val.count; i++) {
        if (MORPHO_ISEQUAL(list->val.data[i], val)) { /* Remove it if we're not at the end of the list */
            if (i<list->val.count-1) memmove(list->val.data+i, list->val.data+i+1, sizeof(value)*(list->val.count-i-1));
            list->val.count--;
            return true;
        }
    }

    return false;
}

/** Gets an element from a list
 * @param[in] list a list object
 * @param[in] i the index (may be negative)
 * @param[in] out filled out on exit if index is in bounds
 * @returns true on success */
bool list_getelement(objectlist *list, int i, value *out) {
    if (!(i>=-(int) list->val.count && i<(int) list->val.count)) return false;
    if (i>=0) *out=list->val.data[i];
    else *out=list->val.data[list->val.count+i];
    return true;
}

/** Sort function for list_sort */
int list_sortfunction(const void *a, const void *b) {
    value l=*(value *) a, r=*(value *) b;
    MORPHO_CMPPROMOTETYPE(l, r);
    return -morpho_comparevalue(l, r);
}

/** Sort the contents of a list */
void list_sort(objectlist *list) {
    qsort(list->val.data, list->val.count, sizeof(value), list_sortfunction);
}

static vm *list_sortwithfn_vm;
static value list_sortwithfn_fn;
static bool list_sortwithfn_err;

/** Sort function for list_sort */
int list_sortfunctionwfn(const void *a, const void *b) {
    value args[2] = {*(value *) a, *(value *) b};
    value ret;

    if (morpho_call(list_sortwithfn_vm, list_sortwithfn_fn, 2, args, &ret)) {
        if (MORPHO_ISINTEGER(ret)) return MORPHO_GETINTEGERVALUE(ret);
        if (MORPHO_ISFLOAT(ret)) return morpho_comparevalue(MORPHO_FLOAT(0), ret);
    }

    list_sortwithfn_err=true;
    return 0;
}

/** Sort the contents of a list */
bool list_sortwithfn(vm *v, value fn, objectlist *list) {
    list_sortwithfn_vm=v;
    list_sortwithfn_fn=fn;
    list_sortwithfn_err=false;
    qsort(list->val.data, list->val.count, sizeof(value), list_sortfunctionwfn);
    return !list_sortwithfn_err;
}

/** Sort function for list_order */
typedef struct {
    unsigned int indx;
    value val;
} listorderstruct;

/** Sort function for list_order */
int list_orderfunction(const void *a, const void *b) {
    return -morpho_comparevalue(((listorderstruct *) a)->val, ((listorderstruct *) b)->val);
}

/* Returns a list of indices giving the ordering of a list */
objectlist *list_order(objectlist *list) {
    listorderstruct *order = MORPHO_MALLOC(list->val.count*sizeof(listorderstruct));
    objectlist *new = NULL;

    if (order) {
        for (unsigned int i=0; i<list->val.count; i++) {
            order[i].indx=i;
            order[i].val=list->val.data[i];
        }
        qsort(order, list->val.count, sizeof(listorderstruct), list_orderfunction);

        new=object_newlist(list->val.count, NULL);
        if (new) {
            for (unsigned int i=0; i<list->val.count; i++) {
                new->val.data[i]=MORPHO_INTEGER(order[i].indx);
            }
            new->val.count=list->val.count;
        }

        MORPHO_FREE(order);
    }
    return new;
}

/** Reverses a list in place */
void list_reverse(objectlist *list) {
    unsigned int hlen = list->val.count / 2;
    for (unsigned int i=0; i<hlen; i++) {
        value swp = list->val.data[i];
        unsigned int j=list->val.count-i-1;
        list->val.data[i]=list->val.data[j];
        list->val.data[j]=swp;
    }
}

/** Tests if a value is a member of a list */
bool list_ismember(objectlist *list, value v) {
    for (unsigned int i=0; i<list->val.count; i++) {
        if (MORPHO_ISEQUAL(list->val.data[i], v)) return true;
    }
    return false;
}

/** Clones a list */
objectlist *list_clone(objectlist *list) {
    return object_newlist(list->val.count, list->val.data);
}

/* Copies data from list a at position indx to list out at position newindx with a generic interface */
objectarrayerror list_slicecopy(value * a,value * out, unsigned int ndim, unsigned int *indx,unsigned int *newindx){
    value data;
    objectlist *outList = MORPHO_GETLIST(*out);

    if (list_getelement(MORPHO_GETLIST(*a),indx[0],&data)){
        outList->val.data[newindx[0]] = data;
    } else return ARRAY_OUTOFBOUNDS;
    return ARRAY_OK;
}

/** Concatenates two lists */
objectlist *list_concatenate(objectlist *a, objectlist *b) {
    objectlist *new=object_newlist(a->val.count+b->val.count, NULL);

    if (new) {
        memcpy(new->val.data, a->val.data, sizeof(value)*a->val.count);
        memcpy(new->val.data+a->val.count, b->val.data, sizeof(value)*b->val.count);
        new->val.count=a->val.count+b->val.count;
    }

    return new;
}

/** Rolls a list by a number of elements */
objectlist *list_roll(objectlist *a, int nplaces) {
    objectlist *new=object_newlist(a->val.count, NULL);
    
    if (new) {
        new->val.count=a->val.count;
        unsigned int N = a->val.count;
        int n = abs(nplaces);
        if (n>N) n = n % N;
        unsigned int Np = N - n; // Number of elements to roll
        
        if (nplaces<0) {
            memcpy(new->val.data, a->val.data+n, sizeof(value)*Np);
            memcpy(new->val.data+Np, a->val.data, sizeof(value)*n);
        } else {
            memcpy(new->val.data+n, a->val.data, sizeof(value)*Np);
            if (n>0) memcpy(new->val.data, a->val.data+Np, sizeof(value)*n);
        }
    }

    return new;
}

/** Loop function for enumerable initializers */
static bool list_enumerableinitializer(vm *v, indx i, value val, void *ref) {
    objectlist *list = (objectlist *) ref;
    list_append(list, val);
    return true;
}

/* Constructs a new list of a given size with a generic interface */
void list_sliceconstructor(unsigned int *slicesize,unsigned int ndim,value* out){
    objectlist *list = object_newlist(slicesize[0], NULL);
    list->val.count = slicesize[0];
    *out = MORPHO_OBJECT(list);
}

/* Checks that a list is indexed with only one value with a generic interface */
bool list_slicedim(value * a, unsigned int ndim){
    if (ndim>1||ndim<0) return false;
    return true;
}

/** Generate sets/tuples and return as a list of lists */
value list_generatetuples(vm *v, objectlist *list, unsigned int n, tuplemode mode) {
    unsigned int nval=list->val.count;
    unsigned int work[2*n];
    value tuple[n];
    morpho_tuplesinit(list->val.count, n, work, mode);
    objectlist *new = object_newlist(0, NULL);
    if (!new) goto list_generatetuples_cleanup;

    while (morpho_tuples(nval, list->val.data, n, work, mode, tuple)) {
        objectlist *el = object_newlist(n, tuple);
        if (el) {
            list_append(new, MORPHO_OBJECT(el));
        } else {
            goto list_generatetuples_cleanup;
        }
    }

    list_append(new, MORPHO_OBJECT(new));
    morpho_bindobjects(v, new->val.count, new->val.data);
    new->val.count--; // And pop it back off

    return MORPHO_OBJECT(new);

list_generatetuples_cleanup:
    morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);

    if (new) { // Deallocate partially created list
        for (unsigned int i=0; i<new->val.count; i++) {
            value el=new->val.data[i];
            if (MORPHO_ISOBJECT(el)) object_free(MORPHO_GETOBJECT(el));
        }
        object_free((object *) new);
    }

    return MORPHO_NIL;
}

/* **********************************************************************
 * List veneer class
 * ********************************************************************** */

/** Constructor function for Lists */
value list_constructor(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    value init=MORPHO_NIL;
    objectlist *new=NULL;

    if (nargs==1 && MORPHO_ISRANGE(MORPHO_GETARG(args, 0))) {
        init = MORPHO_GETARG(args, 0);
        new = object_newlist(0, NULL);
    } else new = object_newlist(nargs, args+1);

    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);

        if (!MORPHO_ISNIL(init)) {
            builtin_enumerateloop(v, init, list_enumerableinitializer, new);
        }
    }

    return out;
}

/** Append an element to a list */
value List_append(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));

    unsigned int capacity = slf->val.capacity;

    varray_valueadd(&slf->val, args+1, nargs);

    if (slf->val.capacity!=capacity) morpho_resizeobject(v, (object *) slf, capacity*sizeof(value)+sizeof(objectlist), slf->val.capacity*sizeof(value)+sizeof(objectlist));

    return MORPHO_SELF(args);
}

/** Remove a element from a list */
value List_remove(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));

    if (nargs==1) {
        if (!list_remove(slf, MORPHO_GETARG(args, 0))) morpho_runtimeerror(v, LIST_ENTRYNTFND);
    } else morpho_runtimeerror(v, VM_INVALIDARGS, 1, nargs);

    return MORPHO_NIL;
}

/** Inserts an element at a specified position */
value List_insert(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));

    if (nargs>=2) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            int indx = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
            if (!list_insert(slf, indx, nargs-1, &MORPHO_GETARG(args, 1))) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        }
    } else morpho_runtimeerror(v, VM_INVALIDARGS, 2, nargs);

    return MORPHO_NIL;
}

/** Pops an element from the end of a list */
value List_pop(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (slf->val.count>0) {
        if (nargs>0 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            int indx = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
            out=slf->val.data[indx];
            memmove(slf->val.data+indx, slf->val.data+indx+1, sizeof(value)*(slf->val.count-indx-1));
        } else {
            out=slf->val.data[slf->val.count-1];
        }
        slf->val.count--;
    }

    return out;
}

/** Get an element */
value List_getindex(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (nargs==1) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

            if (!list_getelement(slf, i, &out)) {
                morpho_runtimeerror(v, VM_OUTOFBOUNDS);
            }
        } else {
            objectarrayerror err = getslice(&MORPHO_SELF(args),&list_slicedim,&list_sliceconstructor,&list_slicecopy,nargs,&MORPHO_GETARG(args, 0),&out);
            if (err!=ARRAY_OK) MORPHO_RAISE(v, array_to_list_error(err) );
            if (MORPHO_ISOBJECT(out)){
                morpho_bindobjects(v,1,&out);
            } else MORPHO_RAISE(v, VM_NONNUMINDX);

        }
    } else MORPHO_RAISE(v, LIST_NUMARGS)

    return out;
}

/** Sets an element */
value List_setindex(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));

    if (nargs==2) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
            int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
            if (i<slf->val.count) slf->val.data[i]=MORPHO_GETARG(args, 1);
            else morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        } else morpho_runtimeerror(v, SETINDEX_ARGS);
    } else morpho_runtimeerror(v, SETINDEX_ARGS);

    return MORPHO_SELF(args);
}

/** Print a list */
value List_print(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    if (!MORPHO_ISLIST(self)) return Object_print(v, nargs, args);
    
    objectlist *lst=MORPHO_GETLIST(self);

    morpho_printf(v, "[ ");
    for (unsigned int i=0; i<lst->val.count; i++) {
        morpho_printvalue(v, lst->val.data[i]);
        if (i<lst->val.count-1) morpho_printf(v, ", ");
    }
    morpho_printf(v, " ]");

    return MORPHO_NIL;
}

/** Convert a list to a string */
value List_tostring(vm *v, int nargs, value *args) {
    objectlist *lst=MORPHO_GETLIST(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    varray_char buffer;
    varray_charinit(&buffer);

    varray_charadd(&buffer, "[ ", 2);
    for (unsigned int i=0; i<lst->val.count; i++) {
        morpho_printtobuffer(v, lst->val.data[i], &buffer);
        if (i<lst->val.count-1) varray_charadd(&buffer, ", ", 2);
    }
    varray_charadd(&buffer, " ]", 2);

    out = object_stringfromvarraychar(&buffer);
    if (MORPHO_ISSTRING(out)) {
        morpho_bindobjects(v, 1, &out);
    }
    varray_charclear(&buffer);

    return out;
}

/** Enumerate members of a list */
value List_enumerate(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        int n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));

        if (n<0) {
            out=MORPHO_INTEGER(slf->val.count);
        } else if (n<slf->val.count) {
            return slf->val.data[n];
        } else {
            morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        }
    } else MORPHO_RAISE(v, ENUMERATE_ARGS);

    return out;
}

/** Get number of entries */
value List_count(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));

    return MORPHO_INTEGER(slf->val.count);
}

/** Generate a list of n-tuples from a list  */
value List_tuples(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    unsigned int n=2;

    if (nargs>0 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        if (n<2) n=2;
    }

    return list_generatetuples(v, slf, n, MORPHO_TUPLEMODE);
}

/** Generate a list of n-tuples from a list  */
value List_sets(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    unsigned int n=2;

    if (nargs>0 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        n=MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        if (n<2) n=2;
        if (n>slf->val.capacity) n = slf->val.capacity;
    }

    return list_generatetuples(v, slf, n, MORPHO_SETMODE);
}

/** Clones a list */
value List_clone(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    objectlist *new = list_clone(slf);
    if (!new) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    value out = MORPHO_OBJECT(new);
    morpho_bindobjects(v, 1, &out);
    return out;
}

/** Arithmetic add of two lists  */
value List_add(vm *v, int nargs, value *args) {
    UNREACHABLE("API for list add has changed.\n");
    return MORPHO_NIL;
}

/** Joins two lists together  */
value List_join(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    if (nargs==1 && MORPHO_ISLIST(MORPHO_GETARG(args, 0))) {
        objectlist *operand = MORPHO_GETLIST(MORPHO_GETARG(args, 0));
        objectlist *new = list_concatenate(slf, operand);

        if (new) {
            out = MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        }

    } else morpho_runtimeerror(v, LIST_ADDARGS);

    return out;
}

/** Roll a list */
value List_roll(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    if (nargs==1 &&
        MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        int roll;
        morpho_valuetoint(MORPHO_GETARG(args, 0), &roll);
        
        objectlist *new = list_roll(slf, roll);

        if (new) {
            out = MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        }

    } else morpho_runtimeerror(v, LIST_ADDARGS);

    return out;
}

/** Sorts a list */
value List_sort(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));

    if (nargs==0) {
        list_sort(slf);
    } else if (nargs==1 && MORPHO_ISCALLABLE(MORPHO_GETARG(args, 0))) {
        if (!list_sortwithfn(v, MORPHO_GETARG(args, 0), slf)) {
            morpho_runtimeerror(v, LIST_SRTFN);
        }
    }

    return MORPHO_NIL;
}

/** Returns a list of indices that would sort the list self */
value List_order(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    objectlist *new=list_order(slf);
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);

    return out;
}

/** Returns a list with the order reversed */
value List_reverse(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));
    
    list_reverse(slf);
    
    return MORPHO_NIL;
}

/** Tests if a list has a value as a member */
value List_ismember(vm *v, int nargs, value *args) {
    objectlist *slf = MORPHO_GETLIST(MORPHO_SELF(args));

    if (nargs==1) {
        return MORPHO_BOOL(list_ismember(slf, MORPHO_GETARG(args, 0)));
    } else morpho_runtimeerror(v, ISMEMBER_ARG, 1, nargs);

    return MORPHO_NIL;
}

MORPHO_BEGINCLASS(List)
MORPHO_METHOD(MORPHO_APPEND_METHOD, List_append, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_REMOVE_METHOD, List_remove, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_INSERT_METHOD, List_insert, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_POP_METHOD, List_pop, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, List_getindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, List_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, List_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_TOSTRING_METHOD, List_tostring, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ENUMERATE_METHOD, List_enumerate, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, List_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_TUPLES_METHOD, List_tuples, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_SETS_METHOD, List_sets, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, List_clone, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADD_METHOD, List_add, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_JOIN_METHOD, List_join, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ROLL_METHOD, List_roll, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_SORT_METHOD, List_sort, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_ORDER_METHOD, List_order, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_REVERSE_METHOD, List_reverse, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(LIST_ISMEMBER_METHOD, List_ismember, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CONTAINS_METHOD, List_ismember, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

objecttype objectlisttype;

void list_initialize(void) {
    // Define list objecttype
    objectlisttype=object_addtype(&objectlistdefn);
    
    // Locate the Object class to use as the parent class of Range
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // List constructor function
    builtin_addfunction(LIST_CLASSNAME, list_constructor, BUILTIN_FLAGSEMPTY);
    
    // List constructor function
    value listclass=builtin_addclass(LIST_CLASSNAME, MORPHO_GETCLASSDEFINITION(List), objclass);
    object_setveneerclass(OBJECT_LIST, listclass);
    
    // List error messages
    morpho_defineerror(LIST_ENTRYNTFND, ERROR_HALT, LIST_ENTRYNTFND_MSG);
    morpho_defineerror(LIST_ADDARGS, ERROR_HALT, LIST_ADDARGS_MSG);
    morpho_defineerror(LIST_SRTFN, ERROR_HALT, LIST_SRTFN_MSG);
    morpho_defineerror(LIST_ARGS, ERROR_HALT, LIST_ARGS_MSG);
    morpho_defineerror(LIST_NUMARGS, ERROR_HALT, LIST_NUMARGS_MSG);
}

void list_finalize(void) {
}
