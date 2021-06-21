/** @file object.h
 *  @author T J Atherton
 *
 *  @brief Provide functionality for extended and mutable data types.
*/

#ifndef object_h
#define object_h

#include <stddef.h>
#include "value.h"
#include "dictionary.h"

typedef ptrdiff_t indx;

void object_initialize(void);
void object_finalize(void);

/* ---------------------------
 * Generic objects
 * --------------------------- */

/** The type of an object */
typedef enum {
    OBJECT_STRING,
    OBJECT_FUNCTION,
    OBJECT_BUILTINFUNCTION,
    OBJECT_CLOSURE,
    OBJECT_UPVALUE,
    OBJECT_CLASS,
    OBJECT_INSTANCE,
    OBJECT_INVOCATION,
    OBJECT_RANGE,
    OBJECT_LIST,
    OBJECT_DICTIONARY,
    OBJECT_ARRAY,
    OBJECT_MATRIX,
    OBJECT_SPARSE,
    OBJECT_DOKKEY,
    
    /* Geometry classes */
    OBJECT_MESH,
    OBJECT_SELECTION,
    OBJECT_FIELD,
    
    OBJECT_EXTERN /* Intended for objects that are only visible to morpho and not involved in the runtime e.g. the help system.  */
} objecttype;

/** Simplest object */
struct sobject {
    objecttype type;
    enum {
        OBJECT_ISUNMANAGED,
        OBJECT_ISUNMARKED,
        OBJECT_ISMARKED
    } status;
    hash hsh;
    struct sobject *next; 
};

/** Gets the type of the object associated with a value */
#define MORPHO_GETOBJECTTYPE(val)           (MORPHO_GETOBJECT(val)->type)

/** Gets an objects key */
#define MORPHO_GETOBJECTHASH(val)           (MORPHO_GETOBJECT(val)->hsh)
/** Sets an objects key */
#define MORPHO_SETOBJECTHASH(val, newhash)  (MORPHO_GETOBJECT(val)->hsh = newhash)

/** Tests whether an object is of a specified type */
static inline bool object_istype(value val, objecttype type) {
    return (MORPHO_ISOBJECT(val) && MORPHO_GETOBJECTTYPE(val)==type);
}

void object_init(object *obj, objecttype type);
void object_free(object *obj);
void object_freeunmanaged(object *obj);
void object_print(value v);
void object_printtobuffer(value v, varray_char *buffer);
object *object_new(size_t size, objecttype type);
size_t object_size(object *obj);

static inline void morpho_freeobject(value val) {
    if (MORPHO_ISOBJECT(val)) object_free(MORPHO_GETOBJECT(val));
}

/* ---------------------------
 * Strings
 * --------------------------- */

/** A string object */
typedef struct {
    object obj;
    size_t length;
    char *string; 
    char stringdata[];
} objectstring;

#define MORPHO_GETSTRING(val)             ((objectstring *) MORPHO_GETOBJECT(val))
#define MORPHO_GETCSTRING(val)            (((objectstring *) MORPHO_GETOBJECT(val))->string)
#define MORPHO_GETSTRINGLENGTH(val)       (((objectstring *) MORPHO_GETOBJECT(val))->length)

/** Tests whether an object is a string */
#define MORPHO_ISSTRING(val) object_istype(val, OBJECT_STRING)

/** Use to create static strings on the C stack */
#define MORPHO_STATICSTRING(cstring)      { .obj.type=OBJECT_STRING, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .string=cstring, .length=strlen(cstring) }

/** Use to create static strings on the C stack */
#define MORPHO_STATICSTRINGWITHLENGTH(cstring, len)      { .obj.type=OBJECT_STRING, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .string=cstring, .length=len }


#define OBJECT_STRINGLABEL "string"
#define OBJECT_SYMBOLLABEL "symbol"

value object_stringfromcstring(const char *in, size_t length);
value object_stringfromvarraychar(varray_char *in);
value object_clonestring(value val);
value object_concatenatestring(value a, value b);

/* -------------------------------------------------------
 * Upvalues
 * ------------------------------------------------------- */

/** Each upvalue */
typedef struct {
    bool islocal; /** Set if the upvalue is local to this function */
    indx reg; /** An index that either:
                  if islocal - refers to the register
               OR otherwise  - refers to the upvalue array in the current closure */
} upvalue;

DECLARE_VARRAY(upvalue, upvalue)

DECLARE_VARRAY(varray_upvalue, varray_upvalue)

/* ---------------------------
 * Functions
 * --------------------------- */

typedef struct {
    value symbol; /** Symbol associated with the variable */
    indx def; /** Default value as constant */
    indx reg; /** Associated register */
} optionalparam;

DECLARE_VARRAY(optionalparam, optionalparam)

/** A function object */
typedef struct sobjectfunction {
    object obj;
    int nargs;
    value name;
    indx entry;
    struct sobjectfunction *parent;
    int nupvalues;
    int nregs; 
    varray_value konst;
    varray_varray_upvalue prototype;
    varray_optionalparam opt;
} objectfunction;

/** Gets an objectfunction from a value */
#define MORPHO_GETFUNCTION(val)   ((objectfunction *) MORPHO_GETOBJECT(val))

/** Tests whether an object is a function */
#define MORPHO_ISFUNCTION(val) object_istype(val, OBJECT_FUNCTION)

void object_functioninit(objectfunction *func);
void object_functionclear(objectfunction *func);
bool object_functionaddprototype(objectfunction *func, varray_upvalue *v, indx *ix);
objectfunction *object_getfunctionparent(objectfunction *func);
value object_getfunctionname(objectfunction *func);
varray_value *object_functiongetconstanttable(objectfunction *func);
objectfunction *object_newfunction(indx entry, value name, objectfunction *parent, unsigned int nargs);

/* ---------------------------
 * Upvalue objects
 * --------------------------- */

typedef struct sobjectupvalue {
    object obj;
    value* location; /** Pointer to the location of the upvalue */
    value  closed; /** Closed value of the upvalue */
    struct sobjectupvalue *next;
} objectupvalue;

void object_upvalueinit(objectupvalue *c);
objectupvalue *object_newupvalue(value *reg);

/** Gets an objectfunction from a value */
#define MORPHO_GETUPVALUE(val)   ((objectupvalue *) MORPHO_GETOBJECT(val))

/** Tests whether an object is a function */
#define MORPHO_ISUPVALUE(val) object_istype(val, OBJECT_UPVALUE)

/* ---------------------------
 * Closures
 * --------------------------- */

typedef struct {
    object obj;
    objectfunction *func;
    int nupvalues;
    objectupvalue *upvalues[];
} objectclosure;

objectclosure *object_newclosure(objectfunction *sf, objectfunction *func, indx np);

/** Tests whether an object is a closure */
#define MORPHO_ISCLOSURE(val) object_istype(val, OBJECT_CLOSURE)

/** Gets the object as a closure */
#define MORPHO_GETCLOSURE(val)   ((objectclosure *) MORPHO_GETOBJECT(val))

/** Retrieve the function object from a closure */
#define MORPHO_GETCLOSUREFUNCTION(val)  (((objectclosure *) MORPHO_GETOBJECT(val))->func)

/* ---------------------------
 * Classes
 * --------------------------- */

typedef struct sobjectclass {
    object obj;
    struct sobjectclass *superclass;
    value name;
    dictionary methods;
} objectclass;

/** Tests whether an object is a class */
#define MORPHO_ISCLASS(val) object_istype(val, OBJECT_CLASS)

/** Gets the object as a class */
#define MORPHO_GETCLASS(val)   ((objectclass *) MORPHO_GETOBJECT(val))

/** Gets the superclass */
#define MORPHO_GETSUPERCLASS(val)   (MORPHO_GETCLASS(val)->superclass)

objectclass *object_newclass(value name);

/* ---------------------------
 * Instances
 * --------------------------- */

typedef struct {
    object obj;
    objectclass *klass;
    dictionary fields;
} objectinstance;

/** Tests whether an object is a class */
#define MORPHO_ISINSTANCE(val) object_istype(val, OBJECT_INSTANCE)

/** Gets the object as a class */
#define MORPHO_GETINSTANCE(val)   ((objectinstance *) MORPHO_GETOBJECT(val))

objectinstance *object_newinstance(objectclass *klass);

bool objectinstance_setproperty(objectinstance *obj, value key, value val);
bool objectinstance_getproperty(objectinstance *obj, value key, value *val);

/* ---------------------------
 * Bound methods
 * --------------------------- */

typedef struct {
    object obj;
    value receiver;
    value method;
} objectinvocation;

/** Tests whether an object is an invocation */
#define MORPHO_ISINVOCATION(val) object_istype(val, OBJECT_INVOCATION)

/** Gets the object as an invocation */
#define MORPHO_GETINVOCATION(val)   ((objectinvocation *) MORPHO_GETOBJECT(val))

objectinvocation *object_newinvocation(value receiver, value method);

bool objectinstance_insertpropertybycstring(objectinstance *obj, char *property, value val);

bool objectinstance_getpropertybycstring(objectinstance *obj, char *property, value *val);

/* -------------------------------------------------------
 * Ranges
 * ------------------------------------------------------- */

typedef struct {
    object obj;
    unsigned int nsteps; 
    value start;
    value end;
    value step;
} objectrange;

/** Tests whether an object is a range */
#define MORPHO_ISRANGE(val) object_istype(val, OBJECT_RANGE)

/** Gets the object as a range */
#define MORPHO_GETRANGE(val)   ((objectrange *) MORPHO_GETOBJECT(val))

objectrange *object_newrange(value start, value end, value step);

/* -------------------------------------------------------
 * Lists
 * ------------------------------------------------------- */

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
 * Dictionaries
 * ------------------------------------------------------- */

typedef struct {
    object obj;
    dictionary dict;
} objectdictionary;

/** Tests whether an object is a dictionary */
#define MORPHO_ISDICTIONARY(val) object_istype(val, OBJECT_DICTIONARY)

/** Gets the object as a dictionary */
#define MORPHO_GETDICTIONARY(val)   ((objectdictionary *) MORPHO_GETOBJECT(val))

/** Extract the dictionary from an objectdictionary */
/*static dictionary *object_dictionaryfromobject(objectdictionary *dict) {
    return &dict->dict;
}*/

objectdictionary *object_newdictionary(void);

/* -------------------------------------------------------
 * Arrays
 * ------------------------------------------------------- */

typedef struct {
    object obj;
    unsigned int dimensions;
    unsigned int nelements;
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
 * Matrices
 * ------------------------------------------------------- */

/** Matrices are a purely numerical collection type oriented toward linear algebra.
    Elements are stored in column-major format, i.e.
        [ 1 2 ]
        [ 3 4 ]
    is stored ( 1, 3, 2, 4 ) in memory. This is for compatibility with standard linear algebra packages */

typedef struct {
    object obj;
    unsigned int nrows;
    unsigned int ncols;
    double *elements;
    double matrixdata[];
} objectmatrix;

/** Tests whether an object is a matrix */
#define MORPHO_ISMATRIX(val) object_istype(val, OBJECT_MATRIX)

/** Gets the object as an matrix */
#define MORPHO_GETMATRIX(val)   ((objectmatrix *) MORPHO_GETOBJECT(val))

/** Creates a matrix object */
objectmatrix *object_newmatrix(unsigned int nrows, unsigned int ncols, bool zero);

/** Creates a new matrix from an array */
objectmatrix *object_matrixfromarray(objectarray *array);

/** Creates a new matrix from an existing matrix */
objectmatrix *object_clonematrix(objectmatrix *array);

/** @brief Use to create static matrices on the C stack
    @details Intended for small matrices; Caller needs to supply a double array of size nr*nc. */
#define MORPHO_STATICMATRIX(darray, nr, nc)      { .obj.type=OBJECT_MATRIX, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .elements=darray, .nrows=nr, .ncols=nc }

/* -------------------------------------------------------
 * Sparse matrices
 * ------------------------------------------------------- */

/** The dictionary of keys format uses this special object type to store indices, enabling use of the existing dictionary type.
    @warning These are for internal use only and should never be  returned to user code */
typedef struct {
    object obj;
    unsigned int row;
    unsigned int col;
} objectdokkey;

/** Create */
#define MORPHO_STATICDOKKEY(i,j)      { .obj.type=OBJECT_DOKKEY, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .row=i, .col=j }

/** Tests whether an object is a dok key */
#define MORPHO_ISDOKKEY(val) object_istype(val, OBJECT_DOKKEY)

/** Gets the object as a dok key */
#define MORPHO_GETDOKKEY(val)   ((objectdokkey *) MORPHO_GETOBJECT(val))

/** Gets the row and column from a objectdokkey */
#define MORPHO_GETDOKKEYROW(objptr)    ((unsigned int) (objptr)->row)
#define MORPHO_GETDOKKEYCOL(objptr)    ((unsigned int) (objptr)->col)

#define MORPHO_GETDOKROWWVAL(val)    ((unsigned int) (MORPHO_GETDOKKEY(val)->row))
#define MORPHO_GETDOKCOLWVAL(val)    ((unsigned int) (MORPHO_GETDOKKEY(val)->col))

DECLARE_VARRAY(dokkey, objectdokkey);

typedef struct {
    int nrows;
    int ncols;
    dictionary dict;
    objectdokkey *keys;
} sparsedok;

typedef struct {
    int nentries;
    int nrows;
    int ncols;
    int *cptr; // Pointers to column entries
    int *rix; // Row indices
    double *values; // Values
} sparseccs;

typedef struct {
    object obj;
    sparsedok dok;
    sparseccs ccs;
} objectsparse;

/** Tests whether an object is a sparse matrix */
#define MORPHO_ISSPARSE(val) object_istype(val, OBJECT_SPARSE)

/** Gets the object as a sparse matrix */
#define MORPHO_GETSPARSE(val)   ((objectsparse *) MORPHO_GETOBJECT(val))

objectsparse *object_newsparse(int *nrows, int *ncols);
objectsparse *sparse_sparsefromarray(objectarray *array);

/* -------------------------------------------------------
 * Mesh
 * ------------------------------------------------------- */

typedef struct {
    object obj;
    unsigned int dim;
    objectmatrix *vert;
    objectarray *conn;
    object *link; 
} objectmesh;

/** Tests whether an object is a mesh */
#define MORPHO_ISMESH(val) object_istype(val, OBJECT_MESH)

/** Gets the object as a mesh */
#define MORPHO_GETMESH(val)   ((objectmesh *) MORPHO_GETOBJECT(val))

/** Creates a mesh object */
objectmesh *object_newmesh(unsigned int dim, unsigned int nv, double *v);

/* -------------------------------------------------------
 * Selection
 * ------------------------------------------------------- */

typedef struct {
    object obj;
    objectmesh *mesh; /** The mesh the selection is referring to */
    
    enum {
        SELECT_ALL, SELECT_NONE, SELECT_SOME
    } mode; /** What is selected? */
    
    unsigned int ngrades; /** Number of grades */
    dictionary selected[]; /** Selections */
} objectselection;

/** Tests whether an object is a selection */
#define MORPHO_ISSELECTION(val) object_istype(val, OBJECT_SELECTION)

/** Gets the object as a selection */
#define MORPHO_GETSELECTION(val)   ((objectselection *) MORPHO_GETOBJECT(val))

/** Creates an empty selection object */
objectselection *object_newselection(objectmesh *mesh);

/* -------------------------------------------------------
 * Field
 * ------------------------------------------------------- */

typedef struct {
    object obj;
    objectmesh *mesh; /** The mesh the selection is referring to */
    
    unsigned int ngrades; /** Number of grades */
    unsigned int *dof; /** number of degrees of freedom per entry in each grade */
    unsigned int *offset; /** Offsets into the store for each grade */
    
    value prototype; /** Prototype object */
    unsigned int psize; /** Number of dofs per copy of the prototype */
    unsigned int nelements; /** Total number of elements in the fireld */
    void *pool; /** Pool of statically allocated objects */
    
    objectmatrix data; /** Underlying data store */
} objectfield;

/** Tests whether an object is a field */
#define MORPHO_ISFIELD(val) object_istype(val, OBJECT_FIELD)

/** Gets the object as a field */
#define MORPHO_GETFIELD(val)   ((objectfield *) MORPHO_GETOBJECT(val))

/** Creates an empty field object */
objectfield *object_newfield(objectmesh *mesh, value prototype, unsigned int *dof);

#endif /* object_h */
