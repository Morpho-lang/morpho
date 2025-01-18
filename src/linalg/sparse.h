/** @file sparse.h
 *  @author T J Atherton
 *
 *  @brief Veneer class over the objectsparse type that provides sparse matrices
 */

#ifndef sparse_h
#define sparse_h

#include "build.h"
#ifdef MORPHO_INCLUDE_SPARSE

#include <stdio.h>
#include "object.h"
#include "morpho.h"
#include "matrix.h"

/* -------------------------------------------------------
 * Sparse objects
 * ------------------------------------------------------- */

extern objecttype objectdokkeytype;
#define OBJECT_DOKKEY objectdokkeytype

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

extern objecttype objectsparsetype;
#define OBJECT_SPARSE objectsparsetype

typedef struct {
    object obj;
    sparsedok dok;
    sparseccs ccs;
} objectsparse;

/** Tests whether an object is a sparse matrix */
#define MORPHO_ISSPARSE(val) object_istype(val, OBJECT_SPARSE)

/** Gets the object as a sparse matrix */
#define MORPHO_GETSPARSE(val)   ((objectsparse *) MORPHO_GETOBJECT(val))

/** @brief Use to create static sparse matrices on the C stack. Note that the entries should be initialized */
#define MORPHO_STATICSPARSE()      { .obj.type=OBJECT_SPARSE, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL }

objectsparse *object_newsparse(int *nrows, int *ncols);
objectsparse *sparse_sparsefromarray(objectarray *array);

/* -------------------------------------------------------
 * Sparse veneer class
 * ------------------------------------------------------- */

#define SPARSE_CLASSNAME "Sparse"

#define SPARSE_ROWINDICES_METHOD "rowindices"
#define SPARSE_SETROWINDICES_METHOD "setrowindices"
#define SPARSE_COLINDICES_METHOD "colindices"
#define SPARSE_INDICES_METHOD "indices"

/* -------------------------------------------------------
 * Sparse errors
 * ------------------------------------------------------- */

#define SPARSE_CONSTRUCTOR                "SprsCns"
#define SPARSE_CONSTRUCTOR_MSG            "Sparse() should be called either with dimensions or an array initializer."

#define SPARSE_SETFAILED                  "SprsSt"
#define SPARSE_SETFAILED_MSG              "Attempt to set sparse matrix element failed."

#define SPARSE_INVLDARRAYINIT             "SprsInvldInit"
#define SPARSE_INVLDARRAYINIT_MSG         "Invalid initializer passed to Sparse()."

#define SPARSE_CONVFAILEDERR              "SprsCnvFld"
#define SPARSE_CONVFAILEDERR_MSG          "Sparse format conversion failed."

#define SPARSE_OPFAILEDERR                "SprsOpFld"
#define SPARSE_OPFAILEDERR_MSG            "Sparse matrix operation failed."

/* -------------------------------------------------------
 * Sparse interface
 * ------------------------------------------------------- */

/* ***************************************
 * Dictionary of keys format
 * *************************************** */

void sparsedok_init(sparsedok *dok);
void sparsedok_clear(sparsedok *dok);
bool sparsedok_insert(sparsedok *dok, int i, int j, value val);
bool sparsedok_get(sparsedok *dok, int i, int j, value *val);
bool sparsedok_remove(sparsedok *dok, int i, int j, value *val);
bool sparsedok_setdimensions(sparsedok *dok, int nrows, int ncols);
unsigned int sparsedok_count(sparsedok *dok);
void *sparsedok_loopstart(sparsedok *dok);
bool sparsedok_loop(sparsedok *dok, void **cntr, int *i, int *j);
bool sparsedok_copy(sparsedok *src, sparsedok *dest);
bool sparsedok_copyat(sparsedok *src, sparsedok *dest, int row0, int col0);
bool sparsedok_copymatrixat(objectmatrix *src, sparsedok *dest, int row0, int col0);
bool sparsedok_copytomatrix(sparsedok *src, objectmatrix *dest, int row0, int col0);
void sparsedok_print(vm *v, sparsedok *dok);

/* ***************************************
 * Compressed Column Storage Format
 * *************************************** */

void sparseccs_init(sparseccs *ccs);
void sparseccs_clear(sparseccs *ccs);
bool sparseccs_resize(sparseccs *ccs, int nrows, int ncols, unsigned int nentries, bool values);
bool sparseccs_get(sparseccs *ccs, int i, int j, double *val);

bool sparseccs_getrowindices(sparseccs *ccs, int col, int *nentries, int **entries);
bool sparseccs_getrowindiceswithvalues(sparseccs *ccs, int col, int *nentries, int **entries, double **vals);
bool sparseccs_setrowindices(sparseccs *ccs, int col, int nentries, int *entries);
bool sparseccs_getcolindices(sparseccs *ccs, int maxentries, int *nentries, int *entries);
bool sparseccs_getcolindicesforrow(sparseccs *ccs, int row, int maxentries, int *nentries, int *entries);
bool sparseccs_doktoccs(sparsedok *in, sparseccs *out, bool copyvals);
bool sparseccs_copy(sparseccs *src, sparseccs *dest);
bool sparseccs_copytodok(sparseccs *src, sparsedok *dest, int row0, int col0);
bool sparseccs_copytomatrix(sparseccs *src, objectmatrix *dest, int row0, int col0);
void sparseccs_print(vm *v, sparseccs *ccs);

typedef enum { SPARSE_DOK, SPARSE_CCS } objectsparseformat;

typedef enum { SPARSE_OK, SPARSE_INCMPTBLDIM, SPARSE_INVLDINIT, SPARSE_CONVFAILED, SPARSE_FAILED } objectsparseerror;

/* ***************************************
 * Generic sparse functions
 * *************************************** */

void sparse_raiseerror(vm *v, objectsparseerror err);

bool sparse_checkformat(objectsparse *sparse, objectsparseformat format, bool force, bool copyvals);

objectsparseerror sparse_tomatrix(objectsparse *in, objectmatrix **out);
objectsparse *sparse_clone(objectsparse *s);
bool sparse_setelement(objectsparse *matrix, int row, int col, value value);
bool sparse_getelement(objectsparse *matrix, int row, int col, value *value);
void sparse_getdimensions(objectsparse *s, int *nrows, int *ncols);

objectsparseerror sparse_add(objectsparse *a, objectsparse *b, double alpha, double beta, objectsparse *out);
objectsparseerror sparse_mul(objectsparse *a, objectsparse *b, objectsparse *out);
objectsparseerror sparse_mulsxd(objectsparse *a, objectmatrix *b, objectmatrix *out);
objectsparseerror sparse_muldxs(objectmatrix *a, objectsparse *b, objectmatrix *out);
objectsparseerror sparse_transpose(objectsparse *a, objectsparse *out);

void sparse_clear(objectsparse *a);
size_t sparse_size(objectsparse *a);

objectsparseerror sparse_cat(objectlist *in, objectsparse *dest);
objectsparseerror sparse_catmatrix(objectlist *in, objectmatrix **out);

/* ***************************************
 * Sparse class methods
 * *************************************** */

value Sparse_divr(vm *v, int nargs, value *args);

/** Intialization */
void sparse_initialize(void);

#endif

#endif /* sparse_h */
