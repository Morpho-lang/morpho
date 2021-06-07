/** @file sparse.h
 *  @author T J Atherton
 *
 *  @brief Veneer class over the objectsparse type that provides sparse matrices
 */

#ifndef sparse_h
#define sparse_h

#include <stdio.h>
#include "object.h"
#include "morpho.h"

/* ***************************************
 * The Sparse class
 * *************************************** */

#define SPARSE_CLASSNAME "Sparse"

#define SPARSE_ROWINDICES_METHOD "rowindices"
#define SPARSE_COLINDICES_METHOD "colindices"
#define SPARSE_INDICES_METHOD "indices"

#define SPARSE_CONSTRUCTOR                "SprsCns"
#define SPARSE_CONSTRUCTOR_MSG            "Sparse() should be called either with dimensions or an array initializer."

#define SPARSE_SETFAILED                  "SprsSt"
#define SPARSE_SETFAILED_MSG              "Attempt to set sparse matrix element failed."

#define SPARSE_INVLDARRAYINIT             "SprsInvldInit"
#define SPARSE_INVLDARRAYINIT_MSG         "Array initializer passed to Sparse() must be a 1 or 2 dimensional array."

#define SPARSE_CONVFAILEDERR              "SprsCnvFld"
#define SPARSE_CONVFAILEDERR_MSG          "Sparse format conversion failed."

#define SPARSE_OPFAILEDERR                "SprsOpFld"
#define SPARSE_OPFAILEDERR_MSG            "Sparse matrix operation failed."

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

/* ***************************************
 * Compressed Column Storage Format
 * *************************************** */

void sparseccs_init(sparseccs *ccs);
void sparseccs_clear(sparseccs *ccs);
bool sparseccs_resize(sparseccs *ccs, int nrows, int ncols, unsigned int nentries, bool values);
bool sparseccs_get(sparseccs *ccs, int i, int j, double *val);

bool sparseccs_getrowindices(sparseccs *ccs, int col, int *nentries, int **entries);
bool sparseccs_getcolindices(sparseccs *ccs, int *nentries, int *entries);
bool sparseccs_doktoccs(sparsedok *in, sparseccs *out, bool copyvals);

/* ***************************************
 * Object sparse interface
 * *************************************** */

typedef enum { SPARSE_DOK, SPARSE_CCS } objectsparseformat;

typedef enum { SPARSE_OK, SPARSE_INCMPTBLDIM, SPARSE_CONVFAILED, SPARSE_FAILED } objectsparseerror;

bool sparse_checkformat(objectsparse *sparse, objectsparseformat format, bool force, bool copyvals);

bool sparse_setelement(objectsparse *matrix, int row, int col, value value);
bool sparse_getelement(objectsparse *matrix, int row, int col, value *value);

objectsparseerror sparse_add(objectsparse *a, objectsparse *b, double alpha, double beta, objectsparse *out);
objectsparseerror sparse_mul(objectsparse *a, objectsparse *b, objectsparse *out);
objectsparseerror sparse_transpose(objectsparse *a, objectsparse *out);

void sparse_clear(objectsparse *a);
size_t sparse_size(objectsparse *a);

/* ***************************************
 * Sparse class methods
 * *************************************** */

value Sparse_divr(vm *v, int nargs, value *args);

/* ***************************************
 * Initialization
 * *************************************** */

void sparse_initialize(void);

#endif /* sparse_h */
