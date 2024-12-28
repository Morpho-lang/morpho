/** @file xmatrix.h
 *  @author T J Atherton
 *
 *  @brief New linear algebra library
*/

#ifndef xmatrix_h
#define xmatrix_h

/* -------------------------------------------------------
 * Matrix object type
 * ------------------------------------------------------- */

extern objecttype objectxmatrixtype;
#define OBJECT_XMATRIX objectxmatrixtype

typedef struct {
    object obj;
    int nrows;
    int ncols;
    int nels;
    double *elements;
    double matrixdata[];
} objectxmatrix;

/* -------------------------------------------------------
 * Matrix veneer class
 * ------------------------------------------------------- */

#define XMATRIX_CLASSNAME                   "XMatrix"

void xmatrix_initialize(void);

#endif
