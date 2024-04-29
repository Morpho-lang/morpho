/** @file signature.h
 *  @author T J Atherton
 *
 *  @brief Function signatures and their declarations
*/

#ifndef signature_h
#define signature_h

#include <stdbool.h>
#include "varray.h"

typedef struct {
    varray_value types;
} signature;

void signature_init(signature *s);
void signature_clear(signature *s);

bool signature_istyped(signature *s);
bool signature_isequal(signature *a, signature *b);
bool signature_paramlist(signature *s, int *nparams, value **ptypes);
bool signature_getparamtype(signature *s, int i, value *type);
int signature_countparams(signature *s);

void signature_set(signature *s, int nparam, value *types);
bool signature_parse(char *sig, signature *out);

void signature_print(signature *s);

#endif
