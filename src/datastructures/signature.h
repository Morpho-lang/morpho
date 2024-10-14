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
    varray_value types; /** Signature of parameters */
    value ret; /** Return type */
    bool varg; /** Is the function variadic? */
} signature;

void signature_init(signature *s);
void signature_clear(signature *s);

void signature_setvarg(signature *s, bool varg);
bool signature_isvarg(signature *s);

bool signature_istyped(signature *s);
bool signature_isequal(signature *a, signature *b);
bool signature_paramlist(signature *s, int *nparams, value **ptypes);
bool signature_getparamtype(signature *s, int i, value *type);
value signature_getreturntype(signature *s);
int signature_countparams(signature *s);

void signature_set(signature *s, int nparam, value *types);
bool signature_parse(char *sig, signature *out);

void signature_print(signature *s);

#endif
