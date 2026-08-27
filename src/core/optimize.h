/** @file optimize.h
 *  @brief Conservative dependency queries on Morpho callables.
 *
 *  If a callable cannot be inspected, queries report a dependency
 *  (fnaccessesarg → true; fnloadsconstants → all hit).
 */

#ifndef optimize_h
#define optimize_h

#include "morpho.h"

bool optimize_fnaccessesarg(vm *v, value f, int arg);
void optimize_fnloadsconstants(vm *v, value f, int nvals, value *konsts, bool *hit);

#endif
