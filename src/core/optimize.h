/** @file optimize.h
 *  @brief Minimal dependency queries on Morpho callables.
 */

#ifndef optimize_h
#define optimize_h

#include "morpho.h"

bool optimize_fnaccessesarg(vm *v, value f, int arg);
bool optimize_fnloadsconstant(vm *v, value f, int nvals, value *konsts);

#endif
