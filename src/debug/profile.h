/** @file profile.h
 *  @author T J Atherton
 *
 *  @brief Profiler
 */

#ifndef profile_h
#define profile_h

#include "compile.h"
#include "vm.h"
#include "morpho.h"

bool morpho_profile(vm *v, program *p);

#endif /* profile_h */
