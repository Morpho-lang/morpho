/** @file vm.h
 *  @author T J Atherton
 *
 *  @brief The Morpho virtual machine
 */

#ifndef vm_h
#define vm_h

#define MORPHO_CORE
#include "core.h"

/* **********************************************************************
* Interface
* ********************************************************************** */

void morpho_initialize(void);
void morpho_finalize(void);

instructionindx vm_previnstruction(vm *v);
instructionindx vm_currentinstruction(vm *v);

#endif /* vm_h */
