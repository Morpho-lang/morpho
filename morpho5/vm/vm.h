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
* Prototypes
* ********************************************************************** */

#define MORPHO_PROGRAMSTART 0
void program_setentry(program *p, instructionindx entry);
instructionindx program_getentry(program *p);
varray_value *program_getconstanttable(program *p);
void program_bindobject(program *p, object *obj);

value program_internsymbol(program *p, value symbol);

void vm_unbindobject(vm *v, value obj);
void vm_freeobjects(vm *v);
void vm_collectgarbage(vm *v);

void morpho_initialize(void);
void morpho_finalize(void);

#endif /* vm_h */
