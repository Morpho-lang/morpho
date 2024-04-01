/** @file gc.h
 *  @author T J Atherton
 *
 *  @brief Morpho garbage collector
 */

#ifndef gc_h
#define gc_h

/* **********************************************************************
* Interface
* ********************************************************************** */

void vm_graylistinit(graylist *g);
void vm_graylistclear(graylist *g);
void vm_graylistadd(graylist *g, object *obj);

void vm_unbindobject(vm *v, value obj);
void vm_freeobjects(vm *v);
void vm_collectgarbage(vm *v);

#endif /* vm_h */
