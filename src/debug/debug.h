/** @file debug.h
 *  @author T J Atherton
 *
 *  @brief Debugger, dissassembly and other tools
 */

#ifndef debug_h
#define debug_h

#include <stdio.h>
#include "syntaxtree.h"
#include "object.h"
#include "program.h"
#include "debugannotation.h"

#define DEBUG_ISSINGLESTEP(d) ((d) && (d->singlestep))

debugannotation *debug_lastannotation(varray_debugannotation *list);
void debug_addannotation(varray_debugannotation *list, debugannotation *annotation);
void debug_stripend(varray_debugannotation *list);
void debug_setfunction(varray_debugannotation *list, objectfunction *func);
void debug_setclass(varray_debugannotation *list, objectclass *klass);
void debug_setmodule(varray_debugannotation *list, value module);
void debug_setreg(varray_debugannotation *list, indx reg, value symbol);
void debug_setglobal(varray_debugannotation *list, indx gindx, value symbol);
void debug_pusherr(varray_debugannotation *list, objectdictionary *dict);
void debug_poperr(varray_debugannotation *list);
void debug_addnode(varray_debugannotation *list, syntaxtreenode *node);
void debug_clearannotationlist(varray_debugannotation *list);

void debug_disassembleinstruction(instruction instruction, instructionindx indx, value *konst, value *reg);
void debug_disassemble(program *code, int *matchline);

bool debug_infofromindx(program *code, instructionindx indx, value *module, int *line, int *posn, objectfunction **func, objectclass **klass);

void debug_showannotations(varray_debugannotation *list);

void debug_showstack(vm *v);

void debugger_init(debugger *d, program *p);;
void debugger_clear(debugger *d);
bool debugger_insinglestep(debugger *d);

bool debug_shouldbreakatpc(vm *v, instruction *pc);
bool debugger_isactive(debugger *d);

#endif /* debug_h */
