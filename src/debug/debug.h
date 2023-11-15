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
