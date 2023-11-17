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

bool debug_infofromindx(program *code, instructionindx indx, value *module, int *line, int *posn, objectfunction **func, objectclass **klass);
bool debug_indxfromline(program *code, int line, instructionindx *out);
bool debug_indxfromfunction(program *code, value klassname, value fname, instructionindx *indx);
bool debug_symbolsforfunction(program *code, objectfunction *func, instructionindx *indx, value *symbols);

void debug_disassembleinstruction(instruction instruction, instructionindx indx, value *konst, value *reg);
void debug_disassemble(program *code, int *matchline);

void debug_showstack(vm *v);

void debugger_init(debugger *d, program *p);
void debugger_clear(debugger *d);
void debugger_setsinglestep(debugger *d, bool singlestep);
bool debugger_insinglestep(debugger *d);
void debugger_setbreakpoint(debugger *d, instructionindx indx);
void debugger_clearbreakpoint(debugger *d, instructionindx indx);
bool debugger_shouldbreakat(debugger *d, instructionindx indx);
bool debugger_isactive(debugger *d);

bool debugger_showaddress(debugger *debug, indx reg);
bool debugger_showbreakpoints(debugger *debug);
bool debugger_showglobals(debugger *debug);
bool debugger_showglobal(debugger *debug, indx g);
bool debugger_showregisters(debugger *debug);
bool debugger_showstack(debugger *debug);

bool debugger_enter(vm *v, debugger *debug);

#endif /* debug_h */
