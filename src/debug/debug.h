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

/* -------------------------------------------------------
 * Debugger error messages
 * ------------------------------------------------------- */

#define DEBUGGER_FINDSYMBOL                "DbgSymbl"
#define DEBUGGER_FINDSYMBOL_MSG            "Can't find symbol '%s' in current context."

#define DEBUGGER_SETPROPERTY               "DbgStPrp"
#define DEBUGGER_SETPROPERTY_MSG           "Object does not support setting properties."

#define DEBUGGER_INVLDREGISTER             "DbgInvldRg"
#define DEBUGGER_INVLDREGISTER_MSG         "Invalid register."

#define DEBUGGER_INVLDGLOBAL               "DbgInvldGlbl"
#define DEBUGGER_INVLDGLOBAL_MSG           "Invalid global."

#define DEBUGGER_INVLDINSTR                "DbgInvldInstr"
#define DEBUGGER_INVLDINSTR_MSG            "Invalid instruction."

#define DEBUGGER_REGISTEROBJ               "DbgRgObj"
#define DEBUGGER_REGISTEROBJ_MSG           "Register %i does not contain an object."

#define DEBUGGER_SYMBOLPROP                "DbgSymblPrpty"
#define DEBUGGER_SYMBOLPROP_MSG            "Symbol lacks property '%s'."

/* -------------------------------------------------------
 * Debugger interface
 * ------------------------------------------------------- */

bool debug_infofromindx(program *code, instructionindx indx, value *module, int *line, int *posn, objectfunction **func, objectclass **klass);

void debugger_init(debugger *d, program *p);
void debugger_clear(debugger *d);

void debugger_seterror(debugger *d, error *err);

vm *debugger_currentvm(debugger *d);
bool debugger_isactive(debugger *d);

void debugger_setsinglestep(debugger *d, bool singlestep);
bool debugger_insinglestep(debugger *d);

bool debugger_setbreakpoint(debugger *d, instructionindx indx);
bool debugger_clearbreakpoint(debugger *d, instructionindx indx);
bool debugger_shouldbreakat(debugger *d, instructionindx indx);

void debugger_disassembleinstruction(vm *v, instruction instruction, instructionindx indx, value *konst, value *reg);
void debugger_disassemble(vm *v, program *code, int *matchline);

void debugger_garbagecollect(debugger *debug);
void debugger_quit(debugger *debug);

bool debugger_breakatinstruction(debugger *debug, bool set, instructionindx indx);
bool debugger_breakatline(debugger *debug, bool set, value file, int line);
bool debugger_breakatfunction(debugger *debug, bool set, value klass, value function);

void debugger_showlocation(debugger *debug, instructionindx indx);
bool debugger_showaddress(debugger *debug, indx rindx);
void debugger_showbreakpoints(debugger *debug);
void debugger_showglobals(debugger *debug);
bool debugger_showglobal(debugger *debug, indx g);
void debugger_showregisters(debugger *debug);
void debugger_showstack(debugger *debug);

bool debugger_showsymbol(debugger *debug, value symbol);
bool debugger_showproperty(debugger *debug, value obj, value property);
void debugger_showsymbols(debugger *debug);

bool debugger_setregister(debugger *debug, indx reg, value val);
bool debugger_setsymbol(debugger *debug, value symbol, value val);
bool debugger_setproperty(debugger *debug, value symbol, value property, value val);

bool debugger_enter(debugger *debug, vm *v);

void debugger_initialize(void);

#endif /* debug_h */
