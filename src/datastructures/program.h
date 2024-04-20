/** @file program.h
 *  @author T J Atherton
 *
 *  @brief Morpho program data structure
*/

#ifndef program_h
#define program_h

#include "varray.h"
#include "object.h"
#include "classes.h"
#include "debugannotation.h"

#ifdef MORPHO_CORE

/* -------------------------------------------------------
 * Instructions are the basic unit of execution
 * ------------------------------------------------------- */

typedef unsigned int instruction;
DECLARE_VARRAY(instruction, instruction);

/** @brief Index into instructions */
typedef indx instructionindx;

/* -------------------------------------------------------
 * Global variables
 * ------------------------------------------------------- */

/** @brief Index of a global */
typedef int globalindx;

/** @brief Record information about each global variable */
typedef struct {
    value symbol;
    value type;
} globalinfo;

DECLARE_VARRAY(globalinfo, globalinfo)

/* -------------------------------------------------------
 * Programs comprise instructions and debugging information
 * ------------------------------------------------------- */

/** @brief Morpho code program and associated data */
typedef struct {
    varray_instruction code; /** Compiled instructions */
    varray_debugannotation annotations; /** Information about how the code connects to the source */
    objectfunction *global;  /** Pseudofunction containing global data */
    varray_globalinfo globals; /** Global variables */
    object *boundlist; /** Linked list of static objects bound to this program */
    dictionary symboltable; /** The symbol table */
} program;

#define MORPHO_PROGRAMSTART 0
void program_setentry(program *p, instructionindx entry);
instructionindx program_getentry(program *p);
varray_value *program_getconstanttable(program *p);
void program_bindobject(program *p, object *obj);

value program_internsymbol(program *p, value symbol);

globalindx program_addglobal(program *p, value symbol);
int program_countglobals(program *p);

#endif /* MORPHO_CORE */

#endif /* error_h */
