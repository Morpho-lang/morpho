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

#ifdef MORPHO_CORE

/* -------------------------------------------------------
 * Instructions are the basic unit of execution
 * ------------------------------------------------------- */

typedef unsigned int instruction;
DECLARE_VARRAY(instruction, instruction);

/* -------------------------------------------------------
 * Debug annotations contain debugging information
 * ------------------------------------------------------- */

/** Annotations for the compiled code to link back to the source */
typedef struct {
    enum {
        DEBUG_FUNCTION, // Set the current function
        DEBUG_CLASS, // Set the current class
        DEBUG_MODULE, // Set the current module
        DEBUG_REGISTER, // Associates a symbol with a register
        DEBUG_GLOBAL, // Associates a symbol with a global
        DEBUG_ELEMENT, // Associates a sequence of instructions with a code element
        DEBUG_PUSHERR, // Push an error handler
        DEBUG_POPERR // Pop an error handler
    } type;
    union {
        struct {
            objectdictionary *handler;
        } errorhandler;
        struct {
            objectfunction *function;
        } function;
        struct {
            objectclass *klass;
        } klass;
        struct {
            value module;
        } module;
        struct {
            indx reg;
            value symbol;
        } reg;
        struct {
            indx gindx;
            value symbol;
        } global;
        struct {
            int ninstr;
            int line;
            int posn;
        } element;
    } content;
} debugannotation;

DECLARE_VARRAY(debugannotation, debugannotation)

/* -------------------------------------------------------
 * Programs comprise instructions and debugging information
 * ------------------------------------------------------- */

/** @brief Morpho code program and associated data */
typedef struct {
    varray_instruction code; /** Compiled instructions */
    varray_debugannotation annotations; /** Information about how the code connects to the source */
    objectfunction *global;  /** Pseudofunction containing global data */
    unsigned int nglobals;
    object *boundlist; /** Linked list of static objects bound to this program */
    dictionary symboltable; /** The symbol table */
} program;

#endif /* MORPHO_CORE */

#endif /* error_h */
