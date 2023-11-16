/** @file debugannotation.h
 *  @author T J Atherton
 *
 *  @brief Debugging annotations for programs
*/

#ifndef debugannotation_h
#define debugannotation_h

#include "syntaxtree.h"
#include "object.h"
#include "program.h"

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
 * Work with debug annotations
 * ------------------------------------------------------- */

debugannotation *debugannotation_last(varray_debugannotation *list);
void debugannotation_add(varray_debugannotation *list, debugannotation *annotation);
void debugannotation_stripend(varray_debugannotation *list);
void debugannotation_setfunction(varray_debugannotation *list, objectfunction *func);
void debugannotation_setclass(varray_debugannotation *list, objectclass *klass);
void debugannotation_setmodule(varray_debugannotation *list, value module);
void debugannotation_setreg(varray_debugannotation *list, indx reg, value symbol);
void debugannotation_setglobal(varray_debugannotation *list, indx gindx, value symbol);
void debugannotation_pusherr(varray_debugannotation *list, objectdictionary *dict);
void debugannotation_poperr(varray_debugannotation *list);
void debugannotation_addnode(varray_debugannotation *list, syntaxtreenode *node);
void debugannotation_clear(varray_debugannotation *list);

void debugannotation_showannotations(varray_debugannotation *list);

#endif /* debugannotation_h */
