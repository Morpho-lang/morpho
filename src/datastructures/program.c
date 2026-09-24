/** @file program.c
*  @author T J Atherton
*
*  @brief Data structure for a morpho program
*/

#define MORPHO_CORE

#include "core.h"
#include "debug.h"
#include "compile.h"
#include "morpho.h"

/* **********************************************************************
* Programs
* ********************************************************************** */

DEFINE_VARRAY(instruction, instruction);
DEFINE_VARRAY(globalinfo, globalinfo);

/** @brief Initializes a program */
void program_init(program *p) {
    varray_instructioninit(&p->code);
    varray_debugannotationinit(&p->annotations);
    p->global=object_newfunction(MORPHO_PROGRAMSTART, MORPHO_NIL, NULL, 0);
    p->boundlist=NULL;
    varray_globalinfoinit(&p->globals);
    varray_valueinit(&p->classes);
}

/** @brief Clears a program, freeing associated data structures */
void program_clear(program *p) {
    if (p->global) object_free((object *) p->global);
    varray_instructionclear(&p->code);
    debugannotation_clear(&p->annotations);
    p->global=NULL;
    /* Free any objects bound to the program */
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    fprintf(stderr, "--Freeing objects bound to program.\n");
#endif
    while (p->boundlist!=NULL) {
        object *next = p->boundlist->next;
        object_free(p->boundlist);
        p->boundlist=next;
    }
    #ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
        fprintf(stderr, "------\n");
    #endif
    /* Globals hold interned symbols owned by the global symbol table */
    varray_globalinfoclear(&p->globals);
    varray_valueclear(&p->classes);
}

/** @brief Creates and initializes a new program */
program *morpho_newprogram(void) {
    program *new = MORPHO_MALLOC(sizeof(program));

    if (new) program_init(new);

    return new;
}

/** @brief Frees a program */
void morpho_freeprogram(program *p) {
    program_clear(p);
    MORPHO_FREE(p);
}

/** Sets the entry point of a program */
void program_setentry(program *p, instructionindx entry) {
    if (p->global) p->global->entry=entry;
}

/** Gets the entry point of a program  */
instructionindx program_getentry(program *p) {
    instructionindx out = MORPHO_PROGRAMSTART;
    if (p->global) out=p->global->entry;
    return out;
}

/** Retrieves the bytecode associated with the program */
varray_instruction *program_getbytecode(program *p) {
    return &p->code;
}

/** Retrieves instruction i; returns false if out of bounds or out is NULL. */
bool program_getinstruction(program *p, instructionindx i, instruction *out) {
    if (!p || !out || i<0 || i>=p->code.count) return false;
    *out=p->code.data[i];
    return true;
}

/** Retrieves the global function */
objectfunction *program_getglobalfn(program *p) {
    return p->global;
}

/** @brief Binds an object to a program
 *  @details Objects bound to the program are freed with the program; use for static data (e.g. held in constant tables) */
void program_bindobject(program *p, object *obj) {
    if (!obj->next && /* Object is not already bound to the program (or something else) */
        p->boundlist!=obj &&
        obj->status==OBJECT_ISUNMANAGED) {
        obj->status=OBJECT_ISPROGRAM;
        obj->next=p->boundlist;
        p->boundlist=obj;
    }
}

/** @brief Interns a symbol.
 *  @details Selectors live in the global symbol table so a name compiled from
 *           Morpho source and the same name registered later by an extension
 *           are the same object. The program does not own the string. */
value program_internsymbol(program *p, value symbol) {
    (void) p;
#ifdef MORPHO_DEBUG_SYMBOLTABLE
    fprintf(stderr, "Interning symbol '");
    morpho_printvalue(NULL, symbol);
#endif
    value out = builtin_internsymbol(symbol);
#ifdef MORPHO_DEBUG_SYMBOLTABLE
    fprintf(stderr, "' at %p\n", (void *) MORPHO_GETOBJECT(out));
#endif
    return out;
}

/** @brief Adds a global to the program */
globalindx program_addglobal(program *p, value symbol) {
    globalinfo info = { .symbol = program_internsymbol(p, symbol), .type=MORPHO_NIL };
    
    return (globalindx) varray_globalinfowrite(&p->globals, info);
}

/** @brief Sets the type associated with a global variable */
void program_globalsettype(program *p, globalindx indx, value type) {
    if (indx<0 || indx>p->globals.count) return;
    
    p->globals.data[indx].type=type;
}

/** @brief Gets the type associated with a global variable */
bool program_globaltype(program *p, globalindx indx, value *type) {
    if (indx<0 || indx>p->globals.count) return false;
    *type = p->globals.data[indx].type;
    return true; 
}

/** @brief Gets the symbol associated with a global variable */
bool program_globalsymbol(program *p, globalindx indx, value *symbol) {
    if (indx<0 || indx>p->globals.count) return false;
    *symbol = p->globals.data[indx].symbol;
    return true; 
}

/** @brief Returns the number of globals allocated in the program */
int program_countglobals(program *p) {
    return p->globals.count;
}

/** @brief Adds a class to the program's class list */
int program_addclass(program *p, value klass) {
    return varray_valuewrite(&p->classes, klass);
}

/** @brief Returns the number of classes allocated in the program */
int program_countclasses(program *p, value klass) {
    return p->classes.count;
}
