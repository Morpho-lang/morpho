/** @file program.c
*  @author T J Atherton
*
*  @brief Data structure for a morpho program
*/

#define MORPHO_CORE

#include "core.h"
#include "debug.h"

/* **********************************************************************
* Programs
* ********************************************************************** */

DEFINE_VARRAY(instruction, instruction);

/** @brief Initializes a program */
static void vm_programinit(program *p) {
    varray_instructioninit(&p->code);
    varray_debugannotationinit(&p->annotations);
    p->global=object_newfunction(MORPHO_PROGRAMSTART, MORPHO_NIL, NULL, 0);
    p->boundlist=NULL;
    dictionary_init(&p->symboltable);
    //builtin_copysymboltable(&p->symboltable);
    p->nglobals=0;
}

/** @brief Clears a program, freeing associated data structures */
static void vm_programclear(program *p) {
    if (p->global) object_free((object *) p->global);
    varray_instructionclear(&p->code);
    debug_clearannotationlist(&p->annotations);
    p->global=NULL;
    /* Free any objects bound to the program */
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("--Freeing objects bound to program.\n");
#endif
    while (p->boundlist!=NULL) {
        object *next = p->boundlist->next;
        object_free(p->boundlist);
        p->boundlist=next;
    }
    #ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
        printf("------\n");
    #endif
    dictionary_clear(&p->symboltable); /* Note we don't free the contents as they should be bound to the program */
}

/** @brief Creates and initializes a new program */
program *morpho_newprogram(void) {
    program *new = MORPHO_MALLOC(sizeof(program));

    if (new) vm_programinit(new);

    return new;
}

/** @brief Frees a program */
void morpho_freeprogram(program *p) {
    vm_programclear(p);
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

/** @brief Binds an object to a program
 *  @details Objects bound to the program are freed with the program; use for static data (e.g. held in constant tables) */
void program_bindobject(program *p, object *obj) {
    if (!obj->next && /* Object is not already bound to the program (or something else) */
        obj->status==OBJECT_ISUNMANAGED && /* Object is unmanaged */
        (!MORPHO_ISBUILTINFUNCTION(MORPHO_OBJECT(obj))) && /* Object is not a built in function that is freed separately */
        (p->boundlist!=obj->next && p->boundlist!=NULL) /* To handle the case where the object is the only object */
        ) {

        obj->next=p->boundlist;
        p->boundlist=obj;
    }
}

/** @brief Interns a symbol into the programs symbol table.
 *  @details Note that the string is cloned if it does not exist already.
 *           Interning is used to accelerate dynamic lookups as the same string for a symbol will be used universally */
value program_internsymbol(program *p, value symbol) {
    value new = symbol, out;
#ifdef MORPHO_DEBUG_SYMBOLTABLE
    printf("Interning symbol '");
    morpho_printvalue(symbol);
#endif
    
    if (builtin_checksymbol(symbol)) { // Check if this is part of the built in symbol table already
        return builtin_internsymbol(symbol);
    }
    
    if (!dictionary_get(&p->symboltable, symbol, NULL)) {
       new = object_clonestring(symbol);
    }
    out = dictionary_intern(&p->symboltable, new);
#ifdef MORPHO_DEBUG_SYMBOLTABLE
    printf("' at %p\n", (void *) MORPHO_GETOBJECT(out));
#endif
    program_bindobject(p, MORPHO_GETOBJECT(out));
    return out;
}
