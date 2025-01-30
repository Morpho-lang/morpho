/** @file debugannotation.c
*  @author T J Atherton
*
*  @brief Debugging annotations for programs
*/

#include "debugannotation.h"

/* **********************************************************************
 * Debugging annotations
 * ********************************************************************** */

DEFINE_VARRAY(debugannotation, debugannotation);

/** Retrieve the last annotation */
debugannotation *debugannotation_last(varray_debugannotation *list) {
    if (list->count>0) return &list->data[list->count-1];
    return NULL;
}

/** Adds an annotation to a list */
void debugannotation_add(varray_debugannotation *list, debugannotation *annotation) {
    varray_debugannotationadd(list, annotation, 1);
}

/** Removes the last annotation */
void debugannotation_stripend(varray_debugannotation *list) {
    if (list->count>0) list->data[list->count-1].content.element.ninstr--;
}

/** Sets the current function */
void debugannotation_setfunction(varray_debugannotation *list, objectfunction *func) {
    debugannotation ann = { .type = DEBUG_FUNCTION, .content.function.function = func};
    debugannotation_add(list, &ann);
}

/** Sets the current class */
void debugannotation_setclass(varray_debugannotation *list, objectclass *klass) {
    debugannotation ann = { .type = DEBUG_CLASS, .content.klass.klass = klass};
    debugannotation_add(list, &ann);
}

/** Sets the current module */
void debugannotation_setmodule(varray_debugannotation *list, value module) {
    debugannotation ann = { .type = DEBUG_MODULE, .content.module.module = module };
    debugannotation_add(list, &ann);
}

/** Associates a register with a symbol */
void debugannotation_setreg(varray_debugannotation *list, indx reg, value symbol) {
    if (!MORPHO_ISSTRING(symbol)) return;
    value sym = object_clonestring(symbol);
    debugannotation ann = { .type = DEBUG_REGISTER, .content.reg.reg = reg, .content.reg.symbol = sym };
    debugannotation_add(list, &ann);
}

/** Associates a global with a symbol */
void debugannotation_setglobal(varray_debugannotation *list, indx gindx, value symbol) {
    if (!MORPHO_ISSTRING(symbol)) return;
    value sym = object_clonestring(symbol);
    debugannotation ann = { .type = DEBUG_GLOBAL, .content.global.gindx = gindx, .content.global.symbol = sym };
    debugannotation_add(list, &ann);
}

/** Pushes an error handler onto the stack */
void debugannotation_pusherr(varray_debugannotation *list, objectdictionary *dict) {
    debugannotation ann = { .type = DEBUG_PUSHERR, .content.errorhandler.handler = dict};
    debugannotation_add(list, &ann);
}

/** Pops an error handler from the stack */
void debugannotation_poperr(varray_debugannotation *list) {
    debugannotation ann = { .type = DEBUG_POPERR };
    debugannotation_add(list, &ann);
}

/** Uses information from a syntaxtreenode to associate a sequence of instructions with source */
void debugannotation_addnode(varray_debugannotation *list, syntaxtreenode *node) {
    if (!node) return;
    debugannotation *last = debugannotation_last(list);
    if (last && last->type==DEBUG_ELEMENT &&
        node->line==last->content.element.line &&
        node->posn==last->content.element.posn) {
        last->content.element.ninstr++;
    } else {
        debugannotation ann = { .type = DEBUG_ELEMENT, .content.element.line = node->line, .content.element.posn = node->posn, .content.element.ninstr=1 };
        debugannotation_add(list, &ann);
    }
}

/** Clear debugging list, freeing attached info */
void debugannotation_clear(varray_debugannotation *list) {
    for (unsigned int j=0; j<list->count; j++) {
        value sym=MORPHO_NIL;
        switch (list->data[j].type) {
            case DEBUG_REGISTER:
                sym = list->data[j].content.reg.symbol;
                break;
            case DEBUG_GLOBAL:
                sym=list->data[j].content.global.symbol;
                break;
            default: break;
        }
        if (MORPHO_ISOBJECT(sym)) object_free(MORPHO_GETOBJECT(sym));
    }
    varray_debugannotationclear(list);
}

/* **********************************************************************
 * Display annotations
 * ********************************************************************** */

/** Prints all the annotations for a program */
void debugannotation_showannotations(varray_debugannotation *list) {
    indx ix = 0;
    printf("Showing %u annotations.\n", list->count);
    for (unsigned int j=0; j<list->count; j++) {
        printf("%u: ", j);
        debugannotation *ann = &list->data[j];
        switch (ann->type) {
            case DEBUG_CLASS:
                printf("Class: ");
                if (!ann->content.klass.klass) {
                    printf("(none)");
                } else {
                    morpho_printvalue(NULL, MORPHO_OBJECT(ann->content.klass.klass));
                }
                break;
            case DEBUG_ELEMENT:
                printf("Element: [%ti] instructions: %i line: %i posn: %i",
                       ix, ann->content.element.ninstr, ann->content.element.line, ann->content.element.posn);
                ix+=ann->content.element.ninstr;
                break;
            case DEBUG_FUNCTION:
                printf("Function: ");
                morpho_printvalue(NULL, MORPHO_OBJECT(ann->content.function.function));
                break;
            case DEBUG_MODULE:
                printf("Module: ");
                morpho_printvalue(NULL, ann->content.module.module);
                break;
            case DEBUG_PUSHERR:
                printf("Pusherr: ");
                morpho_printvalue(NULL, MORPHO_OBJECT(ann->content.errorhandler.handler));
                break;
            case DEBUG_POPERR:
                printf("Poperr: ");
                break;
            case DEBUG_REGISTER:
                printf("Register: %ti ", ann->content.reg.reg);
                morpho_printvalue(NULL, ann->content.reg.symbol);
                break;
            case DEBUG_GLOBAL:
                printf("Global: %ti ", ann->content.global.gindx);
                morpho_printvalue(NULL, ann->content.reg.symbol);
                break;
        }
        printf("\n");
    }
}
