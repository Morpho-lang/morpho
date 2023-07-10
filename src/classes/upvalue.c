/** @file upvalue.c
 *  @author T J Atherton
 *
 *  @brief Implements upvalue object type
 */

#include "morpho.h"
#include "object.h"
#include "classes.h"

/* **********************************************************************
 * objectupvalue definitions
 * ********************************************************************** */

/** Upvalue object definitions */
void objectupvalue_printfn(object *obj) {
    printf("upvalue");
}

void objectupvalue_markfn(object *obj, void *v) {
    morpho_markvalue(v, ((objectupvalue *) obj)->closed);
}

size_t objectupvalue_sizefn(object *obj) {
    return sizeof(objectupvalue);
}

objecttypedefn objectupvaluedefn = {
    .printfn=objectupvalue_printfn,
    .markfn=objectupvalue_markfn,
    .freefn=NULL,
    .sizefn=objectupvalue_sizefn
};


/** Initializes a new upvalue object. */
void object_upvalueinit(objectupvalue *c) {
    object_init(&c->obj, OBJECT_UPVALUE);
    c->location=NULL;
    c->closed=MORPHO_NIL;
    c->next=NULL;
}

/* **********************************************************************
 * objectupvalue utility functions
 * ********************************************************************** */

/** Creates a new upvalue for the register pointed to by reg. */
objectupvalue *object_newupvalue(value *reg) {
    objectupvalue *new = (objectupvalue *) object_new(sizeof(objectupvalue), OBJECT_UPVALUE);

    if (new) {
        object_upvalueinit(new);
        new->location=reg;
    }

    return new;
}

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

DEFINE_VARRAY(upvalue, upvalue);
DEFINE_VARRAY(varray_upvalue, varray_upvalue);

objecttype objectupvaluetype;

void upvalue_initialize(void) {
    // Define upvalue object type
    objectupvaluetype=object_addtype(&objectupvaluedefn);
}

void upvalue_finalize(void) {
}
