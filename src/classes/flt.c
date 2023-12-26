/** @file flt.c
 *  @author T J Atherton
 *
 *  @brief Veneer class for float values
 */

#include "morpho.h"

/* **********************************************************************
 * objectclosure definitions
 * ********************************************************************** */

void objectclosure_printfn(object *obj, void *v) {
    objectclosure *f = (objectclosure *) obj;
    morpho_printf(v, "<");
    objectfunction_printfn((object *) f->func, v);
    morpho_printf(v, ">");
}

void objectclosure_markfn(object *obj, void *v) {
    objectclosure *c = (objectclosure *) obj;
    morpho_markobject(v, (object *) c->func);
    for (unsigned int i=0; i<c->nupvalues; i++) {
        morpho_markobject(v, (object *) c->upvalues[i]);
    }
}

size_t objectclosure_sizefn(object *obj) {
    return sizeof(objectclosure)+sizeof(objectupvalue *)*((objectclosure *) obj)->nupvalues;
}

objecttypedefn objectclosuredefn = {
    .printfn=objectclosure_printfn,
    .markfn=objectclosure_markfn,
    .freefn=NULL,
    .sizefn=objectclosure_sizefn
};

/** Closure functions */
void object_closureinit(objectclosure *c) {
    c->func=NULL;
}

/** @brief Creates a new closure
 *  @param sf       the objectfunction of the current environment
 *  @param func     a function object to enclose
 *  @param np       the prototype number to use */
objectclosure *object_newclosure(objectfunction *sf, objectfunction *func, indx np) {
    objectclosure *new = NULL;
    varray_upvalue *up = NULL;

    if (np<sf->prototype.count) {
        up = &sf->prototype.data[np];
    }

    if (up) {
        new = (objectclosure *) object_new(sizeof(objectclosure) + sizeof(objectupvalue*)*up->count, OBJECT_CLOSURE);
        if (new) {
            object_closureinit(new);
            new->func=func;
            for (unsigned int i=0; i<up->count; i++) {
                new->upvalues[i]=NULL;
            }
            new->nupvalues=up->count;
        }
    }

    return new;
}

/* **********************************************************************
 * objectclosure utility functions
 * ********************************************************************** */

/* **********************************************************************
 * Closure veneer class
 * ********************************************************************** */

value Closure_tostring(vm *v, int nargs, value *args) {
    objectclosure *self=MORPHO_GETCLOSURE(MORPHO_SELF(args));
    value out = MORPHO_NIL;
    
    varray_char buffer;
    varray_charinit(&buffer);

    if (self->func) {
        varray_charadd(&buffer, "<<fn ", 5);
        morpho_printtobuffer(v, self->func->name, &buffer);
        varray_charadd(&buffer, ">>", 2);
    }

    out = object_stringfromvarraychar(&buffer);
    if (MORPHO_ISSTRING(out)) {
        morpho_bindobjects(v, 1, &out);
    }
    varray_charclear(&buffer);

    return out;
}

MORPHO_BEGINCLASS(Closure)
MORPHO_METHOD(MORPHO_TOSTRING_METHOD, Closure_tostring, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

void float_initialize(void) {
}

void float_finalize(void) {
}
