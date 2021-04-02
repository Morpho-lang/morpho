/** @file scene.c
 *  @author T J Atherton
 *
 *  @brief Scene descriptions
 */

#include <stdlib.h>
#include "scene.h"

/* -------------------------------------------------------
 * Constructor/Destructor
 * ------------------------------------------------------- */

/** Create a new scene */
scene *scene_new(int id, int dim) {
    scene *new = malloc(sizeof(scene));
    if (new) {
        new->id=id;
        new->dim=dim; 
        varray_gobjectinit(&new->objectlist);
        varray_gdrawinit(&new->displaylist);
        varray_floatinit(&new->data);
        varray_intinit(&new->indx);
    }
    return new;
}

/** Free a scene and associated data structures */
void scene_free(scene *s) {
    for (unsigned int i=0; i<s->objectlist.count; i++) {
        gobject *obj = &s->objectlist.data[i];
        if (obj->vertexdata.format) free(obj->vertexdata.format);
        varray_gelementclear(&obj->elements);
    }
    
    varray_gobjectclear(&s->objectlist);
    varray_gdrawclear(&s->displaylist);
    varray_floatclear(&s->data);
    varray_intclear(&s->indx);
    free(s);
}

/** Find a scene from the id */
scene *scene_find(int id) {
    return NULL;
}

/* -------------------------------------------------------
 * Add
 * ------------------------------------------------------- */

/** Adds an object to a scene */
gobject *scene_addobject(scene *s, int id) {
    gobject obj;
    obj.id=id;
    obj.vertexdata.indx=SCENE_EMPTY;
    obj.vertexdata.length=SCENE_EMPTY;
    varray_gelementinit(&obj.elements);
    
    varray_gobjectadd(&s->objectlist, &obj, 1);
    return &s->objectlist.data[s->objectlist.count-1];
}

/** Add vertex data to a scene; returns the starting index of the data */
int scene_adddata(scene *s, float *data, int count) {
    int ret = s->data.count;
    varray_floatadd(&s->data, data, count);
    return ret;
}

/** Add index data to a scene; returns the starting index of the data  */
int scene_addindex(scene *s, int *data, int count) {
    int ret=s->indx.count;
    varray_intadd(&s->indx, data, count);
    return ret;
}

/** Adds element data to an object */
int scene_addelement(gobject *obj, gelement *el) {
    varray_gelementadd(&obj->elements, el, 1);
    return obj->elements.count-1;
}

/* -------------------------------------------------------
 * Find
 * ------------------------------------------------------- */

/** Gets a gobject structure given an id */
gobject *scene_getgobjectfromid(scene *s, int id) {
    for (unsigned int i=0; i<s->objectlist.count; i++) {
        if (s->objectlist.data[i].id==id) return &s->objectlist.data[i];
    }
    return NULL;
}

/* -------------------------------------------------------
 * Varrays
 * ------------------------------------------------------- */

DEFINE_VARRAY(gobject, gobject);
DEFINE_VARRAY(gelement, gelement);
DEFINE_VARRAY(gdraw, gdraw);
DEFINE_VARRAY(float, float);

/* -------------------------------------------------------
 * Initialize/Finalize
 * ------------------------------------------------------- */

void scene_initialize(void) {
}

void scene_finalize(void) {
}
