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
        varray_gcolorinit(&new->colorlist);
        varray_gfontinit(&new->fontlist);
        varray_gtextinit(&new->textlist);
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
    
    for (unsigned int i=0; i<s->fontlist.count; i++) {
        text_fontclear(&s->fontlist.data[i].font);
    }
    
    for (unsigned int i=0; i<s->textlist.count; i++) {
        free(s->textlist.data[i].text);
    }
    
    varray_gobjectclear(&s->objectlist);
    varray_gdrawclear(&s->displaylist);
    varray_gcolorclear(&s->colorlist);
    varray_gfontclear(&s->fontlist);
    varray_gtextclear(&s->textlist);
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

/** Adds a font to a scene
 @param[in] s - The scene
 @param[in] id - font id
 @param[in] file - font file to open
 @param[in] size - in points
 @param[out] fontindx - index to refer to this */
bool scene_addfont(scene *s, int id, char *file, float size, int *fontindx) {
    gfont font;
    
    font.id=id;
    text_fontinit(&font.font, TEXT_DEFAULTWIDTH);
    
    int sizepx = (int) (size / 72.0 * 720.0) /* in pts / points per inch * DPI */;
    
    if (text_openfont(file, sizepx, &font.font)) {
        varray_gfontwrite(&s->fontlist, font);
        if (fontindx) *fontindx = s->fontlist.count-1;
        return true;
    }

    return false;
}

/** Find the textfont object corresponding to a given fontid */
textfont *scene_getfontfromid(scene *s, int fontid) {
    for (int i=0; i<s->fontlist.count; i++) {
        if (s->fontlist.data[i].id==fontid) return &s->fontlist.data[i].font;
    }
    return NULL;
}

/** Adds text to a scene */
int scene_addtext(scene *s, int fontid, char *text) {
    textfont *font = scene_getfontfromid(s, fontid);

    if (!font) {
        fprintf(stderr, "Font id '%i' not found.\n", fontid);
        return false;
    }
    
    text_prepare(font, text);
    
    gtext txt;
    txt.fontid=fontid;
    txt.text=text;
    
    return varray_gtextwrite(&s->textlist, txt);;
}

/** Adds a color to a scene */
int scene_addcolor(scene *s, int colorid, int length, int indx) {
    gcolor color = { .colorid = colorid,
                     .length = length,
                     .indx = indx
        
    };
    
    return varray_gcolorwrite(&s->colorlist, color);
}

void scene_adddraw(scene *scene, gdrawtype type, int id, int matindx) {
    gdraw d = { .type = type, .id = id, .matindx = matindx };
    varray_gdrawwrite(&scene->displaylist, d);
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

/** Gets a gcolor structure given an id */
gcolor *scene_getcolorfromid(scene *s, int id) {
    for (unsigned int i=0; i<s->colorlist.count; i++) {
        if (s->colorlist.data[i].colorid==id) return &s->colorlist.data[i];
    }
    return NULL;
}

/* -------------------------------------------------------
 * Varrays
 * ------------------------------------------------------- */

DEFINE_VARRAY(gobject, gobject);
DEFINE_VARRAY(gelement, gelement);
DEFINE_VARRAY(gcolor, gcolor);
DEFINE_VARRAY(gfont, gfont);
DEFINE_VARRAY(gdraw, gdraw);
DEFINE_VARRAY(gtext, gtext);
DEFINE_VARRAY(float, float);

/* -------------------------------------------------------
 * Initialize/Finalize
 * ------------------------------------------------------- */

void scene_initialize(void) {
}

void scene_finalize(void) {
}
