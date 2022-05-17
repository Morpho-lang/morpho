/** @file scene.h
 *  @author T J Atherton
 *
 *  @brief Scene descriptions
 */

#ifndef scene_h
#define scene_h

#include <stdio.h>
#include "varray.h"
#include "text.h"

#define SCENE_EMPTY -1
DECLARE_VARRAY(float, float);

/* **********************
 * An element of a scene
 * ********************** */

typedef enum {
    POINTS,
    LINES,
    FACETS
} gelementtype;

typedef struct {
    gelementtype type; 
    int indx;
    int length; 
} gelement;

DECLARE_VARRAY(gelement, gelement);

/* **********************
 * Objects
 * ********************** */

typedef struct {
    int id;
    struct {
        char *format;
        int indx;
        int length;
    } vertexdata;
    varray_gelement elements;
} gobject;

DECLARE_VARRAY(gobject, gobject);

/* **********************
 * Fonts
 * ********************** */

typedef struct {
    int id;
    textfont font;
} gfont;

DECLARE_VARRAY(gfont, gfont);

/* **********************
 * Text
 * ********************** */

/*typedef struct {
    int fontid;
    char *text;
} gtext;

DECLARE_VARRAY(gtext, gtext);*/

/* **********************
 * List of things to draw
 * ********************** */

typedef enum {
    OBJECT,
    TEXT
} gdrawtype;

typedef struct {
    gdrawtype type;
    int id;
    int matindx;
} gdraw;

DECLARE_VARRAY(gdraw, gdraw);

/* ***************************
 * The overall scene structure
 * *************************** */

typedef struct sscene {
    struct sscene *next; /** Linked list */
    
    int id; /** The scene ID */
    int dim; /** Number of dimensions; 2 or 3 */
    
    varray_float data;
    varray_int indx;
    varray_gobject objectlist;
    varray_gdraw displaylist;
    varray_gfont fontlist; 
} scene;

scene *scene_new(int id, int dim);
scene *scene_find(int id);
void scene_free(scene *s);

gobject *scene_addobject(scene *s, int id);
int scene_adddata(scene *s, float *data, int count);
int scene_addindex(scene *s, int *data, int count);
int scene_addelement(gobject *obj, gelement *el);
gfont *scene_addfont(scene *s, int id, char *file, float size);

gobject *scene_getgobjectfromid(scene *s, int id);

void scene_initialize(void);
void scene_finalize(void);

#endif /* scene_h */
