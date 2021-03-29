/** @file scene.h
 *  @author T J Atherton
 *
 *  @brief Scene descriptions
 */

#ifndef scene_h
#define scene_h

#include <stdio.h>
#include "varray.h"

#define SCENE_EMPTY -1
DECLARE_VARRAY(float, float);
DECLARE_VARRAY(int, int);

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

typedef struct {
    int id;
    int matindx;
} gdraw;

DECLARE_VARRAY(gdraw, gdraw);

typedef struct sscene {
    struct sscene *next; /** Linked list */
    
    int id; /** The scene ID */
    int dim; /** Number of dimensions; 2 or 3 */
    
    varray_float data;
    varray_int indx;
    varray_gobject objectlist;
    varray_gdraw displaylist;
} scene;

scene *scene_new(int id, int dim);
scene *scene_find(int id);
void scene_free(scene *s);

gobject *scene_addobject(scene *s, int id);
int scene_adddata(scene *s, float *data, int count);
int scene_addindex(scene *s, int *data, int count);
int scene_addelement(gobject *obj, gelement *el);

gobject *scene_getgobjectfromid(scene *s, int id);

void scene_initialize(void);
void scene_finalize(void);

#endif /* scene_h */
