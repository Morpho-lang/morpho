/** @file render.h
 *  @author T J Atherton
 *
 *  @brief OpenGL rendering
 */


#ifndef render_h
#define render_h

#include <stdio.h>
#include <stdbool.h>
#include "varray.h"
#include "matrix3d.h"
#include "scene.h"

#define GL_SILENCE_DEPRECATION
#include <glad/glad.h>

DECLARE_VARRAY(GLuint, GLuint)

/** @brief Structure to hold information about OpenGL buffers.
 *  @details Each of these includes several types of OpenGL buffer:
 *  - a vertex array object that saves OpenGL state (e.g. the structure of the vertex buffer) for swift use.
 *  - a vertex buffer object to hold vertex and attribute data.
 *  - element array buffer to hold draw instruction lists.
 * The renderer consolidates objects and references into as few OpenGL objects as possible. */
typedef struct {
    char *format;
    GLuint array; /* Handle for vertex array object */
    GLuint buffer; /* Handle for vertex buffer object */
    GLuint element; /* Handle for element array buffer object */
    int vlength; /* Length of the vertex buffer */
    int elength; /* Length of the element array buffer */
} renderglbuffers;

DECLARE_VARRAY(renderglbuffers, renderglbuffers)

/** @brief An object to be rendered
 *  @details Points to the appropriate OpenGL buffer. */
typedef struct {
    gobject *obj; /* The original object */
    renderglbuffers *buffer; /* Pointer to OpenGL buffer collection */
    int voffset; /* Offset into the vertex buffer */
    int eoffset; /* Offset into the element array buffer */
} renderobject;

DECLARE_VARRAY(renderobject, renderobject)

/** @brief A font to be used */
typedef struct {
    textfont *font;
    GLuint texture;
} renderfont;

DECLARE_VARRAY(renderfont, renderfont)

/** @brief Render instructions */
typedef struct {
    enum {
        RNOP,
        RMODEL, /* Set the model matrix */
        RARRAY, /* Bind a VAO */
        RTRIANGLES, /* Draw triangles */
        RLINES, /* Draw lines */
        RPOINTS, /* Draw points */
        RTEXT, /* Draw text */
        RCOLOR, /* Set the current color */
    } instruction;
    
    union {
        struct {
            float *model;
        } model;
        
        struct {
            GLuint handle;
        } array;
        
        struct {
            int length;
            void *offset;
        } triangles;
        
        struct {
            char *txt;
            int rfontid; 
        } text;
        
        struct {
            float rgb[3]; 
        } color;
    } data;
    
    renderobject *obj;
} renderinstruction;

DECLARE_VARRAY(renderinstruction, renderinstruction)

/** Renderer object. */
typedef struct {
    GLuint shader;
    GLuint textshader;
    varray_renderobject objects;
    varray_renderfont fonts;
    varray_renderglbuffers glbuffers;
    varray_renderinstruction renderlist;
    GLuint fontvao;
    GLuint fontvbo;
} renderer;

bool render_init(renderer *r);
void render_clear(renderer *r);

void render_preparescene(renderer *r, scene *s);
void render_render(renderer *r, float aspectratio, mat4x4 view);

#endif /* render_h */
