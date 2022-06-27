/** @file display.h
 *  @author T J Atherton
 *
 *  @brief Window handling using GLFW
 */

#ifndef display_h
#define display_h

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "matrix3d.h"
#include "scene.h"
#include "render.h"
#include "text.h"

#define GL_SILENCE_DEPRECATION
#include <glad/glad.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#pragma clang diagnostic pop


#define DISPLAY_DEFAULTWIDTH 480
#define DISPLAY_DEFAULTHEIGHT 480

#define DISPLAY_DEFAULTTITLE "Morpho"

typedef GLFWwindow windowref;

/** Display object corresponds to a discrete window */
typedef struct sdisplay {
    struct sdisplay *next; /** Linked list */
    
    windowref *window;
    scene *s;
    
    float width; /** Width of the window */
    float aspectRatio; /** Aspect ratio for the window */
    
    double ox, oy; /** Previous mouse x,y positions */
    
    enum {
        NORMAL,
        DRAGGING_ROT,
        DRAGGING_TRANS
    } state; /** Current action */
    
    mat4x4 view; /** Current view matrix for window */
    
    renderer render;
} display;

display *display_open(scene *s);
void display_setwindowtitle(display *d, char *title);

void display_loop(void);

bool display_initialize(void);
void display_finalize(void);

#endif /* display_h */
