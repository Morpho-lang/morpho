/** @file display.c
 *  @author T J Atherton
 *
 *  @brief Window handling using GLFW
 */

#include <math.h>
#include "display.h"
#include "scene.h"
#include "render.h"

/* -------------------------------------------------------
 * Global variables
 * ------------------------------------------------------- */

display *opendisplays;

/* -------------------------------------------------------
 * Utility functions
 * ------------------------------------------------------- */

/** Identify the display structure from a given window ref. */
static display *display_fromwindow(windowref *window) {
    return (display *) glfwGetWindowUserPointer(window);
}

/** Add to list of open displays */
void display_add(display *d) {
    d->next=opendisplays;
    opendisplays=d;
}

/** Frees data attached to a display */
void display_free(display *d) {
    scene_free(d->s);
    render_clear(&d->render);
    free(d);
}

/** Remove from list of open displays */
void display_remove(display *d) {
    if (opendisplays==d) {
        opendisplays=d->next;
        display_free(d);
    } else {
        display *prev = NULL;
        for (display *e = opendisplays; e!=NULL; e=e->next) {
            if (d==e) {
                prev->next=d->next;
                display_free(d);
                return;
            }
            prev=d;
        }
    }
}

/* -------------------------------------------------------
 * Event callbacks
 * ------------------------------------------------------- */

/** Error callback */
static void display_errorcallback(int error, const char* description) {
    fprintf(stderr, "morphoview: GLFW error '%s'\n", description);
}

/** Framebuffer resize callback */
static void display_framebuffersizecallback(windowref *window, int width, int height) {
    display *d=display_fromwindow(window);
    
    d->aspectRatio=(float) width/(float) height;
    //glViewport(0, 0, width, height);
    d->width = (float) width;
    glViewport(0, 0, width, height);
}

/** Keypress callback function */
static void display_keycallback(windowref *window, int key, int scancode, int action, int mods) {
    if (action!=GLFW_PRESS) return;
    display *d=display_fromwindow(window);
    
    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            break;
        case GLFW_KEY_TAB:
        { /* Reset the view */
            d->ox=0.0; d->oy=0.0;
            mat3d_identity4x4(d->view);
        }
            break;
        case GLFW_KEY_LEFT:
            if (mods & GLFW_MOD_ALT) {
                /* Translate left */
                vec3 a = {-0.1, 0.0, 0.0};
                mat3d_translate(d->view, a, d->view);
            } else { /* Rotate left */
                vec3 a = {0.0, 1.0, 0.0};
                mat3d_rotate(d->view, a, -0.1, d->view);
            }
            break;
        case GLFW_KEY_RIGHT:
            if (mods & GLFW_MOD_ALT) {
                /* Translate right */
                vec3 a = {0.1, 0.0, 0.0};
                mat3d_translate(d->view, a, d->view);
            } else { /* Rotate right */
                vec3 a = {0.0, 1.0, 0.0};
                mat3d_rotate(d->view, a, +0.1, d->view);
            }
            break;
        case GLFW_KEY_DOWN:
            if (mods & GLFW_MOD_ALT) {
                /* Translate down */
                vec3 a = {0.0, -0.1, 0.0};
                mat3d_translate(d->view, a, d->view);
            } else { /* Rotate down */
                vec3 a = {1.0, 0.0, 0.0};
                mat3d_rotate(d->view, a, +0.1, d->view);
            }
            break;
        case GLFW_KEY_UP:
            if (mods & GLFW_MOD_ALT) {
                /* Translate up */
                vec3 a = {0.0, 0.1, 0.0};
                mat3d_translate(d->view, a, d->view);
            } else { /* Rotate up */
                vec3 a = {1.0, 0.0, 0.0};
                mat3d_rotate(d->view, a, -0.1, d->view);
            }
            break;
        case GLFW_KEY_PAGE_DOWN:
        { /* Rotate clockwise */
            vec3 a = {0.0, 0.0, 1.0};
            mat3d_rotate(d->view, a, -0.1, d->view);
        }
            break;
        case GLFW_KEY_PAGE_UP:
        { /* Rotate anticlockwise */
            vec3 a = {0.0, 0.0, 1.0};
            mat3d_rotate(d->view, a, +0.1, d->view);
        }
            break;
        case GLFW_KEY_EQUAL: mat3d_scale(d->view, 1.05, d->view); break;
        case GLFW_KEY_MINUS: mat3d_scale(d->view, 0.95, d->view); break;
    }
    
}

/** Mouse click callback */
static void display_mousebuttoncallback(windowref *window, int button, int action, int mods) {
    display *d=display_fromwindow(window);
    
    if (action == GLFW_PRESS) {
        d->state = (button==GLFW_MOUSE_BUTTON_LEFT ? DRAGGING_ROT : DRAGGING_TRANS);
    } else {
        d->state = NORMAL;
    }
}

/** Cursor position callback */
static void display_cursorposncallback(windowref *window, double x, double y) {
    display *d=display_fromwindow(window);

    if (d->state==DRAGGING_ROT) {
        float dx=2.0*(x-d->ox)/d->width;
        float dy=-2.0*(y-d->oy)/d->width;
        
        vec3 axis = {-dy, dx, 0};
        mat3d_rotate(d->view, axis, 1.5*sqrt(dx*dx+dy*dy), d->view);
    } else if (d->state==DRAGGING_TRANS) {
        float dx=2.0*((float)(x-d->ox))/d->width;
        float dy=2.0*((float)(y-d->oy))/d->width;
        
        vec3 a = {dx, -dy, 0.0};
        mat3d_translate(d->view, a, d->view);
    }
    d->ox=x; d->oy=y;
}

/** Scroll callback */
static void display_scrollcallback(windowref *window, double x, double y) {
    display *d=display_fromwindow(window);
    
    mat3d_scale(d->view, 1.0-0.25*y, d->view);
}

/* -------------------------------------------------------
 * Create a window
 * ------------------------------------------------------- */

/** Initializes a display structure */
void display_init(display *d, scene *s) {
    d->s=s;
    d->width=0.0;
    d->aspectRatio=1.0;
    d->ox=0.0;
    d->oy=0.0;
    d->state=NORMAL;
    d->window=NULL;
    mat3d_identity4x4(d->view);
}

/** Create a new display */
display *display_open(scene *s) {
    display *new = malloc(sizeof(display));
    if (!new) {
        fprintf(stderr, "morphoview: Couldn't allocate display structure");
        return NULL;
    }
    
    display_init(new, s);
    
    windowref *window = NULL;
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
#endif
    glfwWindowHint(GLFW_SAMPLES, 4);
    
    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(DISPLAY_DEFAULTWIDTH, DISPLAY_DEFAULTHEIGHT, DISPLAY_DEFAULTTITLE, NULL, NULL);
    if (!window) return NULL;

    new->width=DISPLAY_DEFAULTWIDTH;
    new->aspectRatio=((float) DISPLAY_DEFAULTWIDTH)/((float) DISPLAY_DEFAULTHEIGHT);

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, display_keycallback);
    glfwSetFramebufferSizeCallback(window, display_framebuffersizecallback);
    glfwSetScrollCallback(window, display_scrollcallback);
    glfwSetCursorPosCallback(window, display_cursorposncallback);
    glfwSetMouseButtonCallback(window, display_mousebuttoncallback);
    glfwSetWindowUserPointer(window, new);
    
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        fprintf(stderr, "morphoview: Failed to initialize GLAD");
    }
    
    /** Initialize the display */
    render_init(&new->render);
    new->window=window;
    
    /** Add this to the display list */
    display_add(new);
    
    return new;
}

/** Sets the window title */
void display_setwindowtitle(display *d, char *title) {
    if (d) glfwSetWindowTitle(d->window, title);
}

/* -------------------------------------------------------
 * Main loop
 * ------------------------------------------------------- */

void display_loop(void) {
    while (opendisplays!=NULL) {
        for (display *d=opendisplays; d!=NULL; d=d->next) {
            if (glfwWindowShouldClose(d->window)) {
                glfwDestroyWindow(d->window);
                display_remove(d);
                break;
            } else {
                glfwMakeContextCurrent(d->window);
                render_render(&d->render, d->aspectRatio, d->view);
                
                glfwSwapBuffers(d->window);
            }
        }
        
        glfwPollEvents();
    }
}

/* -------------------------------------------------------
 * Initialization/Finalization
 * ------------------------------------------------------- */

bool display_initialize(void) {
    bool success = glfwInit();
    if (!success) fprintf(stderr, "morphoview: Could not launch GLFW.\n");
        
    glfwSetErrorCallback(display_errorcallback);
    
    opendisplays=NULL;
    
    return success;
}

void display_finalize(void) {
    glfwTerminate();
}
