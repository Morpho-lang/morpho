/** @file render.c
 *  @author T J Atherton
 *
 *  @brief OpenGL rendering
 */

#include <string.h>
#include "render.h"

/* -------------------------------------------------------
 * Varrays
 * ------------------------------------------------------- */

DEFINE_VARRAY(renderobject, renderobject)

DEFINE_VARRAY(renderfont, renderfont)

DEFINE_VARRAY(renderglbuffers, renderglbuffers)

DEFINE_VARRAY(renderinstruction, renderinstruction)

/* -------------------------------------------------------
 * Shaders
 * ------------------------------------------------------- */

/* Default shader */

const char *vertexshader = "#version 330 core\n"
    "layout (location = 0) in vec3 vPos;"
    "layout (location = 1) in vec3 vColor;"
    "layout (location = 2) in vec3 vNormal;"
    "out vec3 fragColor;"
    "out vec3 fragPos;"
    "out vec3 normal;"
    "uniform mat4 model;"
    "uniform mat4 view;"
    "uniform mat4 proj;"

    "void main() {"
    "   gl_Position = proj * view * model * vec4(vPos, 1.0);"
    "   fragColor = vColor;"
    "   fragPos = vPos;"
    "   normal = mat3(transpose(inverse(view * model))) * vNormal;"
    "}";

const char *fragmentshader = "#version 330 core\n"
    "out vec4 FragColor;"
    "in vec3 fragColor;"
    "in vec3 fragPos;"
    "in vec3 normal;"
    "uniform vec3 lightColor;"
    "uniform vec3 lightPos;"
    "uniform vec3 viewPos;"
    ""
    "void main() {"
    "   float ambientStrength = 0.1;"
    "   vec3 ambient = ambientStrength * lightColor;"
    ""
    "   vec3 norm = normalize(normal);"
    "   vec3 lightDir = normalize(lightPos-fragPos);"
    "   float diff = max(dot(norm, lightDir), 0.0);"
    "   vec3 diffuse = diff * lightColor;"
    ""
    "   float specularStrength = 0.2;"
    "   vec3 viewDir = normalize(viewPos-fragPos);"
    "   vec3 reflectDir = reflect(-lightDir, norm);"
    "   float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);"
    "   vec3 specular = specularStrength * spec * lightColor;"
    ""
    "   vec3 result = (ambient + diffuse + specular) * fragColor;"
    "   FragColor = vec4(result, 1.0f);"
    "}";

/* Flat shader */

const char *flatvertexshader = "#version 330 core\n"
    "layout (location = 0) in vec3 vPos;"
    "layout (location = 1) in vec3 vColor;"
    "out vec3 fragColor;"
    "uniform mat4 model;"
    "uniform mat4 view;"
    "uniform mat4 proj;"

    "void main() {"
    "   gl_Position = proj * view * model * vec4(vPos, 1.0);"
    "   fragColor = vColor;"
    "   fragPos = vPos;"
    "}";

const char *flatfragmentshader = "#version 330 core\n"
    "out vec4 FragColor;"
    "in vec3 fragColor;"
    ""
    "void main() {"
    "   FragColor = vec4(fragColor, 1.0f);"
    "}";

/* Text shader */

const char *textvertexshader =
    "#version 330 core\n"
    "layout (location = 0) in vec3 vertex;"
    "layout (location = 1) in vec2 tex;"
    "out vec2 TexCoords;"

    "uniform mat4 model;"
    "uniform mat4 view;"
    "uniform mat4 proj;"

    "void main() {"
    "    gl_Position = proj * view * model * vec4(vertex, 1.0);"
    "    TexCoords = tex;"
    "}";

const char *textfragmentshader =
    "#version 330 core\n"
    "in vec2 TexCoords;"
    "out vec4 color;"
    "uniform sampler2D text;"
    "uniform vec3 textColor;"

    "void main() {"
    "   vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);"
    "   color = vec4(textColor, 1.0) * sampled;"
    "}";

/* -------------------------------------------------------
 * Compile shaders
 * ------------------------------------------------------- */

/** Compiles and links shaders
 * @param[in] vertexshadersource - vertex shader
 * @param[in] fragmentshadersource - fragment shader
 * @param[out] program - compiled program id
 * @returns true on success, false if compilation failed */
bool render_compileprogram(const char *vertexshadersource, const char *fragmentshadersource, GLuint *program) {
    int success;
    
    /* Create and compile vertex shader */
    unsigned int vertexshader;
    vertexshader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexshader, 1, &vertexshadersource, NULL);
    glCompileShader(vertexshader);
    
    /* Check shader compilation was successful */
    char infoLog[512];
    glGetShaderiv(vertexshader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(vertexshader, 512, NULL, infoLog);
        fprintf(stderr, "morphoview: Vertex shader failed to compile with error '%s'\n", infoLog);
        return false;
    }
    
    /* Create and compile fragment shader */
    unsigned int fragmentshader;
    fragmentshader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentshader, 1, &fragmentshadersource, NULL);
    glCompileShader(fragmentshader);
    glGetShaderiv(vertexshader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(fragmentshader, 512, NULL, infoLog);
        fprintf(stderr,"Fragment shader failed to compile with error '%s'\n", infoLog);
        return false;
    }
    
    /* Link shader program */
    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();
    
    glAttachShader(shaderProgram, vertexshader);
    glAttachShader(shaderProgram, fragmentshader);
    glLinkProgram(shaderProgram);
    
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        fprintf(stderr, "Shader link failure with error '%s'\n", infoLog);
        return false;
    }
    
    if (program) *program = shaderProgram;
    
    /* Delete the compiled vertex and fragment shaders */
    glDeleteShader(vertexshader);
    glDeleteShader(fragmentshader);
    
    return true;
}

/* -------------------------------------------------------
 * Initialize/finalize display
 * ------------------------------------------------------- */

/** Initializes a display, compiling shaders */
bool render_init(renderer *r) {
    
    render_compileprogram(vertexshader, fragmentshader, &r->shader);
    render_compileprogram(textvertexshader, textfragmentshader, &r->textshader);
    
    /* Enable OpenGL features */
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    varray_renderobjectinit(&r->objects);
    varray_renderfontinit(&r->fonts);
    varray_renderglbuffersinit(&r->glbuffers);
    varray_renderinstructioninit(&r->renderlist);
    
    return true;
}

void render_clear(renderer *r) {
    for (unsigned int i=0; i<r->glbuffers.count; i++) {
        renderglbuffers *b=&r->glbuffers.data[i];
        
        glDeleteVertexArrays(1, &b->array);
        glDeleteBuffers(1, &b->buffer);
        glDeleteBuffers(1, &b->element);
    }
    
    varray_renderglbuffersclear(&r->glbuffers);
    varray_renderfontclear(&r->fonts);
    varray_renderobjectclear(&r->objects);
    varray_renderinstructionclear(&r->renderlist);
    
    glDeleteProgram(r->shader);
}


/* -------------------------------------------------------
 * Text rendering
 * ------------------------------------------------------- */

//GLuint fonttexture;
//GLuint fontvao;
//GLuint fontvbo;

/** Creates an OpenGL texture from the texture atlas */
void render_fonttexture(renderer *r, textfont *font, GLuint *out) {
    /* Now create an OpenGL texture from this */
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // disable byte-alignment restriction
    
    glGenTextures(1, out); // Create and define the texture
    glBindTexture(GL_TEXTURE_2D, *out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, font->skyline.width, font->skyline.height,
                 0, GL_RED, GL_UNSIGNED_BYTE, font->texturedata);
    // set texture options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

/** Prepare fonts for display */
void render_preparefonts(renderer *r, scene *scene) {
    
    for (int i=0; i<scene->fontlist.count; i++) {
        gfont *f=&scene->fontlist.data[i];
        text_generatetexture(&f->font);
        
        renderfont font;
        font.font=&f->font;
        render_fonttexture(r, &f->font, &font.texture);
        varray_renderfontwrite(&r->fonts, font);
    }
    
    glGenVertexArrays(1, &r->fontvao);
    glGenBuffers(1, &r->fontvbo);
    
    glBindVertexArray(r->fontvao);
    glBindBuffer(GL_ARRAY_BUFFER, r->fontvbo);
    
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 5, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *) (sizeof(GLfloat)*3));
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

/** Prepares text for display */
void render_preparetext(renderer *r, scene *s, gdraw *drw, GLuint *carray) {
    /* Change the model matrix if provided */
    if (drw->matindx!=SCENE_EMPTY) {
        renderinstruction ins = { .instruction = RMODEL,
                                  .data.model.model = &s->data.data[drw->matindx],
                                  .obj=NULL };
        varray_renderinstructionwrite(&r->renderlist, ins);
    }
    
    gtext *txt = &s->textlist.data[drw->id];
    
    // Store font id in the render list
    textfont *font = scene_getfontfromid(s, txt->fontid);
    if (!font) return;
    
    int rfontid;
    for (rfontid=0; rfontid<r->fonts.count; rfontid++) {
        if (r->fonts.data[rfontid].font==font) break;
    }
    if (rfontid>=r->fonts.count) return;
    
    renderinstruction ins = { .instruction = RTEXT,
                              .data.text.txt = txt->text,
                              .data.text.rfontid = rfontid,
                              .obj = NULL };

    varray_renderinstructionwrite(&r->renderlist, ins);
}

/** Draws a text element */
void render_rendertext(renderer *r, int rfontid, char *text) {
    textglyph glyph;
    float x=0.0, y=0.0, z=0.0;
    
    renderfont *font = &r->fonts.data[rfontid];
    
    float scale = 1.0/720.0;

    glBindTexture(GL_TEXTURE_2D, font->texture);
    
    for (char *c = text, *next; *c!='\0'; c=next) {
        if (!text_findglyph(font->font, c, &glyph, &next)) return;
        
        float xpos = x + glyph.bearingx * scale;
        float ypos = y - (glyph.height - glyph.bearingy) * scale;
        float w = glyph.width * scale;
        float h = glyph.height * scale;
        
        float txpos = (float) glyph.x/(float) font->font->skyline.width;
        float typos = (float) glyph.y/(float) font->font->skyline.height;
        float tw = (float) glyph.width/(float) font->font->skyline.width;
        float th = (float) glyph.height/(float) font->font->skyline.height;
        
        float vertices[6][5] = {
                    { xpos,     ypos,     z, txpos,      typos + th },
                    { xpos,     ypos + h, z, txpos,      typos      },
                    { xpos + w, ypos + h, z, txpos + tw, typos      },

                    { xpos,     ypos,     z, txpos,      typos + th },
                    { xpos + w, ypos + h, z, txpos + tw, typos      },
                    { xpos + w, ypos,     z, txpos + tw, typos + th }
                };
        
        glBindBuffer(GL_ARRAY_BUFFER, r->fontvbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        // render quad
        glDrawArrays(GL_TRIANGLES, 0, 6) ;
        
        x += (glyph.advance >> 6) * scale; // bitshift by 6 to get value in pixels (2^6 = 64)
        z += 1e-4; // Advance z by a tiny amount so that successive glyphs are drawn over previous glyphs (for languages where overlapping glyphs exist)
    }
}

/** Renders the texture atlas for a font (for debugging purposes) */
void render_renderfonttextureatlas(renderer *r, int rfontid) {
    renderfont *font = &r->fonts.data[rfontid];
    
    glBindTexture(GL_TEXTURE_2D, font->texture);
    
    float xpos = 0, ypos = 0, w = 20, h = 20;
    float txpos = 0, typos = 0, tw = 1, th = 1;
    
    float vertices[6][5] = {
                { xpos,     ypos,     0.0f, txpos,      typos + th },
                { xpos,     ypos + h, 0.0f, txpos,      typos      },
                { xpos + w, ypos + h, 0.0f, txpos + tw, typos      },

                { xpos,     ypos,     0.0f, txpos,      typos + th },
                { xpos + w, ypos + h, 0.0f, txpos + tw, typos      },
                { xpos + w, ypos,     0.0f, txpos + tw, typos + th }
            };
    
    glBindBuffer(GL_ARRAY_BUFFER, r->fontvbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // render quad
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

/* -------------------------------------------------------
 * Object rendering
 * ------------------------------------------------------- */

/** Checks if an object is present in the render object list */
renderobject *render_findrenderobject(varray_renderobject *list, gobject *obj) {
    for (unsigned int i=0; i<list->count; i++) {
        if (list->data[i].obj==obj) return &list->data[i];
    }
    return NULL;
}

/** Finds a render object based on an object id */
renderobject *render_findrenderobjectwithid(varray_renderobject *list, int id) {
    for (unsigned int i=0; i<list->count; i++) {
        if (list->data[i].obj->id==id) return &list->data[i];
    }
    return NULL;
}

/** Adds an id to an id list only if it is not present already */
renderobject *render_addobject(varray_renderobject *list, gobject *obj) {
    renderobject *out = render_findrenderobject(list, obj);
    if (!out) {
        renderobject robj = { .obj = obj, .buffer = NULL, .voffset = 0, .eoffset = 0 };
        if (varray_renderobjectadd(list, &robj, 1)) {
            out = &list->data[list->count-1];
        }
    }
    return out;
}

/** Finds the appropriate vertex buffer from  */
renderglbuffers *render_findglbuffer(varray_renderglbuffers *list, char *format) {
    for (unsigned int i=0; i<list->count; i++) {
        if (strcmp(list->data[i].format, format)==0) return &list->data[i];
    }
    return NULL;
}

/** Adds an object to appropriate OpenGL buffers if it hasn't already been added. */
void render_addobjecttoglbuffer(varray_renderglbuffers *list, renderobject *robj) {
    if (!robj || robj->buffer!=NULL) return; /* The renderobject already has been allocated to a buffer */
    
    /* First find if an appropriate OpenGL buffer exists for the given format */
    renderglbuffers *buffer = render_findglbuffer(list, robj->obj->vertexdata.format);
    if (!buffer) {
        renderglbuffers new = { .format = robj->obj->vertexdata.format, .array = 0, .buffer = 0, .element = 0, .vlength = 0, .elength = 0};
        if (varray_renderglbuffersadd(list, &new, 1)) {
            buffer = &list->data[list->count-1];
        }
    }
    
    if (buffer) {
        /* Store buffer information in the render object */
        robj->buffer=buffer;
        /* Offset and size of vertex buffer entries */
        robj->voffset=buffer->vlength;
        buffer->vlength+=robj->obj->vertexdata.length;
        
        /* Offset and size of element buffer entries */
        robj->eoffset=buffer->elength;
        /* Loop over the objects separate elements */
        for (unsigned int i=0; i<robj->obj->elements.count; i++) {
            gelement *el=&robj->obj->elements.data[i];
            buffer->elength+=el->length;
        }
    }
}

/** Calculate the size of vertex data given a format string */
int render_entrysizefromformat(scene *s, char *format) {
    int size = 0;
    for (char *c = format; *c != '\0'; c++) {
        switch (*c) {
            case 'x':
            case 'n': size+=s->dim; break;
            case 'c': size+=3; break;
            default: break;
        }
    }
    return size;
}

/** Draws an object to  newly allocated OpenGL buffers */
void render_drawobject(renderer *r, scene *s, unsigned int i) {
    renderglbuffers *b = &r->glbuffers.data[i];
    int entrysize = render_entrysizefromformat(s, b->format);
    
    glGenVertexArrays(1, &b->array);
    glGenBuffers(1, &b->buffer);
    glGenBuffers(1, &b->element);
    
    glBindVertexArray(b->array);
    
    glBindBuffer(GL_ARRAY_BUFFER, b->buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*b->vlength, NULL, GL_STATIC_DRAW);
    
    /* Copy all the object data into the buffer */
    for (unsigned int j=0; j<r->objects.count; j++) {
        renderobject *obj = &r->objects.data[j];
        if (obj && obj->buffer==b) {
            glBufferSubData( GL_ARRAY_BUFFER,
                            sizeof(GLfloat)*obj->voffset,
                            sizeof(GLfloat)*obj->obj->vertexdata.length,
                            s->data.data+obj->obj->vertexdata.indx);
        }
    }
    
    unsigned int offset = 0;
    for (unsigned int j=0; b->format[j]!='\0'; j++) {
        if (b->format[j]=='x') {
            glVertexAttribPointer(0, s->dim, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*entrysize, (void*) (sizeof(GLfloat)*offset));
            glEnableVertexAttribArray(0);
            offset += s->dim;
        } else if (b->format[j]=='c') {
            glVertexAttribPointer(1, s->dim, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*entrysize, (void*) (sizeof(GLfloat)*offset));
            glEnableVertexAttribArray(1);
            offset += 3;
        } else if (b->format[j]=='n') {
            glVertexAttribPointer(2, s->dim, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*entrysize, (void*) (sizeof(GLfloat)*offset));
            glEnableVertexAttribArray(2);
            offset += s->dim;
        }
    }
    
    /* Unbind vertex array buffer */
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    /* Now for the element array buffer */
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b->element);
    /* Size the element buffer */
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*b->elength, NULL, GL_STATIC_DRAW);
    
    /* Copy all the object element data into the buffer */
    for (unsigned int j=0; j<r->objects.count; j++) {
        renderobject *obj = &r->objects.data[j];
        
        if (obj->buffer==b) {
            int offset = obj->eoffset;
            
            /* Loop over elements */
            for (unsigned int j=0; j<obj->obj->elements.count; j++) {
                gelement *el=&obj->obj->elements.data[j];
                
                /* Offset the vertex indices by the vertex offset */
                if (obj->voffset>0) for (unsigned int k=0; k<el->length; k++) {
                    s->indx.data[el->indx+k] += obj->voffset/entrysize;
                }
                
                /* Copy the vertex indices over */
                glBufferSubData( GL_ELEMENT_ARRAY_BUFFER,
                                sizeof(GLuint)*offset,
                                sizeof(GLuint)*el->length,
                                s->indx.data+el->indx);
                
                /* Restore vertex indices */
                if (obj->voffset>0) for (unsigned int k=0; k<el->length; k++) {
                    s->indx.data[el->indx+k] -= obj->voffset/entrysize;
                }
                
                offset+=el->length;
            }
        }
    }
    
    glBindVertexArray(0);
}

/** Prepares an object for rendering, inserting appropriate instructions into the render list */
void render_prepareobject(renderer *r, scene *s, gdraw *drw, GLuint *carray) {
    renderobject *obj = render_findrenderobjectwithid(&r->objects, drw->id);
    
    /* Select the vertex array if necessary */
    renderinstruction ins = { .instruction = RARRAY, .data.array.handle = obj->buffer->array, .obj=obj };
    if (*carray!=obj->buffer->array) varray_renderinstructionadd(&r->renderlist, &ins, 1);
    *carray=obj->buffer->array;
    
    /* Change the model matrix if provided */
    if (drw->matindx!=SCENE_EMPTY) {
        renderinstruction ins = { .instruction = RMODEL,
                                  .data.model.model = &s->data.data[drw->matindx],
                                  .obj=obj };
        varray_renderinstructionadd(&r->renderlist, &ins, 1);
    }
    
    /* Now loop over the elements in the object */
    int offset=obj->eoffset;
    for (unsigned int j=0; j<obj->obj->elements.count; j++) {
        gelement *el = &obj->obj->elements.data[j];
        renderinstruction ins = { .instruction = RNOP, .obj=obj};
        
        switch (el->type) {
            case FACETS:
                ins.instruction=RTRIANGLES;
                ins.data.triangles.offset=(void *) (sizeof(GLuint)*offset);
                ins.data.triangles.length=el->length;
                offset+=el->length;
                break;
            case LINES:
                ins.instruction=RLINES;
                ins.data.triangles.offset=(void *) (sizeof(GLuint)*offset);
                ins.data.triangles.length=el->length;
                offset+=el->length;
                break;
            default:
                break;
        }
        
        if (ins.instruction!=RNOP) varray_renderinstructionadd(&r->renderlist, &ins, 1);
    }
}

/* -------------------------------------------------------
 * Prepare scene
 * ------------------------------------------------------- */

/** Prepares a scene for rendering */
void render_preparescene(renderer *r, scene *s) {
    render_preparefonts(r, s);
    
    /* Loop over the display list to identify objects */
    for (unsigned int i=0; i<s->displaylist.count; i++) {
        gdraw *drw=&s->displaylist.data[i];
        switch (drw->type) {
            case OBJECT:
            {
                renderobject *robj = NULL;
                
                /* Add the object to the scene if not already present */
                gobject *obj = scene_getgobjectfromid(s, s->displaylist.data[i].id);
                if (obj) robj=render_addobject(&r->objects, obj);
                
                /* Add vertex data to a suitable vertex buffer, or create one if necessary */
                if (robj) render_addobjecttoglbuffer(&r->glbuffers, robj);
            }
                break;
            default:
                break;
        }
    }
    
    /* Now allocate OpenGL buffers and arrays */
    for (unsigned int i=0; i<r->glbuffers.count; i++) {
        render_drawobject(r, s, i);
    }
    
    /* Now create the object render list */
    GLuint carray=0;
    for (unsigned int i=0; i<s->displaylist.count; i++) {
        gdraw *drw=&s->displaylist.data[i];
        switch (drw->type) {
            case OBJECT:
                render_prepareobject(r, s, drw, &carray);
                break;
            case TEXT:
                render_preparetext(r, s, drw, &carray);
                break;
            case COLOR:
            { // Set current color
                gcolor *color = scene_getcolorfromid(s, drw->id);
                
                if (color) {
                    renderinstruction ins = { .instruction = RCOLOR };
                    for (int i=0; i<3; i++) ins.data.color.rgb[i]=s->data.data[color->indx+i];
                
                    varray_renderinstructionadd(&r->renderlist, &ins, 1);
                } else {
                    printf("Color %i not found.\n", drw->id);
                }
            }
                break;
        }
    }
}

/* -------------------------------------------------------
 * Render the scene
 * ------------------------------------------------------- */

void render_render(renderer *r, float aspectratio, mat4x4 view) {
    /* Clear the display */
    glClearColor(0.160784f, 0.164706f, 0.188235f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    /* Load the shader */
    glUseProgram(r->shader);
    
    /* Location of shader properties */
    GLint modeluniform = glGetUniformLocation(r->shader, "model");
    GLint viewuniform = glGetUniformLocation(r->shader, "view");
    GLint projuniform = glGetUniformLocation(r->shader, "proj");
    
    GLint lightcoloruniform = glGetUniformLocation(r->shader, "lightColor");
    GLint lightposuniform = glGetUniformLocation(r->shader, "lightPos");
    GLint viewposuniform = glGetUniformLocation(r->shader, "viewPos");
    
    /* Set up the lighting */
    vec3 lightcolor = {1.0f, 1.0f, 1.0f};
    vec3 lightposn = {2.0f, 1.0f, 5.0f};
    vec3 viewposn = {0.0f, 0.0f, 1.0f};
    
    glUniform3fv(lightcoloruniform, 1, lightcolor);
    glUniform3fv(lightposuniform, 1, lightposn);
    glUniform3fv(viewposuniform, 1, viewposn);
    
    /* Set up the view matrix */
    glUniformMatrix4fv(viewuniform, 1, GL_FALSE, view);
    
    /* Set up the projection matrix */
    mat4x4 proj;
    mat3d_ortho(NULL, proj, -1.0*aspectratio, 1.0*aspectratio, -1.0, 1.0, 1.0, 10.0);
    glUniformMatrix4fv(projuniform, 1, GL_FALSE, proj);
    
    /* Render objects */
    for (unsigned i=0; i<r->renderlist.count; i++) {
        renderinstruction *ins=&r->renderlist.data[i];
        switch (ins->instruction) {
            case RNOP: break;
            case RMODEL:
                glUniformMatrix4fv(modeluniform, 1, GL_FALSE, ins->data.model.model);
                break;
            case RARRAY:
                glBindVertexArray(ins->data.array.handle);
                break;
            case RTRIANGLES:
                glDrawElements(GL_TRIANGLES, ins->data.triangles.length, GL_UNSIGNED_INT, ins->data.triangles.offset);
                break;
            case RLINES:
                glDrawElements(GL_LINES, ins->data.triangles.length, GL_UNSIGNED_INT, ins->data.triangles.offset);
                break;
            case RPOINTS:
                glDrawElements(GL_POINTS, ins->data.triangles.length, GL_UNSIGNED_INT, ins->data.triangles.offset);
                break;
            case RTEXT: case RCOLOR:
                break;
        }
    }
    
    /* Now text rendering pass */
    glUseProgram(r->textshader);
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    GLint textcoloruniform = glGetUniformLocation(r->textshader, "textColor");
    vec3 textcolor = {1.0f, 1.0f, 1.0f};
    glUniform3fv(textcoloruniform, 1, textcolor);
    
    modeluniform = glGetUniformLocation(r->textshader, "model");
    viewuniform = glGetUniformLocation(r->textshader, "view");
    projuniform = glGetUniformLocation(r->textshader, "proj");
    
    glUniformMatrix4fv(viewuniform, 1, GL_FALSE, view);
    glUniformMatrix4fv(projuniform, 1, GL_FALSE, proj);
    
    mat4x4 model;
    mat3d_identity4x4(model);
    glUniformMatrix4fv(modeluniform, 1, GL_FALSE, model);
    
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(r->fontvao);
    
    /* Render objects */
    for (unsigned i=0; i<r->renderlist.count; i++) {
        renderinstruction *ins=&r->renderlist.data[i];
        switch (ins->instruction) {
            case RMODEL:
                glUniformMatrix4fv(modeluniform, 1, GL_FALSE, ins->data.model.model);
                break;
            case RTEXT:
                render_rendertext(r, ins->data.text.rfontid, ins->data.text.txt);
                //render_renderfonttextureatlas(r, ins->data.text.rfontid);
                break;
            case RCOLOR:
                glUniform3fv(textcoloruniform, 1, ins->data.color.rgb);
                break;
            default:
                break;
        }
    }
    /* End text rendering */
    
    GLenum er = glGetError();
    if (er!=0) {
        fprintf(stderr, "OpenGL error %u\n",er);
    }
    
    glBindVertexArray(0);
}
