/** @file text.h
 *  @author T J Atherton
 *
 *  @brief Text rendering using freetype 
 */

#ifndef text_h
#define text_h

#include "varray.h"
#include "dictionary.h"
#include <stdio.h>
#include <stdbool.h>

#include <ft2build.h>
#include FT_FREETYPE_H

#define GL_SILENCE_DEPRECATION
#include <glad/glad.h>

#define TEXTSKYLINE_EMPTY -1

/** Skyline data structure for rectangle packing */
typedef struct slentry {
    int xpos, ypos, width;
    int next; 
} textskylineentry;

DECLARE_VARRAY(textskylineentry, textskylineentry);

typedef struct {
    int width, height; // Width and height of the overall container
    
    varray_textskylineentry skyline;
} textskyline;

#define TEXT_SKYLINEENTRY(a, i) (&a->skyline.data[i])

/** Glyphs */

typedef struct {
    int code;
    FT_Int width;
    FT_Int height;
    int x, y; // Location in texture 
} textglyph;

DECLARE_VARRAY(textglyph, textglyph);

typedef struct {
    FT_Face face;
    
    textskyline skyline;
    varray_textglyph glyphs;
    
    unsigned int texture;
    char *texturedata; 
} textfont;

void text_test(textfont *font);

bool text_openfont(char *file, int size, textfont *font);
void text_clearfont(textfont *font);
bool text_prepare(textfont *font, char *text);
bool text_setfont(textfont *font);
void text_draw(textfont *font, char *text);

void text_initialize(void);
void text_finalize(void);

#endif /* text_h */
