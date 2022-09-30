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

#define TEXT_DEFAULTWIDTH 1280
#define TEXT_DEFAULTHEIGHT 960
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
    int width, height;
    int bearingx, bearingy; // Offset to top left point of a character from the origin
    unsigned int advance; // Offset to advance to next glyph
    
    int x, y; // Location in texture
} textglyph;

DECLARE_VARRAY(textglyph, textglyph);

typedef struct {
    FT_Face face;
    
    textskyline skyline;
    varray_textglyph glyphs;
    
    char *texturedata; 
} textfont;

void text_test(textfont *font);
void text_showtexture(textfont *font);

void text_fontinit(textfont *font, int width);
bool text_openfont(char *file, int size, textfont *font);
void text_fontclear(textfont *font);
bool text_prepare(textfont *font, char *text);
bool text_generatetexture(textfont *font);

bool text_findglyph(textfont *font, char *string, textglyph *glyph, char **next);

void text_initialize(void);
void text_finalize(void);

#endif /* text_h */
