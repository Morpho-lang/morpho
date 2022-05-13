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

/** Skyline data structures for rectangle packing */

typedef struct slentry {
    int xpos, ypos, width;
    struct slentry *next;
} textskylineentry;

DECLARE_VARRAY(textskylineentry, textskylineentry);

typedef struct {
    int width, height;
    
    varray_textskylineentry skyline;
} textskyline;

/** Skyline data structures for rectangle packing */

typedef struct {
    FT_Int width;
    FT_Int height;
} textglyph;

DECLARE_VARRAY(textglyph, textglyph);

typedef struct {
    FT_Face face;
    
} textfont;

bool text_openfont(char *file, int size, textfont *font);
void text_clearfont(textfont *font);
void text_prepare(textfont *font, char *text);
bool text_setfont(textfont *font);
void text_draw(textfont *font, char *text);

void text_initialize(void);
void text_finalize(void);

#endif /* text_h */
