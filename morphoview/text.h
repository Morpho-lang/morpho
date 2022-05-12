/** @file text.h
 *  @author T J Atherton
 *
 *  @brief Text rendering 
 */

#ifndef text_h
#define text_h

#include <stdio.h>
#include <stdbool.h>

#include <ft2build.h>
#include FT_FREETYPE_H

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
