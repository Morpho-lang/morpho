/** @file text.c
 *  @author T J Atherton
 *
 *  @brief Text rendering
 */

#include "text.h"

FT_Library ftlibrary;

/** Prints a bitmap (for testing purposes only) */
void text_drawbitmap(FT_Bitmap *bitmap) {
  FT_Int  i, j;
  FT_Int  x_max = bitmap->width;
  FT_Int  y_max = bitmap->rows;

  for (i=0; i<y_max; i++) {
    for (j=0; j<x_max; j++) {
        char c = bitmap->buffer[i * bitmap->width + j];
        printf("%c", (c==0 ? ' ' : '*'));
    }
    printf("\n");
  }
}

/* Opens a font
 * @param[in] file - Font file
 * @param[in] size - Font size
 * @param[out] font - Font record filled out
 * @returns true on success */
bool text_openfont(char *file, int size, textfont *font) {
    FT_Error error = FT_New_Face(ftlibrary, file, 0, &font->face);
    if (error) return false;
    
    error = FT_Set_Pixel_Sizes(font->face, 0, size);
    if (error) return false;
    
    return true;
}

/* Clears a font
 * @param[in] font - Font record filled out */
void text_clearfont(textfont *font) {
    FT_Done_Face(font->face);
}

/* Prepares a font to display a particular piece of text
 * @param[in] font - Font record filled out
 * @param[in] text - */
void text_prepare(textfont *font, char *text) {
    FT_Error error;
    for (char *c = text; *c!='\0'; c++) {
        error = FT_Load_Char(font->face, *c, FT_LOAD_RENDER);
        
        text_drawbitmap(&font->face->glyph->bitmap);
        if (error) return;
        printf("%c\n", *c);
    }
    printf("\n");
}

/* Initialize the text library */
void text_initialize(void) {
    FT_Init_FreeType(&ftlibrary);
    
    textfont font;
    text_openfont("/Library/Fonts/Arial Unicode.ttf", 64, &font);
    //text_openfont("/System/Library/Fonts/Helvetica.ttc", 64, &font);
    
    text_prepare(&font, "Hello World!");
    
    text_clearfont(&font);
}

void text_finalize(void) {
    FT_Done_FreeType(ftlibrary);
}
