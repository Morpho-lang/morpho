/** @file shortstring.c
 *  @author J Overholt
 *
 *  @brief Defines ShortString veneer class
 */
#ifndef shortstring_h
#define shortstring_h

#include <string.h>
#include "value.h"

/* -------------------------------------------------------
 * ShortString veneer class
 * ------------------------------------------------------- */

#define SSTRING_CLASSNAME                 "ShortString"
#define SSTRING_CAPACITY 6
#define SSTRING_LENGTH 6

/* -------------------------------------------------------
 * ShortString errors
 * ------------------------------------------------------- */

#define SSTRING_TOOLONG                   "ShrStrTooLng"
#define SSTRING_TOOLONG_MSG               "Input size exceeds short string capacity."

#define SSTRING_NONANBOX                  "ShrStrNanBxDsbl"
#define SSTRING_NONANBOX_MSG              "Short strings cannot be used with NaN boxing disabled."

/* -------------------------------------------------------
 * ShortString interface
 * ------------------------------------------------------- */
#ifdef MORPHO_NAN_BOXING
value value_shortstringfromnulcstring(const char *in);
value value_shortstringfromcstring(const char *in, size_t length);
int shortstring_length(value v);
#endif
void shortstring_initialize(void);
#endif