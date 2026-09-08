/** @file hint.h
 *  @author J Overholt
 *
 *  @brief Morpho hint data structure
*/

#ifndef hint_h
#define hint_h

#include <stdarg.h>
#include "varray.h"

typedef struct {
    int type;
    size_t length;
    char *msg;
    char data[];
} hint;

#define HINT_BUFFERSIZE 255

hint *hint_createfromcstring(const char *str, size_t len);
hint *hint_createfromformatstring(const char *format, ...);
hint *hint_createfromformatstringvalist(const char *format, va_list args);
void hint_free(hint **hnt);

#endif
