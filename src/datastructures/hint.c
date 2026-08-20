/** @file hint.c
 *  @author J Overholt
 *
 *  @brief Morpho hint data structure
*/
#include "hint.h"

#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include "memory.h"

hint *hint_createfromcstring(const char *str, size_t len) {
    hint *new = MORPHO_MALLOC(sizeof(hint) + sizeof(char)*(len+1));

    if (new) {
        new->msg = new->data;
        memcpy(new->msg, str, len);
        new->msg[len] = '\0';
        new->length = len;
        return new;
    }
    return NULL;
}

hint *hint_createfromformatstringvalist(const char *format, va_list args) {
    // Initial guess allocation
    char *buffer = MORPHO_MALLOC(sizeof(char)*HINT_BUFFERSIZE);
    if (!buffer) return NULL;
    size_t n = vsnprintf(buffer, HINT_BUFFERSIZE, format, args);
    if (n > HINT_BUFFERSIZE) { // Fix if underallocated
        MORPHO_FREE(buffer);
        buffer = MORPHO_MALLOC(sizeof(char)*(n+1));
        if (!buffer) return NULL;
        vsnprintf(buffer, n+1, format, args);
    }
    hint *out = hint_createfromcstring(buffer, n);
    MORPHO_FREE(buffer);

    return out;
}

hint *hint_createfromformatstring(const char *format, ...) {
    va_list args;
    va_start(args, format);
    hint *out = hint_createfromformatstringvalist(format, args);
    va_end(args);

    return out;
}

void hint_free(hint **hnt) {
    MORPHO_FREE(*hnt);
    *hnt = NULL;
}

