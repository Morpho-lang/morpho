/** @file format.c
 *  @author T J Atherton
 *
 *  @brief Formatting of values
*/

#include <string.h>
#include <ctype.h>
#include <stdio.h>

#include "value.h"
#include "format.h"
#include "varray.h"

#define SPRINTFBUFFER 255
#define ERROR_CHECK(f) if (!(f)) return false;

/* **********************************************************************
 * Format utility functions
 * ********************************************************************** */

#define UNSET -1

typedef struct fformat {
    int width;
    int precision;
    char type;
} format;

/** Parses a format string
 @param[in] formatstring - format string to parse
 @param[in] valid - valid type characters */
bool _format_parse(char *formatstring, char *validtypes, char **endptr, format *f) {
    f->width=UNSET;
    f->precision=UNSET;
    f->type=' ';
    
    char *c = formatstring;
    
    if (*c=='%') c++;
    else return false;
    
    if (isdigit(*c)) { // Field width
        f->width = (int) strtol(c, &c, 10);
    }
    
    if (*c=='.') { // Precision
        c++;
        if (isdigit(*c)) {
            f->precision = (int) strtol(c, &c, 10);
        } else return false;
    }
    
    if (!strchr(validtypes, *c)) return false; // Check the type is valid
    f->type=*c;
    c++;
    
    if (endptr) *endptr = c;
    
    return true;
}

/** Prints a value to a buffer using a format specifier */
bool _format_printtobuffer(value v, format *f, varray_char *out) {
    if (!(MORPHO_ISFLOAT(v) || MORPHO_ISINTEGER(v))) return false;
    
    char format[SPRINTFBUFFER];
    char buffer[SPRINTFBUFFER];
    
    char *c = format;
    *c='%'; c++;
    if (f->width>=0) c+=snprintf(c, format+SPRINTFBUFFER-c, "%i", f->width);
    if (f->precision>=0) c+=snprintf(c, format+SPRINTFBUFFER-c, ".%i", f->precision);
    *c=f->type; c++;
    *c='\0';
    
    int nchars=0;
    
    if (MORPHO_ISFLOAT(v)) nchars=snprintf(buffer, SPRINTFBUFFER, format, MORPHO_GETFLOATVALUE(v));
    else nchars=snprintf(buffer, SPRINTFBUFFER, format, MORPHO_GETINTEGERVALUE(v));
    
    varray_charadd(out, buffer, nchars);
    
    return true;
}

/* **********************************************************************
 * Format public functions
 * ********************************************************************** */

/** Prints a quantity to a buffer */
bool format_printtobuffer(value v, char *formatstring, varray_char *out) {
    char *validtypes=FORMAT_INTTYPES;
    if (MORPHO_ISFLOAT(v)) validtypes=FORMAT_FLOATTYPES;
    
    for (char *c = formatstring; *c!='\0'; ) { // Loop over format string
        if (*c!='%') { // Output any characters unconnected to the format
            varray_charwrite(out, *c);
            c++;
        } else {
            format f;
            ERROR_CHECK(_format_parse(c, validtypes, &c, &f));
            ERROR_CHECK(_format_printtobuffer(v, &f, out));
        }
    }
    return true;
}
