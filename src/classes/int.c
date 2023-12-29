/** @file int.c
 *  @author T J Atherton
 *
 *  @brief Veneer class for float values
 */

#include "morpho.h"
#include "classes.h"

#define SPRINTFBUFFER 255
#define ERROR_CHECK(f) if (!(f)) return false;

/* **********************************************************************
 * float utility functions
 * ********************************************************************** */

typedef struct sfformat {
    int width;
    int precision;
    char type;
} fformat;

#define UNSET -1

/** Parses a format string */
bool float_parseformat(char *format, char **endptr, fformat *f) {
    f->width=UNSET;
    f->precision=UNSET;
    f->type=' ';
    
    char *c = format;
    
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
    
    if (!strchr("efgEG", *c)) return false; // Check the type is valid
    f->type=*c;
    c++;
    
    if (endptr) *endptr = c;
    
    return true;
}

/** Prints a value using float formatting */
bool float_printtobufferwithformat(value v, fformat *f, varray_char *out) {
    double p;
    ERROR_CHECK(morpho_valuetofloat(v, &p));
    
    char format[SPRINTFBUFFER];
    char buffer[SPRINTFBUFFER];
    
    char *c = format;
    *c='%'; c++;
    if (f->width>=0) c+=snprintf(c, format+SPRINTFBUFFER-c, "%i", f->width);
    if (f->precision>=0) c+=snprintf(c, format+SPRINTFBUFFER-c, ".%i", f->precision);
    *c=f->type; c++;
    *c='\0';
    
    int nchars = snprintf(buffer, format+SPRINTFBUFFER-c, format, p);
    varray_charadd(out, buffer, nchars);
    
    return true;
}

/** Prints a float to a buffer */
bool float_printtobuffer(value v, char *format, varray_char *out) {
    for (char *c = format; *c!='\0'; ) { // Loop over format string
        if (*c!='%') { // Output any characters unconnected to the format
            varray_charwrite(out, *c);
            c++;
        } else {
            fformat f;
            ERROR_CHECK(float_parseformat(c, &c, &f));
            ERROR_CHECK(float_printtobufferwithformat(v, &f, out));
        }
    }
    return true;
}

/* **********************************************************************
 * Float veneer class
 * ********************************************************************** */

value Float_format(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;

    if (nargs==1 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        varray_char str;
        varray_charinit(&str);
        
        float_printtobuffer(MORPHO_SELF(args),
                            MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)),
                            &str);
        
        out = object_stringfromvarraychar(&str);
        varray_charclear(&str);
    } else {
        
    }
    
    return out;
}

MORPHO_BEGINCLASS(Float)
MORPHO_METHOD(MORPHO_FORMAT_METHOD, Float_format, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

void float_initialize(void) {
    // Create Float veneer class
    value floatclass=builtin_addclass(FLOAT_CLASSNAME, MORPHO_GETCLASSDEFINITION(Float), NULL);
    value_setveneerclass(MORPHO_FLOAT(0.0), floatclass);
}

void float_finalize(void) {
}
