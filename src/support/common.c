/** @file common.c
 *  @author T J Atherton
 *
 *  @brief Utility functions for the Morpho VM
 */

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <ctype.h>

#include "common.h"

/* **********************************************************************
* Printing
* ********************************************************************** */
 
/** @brief Prints a value
 * @param v The value to print */
void morpho_printvalue(vm *v, value val) {
    if (MORPHO_ISFLOAT(val)) {
        morpho_printf(v, "%g", MORPHO_GETFLOATVALUE(val));
        return;
    } else {
        switch (MORPHO_GETTYPE(val)) {
            case VALUE_NIL:
                morpho_printf(v, MORPHO_NILSTRING);
                return;
            case VALUE_BOOL:
                morpho_printf(v, "%s", (MORPHO_GETBOOLVALUE(val) ? MORPHO_TRUESTRING : MORPHO_FALSESTRING));
                return;
            case VALUE_INTEGER:
                morpho_printf(v, "%i", MORPHO_GETINTEGERVALUE(val));
                return;
            case VALUE_OBJECT:
                object_print(v, val);
                return;
            default:
                return;
        }
    }
}

/** @brief Prints a value to a buffer */
#define MORPHO_TOSTRINGTMPBUFFERSIZE   64
bool morpho_printtobuffer(vm *v, value val, varray_char *buffer) {
    bool success=false;
    char tmp[MORPHO_TOSTRINGTMPBUFFERSIZE];
    int nv;

    if (MORPHO_ISSTRING(val)) {
        objectstring *s = MORPHO_GETSTRING(val);
        success=varray_charadd(buffer, s->string, (int) s->length);
    } else if (MORPHO_ISCLASS(val)) {
        objectclass *klass = MORPHO_GETCLASS(val);
        varray_charwrite(buffer, '@');
        success=morpho_printtobuffer(v, klass->name, buffer);
    } else if (MORPHO_ISOBJECT(val)) {
        objectclass *klass = morpho_lookupclass(val);

        if (klass) {
            objectstring str = MORPHO_STATICSTRING(MORPHO_TOSTRING_METHOD);
            value label = MORPHO_OBJECT(&str);
            value method, ret;

            if (morpho_lookupmethod(val, label, &method) &&
                morpho_invoke(v, val, method, 0, NULL, &ret)) {
                if (MORPHO_ISSTRING(ret)) {
                    success=varray_charadd(buffer, MORPHO_GETCSTRING(ret), (int) MORPHO_GETSTRINGLENGTH(ret));
                }
            } else {
                varray_charwrite(buffer, '<');
                success=morpho_printtobuffer(v, klass->name, buffer);
                varray_charwrite(buffer, '>');
            }
        } else if (MORPHO_ISBUILTINFUNCTION(val)) {
            objectbuiltinfunction *fn = MORPHO_GETBUILTINFUNCTION(val);
            varray_charadd(buffer, "<fn ", 4);
            success=morpho_printtobuffer(v, fn->name, buffer);
            varray_charwrite(buffer, '>');
        }
    } else if (MORPHO_ISFLOAT(val)) {
        nv=snprintf(tmp, MORPHO_TOSTRINGTMPBUFFERSIZE, "%g", MORPHO_GETFLOATVALUE(val));
        success=varray_charadd(buffer, tmp, nv);
    } else if (MORPHO_ISINTEGER(val)) {
        nv=snprintf(tmp, MORPHO_TOSTRINGTMPBUFFERSIZE, "%i", MORPHO_GETINTEGERVALUE(val));
        success=varray_charadd(buffer, tmp, nv);
    } else if (MORPHO_ISBOOL(val)) {
        nv=snprintf(tmp, MORPHO_TOSTRINGTMPBUFFERSIZE, "%s", (MORPHO_ISTRUE(val) ? MORPHO_TRUESTRING : MORPHO_FALSESTRING));
        success=varray_charadd(buffer, tmp, nv);
    } else if (MORPHO_ISNIL(val)) {
        nv=snprintf(tmp, MORPHO_TOSTRINGTMPBUFFERSIZE, "%s", MORPHO_NILSTRING);
        success=varray_charadd(buffer, tmp, nv);
    }
    
    return success; 
}

/** @brief Concatenates a sequence of values as a string */
value morpho_concatenate(vm *v, int nval, value *val) {
    varray_char buffer;
    varray_charinit(&buffer);

    for (unsigned int i=0; i<nval; i++) {
        morpho_printtobuffer(v, val[i], &buffer);
    }

    value out=object_stringfromcstring(buffer.data, buffer.count);

    varray_charclear(&buffer);

    return out;
}

/** @brief   Duplicates a string.
 *  @param   string String to duplicate
 *  @warning Caller must call MALLOC_FREE on the allocated string */
char *morpho_strdup(char *string) {
    size_t len = strlen(string) + 1;
    char* output = (char *) MORPHO_MALLOC((len + 1) * sizeof(char));
    if (output) memcpy(output, string, len);

    return output;
}

/* **********************************************************************
* UTF8 support
* ********************************************************************** */

/** @brief Returns the number of bytes in the next character of a given utf8 string
    @returns number of bytes */
int morpho_utf8numberofbytes(const char *string) {
    uint8_t byte = * ((uint8_t *) string);

    if ((byte & 0xc0) == 0x80) return 0; // In the middle of a utf8 string

    // Get the number of bytes from the first character
    if ((byte & 0xf8) == 0xf0) return 4;
    if ((byte & 0xf0) == 0xe0) return 3;
    if ((byte & 0xe0) == 0xc0) return 2;
    return 1;
}

/** Decodes a utf8 encoded character pointed to by c into an int */
int morpho_utf8toint(const char *c) {
    unsigned int ret = -1;
    int nbytes=morpho_utf8numberofbytes(c);
    switch (nbytes) {
        case 1: ret=(c[0] & 0x7f); break;
        case 2: ret=((c[0] & 0x1f)<<6) | (c[1] & 0x3f); break;
        case 3: ret=((c[0] & 0x0f)<<12) | ((c[1] & 0x3f)<<6) | (c[2] & 0x3f); break;
        case 4: ret=((c[0] & 0x0f)<<18) | ((c[1] & 0x3f)<<12) | ((c[2] & 0x3f)<<6) | (c[3] & 0x3f) ; break;
        default: break;
    }
    
    return ret;
}

/** Encodes a unicode character c into a utf8 encoded string, returning the number of bytes written.
  @param[in] c - character to encode
  @param[out] out - buffer to hold string, which must be at least 4 bytes
  @returns the number of bytes written, including 0 on failure */
int morpho_encodeutf8(int c, char *out) {
    if (c<=0x7f) { // 1 byte unicode -> ascii
        out[0]=c; // b 0XXXXXXX
        return 1;
    } else if (c<=0x07ff) { // 2 byte
        out[0]=(char) (((c >>  6) & 0x1f) | 0xc0); // b 110XXXXX
        out[1]=(char) (((c >>  0) & 0x3f) | 0x80); // b 10XXXXXX
        return 2;
    } else if (c<=0xffff) { // 3 byte
        out[0]=(char) (((c >> 12) & 0x0f) | 0xe0); // b 1110XXXX
        out[1]=(char) (((c >>  6) & 0x3f) | 0x80); // b 10XXXXXX
        out[2]=(char) (((c >>  0) & 0x3f) | 0x80); // b 10XXXXXX
        return 3;
    } else if (c<=0x10ffff) {
        out[0]=(char) (((c >> 18) & 0x07) | 0xf0); // b 11110XXX
        out[1]=(char) (((c >> 12) & 0x3f) | 0x80); // b 10XXXXXX
        out[2]=(char) (((c >>  6) & 0x3f) | 0x80); // b 10XXXXXX
        out[3]=(char) (((c >>  0) & 0x3f) | 0x80); // b 10XXXXXX
        return 4;
    }
    return 0;
}

/* **********************************************************************
* Other utility functions
* ********************************************************************** */

/** @brief Computes the nearest power of 2 above an integer
 * @param   n An integer
 * @returns Nearest power of 2 above n
 * See: http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float */
unsigned int morpho_powerof2ceiling(unsigned int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;

    return n;
}

/** Count the number of fixed parameters in a callable object
 * @param[in] f - the function or callable object
 * @param[out] nparams - number of parameters; -1 if unknown
 * @returns true on success, false if f is not callable*/
bool morpho_countparameters(value f, int *nparams) {
    value g = f;
    bool success=false;

    if (MORPHO_ISINVOCATION(g)) { // Unpack invocation
        objectinvocation *inv = MORPHO_GETINVOCATION(g);
        g=inv->method;
    }

    if (MORPHO_ISCLOSURE(g)) { // Unpack closure
        objectclosure *cl = MORPHO_GETCLOSURE(g);
        g=MORPHO_OBJECT(cl->func);
    }

    if (MORPHO_ISFUNCTION(g)) {
        objectfunction *fun = MORPHO_GETFUNCTION(g);
        *nparams=fun->nargs;
        success=true;
    } else if (MORPHO_ISBUILTINFUNCTION(g)) {
        *nparams = -1;
        success=true;
    }

    return success;
}

/** Initialize tuple generator
 @param[in] nval - number of values
 @param[in] n - n-tuples to generate
 @param[in] c - workspace: supply an unsigned integer array of size 2xn  */
void morpho_tuplesinit(unsigned int nval, unsigned int n, unsigned int *c, tuplemode mode) {
    unsigned int *counter=c, *cmax=c+n; // Counters
    for (unsigned int i=0; i<n; i++) {
        counter[i]=(mode == MORPHO_SETMODE ? i : 0 );
        cmax[i]=(mode == MORPHO_SETMODE ? nval-n+i : nval-1);
    }
}

/** Generate n-tuples of unique elements indep of ordering from a list of values
 @param[in] nval - number of values
 @param[in] list - list of values
 @param[in] n - n-tuples to generate
 @param[in] c - workspace: supply an unsigned integer array of size 2xn;
 @param[out] tuple - generated tuple
 @returns true if we returned a valid tuple; false if we're done */
bool morpho_tuples(unsigned int nval, value *list, unsigned int n, unsigned int *c, tuplemode mode, value *tuple) {
    unsigned int *counter=c, *cmax=c+n; // Counters
    int k;

    if (counter[0]>cmax[0]) return false; // Done

    // Generate tuple from counter
    for (unsigned int i=0; i<n; i++) tuple[i]=list[counter[i]];

    // Increment counters
    counter[n-1]++; // Increment last counter
    for (k=n-1; k>0 && counter[k]>cmax[k]; k--) counter[k-1]++; // Carry

    if (k<n-1) {
        if (mode==MORPHO_TUPLEMODE) for (unsigned int i=k+1; i<n; i++) counter[i]=0;
        if (mode==MORPHO_SETMODE) for (unsigned int i=k+1; i<n; i++) counter[i]=counter[i-1]+1;
    }

    return true;
}
