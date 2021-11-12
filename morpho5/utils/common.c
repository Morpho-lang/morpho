/** @file common.c
 *  @author T J Atherton
 *
 *  @brief Common types, data structures and functions for the Morpho VM
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include "common.h"
#include "object.h"

/* **********************************************************************
* Utility functions 
* ********************************************************************** */

/** @brief Prints a value
 * @param v The value to print */
void morpho_printvalue(value v) {
    if (MORPHO_ISFLOAT(v)) {
        printf("%g", MORPHO_GETFLOATVALUE(v) );
        return;
    } else {
        switch (MORPHO_GETTYPE(v)) {
            case VALUE_NIL:
                printf(COMMON_NILSTRING);
                return;
            case VALUE_BOOL:
                printf("%s", ( MORPHO_GETBOOLVALUE(v) ? COMMON_TRUESTRING : COMMON_FALSESTRING ));
                return;
            case VALUE_INTEGER:
                printf("%i", MORPHO_GETINTEGERVALUE(v) );
                return;
            case VALUE_OBJECT:
                object_print(v);
                return;
            default:
                return; 
        }
    }
}

/** @brief Prints a value to a buffer */
#define MORPHO_TOSTRINGTMPBUFFERSIZE   64
void morpho_printtobuffer(vm *v, value val, varray_char *buffer) {
    char tmp[MORPHO_TOSTRINGTMPBUFFERSIZE];
    int nv;
    
    if (MORPHO_ISSTRING(val)) {
        objectstring *s = MORPHO_GETSTRING(val);
        varray_charadd(buffer, s->string, (int) s->length);
    } else if (MORPHO_ISOBJECT(val)) {
        objectclass *klass = morpho_lookupclass(val);
        
        if (klass) {
            objectstring str = MORPHO_STATICSTRING(MORPHO_TOSTRING_METHOD);
            value label = MORPHO_OBJECT(&str);
            value method, ret;
            
            if (morpho_lookupmethod(val, label, &method) &&
                morpho_invoke(v, val, method, 0, NULL, &ret)) {
                if (MORPHO_ISSTRING(ret)) {
                    varray_charadd(buffer, MORPHO_GETCSTRING(ret), (int) MORPHO_GETSTRINGLENGTH(ret));
                }
            } else {
                varray_charwrite(buffer, '<');
                morpho_printtobuffer(v, klass->name, buffer);
                varray_charwrite(buffer, '>');
            }
        } else if (MORPHO_ISFUNCTION(val)) {
            objectfunction *fn = MORPHO_GETFUNCTION(val);
            varray_charadd(buffer, "<fn ", 4);
            morpho_printtobuffer(v, fn->name, buffer);
            varray_charwrite(buffer, '>');
        } else if (MORPHO_ISBUILTINFUNCTION(val)) {
            objectbuiltinfunction *fn = MORPHO_GETBUILTINFUNCTION(val);
            varray_charadd(buffer, "<fn ", 4);
            morpho_printtobuffer(v, fn->name, buffer);
            varray_charwrite(buffer, '>');
        } else if (MORPHO_ISCLASS(val)) {
            objectclass *klass = MORPHO_GETCLASS(val);
            varray_charwrite(buffer, '@');
            morpho_printtobuffer(v, klass->name, buffer);
        }
    } else if (MORPHO_ISFLOAT(val)) {
        nv=sprintf(tmp, "%g", MORPHO_GETFLOATVALUE(val));
        varray_charadd(buffer, tmp, nv);
    } else if (MORPHO_ISINTEGER(val)) {
        nv=sprintf(tmp, "%i", MORPHO_GETINTEGERVALUE(val));
        varray_charadd(buffer, tmp, nv);
    } else if (MORPHO_ISBOOL(val)) {
        nv=sprintf(tmp, "%s", (MORPHO_ISTRUE(val) ? COMMON_TRUESTRING : COMMON_FALSESTRING));
        varray_charadd(buffer, tmp, nv);
    } else if (MORPHO_ISNIL(val)) {
        nv=sprintf(tmp, "%s", COMMON_NILSTRING);
        varray_charadd(buffer, tmp, nv);
    }
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
 *  @warning Caller should call MALLOC_FREE on the allocated string */
char *morpho_strdup(char *string) {
    size_t len = strlen(string) + 1;
    char* output = (char*) malloc ((len + 1) * sizeof(char));
    if (output) memcpy(output, string, len);
    
    return output;
}

/** @brief Returns the number of bytes in the next character of a given utf8 string
    @returns number of bytes */
int morpho_utf8numberofbytes(uint8_t *string) {
    uint8_t byte = * string;
    
    if ((byte & 0xc0) == 0x80) return 0; // In the middle of a utf8 string
    
    // Get the number of bytes from the first character
    if ((byte & 0xf8) == 0xf0) return 4;
    if ((byte & 0xf0) == 0xe0) return 3;
    if ((byte & 0xe0) == 0xc0) return 2;
    return 1;
}

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

/* Tells if an object at path corresponds to a directory */
bool morpho_isdirectory(const char *path) {
   struct stat statbuf;
   if (stat(path, &statbuf) != 0)
       return 0;
   return (bool) S_ISDIR(statbuf.st_mode);
}

/** Determine weather the rest of a string is white space */
bool white_space_remainder(const char *s, int start){
	s += start;
	while (*s){
		if (!isspace(*s)){
			return false;
		}
		s++;
	}
	return true;
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
        objectfunction *fun = MORPHO_GETFUNCTION(f);
        *nparams=fun->nargs;
        success=true;
    } else if (MORPHO_ISBUILTINFUNCTION(f)) {
        *nparams = -1;
        success=true;
    }

    return success;
}
