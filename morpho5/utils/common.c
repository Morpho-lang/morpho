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

/** @brief Concatenates a sequence of values as a string */
#define MORPHO_TOSTRINGTMPBUFFERSIZE   64
value morpho_concatenatestringvalues(int nval, value *v) {
    varray_char buffer;
    varray_charinit(&buffer);
    char tmp[MORPHO_TOSTRINGTMPBUFFERSIZE];
    int nv;
    
    for (unsigned int i=0; i<nval; i++) {
        if (MORPHO_ISFLOAT(v[i])) {
            nv=sprintf(tmp, "%g", MORPHO_GETFLOATVALUE(v[i]));
            varray_charadd(&buffer, tmp, nv);
        } else {
            switch (MORPHO_GETTYPE(v[i])) {
                case VALUE_INTEGER:
                    nv=sprintf(tmp, "%i", MORPHO_GETINTEGERVALUE(v[i]));
                    varray_charadd(&buffer, tmp, nv);
                    break;
                case VALUE_OBJECT:
                    object_printtobuffer(v[i], &buffer);
                    break;
                case VALUE_NIL:
                    break; 
                default:
                    UNREACHABLE("Unhandled type in morpho_tostring.");
            }
        }
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
