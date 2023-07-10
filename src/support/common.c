/** @file common.c
 *  @author T J Atherton
 *
 *  @brief Common types, data structures and functions for the Morpho VM
 */

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>

#include "common.h"
#include "sparse.h"
#include "resources.h"

/* **********************************************************************
* Utility functions
* ********************************************************************** */

/** @brief Compares two values
 * @param a value to compare
 * @param b value to compare
 * @returns 0 if a and b are equal, a positive number if b\>a and a negative number if a\<b
 * @warning does not work if a and b are not the same type (use MORPHO_CHECKCMPTYPE to promote types if ordering is important) */
#define EQUAL 0
#define NOTEQUAL 1
#define BIGGER 1
#define SMALLER -1
int morpho_comparevalue (value a, value b) {

    // if comparing a number to complex cast the number to complex
    // we don't need to bind here beacues the value never needs to exist beyond this scope
    // valgrin/check with tim to be sure
    if (MORPHO_ISCOMPLEX(a) && MORPHO_ISNUMBER(b)){
        // cast b to complex
        double val;
        morpho_valuetofloat(b,&val);
        return (MORPHO_GETCOMPLEX(a)->Z==val ? EQUAL: NOTEQUAL);
    }
    if (MORPHO_ISCOMPLEX(b) && MORPHO_ISNUMBER(a)){
        // cast b to complex
        double val;
        morpho_valuetofloat(a,&val);
        return (MORPHO_GETCOMPLEX(b)->Z==val ? EQUAL: NOTEQUAL);
    }


    if (!morpho_ofsametype(a, b)) return NOTEQUAL;

    if (MORPHO_ISFLOAT(a)) {
        double x = MORPHO_GETFLOATVALUE(b) - MORPHO_GETFLOATVALUE(a);
        if (x>DBL_EPSILON) return BIGGER; /* Fast way out for clear cut cases */
        if (x<-DBL_EPSILON) return SMALLER;
        /* Assumes absolute tolerance is the same as relative tolerance. */
        if (fabs(x)<=DBL_EPSILON*fmax(1.0, fmax(MORPHO_GETFLOATVALUE(a), MORPHO_GETFLOATVALUE(b)))) return EQUAL;
        return (x>0 ? BIGGER : SMALLER);
    } else {
        switch (MORPHO_GETTYPE(a)) {
            case VALUE_NIL:
                return EQUAL; /** Nones are always the same */
            case VALUE_INTEGER:
                return (MORPHO_GETINTEGERVALUE(b) - MORPHO_GETINTEGERVALUE(a));
            case VALUE_BOOL:
                return (MORPHO_GETBOOLVALUE(b) != MORPHO_GETBOOLVALUE(a));
            case VALUE_OBJECT:
                {
                    if (MORPHO_GETOBJECTTYPE(a)!=MORPHO_GETOBJECTTYPE(b)) {
                        return 1; /* Objects of different type are always different */
                    } else if (MORPHO_ISSTRING(a)) {
                        objectstring *astring = MORPHO_GETSTRING(a);
                        objectstring *bstring = MORPHO_GETSTRING(b);
                        size_t len = (astring->length > bstring->length ? astring->length : bstring->length);

                        return -strncmp(astring->string, bstring->string, len);
                    } else if (MORPHO_ISDOKKEY(a) && MORPHO_ISDOKKEY(b)) {
                        objectdokkey *akey = MORPHO_GETDOKKEY(a);
                        objectdokkey *bkey = MORPHO_GETDOKKEY(b);

                        return ((MORPHO_GETDOKKEYCOL(akey)==MORPHO_GETDOKKEYCOL(bkey) &&
                                 MORPHO_GETDOKKEYROW(akey)==MORPHO_GETDOKKEYROW(bkey)) ? EQUAL : NOTEQUAL);
                    } else if (MORPHO_ISCOMPLEX(a) && MORPHO_ISCOMPLEX(b)) {
                        objectcomplex *acomp = MORPHO_GETCOMPLEX(a);
                        objectcomplex *bcomp = MORPHO_GETCOMPLEX(b);
                        return (complex_equality(acomp,bcomp)? EQUAL: NOTEQUAL);
                    } else {
                        return (MORPHO_GETOBJECT(a) == MORPHO_GETOBJECT(b)? EQUAL: NOTEQUAL);
                    }
                }
            default:
                UNREACHABLE("unhandled value type for comparison [Check morpho_comparevalue]");
        }
    }
    return NOTEQUAL;
}
#undef EQUAL
#undef NOTEQUAL
#undef BIGGER
#undef SMALLER

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
