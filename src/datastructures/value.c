/** @file value.c
 *  @author T J Atherton
 *
 *  @brief Fundamental data type for morpho
*/

#include "value.h"
#include "common.h"

/* **********************************************************************
* Comparison of values
* ********************************************************************** */

/** @brief Compares where two values are the same, i.e. are identical or refer to the same object. 
 * @details Faster than morpho_comparevalue
 * @param a value to compare
 * @param b value to compare
 * @returns true if a and b are identical, false otherwise */
bool morpho_issame(value a, value b) {
#ifdef MORPHO_NAN_BOXING
    return (a==b);
#else
    if (a.type!=b.type) return false;

    switch (a.type) {
        case VALUE_NIL:
            return true; /** Nils are always the same */
        case VALUE_INTEGER:
            return (b.as.integer == a.as.integer);
        case VALUE_DOUBLE:
            /* The sign bit comparison is required to distinguish between -0 and 0. */
            return ((b.as.real == a.as.real) && (signbit(b.as.real)==signbit(a.as.real)));
        case VALUE_BOOL:
            return (b.as.boolean == a.as.boolean);
        case VALUE_OBJECT:
            return MORPHO_GETOBJECT(a) == MORPHO_GETOBJECT(b);
        default:
            UNREACHABLE("unhandled value type for comparison [Check morpho_issame]");
    }

    return false;
#endif
}

/** @brief Compares two values
 * @param a value to compare
 * @param b value to compare
 * @returns 0 if a and b are equal, a positive number if b\>a and a negative number if a\<b
 * @warning Requires that both values have the same type */
int morpho_comparevalue(value a, value b) {
    if (!morpho_ofsametype(a, b)) return MORPHO_NOTEQUAL;
    
    if (MORPHO_ISFLOAT(a)) {
        double x = MORPHO_GETFLOATVALUE(b) - MORPHO_GETFLOATVALUE(a);
        if (x>DBL_EPSILON) return MORPHO_BIGGER; /* Fast way out for clear cut cases */
        if (x<-DBL_EPSILON) return MORPHO_SMALLER;
        /* Assumes absolute tolerance is the same as relative tolerance. */
        if (fabs(x)<=DBL_EPSILON*fmax(1.0, fmax(MORPHO_GETFLOATVALUE(a), MORPHO_GETFLOATVALUE(b)))) return MORPHO_EQUAL;
        return (x>0 ? MORPHO_BIGGER : MORPHO_SMALLER);
    } else {
        switch (MORPHO_GETTYPE(a)) {
            case VALUE_NIL:
                return MORPHO_EQUAL; /** Nones are always the same */
            case VALUE_INTEGER:
                return (MORPHO_GETINTEGERVALUE(b) - MORPHO_GETINTEGERVALUE(a));
            case VALUE_BOOL:
                return (MORPHO_GETBOOLVALUE(b) != MORPHO_GETBOOLVALUE(a));
            case VALUE_OBJECT:
                if (MORPHO_GETOBJECTTYPE(a)!=MORPHO_GETOBJECTTYPE(b)) {
                    return 1; /* Objects of different type are always different */
                } else return object_cmp(MORPHO_GETOBJECT(a), MORPHO_GETOBJECT(b));
            default:
                UNREACHABLE("unhandled value type for comparison [Check morpho_comparevalue]");
        }
    }
    
    return MORPHO_NOTEQUAL;
}

/* **********************************************************************
* Type check and conversion
* ********************************************************************** */

/** Detect if a value is a number */
bool morpho_isnumber(value a) {
    return (MORPHO_ISINTEGER(a) || MORPHO_ISFLOAT(a));
}

/** Define notion of falsity/truthyness */
bool morpho_isfalse(value a) {
    return (MORPHO_ISNIL(a) || (MORPHO_ISBOOL(a) && (MORPHO_GETBOOLVALUE(a)==false)));
}

/** Convert a value to an integer */
bool morpho_valuetoint(value v, int *out) {
    if (MORPHO_ISINTEGER(v)) { *out = MORPHO_GETINTEGERVALUE(v); return true; }
    if (MORPHO_ISFLOAT(v)) { *out = (int) MORPHO_GETFLOATVALUE(v); return true; }
    return false;
}

/** Convert a value to a float */
bool morpho_valuetofloat(value v, double *out) {
    if (MORPHO_ISINTEGER(v)) { *out = (double) MORPHO_GETINTEGERVALUE(v); return true; }
    if (MORPHO_ISFLOAT(v)) { *out = MORPHO_GETFLOATVALUE(v); return true; }
    return false;
}

/* **********************************************************************
* Utility functions
* ********************************************************************** */

/** Promotes a list of numbers to floats if any are floating point.
 * @param[in] nv - number of values
 * @param[in] v  - list of values
 * @returns true if successful, false if any values are not numbers */
bool value_promotenumberlist(unsigned int nv, value *v) {
    bool fl=false;
    for (unsigned int i=0; i<nv; i++) {
        if (!MORPHO_ISNUMBER(v[i])) return false;
        if (MORPHO_ISFLOAT(v[i])) fl=true;
    }
    
    if (fl) {
        for (unsigned int i=0; i<nv; i++) {
            if (MORPHO_ISINTEGER(v[i])) v[i]=MORPHO_FLOAT((double) MORPHO_GETINTEGERVALUE(v[i]));
        }
    }
    return true;
}

/** Finds the maximum and minimum of a list of values */
bool value_minmax(unsigned int nval, value *list, value *min, value *max) {
    if (nval==0) return false;
    
    if (min) *min=list[0];
    if (max) *max=list[0];
    
    for (unsigned int i=1; i<nval; i++) {
        if (min) {
            value l=*min, r=list[i];
            MORPHO_CMPPROMOTETYPE(l, r);
            if (morpho_comparevalue(l, r)<0) *min = list[i];
        }
        
        if (max) {
            value l=*max, r=list[i];
            MORPHO_CMPPROMOTETYPE(l, r);
            if (morpho_comparevalue(l, r)>0) *max = list[i];
        }
    }
    
    return true;
}

/* **********************************************************************
* Varray_values and utility functions
* ********************************************************************** */

DEFINE_VARRAY(value, value);

/** @brief Finds a value in an varray using a loose equality test (MORPHO_ISEQUAL)
 *  @param[in]  varray     the array to search
 *  @param[in]  v          value to find
 *  @param[out] out        index of the match
 *  @returns whether the value was found or not. */
bool varray_valuefind(varray_value *varray, value v, unsigned int *out) {
    for (unsigned int i=0; i<varray->count; i++) {
        if (MORPHO_ISEQUAL(varray->data[i], v)) {
            if (out) *out=i;
            return true;
        }
    }
    return false;
}

/** @brief Finds a value in an varray using strict equality test (MORPHO_ISSAME)
 *  @param[in]  varray     the array to search
 *  @param[in]  v          value to find
 *  @param[out] out        index of the match
 *  @returns whether the value was found or not. */
bool varray_valuefindsame(varray_value *varray, value v, unsigned int *out) {
    for (unsigned int i=0; i<varray->count; i++) {
        if (MORPHO_ISSAME(varray->data[i], v)) {
            if (out) *out=i;
            return true;
        }
    }
    return false;
}

/* **********************************************************************
 * Veneer classes
 * ********************************************************************** */

objectclass *_valueveneers[MORPHO_MAXIMUMVALUETYPES];

/** @brief Sets the veneer class for a particular value type */
void value_setveneerclass(value type, value clss) {
    if (!MORPHO_ISCLASS(clss)) {
        UNREACHABLE("Veneer class must be a class.");
    }
    
    if (MORPHO_ISOBJECT(type)) {
        UNREACHABLE("Cannot define a veneer class for generic objects.");
    } else if (MORPHO_ISFLOAT(type)) {
        _valueveneers[0]=MORPHO_GETCLASS(clss);
    } else {
        int k = MORPHO_GETORDEREDTYPE(type);
        _valueveneers[k]=MORPHO_GETCLASS(clss);
    }
}

/** @brief Gets the veneer class for a particular value type */
objectclass *value_getveneerclass(value type) {
    value klss;
    if (MORPHO_ISFLOAT(type)) {
        return _valueveneers[0];
    } else {
        int k = MORPHO_GETORDEREDTYPE(type);
        return _valueveneers[k];
    }
}

/* **********************************************************************
 * Initialization/Finalization
 * ********************************************************************** */

void value_initialize(void) {
    for (int i=0; i<MORPHO_MAXIMUMVALUETYPES; i++) _valueveneers[i]=NULL;
}
