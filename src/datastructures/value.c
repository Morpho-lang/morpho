/** @file value.c
 *  @author T J Atherton
 *
 *  @brief Fundamental data type for morpho
*/

#include <float.h>

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

/**  @brief Compare two doubles for equality using both absolute and relative tolerances */
bool morpho_doubleeqtest(double a, double b) {
    if (a==b) return true; 
    double diff = fabs(a-b);
    double absa = fabs(a), absb=fabs(b);
    double absmax = (absa>absb ? absa : absb);
    return (diff == 0.0) || (absmax > DBL_MIN && diff/absmax <= MORPHO_RELATIVE_EPS);
}

/** @brief Compares two values
 * @param a value to compare
 * @param b value to compare
 * @returns 0 if a and b are equal, a positive number if b\>a and a negative number if a\<b
 * @warning Requires that both values have the same type */
int morpho_comparevalue(value a, value b) {
    if (!morpho_ofsametype(a, b)) return MORPHO_NOTEQUAL;
    
    if (MORPHO_ISFLOAT(a)) {
        double aa = MORPHO_GETFLOATVALUE(a);
        double bb = MORPHO_GETFLOATVALUE(b);
        if (morpho_doubleeqtest(aa,bb)) return MORPHO_EQUAL;
        return (bb>aa ? MORPHO_BIGGER : MORPHO_SMALLER);
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

/** @brief Compares two values, even for inequivalent values e.g. int to float
 * @param a value to compare
 * @param b value to compare
 * @returns 0 if a and b are equal, a positive number if b\>a and a negative number if a\<b*/
int morpho_extendedcomparevalue(value a, value b) {
    if (morpho_ofsametype(a, b)) return morpho_comparevalue(a, b);
    
    value aa=a, bb=b;
    
    if (MORPHO_ISINTEGER(a) && MORPHO_ISFLOAT(b)) {
        aa = MORPHO_INTEGERTOFLOAT(aa);
        return morpho_comparevalue(aa, bb);
    } else if (MORPHO_ISFLOAT(a) && MORPHO_ISINTEGER(b)) {
        bb = MORPHO_INTEGERTOFLOAT(bb);
        return morpho_comparevalue(aa, bb);
    } else if (MORPHO_ISCOMPLEX(bb) && MORPHO_ISNUMBER(aa)) {
        aa=b; bb=a;
    }
    
    if (MORPHO_ISCOMPLEX(aa) && MORPHO_ISNUMBER(bb)) {
        MorphoComplex z = MORPHO_GETDOUBLECOMPLEX(aa);
        if (fabs(cimag(z)) < cabs(z)*MORPHO_RELATIVE_EPS) { // Ensure imaginary part is zero
            aa=MORPHO_FLOAT(creal(z));
            double real;
            morpho_valuetofloat(bb, &real);
            bb=MORPHO_FLOAT(real);
        } else return MORPHO_NOTEQUAL;
        return morpho_comparevalue(aa, bb);
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
            if (morpho_extendedcomparevalue(l, r)<0) *min = list[i];
        }
        
        if (max) {
            value l=*max, r=list[i];
            if (morpho_extendedcomparevalue(l, r)>0) *max = list[i];
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
    if (MORPHO_ISFLOAT(type)) {
        return _valueveneers[0];
    } else {
        int k = MORPHO_GETORDEREDTYPE(type);
        return _valueveneers[k];
    }
}

/** @brief Returns the veneer class given the type index */
objectclass *value_veneerclassfromtype(int type) {
    if (type<MORPHO_MAXIMUMVALUETYPES) {
        return _valueveneers[type];
    } else return NULL;
}

/** @brief Returns an type index for the class */
bool value_veneerclasstotype(objectclass *clss, int *type) {
    for (int i=0; i<MORPHO_MAXIMUMVALUETYPES; i++) {
        if (_valueveneers[i]==clss) {
            if (type) *type = i;
            return true;
        }
    }
    return false;
}

/* **********************************************************************
 * Initialization/Finalization
 * ********************************************************************** */

void value_initialize(void) {
    for (int i=0; i<MORPHO_MAXIMUMVALUETYPES; i++) _valueveneers[i]=NULL;
}
