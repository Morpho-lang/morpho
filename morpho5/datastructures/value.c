/** @file value.c
 *  @author T J Atherton
 *
 *  @brief Fundamental data type for morpho
*/

#include "value.h"
#include "common.h"

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
