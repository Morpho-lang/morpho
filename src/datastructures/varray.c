/** @file varray.c
 *  @author T J Atherton
 *
 *  @brief Dynamically resizing array (varray) data structure
*/

#include "varray.h"

/* **********************************************************************
* Common varray types
* ********************************************************************** */

DEFINE_VARRAY(char, char);
DEFINE_VARRAY(int, int);
DEFINE_VARRAY(double, double);
DEFINE_VARRAY(ptrdiff, ptrdiff_t);

/** @brief Computes the nearest power of 2 above an integer
 * @param   n An integer
 * @returns Nearest power of 2 above n
 See: http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float */
unsigned int varray_powerof2ceiling(unsigned int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    
    return n;
}
