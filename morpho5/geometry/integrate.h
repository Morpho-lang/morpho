/** @file integrate.h
 *  @author T J Atherton
 *
 *  @brief Numerical integration
*/

#ifndef integration_h
#define integration_h

#include <stdio.h>
#include "morpho.h"

#define INTEGRATE_ACCURACYGOAL 1e-6
#define INTEGRATE_MAXRECURSION 100

/** Generic specification for an integrand.
 * @param[in] dim            - The dimension of the space
 * @param[in] lambda      - Barycentric coordinates for the element
 * @param[in] x                 - Coordinates of the point calculated from interpolation
 * @param[in] nquantity - Number of quantities
 * @param[in] quantity - List of quantities evaluated for the point, calculated from interpolation
 * @param[in] ref             - A reference passed by the caller (typically things constant over the domain
 * @returns value of the integrand at the appropriate point with interpolated quantities.
 */
typedef double (integrandfunction) (unsigned int dim, double *lambda, double *x, unsigned int nquantity, value *quantity, void *ref);

bool integrate_lineintegrate(integrandfunction *integrand, unsigned int dim, double *x[2], unsigned int nquantity, value *quantity[2], void *ref, double *out);

#endif /* integration_h */
