/** @file discretization.h
 *  @author T J Atherton
 *
 *  @brief Different finite element discretizations
 */

#ifndef discretization_h
#define discretization_h

#include <stdio.h>

#define DISCRETIZATION_CLASSNAME "Discretization"
#define LAGRANGE_CONSTRUCTORNAME "LagrangeFunctionSpace"

#define DISCRETIZATION_ORDERMETHOD "order"

void discretization_initialize(void);

#endif /* discretization_h */
