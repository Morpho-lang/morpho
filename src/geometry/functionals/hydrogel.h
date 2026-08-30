/** @file hydrogel.h
 *  @author T J Atherton
 *
 *  @brief Hydrogel functional
 */

#ifndef morpho_functionals_hydrogel_h
#define morpho_functionals_hydrogel_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

/* -------------------------------------------------------
 * Hydrogel veneer class
 * ------------------------------------------------------- */

#define HYDROGEL_CLASSNAME             "Hydrogel"

#define HYDROGEL_A_PROPERTY            "a"
#define HYDROGEL_B_PROPERTY            "b"
#define HYDROGEL_C_PROPERTY            "c"
#define HYDROGEL_D_PROPERTY            "d"
#define HYDROGEL_PHIREF_PROPERTY       "phiref"
#define HYDROGEL_PHI0_PROPERTY         "phi0"
#define HYDROGEL_REFERENCE_PROPERTY    "reference"

/* -------------------------------------------------------
 * Hydrogel error messages
 * ------------------------------------------------------- */

#define HYDROGEL_FLDGRD                "HydrglFldGrd"
#define HYDROGEL_FLDGRD_MSG            "Hydrogel has been given phi0 as a Field that lacks scalar elements in grade %u."

#define HYDROGEL_ZEEROREFELEMENT       "HydrglZrRfVl"
#define HYDROGEL_ZEEROREFELEMENT_MSG   "Reference element %u has tiny volume V=%g, V0=%g\n"

#define HYDROGEL_BNDS                  "HydrglBnds"
#define HYDROGEL_BNDS_MSG              "Phi outside bounds at element %u V=%g, V0=%g, phi=%g, 1-phi=%g\n"

void hydrogel_initialize(void);

#endif

#endif /* morpho_functionals_hydrogel_h */
