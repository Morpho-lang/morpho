/** @file functional.h
 *  @author T J Atherton
 *
 *  @brief Functionals
 */

#ifndef functional_h
#define functional_h

#include <stdio.h>

/* Functional properties */
#define FUNCTIONAL_GRADE_PROPERTY      "grade"
#define FUNCTIONAL_ONESIDED_PROPERTY   "onesided"
#define SCALARPOTENTIAL_FUNCTION_PROPERTY     "function"
#define SCALARPOTENTIAL_GRADFUNCTION_PROPERTY     "gradfunction"
#define LINEARELASTICITY_REFERENCE_PROPERTY     "reference"
#define LINEARELASTICITY_POISSON_PROPERTY     "poissonratio"

/* Functional methods */
#define FUNCTIONAL_INTEGRAND_METHOD    "integrand"
#define FUNCTIONAL_GRADIENT_METHOD     "gradient"
#define FUNCTIONAL_TOTAL_METHOD        "total"

/* Functional names */
#define LENGTH_CLASSNAME               "Length"
#define AREA_CLASSNAME                 "Area"
#define AREAENCLOSED_CLASSNAME         "AreaEnclosed"
#define VOLUME_CLASSNAME               "Volume"
#define VOLUMEENCLOSED_CLASSNAME       "VolumeEnclosed"
#define SCALARPOTENTIAL_CLASSNAME      "ScalarPotential"
#define LINEARELASTICITY_CLASSNAME     "LinearElasticity"

/* Errors */
#define FUNC_INTEGRAND_MESH            "FnctlIntMsh"
#define FUNC_INTEGRAND_MESH_MSG        "Method 'integrand' requires a mesh as the argument."

#define FUNC_ELNTFND                   "FnctlELNtFnd"
#define FUNC_ELNTFND_MSG               "Mesh does not provide elements of grade %u."

#define SCALARPOTENTIAL_FNCLLBL        "SclrPtFnCllbl"
#define SCALARPOTENTIAL_FNCLLBL_MSG    "ScalarPotential function is not callable."

#define LINEARELASTICITY_REF           "LnElstctyRef"
#define LINEARELASTICITY_REF_MSG       "LinearElasticity requires a mesh as the argument."

#define LINEARELASTICITY_PRP           "LnElstctyPrp"
#define LINEARELASTICITY_PRP_MSG       "LinearElasticity requires properties 'reference' to be a mesh, 'grade' to be an integer grade and 'poissonratio' to be a number."

void functional_initialize(void);

#endif /* functional_h */
