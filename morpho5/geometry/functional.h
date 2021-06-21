/** @file functional.h
 *  @author T J Atherton
 *
 *  @brief Functionals
 */

#ifndef functional_h
#define functional_h

#include <stdio.h>

/* Functional properties */
#define FUNCTIONAL_GRADE_PROPERTY             "grade"
#define FUNCTIONAL_ONESIDED_PROPERTY          "onesided"
#define FUNCTIONAL_FIELD_PROPERTY             "field"
#define SCALARPOTENTIAL_FUNCTION_PROPERTY     "function"
#define SCALARPOTENTIAL_GRADFUNCTION_PROPERTY "gradfunction"
#define LINEARELASTICITY_REFERENCE_PROPERTY   "reference"
#define LINEARELASTICITY_POISSON_PROPERTY     "poissonratio"
#define EQUIELEMENT_WEIGHT_PROPERTY           "weight"

/* Functional methods */
#define FUNCTIONAL_INTEGRAND_METHOD    "integrand"
#define FUNCTIONAL_TOTAL_METHOD        "total"
#define FUNCTIONAL_GRADIENT_METHOD     "gradient"
#define FUNCTIONAL_FIELDGRADIENT_METHOD     "fieldgradient"

/* Functional names */
#define LENGTH_CLASSNAME               "Length"
#define AREA_CLASSNAME                 "Area"
#define AREAENCLOSED_CLASSNAME         "AreaEnclosed"
#define VOLUME_CLASSNAME               "Volume"
#define VOLUMEENCLOSED_CLASSNAME       "VolumeEnclosed"
#define SCALARPOTENTIAL_CLASSNAME      "ScalarPotential"
#define LINEARELASTICITY_CLASSNAME     "LinearElasticity"
#define EQUIELEMENT_CLASSNAME          "EquiElement"
#define LINECURVATURESQ_CLASSNAME      "LineCurvatureSq"
#define LINETORSIONSQ_CLASSNAME        "LineTorsionSq"
#define GRADSQ_CLASSNAME               "GradSq"
#define NORMSQ_CLASSNAME               "NormSq"
#define LINEINTEGRAL_CLASSNAME         "LineIntegral"

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

#define EQUIELEMENT_ARGS               "EquiElArgs"
#define EQUIELEMENT_ARGS_MSG           "EquiElement allows 'grade' and 'weight' as optional arguments."

#define GRADSQ_ARGS                    "GradSqArgs"
#define GRADSQ_ARGS_MSG                "GradSq requires a field as the argument."

void functional_initialize(void);

#endif /* functional_h */
