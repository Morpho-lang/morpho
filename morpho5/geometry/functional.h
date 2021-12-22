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
#define FLORYHUGGINS_A_PROPERTY               "a"
#define FLORYHUGGINS_B_PROPERTY               "b"
#define FLORYHUGGINS_C_PROPERTY               "c"
#define FLORYHUGGINS_PHI0_PROPERTY            "phi0"
#define EQUIELEMENT_WEIGHT_PROPERTY           "weight"

#define NEMATIC_KSPLAY_PROPERTY               "ksplay"
#define NEMATIC_KTWIST_PROPERTY               "ktwist"
#define NEMATIC_KBEND_PROPERTY                "kbend"
#define NEMATIC_PITCH_PROPERTY                "pitch"
#define NEMATIC_DIRECTOR_PROPERTY             "director"

#define CURVATURE_INTEGRANDONLY_PROPERTY      "integrandonly"

/* Functional methods */
#define FUNCTIONAL_INTEGRAND_METHOD    "integrand"
#define FUNCTIONAL_TOTAL_METHOD        "total"
#define FUNCTIONAL_GRADIENT_METHOD     "gradient"
#define FUNCTIONAL_FIELDGRADIENT_METHOD     "fieldgradient"

/* Special functions that can be used in integrands */
#define TANGENT_FUNCTION               "tangent"
#define NORMAL_FUNCTION                "normal"

/* Functional names */
#define LENGTH_CLASSNAME               "Length"
#define AREA_CLASSNAME                 "Area"
#define AREAENCLOSED_CLASSNAME         "AreaEnclosed"
#define VOLUME_CLASSNAME               "Volume"
#define VOLUMEENCLOSED_CLASSNAME       "VolumeEnclosed"
#define SCALARPOTENTIAL_CLASSNAME      "ScalarPotential"
#define LINEARELASTICITY_CLASSNAME     "LinearElasticity"
#define FLORYHUGGINS_CLASSNAME         "FloryHuggins"
#define EQUIELEMENT_CLASSNAME          "EquiElement"
#define LINECURVATURESQ_CLASSNAME      "LineCurvatureSq"
#define LINETORSIONSQ_CLASSNAME        "LineTorsionSq"
#define MEANCURVATURESQ_CLASSNAME      "MeanCurvatureSq"
#define GAUSSCURVATURE_CLASSNAME       "GaussCurvature"
#define GRADSQ_CLASSNAME               "GradSq"
#define NORMSQ_CLASSNAME               "NormSq"
#define LINEINTEGRAL_CLASSNAME         "LineIntegral"
#define AREAINTEGRAL_CLASSNAME         "AreaIntegral"
#define NEMATIC_CLASSNAME              "Nematic"
#define NEMATICELECTRIC_CLASSNAME      "NematicElectric"

/* Errors */
#define FUNC_INTEGRAND_MESH            "FnctlIntMsh"
#define FUNC_INTEGRAND_MESH_MSG        "Method 'integrand' requires a mesh as the argument."

#define FUNC_ELNTFND                   "FnctlELNtFnd"
#define FUNC_ELNTFND_MSG               "Mesh does not provide elements of grade %u."

#define SCALARPOTENTIAL_FNCLLBL        "SclrPtFnCllbl"
#define SCALARPOTENTIAL_FNCLLBL_MSG    "ScalarPotential function is not callable."

#define LINEINTEGRAL_ARGS              "IntArgs"
#define LINEINTEGRAL_ARGS_MSG          "Integral functionals require a callable argument, followed by zero or more Fields."

#define LINEINTEGRAL_NFLDS             "IntNFlds"
#define LINEINTEGRAL_NFLDS_MSG         "Incorrect number of Fields provided for integrand function."

#define LINEARELASTICITY_REF           "LnElstctyRef"
#define LINEARELASTICITY_REF_MSG       "LinearElasticity requires a mesh as the argument."

#define LINEARELASTICITY_PRP           "LnElstctyPrp"
#define LINEARELASTICITY_PRP_MSG       "LinearElasticity requires properties 'reference' to be a mesh, 'grade' to be an integer grade and 'poissonratio' to be a number."

#define EQUIELEMENT_ARGS               "EquiElArgs"
#define EQUIELEMENT_ARGS_MSG           "EquiElement allows 'grade' and 'weight' as optional arguments."

#define FLORYHUGGINS_ARGS               "FlryHggnsArgs"
#define FLORYHUGGINS_ARGS_MSG           "FloryHuggins requires a reference mesh and allows 'grade', 'a', 'b', 'c' and 'phi0' as optional arguments."

#define FLORYHUGGINS_PRP                "FlryHggnsPrp"
#define FLORYHUGGINS_PRP_MSG            "FloryHuggins requires properties 'reference' to be a mesh, 'grade' to be an integer grade, 'a', 'b' and 'c' to be numbers and 'phi0' as either a number or a field."

#define GRADSQ_ARGS                    "GradSqArgs"
#define GRADSQ_ARGS_MSG                "GradSq requires a field as the argument."

#define NEMATIC_ARGS                   "NmtcArgs"
#define NEMATIC_ARGS_MSG               "Nematic requires a field as the argument."

#define NEMATICELECTRIC_ARGS           "NmtcElArgs"
#define NEMATICELECTRIC_ARGS_MSG       "NematicElectric requires the director and electric field or potential as arguments (in that order)."

#define FUNCTIONAL_ARGS                "FnctlArgs"
#define FUNCTIONAL_ARGS_MSG            "Invalid args passed to method."

void functional_initialize(void);

#endif /* functional_h */
