/** @file jump.h
 *  @author T J Atherton
 *
 *  @brief Jump functional
 */

#ifndef morpho_functionals_jump_h
#define morpho_functionals_jump_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "integral.h"

/* -------------------------------------------------------
 * Jump veneer class
 * ------------------------------------------------------- */

#define JUMP_CLASSNAME                 "Jump"

#define JUMP_STRATEGY_LABEL            "strategy"
#define JUMP_STRATEGY_QUADRATURE       "quadrature"
#define JUMP_STRATEGY_CENTROID         "centroid"

#define JUMPDN_FUNCTION                "jumpdn"

/* -------------------------------------------------------
 * Jump error messages
 * ------------------------------------------------------- */

#define JUMP_UNIMPL                    "JumpUnimpl"
#define JUMP_UNIMPL_MSG                "This Jump evaluation is not implemented yet."

typedef struct jumpref_s jumpref;

typedef struct {
    objectintegralelementref iface;
    jumpref *jref;
    vm *v;
    objectintegralelementref plus;
    objectintegralelementref minus;
} objectjumpinterfaceref;

extern objecttype objectjumpinterfacereftype;
#define OBJECT_JUMPINTERFACEREF objectjumpinterfacereftype
#define MORPHO_ISJUMPINTERFACEREF(val) object_istype(val, OBJECT_JUMPINTERFACEREF)
#define MORPHO_GETJUMPINTERFACEREF(val) ((objectjumpinterfaceref *) MORPHO_GETOBJECT(val))

objectjumpinterfaceref *jump_getinterfaceref(vm *v);

void jump_initialize(void);

#endif

#endif /* morpho_functionals_jump_h */
