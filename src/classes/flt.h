/** @file flt.h
 *  @author T J Atherton
 *
 *  @brief Veneer class for float values
 */

#ifndef float_h
#define float_h

/* -------------------------------------------------------
 * Float veneer class
 * ------------------------------------------------------- */

#define FLOAT_CLASSNAME "Float"

/* -------------------------------------------------------
 * Float error messages
 * ------------------------------------------------------- */

/* -------------------------------------------------------
 * Float interface
 * ------------------------------------------------------- */

/** Method for format strings */
value Value_format(vm *v, int nargs, value *args);

/* Initialization/finalization */
void float_initialize(void);
void float_finalize(void);

#endif
