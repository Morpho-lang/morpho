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

#define VALUE_FRMTARG                     "FrmtArg"
#define VALUE_FRMTARG_MSG                 "Format method requires a format string as its argument."

#define VALUE_INVLDFRMT                   "InvldFrmt"
#define VALUE_INVLDFRMT_MSG               "Invalid format string."

/* -------------------------------------------------------
 * Float interface
 * ------------------------------------------------------- */

/** Method for format strings */
value Value_format(vm *v, int nargs, value *args);

void float_initialize(void);

#endif
