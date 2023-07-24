/** @file err.h
 *  @author T J Atherton
 *
 *  @brief Defines Error veneer class
 */

#ifndef err_h
#define err_h

#include "object.h"

/* -------------------------------------------------------
 * Error class
 * ------------------------------------------------------- */

#define ERROR_CLASSNAME                   "Error"

#define ERROR_TAG_PROPERTY                "tag"
#define ERROR_MESSAGE_PROPERTY            "message"

/* -------------------------------------------------------
 * Error error messages
 * ------------------------------------------------------- */

#define ERROR_ARGS                        "ErrorArgs"
#define ERROR_ARGS_MSG                    "Error much be called with a tag and a default message as arguments."

/* -------------------------------------------------------
 * Error interface
 * ------------------------------------------------------- */

/* Initialization/finalization */
void err_initialize(void);
void err_finalize(void);

#endif
