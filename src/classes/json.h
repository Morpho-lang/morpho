/** @file json.h
 *  @author T J Atherton
 *
 *  @brief JSON parser
 */

#ifndef json_h
#define json_h

#include "object.h"
#include "dictionary.h"

/* -------------------------------------------------------
 * JSON class
 * ------------------------------------------------------- */

#define JSON_CLASSNAME              "JSON"

#define JSON_PARSEMETHOD            "parse"

/* -------------------------------------------------------
 * JSON error messages
 * ------------------------------------------------------- */

#define JSON_OBJCTKEY                    "JSONObjctKey"
#define JSON_OBJCTKEY_MSG                "JSON object keys must be strings."

#define JSON_PRSARGS                     "JSONPrsArgs"
#define JSON_PRSARGS_MSG                 "Method 'parse' requires a string as the argument."

/* -------------------------------------------------------
 * JSON interface
 * ------------------------------------------------------- */

/* Parse JSON into a value */
bool json_parse(char *in, error *err, value *out, varray_value *objects);

/* Initialization/finalization */
void json_initialize(void);
void json_finalize(void);

#endif
