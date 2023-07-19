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

#define JSON_EXTRNSTOK                   "JSONExtrnsTkn"
#define JSON_EXTRNSTOK_MSG               "Extraneous token after JSON element."

#define JSON_PRSARGS                     "JSONPrsArgs"
#define JSON_PRSARGS_MSG                 "Method 'parse' requires a string as the argument."

#define JSON_UNESCPDCTRL                 "JSONUnescpdCtrl"
#define JSON_UNESCPDCTRL_MSG             "Unescaped control character in string literal."

#define JSON_NMBRFRMT                    "JSONNmbrFrmt"
#define JSON_NMBRFRMT_MSG                "Improperly formatted number."

#define JSON_BLNKELMNT                   "JSONBlnkElmnt"
#define JSON_BLNKELMNT_MSG               "Blank element."

/* -------------------------------------------------------
 * JSON interface
 * ------------------------------------------------------- */

/* Parse JSON into a value */
bool json_parse(char *in, error *err, value *out, varray_value *objects);

/* Initialization/finalization */
void json_initialize(void);
void json_finalize(void);

#endif
