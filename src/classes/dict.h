/** @file dict.h
 *  @author T J Atherton
 *
 *  @brief Defines dictionary object type and Dictionary veneer class
 */

#ifndef dict_h
#define dict_h

#include "object.h"
#include "dictionary.h"

/* -------------------------------------------------------
 * Dictionary objects
 * ------------------------------------------------------- */

extern objecttype objectdictionarytype;
#define OBJECT_DICTIONARY objectdictionarytype

typedef struct {
    object obj;
    dictionary dict;
} objectdictionary;

/** Tests whether an object is a dictionary */
#define MORPHO_ISDICTIONARY(val) object_istype(val, OBJECT_DICTIONARY)

/** Gets the object as a dictionary */
#define MORPHO_GETDICTIONARY(val)   ((objectdictionary *) MORPHO_GETOBJECT(val))

/** Gets the object's underlying dictionary structure */
#define MORPHO_GETDICTIONARYSTRUCT(val)   (&(((objectdictionary *) MORPHO_GETOBJECT(val))->dict))

objectdictionary *object_newdictionary(void);

/* -------------------------------------------------------
 * Dictionary veneer class
 * ------------------------------------------------------- */

#define DICTIONARY_CLASSNAME              "Dictionary"

#define DICTIONARY_KEYS_METHOD            "keys"
#define DICTIONARY_CONTAINS_METHOD        "contains"
#define DICTIONARY_REMOVE_METHOD          "remove"
#define DICTIONARY_CLEAR_METHOD           "clear"

/* -------------------------------------------------------
 * Dictionary error messages
 * ------------------------------------------------------- */

#define DICT_DCTKYNTFND                   "DctKyNtFnd"
#define DICT_DCTKYNTFND_MSG               "Key not found in dictionary."

#define DICT_DCTSTARG                     "DctStArg"
#define DICT_DCTSTARG_MSG                 "Dictionary set methods (union, intersection, difference) expect a dictionary as the argument."

/* -------------------------------------------------------
 * Dictionary interface
 * ------------------------------------------------------- */

/* Initialization/finalization */
void dict_initialize(void);
void dict_finalize(void);

#endif
