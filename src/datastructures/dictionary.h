/** @file dictionary.h
 *  @author T J Atherton
 *
 *  @brief Dictionary (hashtable) data structure
 */

#ifndef dictionary_h
#define dictionary_h

#include "value.h"

/* -------------------------------------------------------
 * Hash type definition
 * ------------------------------------------------------- */

typedef uint32_t hash;
#define HASH_EMPTY 0

/* -------------------------------------------------------
 * Dictionary entry type definition
 * ------------------------------------------------------- */

/** @brief A single dictionary entry */
typedef struct {
    value key;
    value val;
} dictionaryentry;

/* -------------------------------------------------------
 * Dictionary type definition
 * ------------------------------------------------------- */

/** @brief dictionary data structure that maps keys to values */
typedef struct {
    unsigned int capacity; /** capacity of the dictionary */
    unsigned int count; /** number of items in the dictionary */
    
    dictionaryentry *contents; /** contents of the dictionary */
} dictionary;

/* -------------------------------------------------------
 * Hash functions that can be used in object definitions
 * ------------------------------------------------------- */

hash dictionary_hashint( uint32_t a);
hash dictionary_hashpointer(void *hash);
hash dictionary_hashcstring(const char* key, size_t length);

/* -------------------------------------------------------
 * Dictionary interface
 * ------------------------------------------------------- */

void dictionary_init(dictionary *dict);
void dictionary_clear(dictionary *dict);
void dictionary_freecontents(dictionary *dict, bool freekeys, bool freevals);
bool dictionary_insert(dictionary *dict, value key, value val);
bool dictionary_insertintern(dictionary *dict, value key, value val);
value dictionary_intern(dictionary *dict, value key);
bool dictionary_get(dictionary *dict, value key, value *val);
bool dictionary_getintern(dictionary *dict, value key, value *val);
bool dictionary_remove(dictionary *dict, value key);
bool dictionary_copy(dictionary *src, dictionary *dest);

bool dictionary_union(dictionary *a, dictionary *b, dictionary *out);
bool dictionary_intersection(dictionary *a, dictionary *b, dictionary *out);
bool dictionary_difference(dictionary *a, dictionary *b, dictionary *out);

#ifdef MORPHO_DEBUG
void dictionary_inspect(dictionary *dict);
void dictionary_test(void);
#endif

#endif /* dictionary_h */
