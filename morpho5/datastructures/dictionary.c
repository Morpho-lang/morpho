/** @file dictionary.h
 *  @author T J Atherton
 *
 *  @brief Dictionary (hashtable) data structure
 */

#include <stdio.h>
#include <inttypes.h>
#include "dictionary.h"
#include "common.h"
#include "memory.h"
#include "object.h"

/*
 * Macros that control the behavior of the dictionary.
 */

/** The initial size of non-empty dictionary */
#define DICTIONARY_DEFAULTSIZE 16

/** An empty value */
#define DICTIONARY_EMPTYVALUE MORPHO_NIL

/** Value used to indicate a tombstone */
#define DICTIONARY_TOMBSTONEVALUE MORPHO_TRUE

/** Literal for an empty entry */
#define DICTIONARY_EMPTYENTRY ((dictionaryentry) { DICTIONARY_EMPTYVALUE, DICTIONARY_EMPTYVALUE})

/** Literal for a tombstone entry */
#define DICTIONARY_TOMBSTONEENTRY ((dictionaryentry) { DICTIONARY_EMPTYVALUE, DICTIONARY_TOMBSTONEVALUE})

/*
 * These macros can be changed to tune the algorithm
 */

/** Define if we need to enforce power of two size in our implementation */
#define DICTIONARY_ENFORCEPOWEROFTWO

/** Test if we should resize - the below triggers a resize at 3/4 capacity */
#define DICTIONARY_SIZEINCREASETHRESHOLD(x) (((x)>>1) + ((x)>>2))

/** Test if we should resize - the below triggers a resize at 1/4 capacity */
#define DICTIONARY_SIZEDECREASETHRESHOLD(x) ((x)>>2)

/** Generate a new larger size given the current size - currently multiplies by 2  */
#define DICTIONARY_INCREASESIZE(x) (x<<1)

/** Generate a new smaller size given the current size - currently divides by 2  */
#define DICTIONARY_DECREASESIZE(x) (x>>1)

/** If defined, use Jenkins integer hash function */
//#define DICTIONARY_INTEGERHASH_JENKINS

/** If defined, use Fibonacci integer hash function */
#define DICTIONARY_INTEGERHASH_FIBONACCI

/** Reduce functions */

/** Integer modulo */
//#define DICTIONARY_REDUCE(x, size) (x % size)
/** Faster version for power of two sizes */
#define DICTIONARY_REDUCE(x, size) (x & (size-1))

/** Faster version for arbitrary sizes - https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/ */
/*static inline uint32_t dictionary_reduce64(uint32_t x, uint32_t N) {
  return ((uint64_t) x * (uint64_t) N) >> 32 ;
}*/
//#define DICTIONARY_REDUCE(x, size) (dictionary_reduce64(x,size))

/*
 * Raw hash functions
 */

/** Integer hash function from 32 ubit int to 32 bit uint due to Robert Jenkins */

#ifdef DICTIONARY_INTEGERHASH_JENKINS
static inline hash dictionary_hashint( uint32_t a) {
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);
   return a;
}
#endif

#ifdef DICTIONARY_INTEGERHASH_FIBONACCI
static inline hash dictionary_hashint(uint32_t hash) {
    return (hash * 2654435769u);
}
#endif

static inline hash dictionary_hashpointer(void *hash) {
    uintptr_t ptr = (uintptr_t) hash;
#if UINTPTR_MAX == UINT64_MAX
    return (ptr * 11400714819323198485llu) >> 32;
#elif UINTPTR_MAX == UINT32_MAX
    return (ptr * 2654435769u);
#else
    UNREACHABLE("dictionary pointer hash function undefined. [Check dictionary_hashpointer]");
#endif
}

/** String hashing function FNV-1a combined with Fibonacci hash */
static inline hash dictionary_hashstring(const char* key, size_t length) {
  uint32_t hash = 2166136261u;

  for (unsigned int i=0; i < length; i++) {
    hash ^= key[i];
    hash *= 16777619u;
  }

  return dictionary_hashint(hash);
}

/** Fibonacci hash function for pairs of integers. */
static inline hash dictionary_hashdokkey(objectdokkey *key) {
    uint64_t i1 = MORPHO_GETDOKKEYROW(key);
    uint64_t i2 = MORPHO_GETDOKKEYCOL(key);
    return ((i1<<32 | i2) * 11400714819323198485llu)>> 32;
}

/*
 * Implementation
 */

/** @brief Hashes a value
 * @param key the key to hash */
static hash dictionary_hash(value key, bool intern) {
    if (MORPHO_ISINTEGER(key)) {
        return dictionary_hashint((uint32_t) MORPHO_GETINTEGERVALUE(key));
    } else if (MORPHO_ISOBJECT(key)){
        if (intern) {
            return MORPHO_GETOBJECTHASH(key);
        } else {
            if (MORPHO_ISSTRING(key)) {
                return dictionary_hashstring(MORPHO_GETCSTRING(key), MORPHO_GETSTRINGLENGTH(key));
            } else if (MORPHO_ISDOKKEY(key)) {
                return dictionary_hashdokkey(MORPHO_GETDOKKEY(key));
            } else {
                return dictionary_hashpointer(MORPHO_GETOBJECT(key));
            }
        }
    }
    return 0;
}

/** @brief Initializes a dictionary
 * @param dict the dictionary to initialize */
void dictionary_init(dictionary *dict) {
    dict->capacity=0;
    dict->count=0;
    dict->contents=NULL;
}

/** @brief Clears a dictionary structure, freeing attached memory
 *  @param dict the dictionary to clear
 *  @warning This doens't free keys or values in the dictionary. */
void dictionary_clear(dictionary *dict) {
    if (dict->contents) MORPHO_FREE(dict->contents);
    dictionary_init(dict);
}

/** @brief Wipes a dictionary's contents
 *  @param dict the dictionary to wipe
 *  @warning This doens't free keys or values in the dictionary. */
void dictionary_wipe(dictionary *dict) {
    for (unsigned int i=0; i<dict->capacity; i++) dict->contents[i].key=MORPHO_NIL;
    dict->count=0;
}


/** @brief Frees a dictionary's contents
 *  @param dict     the dictionary to clear
 *  @param freekeys whether to free the keys
 *  @param freevals whether to free the vals */
void dictionary_freecontents(dictionary *dict, bool freekeys, bool freevals) {
    if (dict->contents) {
        for (unsigned int i=0; i<dict->capacity; i++) {
            dictionaryentry *e = &dict->contents[i];
            if (!MORPHO_ISNIL(e->key)) {
                if (freekeys) morpho_freeobject(e->key);
                if (freevals) morpho_freeobject(e->val);
            }
        }
    }
}

/** @brief Resizes a dictionary.
 *  @param dict the dictionary to resize
 *  @param size a new size for the dictionary
 *  @returns true on success
 */
bool dictionary_resize(dictionary *dict, unsigned int size) {
    dictionaryentry *new=NULL, *old = dict->contents;
    unsigned int oldsize = dict->capacity;
#ifdef DICTIONARY_ENFORCEPOWEROFTWO
    unsigned int newsize = morpho_powerof2ceiling(size);
#else
    unsigned int newsize = size;
#endif
    
    /* Don't resize below the minimum */
    if (dict->contents && newsize<DICTIONARY_DEFAULTSIZE) return false;
    
    new=MORPHO_MALLOC(newsize * sizeof(dictionaryentry));

    /* Clear the newly allocated structure */
    if (new) {
        for (unsigned int i=0; i<newsize; i++) new[i] = DICTIONARY_EMPTYENTRY;
    } else return false;
    
    /* Update the dictionary */
    dict->capacity=newsize;
    dict->contents=new;
    dict->count=0; /* Restart from no entries */
    
    if (old) {
        /* Copy the contents over */
        for (unsigned int i=0; i<oldsize; i++) {
            dictionaryentry *e = &old[i];
            
            if (MORPHO_ISNIL(e->key)) continue;
            
            dictionary_insert(dict, e->key, e->val);
        }
        MORPHO_FREE(old); 
    }
    
    return (bool) dict->contents;
}

/** @brief Searches for an entry in a dictionary
 *  @param[in]  dict   the dictionary to search
 *  @param[in]  key    the key to search for
 *  @param[in]  intern whether to use a strict equality search for objects or a fast search
 *  @param[out] entry  the dictionary entry corresponding to the key or a blank entry.
 *  @returns true if the entry was found, false otherwise */
static bool dictionary_find(dictionary *dict, value key, bool intern, dictionaryentry **entry) {
    /* If there's nothing in the hashtable, return immediately */
    if (!dict->contents) return false;
    
    /* Find the starting point by hashing the key */
    unsigned int start_index = DICTIONARY_REDUCE(dictionary_hash(key, intern), dict->capacity);
    unsigned int indx = start_index;
    
    /* Store a pointer to any tombstones we encounter along the way. */
    dictionaryentry *tombstone = NULL;
    
    /* Loop over entries */
    do {
        dictionaryentry *e = &dict->contents[indx];
        
        if (intern) {
            /* If intern is set, we use MORPHO_SAME (tests for equality depending on whether
            objects are the same not just equivalent. */
            if (MORPHO_ISSAME(e->key, key)) {
                /* We found the key! */
                *entry = e;
                return true;
            }
        } else if (MORPHO_ISEQUAL(e->key, key)) {
            /* If intern is false, we can use the slower equivalence test */
            /* We found the key! */
            *entry = e;
            return true;
        }
        
        /* If the key for this entry is blank, it's empty or could be a tombstone */
        if (MORPHO_ISNIL(e->key)) {
            if (MORPHO_ISEQUAL(e->val, DICTIONARY_TOMBSTONEVALUE)) {
                if (!tombstone) tombstone = e; /* We found a tombstone */
            } else {
                /* Otherwise, this is an empty slot. Return this, or the tombstone */
                *entry = (tombstone ? tombstone : e);
                return false;
            }
        }
        
        indx = DICTIONARY_REDUCE((indx + 1), dict->capacity);
    } while (indx!=start_index); /* Loop will always terminate if we do a full cycle of the entries */
     
    return false;
}

/** @brief Internal function that inserts a value in a hashtable given a key
 * @param[in]  dict the dictionary
 * @param[in]  key  key to insert
 * @param[in]  val  value to insert
 * @param[in]  intern use an interned key
 * @returns true if successful, false otherwise
 * @warning If an entry already exists, it is overwritten. Caller should check for existing keys
 *          if this is necessary. */
static inline bool _dictionary_insert(dictionary *dict, value key, value val, bool intern) {
    dictionaryentry *entry=NULL;
    
    if (!dict->contents) {
        dictionary_resize(dict, DICTIONARY_DEFAULTSIZE);
    } else if (dict->count+1 > DICTIONARY_SIZEINCREASETHRESHOLD(dict->capacity)) {
        /* Trigger a resize */
        dictionary_resize(dict, DICTIONARY_INCREASESIZE(dict->capacity));
    }
    
    if (dict->contents) {
        if (dictionary_find(dict, key, intern, &entry)) {
            /* Entry already exists */
            entry->val=val;
            return true;
        } else {
            /* Entry doesn't exist, */
            if (entry) {
                entry->key=key;
                entry->val=val;
                dict->count++;
                return true;
            }
        }
    }
    
    return false;
}

/** @brief Inserts a value in a hashtable given a key
 * @param[in]  dict the dictionary
 * @param[in]  key  key to insert
 * @param[in]  val  value to insert
 * @returns true if successful, false otherwise
 * @warning If an entry already exists, it is overwritten. Caller should check for existing keys
 *          if this is necessary. */
bool dictionary_insert(dictionary *dict, value key, value val) {
    return _dictionary_insert(dict, key, val, false);
}

/** @brief Inserts a value in a hashtable given a key, assuming the key has been interned
 * @param[in]  dict the dictionary
 * @param[in]  key  key to insert
 * @param[in]  val  value to insert
 * @returns true if successful, false otherwise
 * @warning If an entry already exists, it is overwritten. Caller should check for existing keys
 *          if this is necessary. */
bool dictionary_insertintern(dictionary *dict, value key, value val) {
    return _dictionary_insert(dict, key, val, true);
}
    
/** @brief Interns a new key
 *  @detail Looks to see if a similar key [one that passes MORPHO_ISEQUAL] is already
 *          in the dictionary, and if so returns it. Otherwise, inserts this key into
 *          the dictionary.
 *  @param[in]  dict the dictionary
 *  @param[in]  key  a new key to intern
 *  @returns the internalized key, or MORPHO_NIL on failure. */
value dictionary_intern(dictionary *dict, value key) {
    dictionaryentry *entry=NULL;
    
    /* Is the key already in the dictionary? */
    if (dictionary_find(dict, key, false, &entry)) {
        return entry->key;
    } else {
        /* If not, insert it with a blank value */
        if (dictionary_insert(dict, key, MORPHO_NIL)) {
            MORPHO_SETOBJECTHASH( key, dictionary_hash(key, false));
            return key;
        }
    }
    
    return MORPHO_NIL;
}

/** @brief Internal function that retrieves a value from a dictionary given a key
 * @param[in]  dict   the dictionary to get
 * @param[in]  key    key to locate
 * @param[in]  intern should we assume the key has been interned?
 *                    i.e. that object equivalence is tested with MORPHO_ISSAME rather than MORPHO_ISEQUAL
 * @param[out] val  Stores the result in this value if found.
 * @returns true if found, false otherwise
*/
static inline bool _dictionary_get(dictionary *dict, value key, bool intern, value *val) {
    dictionaryentry *entry=NULL;
    
    if (dictionary_find(dict, key, intern, &entry)) {
        if (val) *val = entry->val;
        return true;
    }
    
    return false;
}

/** @brief Retrieves a value from a dictionary given a key
 * @param[in]  dict the dictionary to get
 * @param[in]  key  key to locate
 * @param[out] val  Stores the result in this value if found.
 * @returns true if found, false otherwise
 */
bool dictionary_get(dictionary *dict, value key, value *val) {
    return _dictionary_get(dict, key, false, val);
}

/** @brief Retrieves a value from a dictionary given a key assuming
 *         that key has been interned.
 * @param[in]  dict the dictionary to get
 * @param[in]  key  key to locate
 * @param[out] val  Stores the result in this value if found.
 * @returns true if found, false otherwise
 */
bool dictionary_getintern(dictionary *dict, value key, value *val) {
    return _dictionary_get(dict, key, true, val);
}

/** @brief Removes a key from a dictionary given a key
 * @param[in]  dict the dictionary to initialize
 * @param[in]  key  key to remove
 * @returns true if the key was found, false otherwise
 */
bool dictionary_remove(dictionary *dict, value key) {
    dictionaryentry *entry=NULL;
    
    if (dictionary_find(dict, key, false, &entry)) {
        *entry = DICTIONARY_TOMBSTONEENTRY;
        dict->count--;
        
        /* If we have lost our last entry, clear the dictionary */
        if (dict->count==0) {
            dictionary_clear(dict);
        } else if (dict->count < DICTIONARY_SIZEDECREASETHRESHOLD(dict->capacity)) {
            /* Trigger a resize if we are below some threshhold */
            dictionary_resize(dict, DICTIONARY_DECREASESIZE(dict->capacity));
        }
        
        return true;
    }
    
    return false;
}

/** @brief Copies the entries of one dictionary to another */
bool dictionary_copy(dictionary *src, dictionary *dest) {
    if (src->contents) {
        for (unsigned int i=0; i<src->capacity; i++) {
            dictionaryentry *e = &src->contents[i];
            if (!MORPHO_ISNIL(e->key)) {
                if (!dictionary_insert(dest, e->key, e->val)) return false;
            }
        }
    }
    return true;
}

/* ---------------------------
 * Set functions
 * --------------------------- */

/** @brief Computes the union of two dictionaries, i.e. the output dictionary contains all keys that occur in either a or b
 * @param[in]  a - input dictionary (values from this dictionary take priority)
 * @param[in]  b - input dictionary
 * @param[out] out - output dictionary
 * @returns true on success. */
bool dictionary_union(dictionary *a, dictionary *b, dictionary *out) {
    dictionary_clear(out);
    if (!dictionary_copy(b, out)) return false;
    if (!dictionary_copy(a, out)) return false;
    return true;
}

/** @brief Computes the intersection of two dictionaries, i.e. the output dictionary contains only keys that occur in both a and b
 * @param[in]  a - input dictionary (values are copied from this dictionary)
 * @param[in]  b - input dictionary
 * @param[out] out - output dictionary
 * @returns true on success. */
bool dictionary_intersection(dictionary *a, dictionary *b, dictionary *out) {
    bool success=true;
    dictionary_clear(out);
    if (a->contents) {
        for (unsigned int i=0; i<a->capacity; i++) {
            dictionaryentry *e = &a->contents[i];
            if (!MORPHO_ISNIL(e->key)) {
                if (dictionary_get(b, e->key, NULL)) {
                    if (!dictionary_insert(out, e->key, e->val)) return false;
                }
            }
        }
    }
    return success;
}

/** @brief Computes the difference of two dictionaries, i.e. the output dictionary contains only keys from a that do NOT occur in B
 * @param[in]  a - input dictionary (keys are copied from this)
 * @param[in]  b - input dictionary
 * @param[out] out - output dictionary
 * @returns true on success. */
bool dictionary_difference(dictionary *a, dictionary *b, dictionary *out) {
    bool success=true;
    dictionary_clear(out);
    if (a->contents) {
        for (unsigned int i=0; i<a->capacity; i++) {
            dictionaryentry *e = &a->contents[i];
            if (!MORPHO_ISNIL(e->key)) {
                if (!dictionary_get(b, e->key, NULL)) {
                    if (!dictionary_insert(out, e->key, e->val)) return false;
                }
            }
        }
    }
    return success;
}

/* ---------------------------
 * Debugging and performance
 * --------------------------- */

#ifdef MORPHO_DEBUG
#include <time.h>

/** @brief Prints the dictionary data structure for debugging purposes */
void dictionary_inspect(dictionary *dict) {
    printf("[");
    for (unsigned int i=0; i<dict->capacity; i++) {
        dictionaryentry *e = &dict->contents[i];
        morpho_printvalue(e->key);
        printf(" ");
    }
    printf("]");
    
    printf("\n");
}

/** @brief Runs a performance test */
void dictionary_testforsize(unsigned int n) {
   clock_t start, end;
   dictionary dict;
   
   /* Test the dictionary */
   dictionary_init(&dict);
   
   printf("%i ", n);
    
   start = clock();
   for (unsigned int i=0; i<n; i++) {
       dictionary_insert(&dict, MORPHO_INTEGER(i), MORPHO_INTEGER(i));
   }
   end = clock();
   
   printf("%g ",((double) end-start)/((double) CLOCKS_PER_SEC)/((double) n));
   
   start = clock();
   value res=MORPHO_INTEGER(0);
   for (unsigned int i=0; i<n; i++) {
       dictionary_get(&dict, MORPHO_INTEGER(i), &res);
   }
   end = clock();
   
   printf("%g ", ((double) end-start)/((double) CLOCKS_PER_SEC)/((double) n));
    
   start = clock();
   for (unsigned int i=0; i<n; i++) {
       dictionary_remove(&dict, MORPHO_INTEGER(i));
   }
   end = clock();
   
   printf("%g", ((double) end-start)/((double) CLOCKS_PER_SEC)/((double) n));
    
   printf("\n");
   
   dictionary_clear(&dict);
}

/** @define Test the hashtable implementation */
void dictionary_test(void) {
    /*for (unsigned int n=1; n<10000000; n=n*2) {
        dictionary_testforsize(n);
    }*/
}
#endif
