/** @file metafunction.h
 *  @author T J Atherton
 *
 *  @brief Defines metafunction object type and Metafunction veneer class
 */

#ifndef metafunction_h
#define metafunction_h

#include "object.h"

/* -------------------------------------------------------
 * Metafunction objects
 * ------------------------------------------------------- */

extern objecttype objectmetafunctiontype;
#define OBJECT_METAFUNCTION objectmetafunctiontype

/** Index type for metafunction resolver */
typedef int mfindx;

/** Compiled metafunction bytecode cell */
typedef int mfinstruction;

DECLARE_VARRAY(mfinstruction, mfinstruction);

/** Metafunction status */
typedef enum {
    METAFUNCTION_BUILDING,
    METAFUNCTION_FROZEN
} metafunctionstate;

/** A metafunction object */
typedef struct sobjectmetafunction {
    object obj;
    value name;
    objectclass *klass; // Parent class for metafunction methods
    metafunctionstate state;
    varray_value fns;
    varray_mfinstruction resolver;
    mfindx entry; // Entry point for the resolver
} objectmetafunction;

/** Gets an objectmetafunction from a value */
#define MORPHO_GETMETAFUNCTION(val)   ((objectmetafunction *) MORPHO_GETOBJECT(val))

/** Tests whether an object is a metafunction */
#define MORPHO_ISMETAFUNCTION(val) object_istype(val, OBJECT_METAFUNCTION)

/* -------------------------------------------------------
 * Metafunction veneer class
 * ------------------------------------------------------- */

#define METAFUNCTION_CLASSNAME "Metafunction"

/* -------------------------------------------------------
 * Metafunction error messages
 * ------------------------------------------------------- */

#define METAFUNCTION_CMPLAMBGS          "MltplDisptchAmbg"
#define METAFUNCTION_CMPLAMBGS_MSG      "Ambiguous or duplicate implementations in multiple dispatch."

#define METAFUNCTION_UNFROZEN           "MltplDisptchUnfrzn"
#define METAFUNCTION_UNFROZEN_MSG       "Metafunction was used before it was finalized."

/* -------------------------------------------------------
 * Metafunction interface
 * ------------------------------------------------------- */

objectmetafunction *object_newmetafunction(value name);
objectmetafunction *metafunction_clone(objectmetafunction *f);

bool metafunction_wrap(value name, value fn, value *out);
bool metafunction_add(objectmetafunction *f, value fn);

void metafunction_setclass(objectmetafunction *f, objectclass *klass);
objectclass *metafunction_class(objectmetafunction *f);

bool metafunction_matchfn(objectmetafunction *fn, value f);
bool metafunction_matchset(objectmetafunction *fn, int n, value *fns);
signature *metafunction_getsignature(value fn);

void metafunction_inferreturntype(objectmetafunction *fn, value *type);

bool metafunction_compile(objectmetafunction *fn, error *err);
bool metafunction_finalize(objectmetafunction *fn, error *err);
bool metafunction_finalizelist(object *list, error *err);
void metafunction_clearinstructions(objectmetafunction *fn);
void metafunction_disassemble(objectmetafunction *fn);

bool metafunction_resolve(objectmetafunction *f, int nargs, value *args, error *err, value *fn);
bool metafunction_reduce(objectmetafunction *f, int nargs, value *args, error *err, value *out);

void metafunction_initialize(void);

#endif
