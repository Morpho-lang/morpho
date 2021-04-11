/** @file builtin.h
 *  @author T J Atherton
 *
 *  @brief Morpho built in functions and classes
*/

#ifndef builtin_h
#define builtin_h

#include "object.h"

#ifndef MORPHO_CORE
#include "morpho.h"
#endif

/* ---------------------------
 * Built in functions
 * --------------------------- */

/** Flags that describe properties of the built in function */
typedef unsigned int builtinfunctionflags;

#define BUILTIN_FLAGSEMPTY  0

/** Type of C function that implements a built in Morpho function */
typedef value (*builtinfunction) (vm *v, int nargs, value *args);

/** A built in function object */
typedef struct  {
    object obj;
    value name;
    builtinfunctionflags flags;
    builtinfunction function;
} objectbuiltinfunction;

/** Gets an objectfunction from a value */
#define MORPHO_GETBUILTINFUNCTION(val)   ((objectbuiltinfunction *) MORPHO_GETOBJECT(val))

/** Tests whether an object is a function */
#define MORPHO_ISBUILTINFUNCTION(val) object_istype(val, OBJECT_BUILTINFUNCTION)

/* ---------------------------
 * Built in classes
 * --------------------------- */
/** A type used to store the entries of a built in class */
typedef struct {
    enum { BUILTIN_METHOD, BUILTIN_PROPERTY } type;
    char *name;
    builtinfunctionflags flags;
    builtinfunction function;
} builtinclassentry;

/** The following macros help to define a built in class. They should be used outside of any function declaration.
 *  To use:
 *  MORPHO_BEGINCLASS(Object)  - Starts the declaration
 *  MORPHO_PROPERTY("test")       - Adds a property called "test" to the definition
 *  MORPHO_METHOD("init", object_init, BUILTIN_FLAGSEMPTY)  - Adds a method called "init" to the definition
 *  MORPHO_ENDCLASS                  - Ends the declaration */

#define MORPHO_BEGINCLASS(name) builtinclassentry builtinclass_##name[] = {

#define MORPHO_PROPERTY(label)  ((builtinclassentry) { .type=(BUILTIN_PROPERTY), .name=(label), .flags=BUILTIN_FLAGSEMPTY, .function=NULL})

#define MORPHO_METHOD(label, func, flg)  ((builtinclassentry) { .type=(BUILTIN_METHOD), .name=(label), .flags=flg, .function=func})

#define MORPHO_ENDCLASS         , MORPHO_PROPERTY(NULL) \
                                };

#define MORPHO_GETCLASSDEFINITION(name)    (builtinclass_##name)

/** Macros and functions for built in classes */

#define MORPHO_GETARG(args, n)  (args[n+1])

/** This macro gets self */
#define MORPHO_SELF(args)       (args[0])

/** Raise an error and return nil */
#define MORPHO_RAISE(v, err)  { morpho_runtimeerror(v, err ); return MORPHO_NIL; }
#define MORPHO_RAISEVARGS(v, err, ...) \
                              { morpho_runtimeerror(v, err, __VA_ARGS__); \
                                return MORPHO_NIL; }

/* ---------------------------
 * Loop functions
 * --------------------------- */

/** Type of C function that implements a built in Morpho function */
typedef bool (*builtin_loopfunction) (vm *v, indx i, value item, void *ref);

/* ---------------------------
 * Prototypes
 * --------------------------- */

value builtin_addfunction(char *name, builtinfunction func, builtinfunctionflags flags);
value builtin_findfunction(value name);
void builtin_printfunction(objectbuiltinfunction *f);

value builtin_addclass(char *name, builtinclassentry desc[], value superclass);
value builtin_findclass(value name);

void builtin_copysymboltable(dictionary *out);

value builtin_internsymbol(value symbol);
value builtin_internsymbolascstring(char *symbol);

void builtin_setveneerclass(objecttype type, value class);
objectclass *builtin_getveneerclass(objecttype type);

bool builtin_options(vm *v, int nargs, value *args, int *nfixed, int noptions, ...);
bool builtin_iscallable(value val);

bool builtin_enumerateloop(vm *v, value obj, builtin_loopfunction fn, void *ref);

void builtin_initialize(void);
void builtin_finalize(void);

#endif /* builtin_h */
