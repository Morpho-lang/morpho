/** @file builtin.h
 *  @author T J Atherton
 *
 *  @brief Morpho built in functions and classes
*/

#ifndef builtin_h
#define builtin_h

#include "object.h"
#include "clss.h"

#ifndef MORPHO_CORE
#include "morpho.h"
#endif

#include "signature.h"

/* -------------------------------------------------------
 * Built in function objects
 * ------------------------------------------------------- */

/** Flags that describe properties of the built in function */
typedef unsigned int builtinfunctionflags;

#define BUILTIN_FLAGSEMPTY    0

#define MORPHO_FN_FLAGSEMPTY  (0)
#define MORPHO_FN_PUREFN      (1<<1)
#define MORPHO_FN_CONSTRUCTOR (1<<2)
#define MORPHO_FN_REENTRANT   (1<<3)
#define MORPHO_FN_OPTARGS     (1<<4)

/** Type of C function that implements a built in Morpho function */
typedef value (*builtinfunction) (vm *v, int nargs, value *args);

/** Object type for built in function */
extern objecttype objectbuiltinfunctiontype;
#define OBJECT_BUILTINFUNCTION objectbuiltinfunctiontype

/** A built in function object */
typedef struct  {
    object obj;
    value name;
    builtinfunctionflags flags;
    builtinfunction function;
    objectclass *klass; 
    signature sig;
} objectbuiltinfunction;

/** Gets an objectfunction from a value */
#define MORPHO_GETBUILTINFUNCTION(val)   ((objectbuiltinfunction *) MORPHO_GETOBJECT(val))

/** Tests whether an object is a function */
#define MORPHO_ISBUILTINFUNCTION(val) object_istype(val, OBJECT_BUILTINFUNCTION)

/* -------------------------------------------------------
 * Built in classes
 * ------------------------------------------------------- */

/** A type used to store the entries of a built in class */
typedef struct {
    enum { BUILTIN_METHOD, BUILTIN_PROPERTY } type;
    char *name;
    char *signature;
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

#define MORPHO_METHOD(label, func, flg)  ((builtinclassentry) { .type=(BUILTIN_METHOD), .name=(label), .signature=NULL, .flags=flg, .function=func})

#define MORPHO_METHOD_SIGNATURE(label, sig, func, flg)  ((builtinclassentry) { .type=(BUILTIN_METHOD), .name=(label), .signature=sig, .flags=flg, .function=func})

#define MORPHO_ENDCLASS         , MORPHO_PROPERTY(NULL) \
                                };

/** Use this macro to retrieve the class definition for calling builtin_addclass */

#define MORPHO_GETCLASSDEFINITION(name)    (builtinclass_##name)

/** Macros and functions for built in classes */

/** Get the nth argument from the args list */
#define MORPHO_GETARG(args, n)  (args[n+1])

/** This macro gets self */
#define MORPHO_SELF(args)       (args[0])

/** Raise an error and return nil */
#define MORPHO_RAISE(v, err)  { morpho_runtimeerror(v, err ); return MORPHO_NIL; }
#define MORPHO_RAISEVARGS(v, err, ...) \
                              { morpho_runtimeerror(v, err, __VA_ARGS__); \
                                return MORPHO_NIL; }

/* -------------------------------------------------------
 * Loop functions to enumerate over enumerable objects
 * ------------------------------------------------------- */

/** Type of C function that implements a built in Morpho function */
typedef bool (*builtin_loopfunction) (vm *v, indx i, value item, void *ref);

/* -------------------------------------------------------
 * Interface
 * ------------------------------------------------------- */

dictionary *builtin_getfunctiontable(void);
void builtin_setfunctiontable(dictionary *dict);

dictionary *builtin_getclasstable(void);
void builtin_setclasstable(dictionary *dict);

value builtin_addfunction(char *name, builtinfunction func, builtinfunctionflags flags);
value builtin_findfunction(value name);

value morpho_addfunction(char *name, char *signature, builtinfunction func, builtinfunctionflags flags);

value builtin_addclass(char *name, builtinclassentry desc[], value superclass);
value builtin_findclass(value name);

void builtin_copysymboltable(dictionary *out);

value builtin_internsymbol(value symbol);
value builtin_internsymbolascstring(char *symbol);
bool builtin_checksymbol(value symbol);

bool builtin_options(vm *v, int nargs, value *args, int *nfixed, int noptions, ...);
bool builtin_iscallable(value val);

bool builtin_enumerateloop(vm *v, value obj, builtin_loopfunction fn, void *ref);

/* -------------------------------------------------------
 * Veneer classes
 * ------------------------------------------------------- */

void object_setveneerclass(objecttype type, value class);
objectclass *object_getveneerclass(objecttype type);
bool object_veneerclasstotype(objectclass *clss, objecttype *type);

void value_setveneerclass(value type, value class);
objectclass *value_getveneerclass(value type);
objectclass *value_veneerclassfromtype(int type);
bool value_veneerclasstotype(objectclass *clss, int *type);

/* -------------------------------------------------------
 * Initialization/finalization
 * ------------------------------------------------------- */

void builtin_initialize(void);
void builtin_finalize(void);

#endif /* builtin_h */
