/** @file builtin.c
 *  @author T J Atherton
 *
 *  @brief Morpho built in functions and classes
*/

#include <stdarg.h>

#include "builtin.h"
#include "common.h"
#include "object.h"
#include "functiondefs.h"
#include "file.h"
#include "system.h"
#include "classes.h"

#include "sparse.h"
#include "geometry.h"

extern objecttypedefn objectmetafunctiondefn;

/* **********************************************************************
 * Global data
 * ********************************************************************** */

/** A table of built in functions */
dictionary builtin_functiontable;

/** A table of built in classes */
dictionary builtin_classtable;

/** A table of symbols used by built in classes */
dictionary builtin_symboltable;

/** Maintain a list of objects created by builtin */
object *builtin_objects;

/** Current function and class tables */
dictionary *_currentfunctiontable;
dictionary *_currentclasstable;

/* **********************************************************************
 * Utility functions
 * ********************************************************************** */

/** Initialize an objectbuiltinfunction */
void builtin_init(objectbuiltinfunction *func) {
    func->flags=BUILTIN_FLAGSEMPTY;
    func->function=NULL;
    func->name=MORPHO_NIL;
    func->klass=NULL;
    signature_init(&func->sig);
}

/** Clear an objectbuiltinfunction */
void builtin_clear(objectbuiltinfunction *func) {
    if (MORPHO_ISOBJECT(func->name)) object_freeifunmanaged(MORPHO_GETOBJECT(func->name));
    signature_clear(&func->sig);
}

/** @brief An enumerate loop.
    @details Successively calls enumerate on obj, passing the result to the supplied function.
    @param[in] v - the virtual machine
    @param[in] obj - object to enumerate over
    @param[in] fn - function to call
    @param[in] ref - reference to pass to the function
    @returns true on success */
bool builtin_enumerateloop(vm *v, value obj, builtin_loopfunction fn, void *ref) {
    value enumerate=MORPHO_NIL;
    value count=MORPHO_NIL, in=MORPHO_INTEGER(-1), val=MORPHO_NIL;
    
    if (morpho_lookupmethod(obj, enumerateselector, &enumerate)) {
        if (!morpho_invoke(v, obj, enumerate, 1, &in, &count)) return false;
        if (!MORPHO_ISINTEGER(count)) return false;
        
        for (indx i=0; i<MORPHO_GETINTEGERVALUE(count); i++) {
            in=MORPHO_INTEGER(i);
            
            if (!morpho_invoke(v, obj, enumerate, 1, &in, &val)) return false;

            if (!(*fn) (v, i, val, ref)) return false;
        }
    }
    
    return true;
}

/** Binds an object to the builtin environment */
void builtin_bindobject(object *obj) {
    if (!obj->next && /* Object is not already bound to the program (or something else) */
        builtin_objects!=obj &&
        obj->status==OBJECT_ISUNMANAGED) {
        obj->status=OBJECT_ISBUILTIN;
        obj->next=builtin_objects;
        builtin_objects=obj;
    }
}

/* **********************************************************************
 * Optional arguments
 * ********************************************************************** */

int vm_getoptionalargs(vm *v);

/** Process optional arguments */
bool builtin_options(vm *v, int nargs, value *args, int *nfixed, int noptions, ...) {
    va_list optlist;
    va_start(optlist, noptions);
    int nopt=vm_getoptionalargs(v);
    
    for (unsigned int i=0; i<noptions; i++) {
        value symbol = va_arg(optlist, value);
        value *dest = va_arg(optlist, value*);
        
        for (int k=0; k<nopt; k++) {
            int r = nargs + 1 + 2*k; // Corresponding register
            if (MORPHO_ISSAME(symbol, args[r])) {
                *dest = args[r+1];
                break;
            }
        }
        // TODO: Should raise an error for unexpected options here by looking for arguments that are strings and unmanaged?
    }
    if (nfixed) *nfixed = nargs; // Exclude register 0
    
    va_end(optlist);
    
    return true;
}

/* **********************************************************************
 * object_builtinfunction definition
 * ********************************************************************** */

/** Instance object definitions */
void objectbuiltinfunction_printfn(object *obj, void *v) {
    objectbuiltinfunction *f = (objectbuiltinfunction *) obj;
    if (f) morpho_printf(v, "<fn %s>", (MORPHO_ISNIL(f->name) ? "" : MORPHO_GETCSTRING(f->name)));
}

void objectbuiltinfunction_freefn(object *obj) {
    builtin_clear((objectbuiltinfunction *) obj);
}

size_t objectbuiltinfunction_sizefn(object *obj) {
    return sizeof(objectbuiltinfunction);
}

objecttypedefn objectbuiltinfunctiondefn = {
    .printfn=objectbuiltinfunction_printfn,
    .markfn=NULL,
    .freefn=objectbuiltinfunction_freefn,
    .sizefn=objectbuiltinfunction_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/* **********************************************************************
 * Signature parsing
 * ********************************************************************** */

/** This mechanism allows builtin classes to cross-reference one another in method signature declarations */

typedef struct _sigparses {
    const char *sig;
    signature *dest;
} _sigparse;

DECLARE_VARRAY(_sigparse, _sigparse)
DEFINE_VARRAY(_sigparse, _sigparse)

varray__sigparse sigparseworklist;

/** Add a signature to be parsed on the next call to builtin_parsesignatures  */
void builtin_addparsesignature(const char *sig, signature *dest) {
    _sigparse s = { .sig = sig, .dest = dest };
    varray__sigparsewrite(&sigparseworklist, s);
}

/** Parses all signatures on the worklist */
bool builtin_parsesignatures(void) {
    _sigparse s;
    while (varray__sigparsepop(&sigparseworklist, &s)) {
        if (!signature_parse(s.sig, s.dest)) {
            return false;
        }
    }
    return true;
}

/* **********************************************************************
 * Create and find builtin functions
 * ********************************************************************** */

/** Gets the current function table */
dictionary *builtin_getfunctiontable(void) {
    return _currentfunctiontable;
}

/** Sets the current function table */
void builtin_setfunctiontable(dictionary *dict) {
    _currentfunctiontable=dict;
}

/** Gets the current class table */
dictionary *builtin_getclasstable(void) {
    return _currentclasstable;
}

/** Sets the current class table */
void builtin_setclasstable(dictionary *dict) {
    _currentclasstable=dict;
}

/** Add a builtin function (old interface)
 * @param name  name of the function
 * @param func  the corresponding C function
 * @param flags flags to define the function
 * @returns value referring to the objectbuiltinfunction */
value builtin_addfunction(char *name, builtinfunction func, builtinfunctionflags flags) {
    value out=MORPHO_NIL;
    morpho_addfunction(name, NULL, func, flags, &out);
    return out;
}

/** Finds a builtin function from its name */
value builtin_findfunction(value name) {
    value out=MORPHO_NIL;
    dictionary_get(&builtin_functiontable, name, &out);
    return out;
}

objectclass *builtin_getparentclass(value fn) {
    if (MORPHO_ISFUNCTION(fn)) return MORPHO_GETFUNCTION(fn)->klass;
    else if (MORPHO_ISBUILTINFUNCTION(fn)) return MORPHO_GETBUILTINFUNCTION(fn)->klass;
    else if (MORPHO_ISMETAFUNCTION(fn)) return MORPHO_GETMETAFUNCTION(fn)->klass;
    else if (MORPHO_ISCLASS(fn)) return MORPHO_GETCLASS(fn)->superclass;
    
    return NULL;
}

/** Adds a new builtinfunction to a given dictionary.
 * @param[in] dict  the dictionary
 * @param[in] name  name of the function to add
 * @param[in] fn function to add
 * @param[in] forcewrap force wrapping the incoming function on first insert
 * @param[out] out the function added (which may be a metafunction)
 * @returns true on success */
bool builtin_addfunctiontodict(dictionary *dict, value name, value fn, bool forcewrap, value *out) {
    bool success=false;
    value entry=MORPHO_NIL, prev=MORPHO_NIL, incoming=fn;
    value selector = dictionary_intern(&builtin_symboltable, name); // Use interned name
    objectclass *klass = builtin_getparentclass(fn);

    if (dictionary_get(dict, selector, &prev) && klass != builtin_getparentclass(prev)) { // Override superclass methods for now
        entry=fn;
        success=dictionary_insert(dict, selector, entry);
    } else {
        if (MORPHO_ISNIL(prev) && forcewrap) {
            if (!metafunction_wrap(name, fn, &incoming)) return false;
        }

        success=metafunction_merge(name, prev, incoming, klass, &entry);
        if (success && MORPHO_ISMETAFUNCTION(entry)) {
            metafunction_setclass(MORPHO_GETMETAFUNCTION(entry), klass);
            if (!MORPHO_ISSAME(prev, entry)) builtin_bindobject(MORPHO_GETOBJECT(entry));
        }
        if (success) success=dictionary_insert(dict, selector, entry);
    }
    
    if (success && out) *out = entry;
    
    return success;
}

/** Add a function to the morpho runtime
 * @param name  name of the function
 * @param signature [optional] signature for the function
 * @param func  the corresponding C function
 * @param flags flags to define the function
 * @param[out] value the function created as usable with morpho_call
 * @returns true on success */
bool morpho_addfunction(char *name, char *signature, builtinfunction func, builtinfunctionflags flags, value *out) {
    objectbuiltinfunction *new = (objectbuiltinfunction *) object_new(sizeof(objectbuiltinfunction), OBJECT_BUILTINFUNCTION);
    if (!new) goto morpho_addfunction_cleanup;
    
    builtin_init(new);
    new->function=func;
    new->flags=flags;
    
    new->name=object_stringfromcstring(name, strlen(name));
    if (!MORPHO_ISSTRING(new->name)) goto morpho_addfunction_cleanup;
    
    // Parse function signature if provided
    if (signature) builtin_addparsesignature(signature, &new->sig);

    value selector = dictionary_intern(&builtin_symboltable, new->name);
    if (MORPHO_ISNIL(selector)) goto morpho_addfunction_cleanup;
    if (!MORPHO_ISSAME(selector, new->name)) morpho_freeobject(new->name);
    new->name=selector;
    if (MORPHO_ISOBJECT(selector)) builtin_bindobject(MORPHO_GETOBJECT(selector));
    
    value newfn = MORPHO_OBJECT(new);
    
    if (!builtin_addfunctiontodict(_currentfunctiontable, new->name, newfn, signature!=NULL, NULL)) {
        UNREACHABLE("Redefinition of function in same extension [in builtin.c]");
    }
    
    // Retain the objectbuiltinfunction in the builtin_objects table
    builtin_bindobject(MORPHO_GETOBJECT(newfn));
    if (out) *out = newfn;
    
    return true;
    
morpho_addfunction_cleanup:
    if (new) {
        object_free((object *) new);
    }
    
    return false;
}

/** Finalize any open metafunctions from the C objects list */
bool builtin_finalizemetafunctions(void) {
    error err;
    error_init(&err);
    if (!metafunction_finalizelist(builtin_objects, &err)) {
        UNREACHABLE("Unable to finalize builtin metafunctions.");
    }
    error_clear(&err);
    
    return true;
}

/* **********************************************************************
 * Create and find builtin classes
 * ********************************************************************** */

/** Defines a built in class
 * @param[in] name          the name of the class
 * @param[in] desc          class description; use MORPHO_GETCLASSDEFINITION(name) to obtain this
 * @param[in] nparents  number of parent classes
 * @param[in] parents    the parent classes
 * @param[out] out          the class object
 * @returns true on success */
bool morpho_addclass(char *name, builtinclassentry desc[], int nparents, value *parents, value *out) {
    value label = object_stringfromcstring(name, strlen(name));
    if (MORPHO_ISOBJECT(label)) builtin_bindobject(MORPHO_GETOBJECT(label));
    objectclass *new = object_newclass(label);
    builtin_bindobject((object *) new);
    bool success=true;
    
    if (!new) return false;
    
    if (dictionary_get(_currentclasstable, label, NULL)) {
        UNREACHABLE("Redefinition of class in same extension [in builtin.c]");
    }
    
    dictionary_insert(_currentclasstable, label, MORPHO_OBJECT(new));
    
    /** Copy methods from superclass */
    for (int i=0; i<nparents; i++) {
        if (MORPHO_ISCLASS(parents[i])) {
            objectclass *parentclass = MORPHO_GETCLASS(parents[i]);
            dictionary_copy(&parentclass->methods, &new->methods);
            if (i==0) new->superclass=parentclass;
            varray_valuewrite(&new->parents, parents[i]);
            varray_valuewrite(&parentclass->children, MORPHO_OBJECT(new));
        }
    }
    
    /** Compute the class linearization */
    if (!class_linearize(new)) {
        UNREACHABLE("Class definition not linearizable.");
    }
    
    for (unsigned int i=0; desc[i].name!=NULL; i++) {
        if (desc[i].type==BUILTIN_METHOD) {
            objectbuiltinfunction *newmethod = (objectbuiltinfunction *) object_new(sizeof(objectbuiltinfunction), OBJECT_BUILTINFUNCTION);
            builtin_init(newmethod);
            newmethod->function=desc[i].function;
            newmethod->klass=new;
            newmethod->name=object_stringfromcstring(desc[i].name, strlen(desc[i].name));
            if (!MORPHO_ISSTRING(newmethod->name)) { success=false; break; }
            newmethod->flags=desc[i].flags;
            if (desc[i].signature) builtin_addparsesignature(desc[i].signature, &newmethod->sig);
            
            value selector = dictionary_intern(&builtin_symboltable, newmethod->name);
            if (MORPHO_ISNIL(selector)) {
                object_free((object *) newmethod);
                success=false;
                break;
            }
            if (!MORPHO_ISSAME(selector, newmethod->name)) morpho_freeobject(newmethod->name);
            newmethod->name=selector;
            if (MORPHO_ISOBJECT(selector)) builtin_bindobject(MORPHO_GETOBJECT(selector));
            value method = MORPHO_OBJECT(newmethod);
            
            builtin_bindobject((object *) newmethod);
            
            builtin_addfunctiontodict(&new->methods, newmethod->name, method, desc[i].signature!=NULL, NULL);
        }
    }
    
    if (success)*out = MORPHO_OBJECT(new);
    return success;
}

/** Defines a built in class (old interface)
 * @param[in] name          the name of the class
 * @param[in] desc          class description; use MORPHO_GETCLASSDEFINITION(name) to obtain this
 * @param[in] superclass the class's superclass
 * @returns the class object */
value builtin_addclass(char *name, builtinclassentry desc[], value superclass) {
    value out = MORPHO_NIL;
    morpho_addclass(name, desc, 1, &superclass, &out);
    return out;
}

/** Finds a builtin class from its label */
value builtin_findclass(value name) {
    value out=MORPHO_NIL;
    if (_currentclasstable) dictionary_get(_currentclasstable, name, &out);
    if (MORPHO_ISNIL(out)) dictionary_get(&builtin_classtable, name, &out);
    return out;
}

/** Finds a builtin class from a cstring label */
value builtin_findclassfromcstring(char *label) {
    objectstring objname = MORPHO_STATICSTRING(label);
    return builtin_findclass(MORPHO_OBJECT(&objname));
}

/** Copies the built in symbol table into a new dictionary */
void builtin_copysymboltable(dictionary *out) {
    dictionary_copy(&builtin_symboltable, out);
}

/** Interns a given symbol. */
value builtin_internsymbol(value symbol) {
    return dictionary_intern(&builtin_symboltable, symbol);
}

/** Interns a symbol given as a C string. */
value builtin_internsymbolascstring(char *symbol) {
    value selector = object_stringfromcstring(symbol, strlen(symbol));
    if (MORPHO_ISOBJECT(selector)) builtin_bindobject(MORPHO_GETOBJECT(selector));
    value internselector = builtin_internsymbol(selector);
    return internselector;
}

/** Checks if a symbol exists in the global symbol table */
bool builtin_checksymbol(value symbol) {
    value val;
    return dictionary_get(&builtin_symboltable, symbol, &val);
}

/* **********************************************************************
 * Initialization/Finalization
 * ********************************************************************** */

extern objecttypedefn objectstringdefn;
extern objecttypedefn objectclassdefn;

objecttype objectbuiltinfunctiontype;

void builtin_initialize(void) {
    dictionary_init(&builtin_functiontable);
    dictionary_init(&builtin_classtable);
    dictionary_init(&builtin_symboltable);
    builtin_objects=NULL;
    
    builtin_setfunctiontable(&builtin_functiontable);
    builtin_setclasstable(&builtin_classtable);
    
    // Initialize core object types
    objectclasstype=object_addtype(&objectclassdefn);
    objectstringtype=object_addtype(&objectstringdefn);
    objectbuiltinfunctiontype=object_addtype(&objectbuiltinfunctiondefn);
    objectmetafunctiontype=object_addtype(&objectmetafunctiondefn);
    
    varray__sigparseinit(&sigparseworklist);

    /* Initialize builtin classes and functions */
    instance_initialize(); // Must initialize first so that Object exists
    
    float_initialize(); // Veneer classes
    int_initialize();
    bool_initialize();
    nil_initialize();
    shortstring_initialize();
    
    string_initialize();  // Classes
    function_initialize();
    cfunction_initialize();
    metafunction_initialize();
    class_initialize();
    upvalue_initialize();
    invocation_initialize();
    dict_initialize();
    list_initialize();
    closure_initialize();
    array_initialize();
    range_initialize();
    complex_initialize();
    err_initialize();
    tuple_initialize();
    
    file_initialize();
    system_initialize();
    json_initialize();
    
    // Initialize function definitions
    functiondefs_initialize();
    
    // Initialize linear algebra
#ifdef MORPHO_INCLUDE_LINALG
    linalg_initialize();
#endif
    
#ifdef MORPHO_INCLUDE_SPARSE
    sparse_initialize();
#endif
    
#ifdef MORPHO_INCLUDE_GEOMETRY
    // Initialize geometry
    geometry_initialize();
#endif
    
    if (!builtin_parsesignatures()) {
        UNREACHABLE("Syntax error in signature.");
    }

    builtin_finalizemetafunctions();
    
    morpho_addfinalizefn(builtin_finalize);
}

void builtin_finalize(void) {
    while (builtin_objects!=NULL) {
        object *next = builtin_objects->next;
        object_free(builtin_objects);
        builtin_objects=next;
    }
    
    varray__sigparseclear(&sigparseworklist);
    
    dictionary_clear(&builtin_functiontable);
    dictionary_clear(&builtin_classtable);
    dictionary_clear(&builtin_symboltable);
}
