/** @file builtin.c
 *  @author T J Atherton
 *
 *  @brief Morpho built in functions and classes
*/

#include "builtin.h"
#include "common.h"
#include "object.h"
#include "functiondefs.h"
#include "file.h"
#include "system.h"
#include "classes.h"

#include "sparse.h"
#include "mesh.h"
#include "selection.h"
#include "functional.h"
#include "field.h"

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
    morpho_freeobject(func->name);
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

/** Tests whether an object is callable */
bool builtin_iscallable(value val) {
    return (MORPHO_ISOBJECT(val) && (MORPHO_ISFUNCTION(val) ||
                                     MORPHO_ISCLOSURE(val) ||
                                     MORPHO_ISINVOCATION(val) ||
                                     MORPHO_ISBUILTINFUNCTION(val) ||
                                     MORPHO_ISMETAFUNCTION(val)));
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

/** Add a builtin function.
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
 * @param[out] out the function added (which may be a metafunction)
 * @returns true on success */
bool builtin_addfunctiontodict(dictionary *dict, value name, value fn, value *out) {
    bool success=false;
    value entry=fn; // Dictionary entry for this name
    value selector = dictionary_intern(&builtin_symboltable, name); // Use interned name
    
    if (dictionary_get(dict, selector, &entry)) { // There was an existing function
        if (MORPHO_ISBUILTINFUNCTION(entry)) { // It was a builtinfunction, so we need to create a metafunction
            if (builtin_getparentclass(fn) !=
                MORPHO_GETBUILTINFUNCTION(entry)->klass) { // Override superclass methods for now
                dictionary_insert(dict, selector, fn);
            } else if (metafunction_wrap(name, entry, &entry)) { // Wrap the old definition in a metafunction
                
                builtin_bindobject(MORPHO_GETOBJECT(entry));
                metafunction_add(MORPHO_GETMETAFUNCTION(entry), fn); // Add the new definition
                success=dictionary_insert(dict, selector, entry);
            }
        } else if (MORPHO_ISMETAFUNCTION(entry)) { // It was already a metafunction so simply add the new function
            success=metafunction_add(MORPHO_GETMETAFUNCTION(entry), fn);
        }
    } else success=dictionary_insert(dict, selector, fn);
    
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
    if (!name) goto morpho_addfunction_cleanup;
    
    // Parse function signature if provided
    if (signature &&
        !signature_parse(signature, &new->sig)) {
        UNREACHABLE("Syntax error in signature definition.");
    }
    
    value newfn = MORPHO_OBJECT(new);
    
    if (!builtin_addfunctiontodict(_currentfunctiontable, new->name, newfn, NULL)) {
        UNREACHABLE("Redefinition of function in same extension [in builtin.c]");
    }
    
    // Retain the objectbuiltinfunction in the builtin_objects table
    builtin_bindobject(MORPHO_GETOBJECT(newfn));
    if (out) *out = newfn;
    
    return true;
    
morpho_addfunction_cleanup:
    if (new) {
        builtin_clear(new);
        object_free((object *) new);
    }
    
    return false;
}

/* **********************************************************************
 * Create and find builtin classes
 * ********************************************************************** */

/** Defines a built in class
 * @param[in] name          the name of the class
 * @param[in] desc          class description; use MORPHO_GETCLASSDEFINITION(name) to obtain this
 * @param[in] superclass the class's superclass
 * @returns the class object */
value builtin_addclass(char *name, builtinclassentry desc[], value superclass) {
    value label = object_stringfromcstring(name, strlen(name));
    builtin_bindobject(MORPHO_GETOBJECT(label));
    objectclass *new = object_newclass(label);
    builtin_bindobject((object *) new);
    objectclass *superklass = NULL;
    
    if (!new) return MORPHO_NIL;
    
    /** Copy methods from superclass */
    if (MORPHO_ISCLASS(superclass)) {
        superklass = MORPHO_GETCLASS(superclass);
        dictionary_copy(&superklass->methods, &new->methods);
        new->superclass=superklass;
    }
    
    for (unsigned int i=0; desc[i].name!=NULL; i++) {
        if (desc[i].type==BUILTIN_METHOD) {
            objectbuiltinfunction *newmethod = (objectbuiltinfunction *) object_new(sizeof(objectbuiltinfunction), OBJECT_BUILTINFUNCTION);
            builtin_init(newmethod);
            newmethod->function=desc[i].function;
            newmethod->klass=new;
            newmethod->name=object_stringfromcstring(desc[i].name, strlen(desc[i].name));
            newmethod->flags=desc[i].flags;
            if (desc[i].signature) {
                signature_parse(desc[i].signature, &newmethod->sig);
            }
            
            dictionary_intern(&builtin_symboltable, newmethod->name);
            value method = MORPHO_OBJECT(newmethod);
            
            builtin_bindobject((object *) newmethod);
            
            builtin_addfunctiontodict(&new->methods, newmethod->name, method, NULL);
        }
    }
    
    if (dictionary_get(_currentclasstable, label, NULL)) {
        UNREACHABLE("Redefinition of class in same extension [in builtin.c]");
    }
    
    dictionary_insert(_currentclasstable, label, MORPHO_OBJECT(new));
    
    return MORPHO_OBJECT(new);
}

/** Finds a builtin class from its name */
value builtin_findclass(value name) {
    value out=MORPHO_NIL;
    dictionary_get(&builtin_classtable, name, &out);
    return out;
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
    builtin_bindobject(MORPHO_GETOBJECT(selector));
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
    objectstringtype=object_addtype(&objectstringdefn);
    objectclasstype=object_addtype(&objectclassdefn);
    objectbuiltinfunctiontype=object_addtype(&objectbuiltinfunctiondefn);
    
    /* Initialize builtin classes and functions */
    instance_initialize(); // Must initialize first so that Object exists
    
    string_initialize();  // Classes
    function_initialize();
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
    
    float_initialize();// Veneer classes
    int_initialize();
    bool_initialize();
    
    file_initialize();
    system_initialize();
    json_initialize();
    
    // Initialize function definitions
    functiondefs_initialize();
    
    // Initialize linear algebra
#ifdef MORPHO_INCLUDE_LINALG
    matrix_initialize();
#endif
    
#ifdef MORPHO_INCLUDE_SPARSE
    sparse_initialize();
#endif
    
#ifdef MORPHO_INCLUDE_GEOMETRY
    // Initialize geometry
    mesh_initialize();
    selection_initialize();
    field_initialize();
    functional_initialize();
#endif
    
    morpho_addfinalizefn(builtin_finalize);
}

void builtin_finalize(void) {
    while (builtin_objects!=NULL) {
        object *next = builtin_objects->next;
        object_free(builtin_objects);
        builtin_objects=next;
    }
    
    dictionary_clear(&builtin_functiontable);
    dictionary_clear(&builtin_classtable);
    dictionary_clear(&builtin_symboltable);
}
