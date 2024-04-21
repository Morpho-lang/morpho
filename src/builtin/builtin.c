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

/** Keep a list of objects created by builtin */
varray_value builtin_objects;

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
    varray_valueinit(&func->signature);
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

/* **********************************************************************
 * Optional arguments
 * ********************************************************************** */

extern value vm_optmarker;

/** Process optional arguments */
bool builtin_options(vm *v, int nargs, value *args, int *nfixed, int noptions, ...) {
    va_list optlist;
    va_start(optlist, noptions);
    int nposn=nargs;
    
    for (unsigned int i=1; i<=nargs; i++) {
        if (MORPHO_ISSAME(args[i], vm_optmarker)) { nposn=i-1; break; }
    }
    
    for (unsigned int i=0; i<noptions; i++) {
        value symbol = va_arg(optlist, value);
        value *dest = va_arg(optlist, value*);
        
        for (int k=nposn+2; k<nargs; k+=2) {
            if (MORPHO_ISSAME(symbol, args[k])) {
                *dest = args[k+1];
                break;
            }
        }
        // TODO: Should raise an error for unexpected options here by looking for arguments that are strings and unmanaged?
    }
    if (nfixed) *nfixed = nposn; // Exclude register 0
    
    va_end(optlist);
    
    return true;
}

/** Tests whether an object is callable */
bool builtin_iscallable(value val) {
    return (MORPHO_ISOBJECT(val) && (MORPHO_ISFUNCTION(val) ||
                                     MORPHO_ISCLOSURE(val) ||
                                     MORPHO_ISINVOCATION(val) ||
                                     MORPHO_ISBUILTINFUNCTION(val)));
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
    objectbuiltinfunction *func = (objectbuiltinfunction *) obj;
    varray_valueclear(&func->signature);
    morpho_freeobject(func->name);
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
    objectbuiltinfunction *new = (objectbuiltinfunction *) object_new(sizeof(objectbuiltinfunction), OBJECT_BUILTINFUNCTION);
    value out = MORPHO_NIL;
    varray_valuewrite(&builtin_objects, MORPHO_OBJECT(new));
    
    if (new) {
        builtin_init(new);
        new->function=func;
        new->name=object_stringfromcstring(name, strlen(name));
        new->flags=flags;
        out = MORPHO_OBJECT(new);
        
        value selector = dictionary_intern(&builtin_symboltable, new->name);
        
        if (dictionary_get(_currentfunctiontable, new->name, NULL)) {
            UNREACHABLE("Redefinition of function in same extension [in builtin.c]");
        }
        
        dictionary_insert(_currentfunctiontable, selector, out);
    }
    
    return out;
}

/** Finds a builtin function from its name */
value builtin_findfunction(value name) {
    value out=MORPHO_NIL;
    dictionary_get(&builtin_functiontable, name, &out);
    return out;
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
    varray_valuewrite(&builtin_objects, label);
    objectclass *new = object_newclass(label);
    varray_valuewrite(&builtin_objects, MORPHO_OBJECT(new));
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
            objectbuiltinfunction *method = (objectbuiltinfunction *) object_new(sizeof(objectbuiltinfunction), OBJECT_BUILTINFUNCTION);
            builtin_init(method);
            method->function=desc[i].function;
            method->klass=new;
            method->name=object_stringfromcstring(desc[i].name, strlen(desc[i].name));
            method->flags=desc[i].flags;
            
            value selector = dictionary_intern(&builtin_symboltable, method->name);
            
            varray_valuewrite(&builtin_objects, MORPHO_OBJECT(method));
            
            if (dictionary_get(&new->methods, method->name, NULL) &&
                ( !superklass || // Ok to redefine methods in the superclass 
                  !dictionary_get(&superklass->methods, method->name, NULL)) ) {
                UNREACHABLE("redefinition of method in builtin class (check builtin.c)");
            }
            
            dictionary_insert(&new->methods, selector, MORPHO_OBJECT(method));
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
    varray_valuewrite(&builtin_objects, selector);
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
    varray_valueinit(&builtin_objects);
    
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
    
    file_initialize();
    system_initialize();
    json_initialize();
    
    // Initialize function definitions
    functiondefs_initialize();
    
    // Initialize linear algebra
    matrix_initialize();
    sparse_initialize();
    
    // Initialize geometry
    mesh_initialize();
    selection_initialize();
    field_initialize();
    functional_initialize();
    
    morpho_addfinalizefn(builtin_finalize);
}

void builtin_finalize(void) {
    for (unsigned int i=0; i<builtin_objects.count; i++) {
        morpho_freeobject(builtin_objects.data[i]);
    }
    dictionary_clear(&builtin_functiontable);
    dictionary_clear(&builtin_classtable);
    dictionary_clear(&builtin_symboltable);
    varray_valueclear(&builtin_objects);
}
