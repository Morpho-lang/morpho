/** @file builtin.c
 *  @author T J Atherton
 *
 *  @brief Morpho built in functions and classes
*/

#include "builtin.h"
#include "common.h"
#include "object.h"
#include "functions.h"
#include "file.h"
#include "builtin.h"
#include "matrix.h"
#include "sparse.h"
#include "mesh.h"
#include "selection.h"
#include "functional.h"
#include "veneer.h"

/* **********************************************************************
 * Global data
 * ********************************************************************** */

/** A table of built in functions */
static dictionary builtin_functiontable;

/** A table of built in classes */
static dictionary builtin_classtable;

/** A table of symbols used by built in classes */
static dictionary builtin_symboltable;

/** Core object types can be provided with a 'veneer' class enabling the user to call methods
    on it, e.g. <string>.length(). This list provides easy access. */
objectclass *objectveneer[OBJECT_EXTERN+1];

/* **********************************************************************
 * Utility functions
 * ********************************************************************** */

/** Initialize an objectbuiltinfunction */
static void builtin_init(objectbuiltinfunction *func) {
    func->flags=BUILTIN_FLAGSEMPTY;
    func->function=NULL;
    func->name=MORPHO_NIL;
}

/** @brief Sets the veneer class for a particular object type */
void builtin_setveneerclass(objecttype type, value class) {
    objectveneer[type]=MORPHO_GETCLASS(class);
}

/** @brief Gets the veneer for a particular object type */
inline objectclass *builtin_getveneerclass(objecttype type) {
    return objectveneer[type];
}

/* **********************************************************************
 * Optional arguments
 * ********************************************************************** */

/** Process optional arguments */
bool builtin_options(vm *v, int nargs, value *args, int *nfixed, int noptions, ...) {
    va_list optlist;
    va_start(optlist, noptions);
    int np=nargs;
    
    for (unsigned int i=0; i<noptions; i++) {
        value symbol = va_arg(optlist, value);
        value *dest = va_arg(optlist, value*);
        for (unsigned int k=0; 2*k<nargs && k<noptions; k+=1) {
            if (MORPHO_ISSAME(symbol, args[nargs-2*k-1])) {
                *dest = args[nargs-2*k];
                if (nargs-2*k-2<np) np=nargs-2*k-2;
            }
        }
        // Should raise an error for unexpected options here by looking for arguments that are strings and unmanaged?
    }
    if (nfixed) *nfixed = np;
    
    va_end(optlist);
    
    return false;
}

/** Tests whether an object is callable */
bool builtin_iscallable(value val) {
    return (MORPHO_ISOBJECT(val) && (MORPHO_ISFUNCTION(val) ||
                                     MORPHO_ISCLOSURE(val) ||
                                     MORPHO_ISINVOCATION(val) ||
                                     MORPHO_ISBUILTINFUNCTION(val)));
}

/* **********************************************************************
 * Create and find builtin functions
 * ********************************************************************** */

/** Add a builtin function.
 * @param name  name of the function
 * @param func  the corresponding C function
 * @param flags flags to define the function
 * @returns value referring to the objectbuiltinfunction */
value builtin_addfunction(char *name, builtinfunction func, builtinfunctionflags flags) {
    objectbuiltinfunction *new = (objectbuiltinfunction *) object_new(sizeof(objectbuiltinfunction), OBJECT_BUILTINFUNCTION);
    value out = MORPHO_NIL;
    
    if (new) {
        builtin_init(new);
        new->function=func;
        new->name=object_stringfromcstring(name, strlen(name));
        new->flags=flags;
        out = MORPHO_OBJECT(new);
        
        if (dictionary_get(&builtin_functiontable, new->name, NULL)) {
            UNREACHABLE("redefinition of builtin function (check builtin.c)");
        }
        
        dictionary_insert(&builtin_functiontable, new->name, out);
    }
    
    return out;
}

/** Finds a builtin function from its name */
value builtin_findfunction(value name) {
    value out=MORPHO_NIL;
    dictionary_get(&builtin_functiontable, name, &out);
    return out;
}

/** Prints a builtin function */
void builtin_printfunction(objectbuiltinfunction *f) {
    if (f) printf("<fn %s>", (MORPHO_ISNIL(f->name) ? "" : MORPHO_GETCSTRING(f->name)));
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
    objectclass *new = object_newclass(label);
    
    if (!new) return MORPHO_NIL;
    
    for (unsigned int i=0; desc[i].name!=NULL; i++) {
        if (desc[i].type==BUILTIN_METHOD) {
            value selector = object_stringfromcstring(desc[i].name, strlen(desc[i].name));
            objectbuiltinfunction *method = (objectbuiltinfunction *) object_new(sizeof(objectbuiltinfunction), OBJECT_BUILTINFUNCTION);
            builtin_init(method);
            method->function=desc[i].function;
            method->name=dictionary_intern(&builtin_symboltable, selector);
            method->flags=desc[i].flags;
            
            /* If interning the symbol changed it, we should free the selector */
            if (!MORPHO_ISSAME(selector, method->name)) object_free(MORPHO_GETOBJECT(selector));
            
            if (dictionary_get(&new->methods, method->name, NULL)) {
                UNREACHABLE("redefinition of method in builtin class (check builtin.c)");
            }
            
            dictionary_insert(&new->methods, method->name, MORPHO_OBJECT(method));
        }
    }
    
    if (dictionary_get(&builtin_classtable, label, NULL)) {
        UNREACHABLE("redefinition of builtin class (check builtin.c)");
    }
    
    dictionary_insert(&builtin_classtable, label, MORPHO_OBJECT(new));
    
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

/** Interns a symbo given as a C string. */
value builtin_internsymbolascstring(char *symbol) {
    value selector = object_stringfromcstring(symbol, strlen(symbol));
    value internselector = builtin_internsymbol(selector);
    if (!MORPHO_ISSAME(selector, internselector)) object_free(MORPHO_GETOBJECT(selector));
    return internselector;
}

/* **********************************************************************
 * Initialization/Finalization
 * ********************************************************************** */

void builtin_initialize(void) {
    for (objecttype i=0; i<OBJECT_EXTERN+1; i++) {
        objectveneer[i]=NULL;
    }
    
    dictionary_init(&builtin_functiontable);
    dictionary_init(&builtin_classtable);
    dictionary_init(&builtin_symboltable);
    
    functions_initialize();
    
    /* Initialize builtin classes and functions */
    file_initialize();
    matrix_initialize();
    sparse_initialize();
    mesh_initialize();
    selection_initialize();
    functional_initialize();
    
    veneer_initialize(); 
}

void builtin_finalize(void) {
    dictionary_freecontents(&builtin_functiontable, false, true);
    dictionary_freecontents(&builtin_classtable, false, true);
    dictionary_clear(&builtin_functiontable);
    dictionary_clear(&builtin_classtable);
    dictionary_clear(&builtin_symboltable);
    
    file_finalize();
}
