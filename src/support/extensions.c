/** @file extensions.c
 *  @author T J Atherton
 *
 *  @brief Morpho extensions
 */

/* **********************************************************************
* Extensions
* ********************************************************************** */

#include <string.h>

#include "varray.h"
#include "value.h"
#include "common.h"
#include "object.h"
#include "builtin.h"
#include "resources.h"
#include "extensions.h"
#include "platform.h"

/* -------------------------------------------------------
 * Extension structure
 * ------------------------------------------------------- */

typedef struct {
    value name;
    value path;
    value functiontable;
    value classtable;
    MorphoDLHandle handle;
} extension;

DECLARE_VARRAY(extension, extension)
DEFINE_VARRAY(extension, extension)

varray_extension extensionlist; // List of loaded extensions

/* -------------------------------------------------------
 * Extension interface
 * ------------------------------------------------------- */

/** Open the dynamic library associated with an extension */
bool extension_dlopen(extension *e) {
    if (e->handle) return true; // Prevent multiple loads
    if (MORPHO_ISSTRING(e->path)) e->handle=platform_dlopen(MORPHO_GETCSTRING(e->path));
    return e->handle;
}

/** Close the dynamic library associated with an extension */
void extension_dlclose(extension *e) {
    if (e->handle) platform_dlclose(e->handle);
    e->handle=NULL;
}

/** Initializes an extension structure with empty values */
void extension_init(extension *e) {
    e->name=MORPHO_NIL;
    e->path=MORPHO_NIL;
    e->functiontable=MORPHO_NIL;
    e->classtable=MORPHO_NIL;
    e->handle=NULL;
}

/** Clears an extension structure */
void extension_clear(extension *e) {
    if (MORPHO_ISOBJECT(e->name)) morpho_freeobject(e->name);
    if (MORPHO_ISOBJECT(e->path)) morpho_freeobject(e->path);
    
    // The functions and classes are freed from the builtin_objects list. 
    if (MORPHO_ISOBJECT(e->functiontable)) morpho_freeobject(e->functiontable);
    if (MORPHO_ISOBJECT(e->classtable)) morpho_freeobject(e->classtable);
    
    extension_dlclose(e);
    extension_init(e);
}

/** Initializes an extension structure with a name and path, creating associated data structures */
bool extension_initwithname(extension *e, char *name, char *path) {
    extension_init(e);
    e->name=object_stringfromcstring(name, strlen(name));
    e->path=object_stringfromcstring(path, strlen(path));
    
    objectdictionary *functiontable = object_newdictionary(),
                     *classtable = object_newdictionary();
    
    if (functiontable) e->functiontable=MORPHO_OBJECT(functiontable);
    if (classtable) e->classtable=MORPHO_OBJECT(classtable);
    e->handle=NULL;
    
    if (!MORPHO_ISSTRING(e->name) ||
        !MORPHO_ISSTRING(e->path) ||
        !MORPHO_ISDICTIONARY(e->functiontable) ||
        !MORPHO_ISDICTIONARY(e->classtable)) {
        extension_clear(e);
        return false;
    }
    return true;
}

/** Trys to locate a function with NAME_FN in extension e, and calls it if found */
bool extension_call(extension *e, const char *name, const char *fn) {
    void (*fptr) (void);
    size_t size = strlen(name) + strlen(fn) + 2;
    char fnname[size];
    strncpy(fnname, name, size);
    strncat(fnname, "_", size);
    strncat(fnname, fn, size);
    
    fptr = platform_dlsym(e->handle, fnname);
    if (fptr) (*fptr) ();
    return fptr;
}

/** Finds the path for an extension using the resource finder */
bool extension_find(char *name, value *path) {
    return morpho_findresource(MORPHO_RESOURCE_EXTENSION, name, path);
}

/** Checks if an extension is already loaded;  returns it in out if found */
bool extension_isloaded(value path, extension *out) {
    for (int i=0; i<extensionlist.count; i++) {
        extension *e = &extensionlist.data[i];
        
        if (MORPHO_ISEQUAL(path, e->path)) {
            if (out) *out = *e;
            return true;
        }
    }
    return false;
}

/** Call the extension's initializer */
bool extension_initialize(extension *e) {
    dictionary *ofunc=builtin_getfunctiontable(),
               *oclss=builtin_getclasstable();
    
    builtin_setfunctiontable(MORPHO_GETDICTIONARYSTRUCT(e->functiontable));
    builtin_setclasstable(MORPHO_GETDICTIONARYSTRUCT(e->classtable));
    
    bool success=extension_call(e, MORPHO_GETCSTRING(e->name), MORPHO_EXTENSIONINITIALIZE);
    
    builtin_setfunctiontable(ofunc);
    builtin_setclasstable(oclss);
    
    return success; 
}

/** Call the extension's finalizer */
bool extension_finalize(extension *e) {
    return extension_call(e, MORPHO_GETCSTRING(e->name), MORPHO_EXTENSIONFINALIZE);
}

/** Load an extension, optionally returning a dictionary of functions and classes defined by the extension */
bool extension_load(char *name, dictionary **functiontable, dictionary **classtable) {
    value path;
    if (!extension_find(name, &path)) return false;
    
    bool success=false;
    
    extension e;
    extension_init(&e);
    
    if (extension_isloaded(path, &e)) {
        success=true;
    } else if (extension_initwithname(&e, name, MORPHO_GETCSTRING(path)) &&
               extension_dlopen(&e)) {
        success=extension_initialize(&e);
        if (success) varray_extensionwrite(&extensionlist, e);
    }
    
    if (success) {
        if (functiontable) *functiontable = MORPHO_GETDICTIONARYSTRUCT(e.functiontable);
        if (classtable) *classtable = MORPHO_GETDICTIONARYSTRUCT(e.classtable);
    } else extension_clear(&e);
    
    morpho_freeobject(path);
    
    return success;
}

/* -------------------------------------------------------
 * Extensions initialization/finalization
 * ------------------------------------------------------- */

void extensions_initialize(void) {
    varray_extensioninit(&extensionlist);
 
    morpho_addfinalizefn(extensions_finalize);
}

void extensions_finalize(void) {
    for (int i=0; i<extensionlist.count; i++) {
        extension *e = &extensionlist.data[i];
        extension_finalize(e);
        extension_clear(e);
    }
    varray_extensionclear(&extensionlist);
}
