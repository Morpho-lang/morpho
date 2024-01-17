/** @file extensions.c
 *  @author T J Atherton
 *
 *  @brief Morpho extensions
 */

/* **********************************************************************
* Extensions
* ********************************************************************** */

#include <dlfcn.h>
#include <string.h>
#include "varray.h"
#include "value.h"
#include "object.h"
#include "builtin.h"
#include "strng.h"
#include "dict.h"
#include "resources.h"
#include "extensions.h"

/* -------------------------------------------------------
 * Extension structure
 * ------------------------------------------------------- */

typedef struct {
    value name;
    value path;
    value functiontable;
    value classtable;
    void *handle;
} extension;

DECLARE_VARRAY(extension, extension)
DEFINE_VARRAY(extension, extension)

varray_extension extensionlist; // List of loaded extensions

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
    
    if (e->handle) dlclose(e->handle);
    extension_init(e);
}

/* -------------------------------------------------------
 * Extensions interface
 * ------------------------------------------------------- */

/** Trys to locate a function with NAME_FN in extension e, and calls it if found */
bool extension_call(extension *e, char *name, char *fn) {
    void (*fptr) (void);
    char fnname[strlen(name)+strlen(fn)+2];
    strcpy(fnname, name);
    strcat(fnname, "_");
    strcat(fnname, fn);
    
    fptr = dlsym(e->handle, fnname);
    if (fptr) (*fptr) ();
    return fptr;
}

/** Attempts to load an extension with given name. Returns true if it was found and loaded successfully */
bool morpho_loadextension(char *name) {
    char *ext[] = { MORPHO_DYLIBEXTENSION, "dylib", "so", "" };
    extension e;
    extension_init(&e);
    
    if (!morpho_findresource(MORPHO_EXTENSIONSDIR, name, ext, true, &e.path)) return false;
    
    e.handle = dlopen(MORPHO_GETCSTRING(e.path), RTLD_LAZY);
    e.name = object_stringfromcstring(name, strlen(name));
    
    objectdictionary *functiontable = object_newdictionary(),
                     *classtable = object_newdictionary();
    if (functiontable) e.functiontable=MORPHO_OBJECT(functiontable);
    if (classtable) e.classtable=MORPHO_OBJECT(classtable);
    
    if (!e.handle || !MORPHO_ISOBJECT(e.name) || !functiontable || !classtable) goto morpho_loadextension_cleanup;
    
    varray_extensionwrite(&extensionlist, e);
    
    dictionary *ofunc=builtin_getfunctiontable(),
               *oclss=builtin_getclasstable();
    
    builtin_setfunctiontable(&functiontable->dict);
    builtin_setfunctiontable(&classtable->dict);
    
    if (!extension_call(&e, name, MORPHO_EXTENSIONINITIALIZE)) {
        varray_extensionpop(&extensionlist, NULL);
        goto morpho_loadextension_cleanup;
    }
    
    builtin_setfunctiontable(ofunc);
    builtin_setclasstable(oclss);
    
    return true;
    
morpho_loadextension_cleanup:
    extension_clear(&e);
    return false;
}

/* -------------------------------------------------------
 * Extensions initialization/finalization
 * ------------------------------------------------------- */

void extensions_initialize(void) {
    varray_extensioninit(&extensionlist);
}

void extensions_finalize(void) {
    for (int i=0; i<extensionlist.count; i++) {
        extension *e = &extensionlist.data[i];
        // Call finalizer
        extension_call(e, MORPHO_GETCSTRING(e->name), MORPHO_EXTENSIONFINALIZE);
        extension_clear(e);
    }
    varray_extensionclear(&extensionlist);
}
