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
#include "strng.h"
#include "resources.h"
#include "extensions.h"

typedef struct {
    value name;
    void *handle;
} extension;

DECLARE_VARRAY(extension, extension)
DEFINE_VARRAY(extension, extension)

#define MORPHO_EXTENSIONINITIALIZE "initialize" // Function to call upon initialization
#define MORPHO_EXTENSIONFINALIZE "finalize"     // Function to call upon finalization

varray_extension extensions;

/** Trys to locate a function with NAME_FN in extension e, and calls it if found */
bool extensions_call(extension *e, char *name, char *fn) {
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
    value out = MORPHO_NIL;
    
    if (!morpho_findresource(MORPHO_EXTENSIONSDIR, name, ext, true, &out)) return false;
    
    extension e;
    e.handle = dlopen(MORPHO_GETCSTRING(out), RTLD_LAZY);
    morpho_freeobject(out);
    
    if (e.handle) {
        e.name = object_stringfromcstring(name, strlen(name));
        varray_extensionwrite(&extensions, e);
        
        if (!extensions_call(&e, name, MORPHO_EXTENSIONINITIALIZE)) {
            dlclose(e.handle);
            return false; // Check extension initialized correctly.
        }
    }
    
    return e.handle;
}

void extensions_initialize(void) {
    varray_extensioninit(&extensions);
}

void extensions_finalize(void) {
    for (int i=0; i<extensions.count; i++) {
        /* Finalize and close each extension */
        value name = extensions.data[i].name;
        extensions_call(&extensions.data[i], MORPHO_GETCSTRING(name), MORPHO_EXTENSIONFINALIZE);
        morpho_freeobject(name);
        dlclose(extensions.data[i].handle);
    }
    varray_extensionclear(&extensions);
}

