/** @file resources.h
 *  @author T J Atherton
 *
 *  @brief Locates resources from installation folders and packages
 */

/** Morpho depends on various supporting files, including help files, modules, extensions, etc. The resources system provides a simple interface to locate these. */

#ifndef resources_h
#define resources_h

#include "value.h"

/* -----------------------------------------
 * Resource types
 * ----------------------------------------- */

typedef enum {
    MORPHO_RESOURCE_HELP,
    MORPHO_RESOURCE_MODULE,
    MORPHO_RESOURCE_EXTENSION
} morphoresourcetype;

/* -----------------------------------------
 * Resource enumerator structure
 * ----------------------------------------- */

/** A resource enumerator contains state information to enable the resources system to recursively search various resource locations (e.g. /usr/local/share/morpho/ ) for a specified query. You initialize the  resourceenumerator with a query,
    then call morpho_enumerateresources until no further resources are found, which is indicated by it returning false. */

typedef struct {
    char *folder; // folder specification to scan
    char *fname;  // filename to match
    char **ext;   // list of possible extensions, terminated by an empty string
    bool recurse; // whether to search recursively
    varray_value resources;
} resourceenumerator;

void morpho_resourceenumeratorinit(resourceenumerator *en, char *folder, char *fname, char *ext[], bool recurse);
void morpho_resourceenumeratorclear(resourceenumerator *en);

void morpho_defaultfolder(resourceenumerator *en, morphoresourcetype type);

/** Locate resources given a current query; returns true if one was found */
bool morpho_enumerateresources(resourceenumerator *en, value *out);

/* -----------------------------------------
 * Resources interface
 * ----------------------------------------- */

/** Interface to locate a specified resource
 @param[in] type - type of resource to find
 @param[in] fname - filename to match
 @param[out] out - an objectstring that contains the resource file location
 @warning: You must free the objectstring after use.*/
bool morpho_findresource(morphoresourcetype type, char *fname, value *out);

void resources_initialize(void);
void resources_finalize(void);

#endif /* resources_h */
