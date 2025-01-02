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
 * Resources interface
 * ----------------------------------------- */

/** Interface to locate a specified resource
 @param[in] type - type of resource to find
 @param[in] fname - filename to match
 @param[out] out - an objectstring that contains the resource file location
 @warning: You must free the objectstring after use.*/
bool morpho_findresource(morphoresourcetype type, char *fname, value *out);

/** Interface to locate all resources of a given type
 @param[in] type - type of resource to find
 @param[out] out - a varray_value that contains resource files
 @warning: You must free the contents of the varray_value after use.*/
bool morpho_listresources(morphoresourcetype type, varray_value *out);

void resources_initialize(void);
void resources_finalize(void);

#endif /* resources_h */
