/** @file resources.c
 *  @author T J Atherton
 *
 *  @brief Locates resources from installation folders and packages
 */

#include <stdio.h>

#include "common.h"
#include "resources.h"
#include "platform.h"
#include "file.h"

/* **********************************************************************
 * Resource enumerator structure
 * ********************************************************************** */

/** A resource enumerator contains state information to enable the resources system to recursively search various resource locations (e.g. /usr/local/share/morpho/ ) for a specified query. You initialize the  resourceenumerator with a query,
    then call morpho_enumerateresources until no further resources are found, which is indicated by it returning false. */

typedef struct {
    char *folder; // folder specification to scan
    char *fname;  // filename to match
    char **ext;   // list of possible extensions, terminated by an empty string
    bool recurse; // whether to search recursively
    varray_value resources;
} resourceenumerator;

/* **********************************************************************
 * Resource types
 * ********************************************************************** */

/** Map morphoresourcetypes to package subfolders.
   @warning: These must match the order of the morphoresourcetypeenum */
static char *_dir[] = {
    MORPHO_HELPDIR,
    MORPHO_MODULEDIR,
    MORPHO_EXTENSIONDIR
};

/* Map morphoresourcetypes to extensions */
static char *_helpext[] =      { MORPHO_HELPEXTENSION, "" };
static char *_moduleext[] =    { MORPHO_EXTENSION, "" };
static char *_extensionext[] = { MORPHO_DYLIBEXTENSION, "dylib", "so", "" };

static char **_ext[] = { _helpext, _moduleext, _extensionext };

/* Map morphoresourcetypes to base folders */
static char *_basedir[] = {
    MORPHO_HELP_BASEDIR,
    MORPHO_MODULE_BASEDIR,
    NULL
};

char *_folderfortype(morphoresourcetype type) { return _dir[type]; }
char **_extfortype(morphoresourcetype type) { return _ext[type]; }
char *_basedirfortype(morphoresourcetype type) { return _basedir[type]; }

/* **********************************************************************
 * Resources
 * ********************************************************************** */

varray_value resourcelocations;

/** Identifies a base folder emanating from path and consistent with resourceenumerator */
void resources_matchbasefolder(resourceenumerator *en, char *path) {
    varray_char fname;
    varray_charinit(&fname);
    varray_charadd(&fname, path, (int) strlen(path));
    varray_charwrite(&fname, MORPHO_DIRSEPARATOR);

    if (en->folder) varray_charadd(&fname, en->folder, (int) strlen(en->folder));
    varray_charwrite(&fname, '\0');

    if (platform_isdirectory(fname.data)) {
        value v = object_stringfromcstring(fname.data, fname.count);
        if (MORPHO_ISSTRING(v)) varray_valuewrite(&en->resources, v);
    }

    varray_charclear(&fname);
}

/** Locates all possible base folders consistent with the current folder specification
 @param[in] en - initialized enumerator */
void resources_basefolders(resourceenumerator *en) {
    for (int i=0; i<resourcelocations.count; i++) { // Loop over possible resource folders
        if (MORPHO_ISSTRING(resourcelocations.data[i])) {
            resources_matchbasefolder(en, MORPHO_GETCSTRING(resourcelocations.data[i]));
        }
    }
}

/** Finds the character at which the extension separator occurs in a filename. Returns NULL if no extension is present */
char *resources_findextension(char *f) {
    for (char *c = f+strlen(f); c>=f; c--) {
        if (*c=='.') return c;
    }
    
    return NULL;
}

/** Checks if a filename matches all criteria in a resourceenumerator
 @param[in] en - initialized enumerator */
bool resources_matchfile(resourceenumerator *en, char *file) {
    // Skip extension
    char *ext = resources_findextension(file);
    if (!ext) ext = file+strlen(file); // If no extension found, just go to the end of the filename

    if (en->fname) { // Match filename if requested
        char *f = ext;
        while (f>=file && *f!=MORPHO_DIRSEPARATOR) f--; // Find last separator
        if (*f==MORPHO_DIRSEPARATOR) f++; // If we stopped at a separator, skip it
        
        size_t len = strlen(en->fname);
        if (strncmp(en->fname, f, len)!=0) return false; // Compare string
        if (!(f[len]=='.' || f[len]=='\0')) return false; // Ensure filename is terminated by null byte or file extension separator.
    }

    if (!en->ext) return true; // Match extension only if requested

    if (*ext!='.') return false;
    for (int k=0; *en->ext[k]!='\0'; k++) { // Check extension against possible extensions
        if (strcmp(ext+1, en->ext[k])==0) return true; // We have a match
    }

    return false;
}

/** Searches a given folder, adding all resources to the enumerator
 @param[in] en - initialized enumerator */
void resources_searchfolder(resourceenumerator *en, char *path) {
    MorphoDirContents contents;
    
    size_t size = platform_maxpathsize();
    char buffer[size];
    char sep[2] = { MORPHO_DIRSEPARATOR, '\0' };
    
    if (platform_directorycontentsinit(&contents, path)) {
        while (platform_directorycontents(&contents, buffer, size)) {
            /* Construct the file name */
            size_t len = strlen(path)+strlen(buffer)+2;
            
            char file[len];
            strcpy(file, path);
            strcat(file, sep);
            strcat(file, buffer);

            if (platform_isdirectory(file)) {
                if (!en->recurse) continue;
            } else {
                if (!resources_matchfile(en, file)) continue;
            }

            /* Add the file or folder to the work list */
            value v = object_stringfromcstring(file, len);
            if (MORPHO_ISSTRING(v)) varray_valuewrite(&en->resources, v);
        }
        platform_directorycontentsclear(&contents);
    }
}

/** Initialize a resource enumerator
 @param[in] en - enumerator to initialize
 @param[in] folder - folder specification to scan
 @param[in] fname - filename to match
 @param[in] ext - list of possible extensions, terminated by an empty string
 @param[in] recurse - search recursively */
void resourceenumerator_init(resourceenumerator *en, char *folder, char *fname, char *ext[], bool recurse) {
    en->folder = folder;
    en->fname = fname;
    en->ext = ext;
    en->recurse = recurse;
    varray_valueinit(&en->resources);
    resources_basefolders(en);
}

/** Clears a resource enumerator
 @param[in] en - enumerator to clear */
void resourceenumerator_clear(resourceenumerator *en) {
    for (int i=0; i<en->resources.count; i++) morpho_freeobject(en->resources.data[i]);
    varray_valueclear(&en->resources);
}

/** Enumerates resources
 @param[in] en - enumerator to use
 @param[out] out - next resource */
bool resourceenumerator_enumerate(resourceenumerator *en, value *out) {
    if (en->resources.count==0) return false;
    value next = en->resources.data[--en->resources.count];

    while (platform_isdirectory(MORPHO_GETCSTRING(next))) {
        resources_searchfolder(en, MORPHO_GETCSTRING(next));
        morpho_freeobject(next);
        if (en->resources.count==0) return false;
        next = en->resources.data[--en->resources.count];
    }

    *out = next;
    return true;
}

/** Adds the default folder for a given resource type */
void resourceenumerator_defaultfolder(resourceenumerator *en, morphoresourcetype type) {
    char *basedir = _basedirfortype(type);
    if (basedir) {
        value v = object_stringfromcstring(basedir, strlen(basedir));
        if (MORPHO_ISSTRING(v)) varray_valuewrite(&en->resources, v);
    }
}

/** Locates a resource
 @param[in] type - type of resource to locate
 @param[in] fname - filename to match
 @param[out] out - an objectstring that contains the resource file location */
bool morpho_findresource(morphoresourcetype type, char *fname, value *out) {
    char *folder = _folderfortype(type);
    char **ext = _extfortype(type);
    
    bool success=false;
    resourceenumerator en;
    resourceenumerator_init(&en, folder, fname, ext, true);
    resourceenumerator_defaultfolder(&en, type);
    success=resourceenumerator_enumerate(&en, out);
    resourceenumerator_clear(&en);
    return success;
}

/** Locates all resources of a given type
 @param[in] type - type of resource to locate
 @param[out] out - a varray_value that contains the resource file locations */
bool morpho_listresources(morphoresourcetype type, varray_value *out) {
    char *folder = _folderfortype(type);
    char **ext = _extfortype(type);
    
    resourceenumerator en;
    resourceenumerator_init(&en, folder, NULL, ext, true);
    resourceenumerator_defaultfolder(&en, type);
    value file;
    while (resourceenumerator_enumerate(&en, &file)) {
        varray_valuewrite(out, file);
    }
    resourceenumerator_clear(&en);
    return (out->count>0);
}

/** Loads a list of packages in ~/.morphopackages */
void resources_loadpackagelist(void) {
    varray_char line;
    varray_charinit(&line);

    size_t len = platform_maxpathsize();
    char home[len];
    if (platform_gethomedirectory(home, len)) {
        varray_charadd(&line, home, (int) strlen(home));
    }

    varray_charwrite(&line, MORPHO_DIRSEPARATOR);
    varray_charadd(&line, MORPHO_PACKAGELIST, (int) strlen(MORPHO_PACKAGELIST));
    varray_charwrite(&line, '\0');

    FILE *f = fopen(line.data, "r");
    if (f) {
        while (!feof(f)) {
            line.count=0;
            if (file_readlineintovarray(f, &line) &&
                line.count>0) {
                value str = object_stringfromvarraychar(&line);
                varray_valuewrite(&resourcelocations, str);
            }
        }
        fclose(f);
    }
    varray_charclear(&line);
}

void resources_initialize(void) {
    varray_valueinit(&resourcelocations);

    resources_loadpackagelist();
    
    morpho_addfinalizefn(resources_finalize);
}

void resources_finalize(void) {
    for (int i=0; i<resourcelocations.count; i++) morpho_freeobject(resourcelocations.data[i]);
    varray_valueclear(&resourcelocations);
}
