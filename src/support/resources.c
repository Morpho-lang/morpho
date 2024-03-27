/** @file resources.c
 *  @author T J Atherton
 *
 *  @brief Locates resources from installation folders and packages
 */

#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>

#include "common.h"
#include "resources.h"
#include "file.h"

/* **********************************************************************
* Resources
* ********************************************************************** */

varray_value resourcelocations;

/** Identifies a base folder emanating from path and consistent with resourceenumerator */
void resources_matchbasefolder(resourceenumerator *en, char *path) {
    varray_char fname;
    varray_charinit(&fname);
    varray_charadd(&fname, path, (int) strlen(path));
    varray_charwrite(&fname, MORPHO_SEPARATOR);

    if (en->folder) {
        int i=0;
        for (; en->folder[i]!='\0' && en->folder[i]!=MORPHO_SEPARATOR; i++) varray_charwrite(&fname, en->folder[i]);

        int nfldr=fname.count;
        varray_charwrite(&fname, MORPHO_SEPARATOR);
        varray_charadd(&fname, MORPHO_MORPHOSUBDIR, strlen(MORPHO_MORPHOSUBDIR));
        varray_charwrite(&fname, '\0');
        if (morpho_isdirectory(fname.data)) {
            fname.count--;
        } else fname.count=nfldr;

        for (; en->folder[i]!='\0'; i++) varray_charwrite(&fname, en->folder[i]);
    }
    varray_charwrite(&fname, '\0');

    if (morpho_isdirectory(fname.data)) {
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
        while (f>=file && *f!=MORPHO_SEPARATOR) f--; // Find last separator
        if (*f==MORPHO_SEPARATOR) f++; // If we stopped at a separator, skip it
        
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
    DIR *d; /* Handle for the directory */
    struct dirent *entry; /* Entries in the directory */
    d = opendir(path);

    if (d) {
        while ((entry = readdir(d)) != NULL) { // Loop over directory entries
            if (strcmp(entry->d_name, ".")==0 ||
                strcmp(entry->d_name, "..")==0) continue;

            /* Construct the file name */
            size_t len = strlen(path)+strlen(entry->d_name)+2;
            char file[len];
            strcpy(file, path);
            strcat(file, "/");
            strcat(file, entry->d_name);

            if (morpho_isdirectory(file)) {
                if (!en->recurse) continue;
            } else {
                if (!resources_matchfile(en, file)) continue;
            }

            /* Add the file or folder to the work list */
            value v = object_stringfromcstring(file, len);
            if (MORPHO_ISSTRING(v)) varray_valuewrite(&en->resources, v);
        }
        closedir(d);
    }
}

/** Initialize a resource enumerator
 @param[in] en - enumerator to initialize
 @param[in] folder - folder specification to scan
 @param[in] fname - filename to match
 @param[in] ext - list of possible extensions, terminated by an empty string
 @param[in] recurse - search recursively */
void morpho_resourceenumeratorinit(resourceenumerator *en, char *folder, char *fname, char *ext[], bool recurse) {
    en->folder = folder;
    en->fname = fname;
    en->ext = ext;
    en->recurse = recurse;
    varray_valueinit(&en->resources);
    resources_basefolders(en);
}

/** Clears a resource enumerator
 @param[in] en - enumerator to clear */
void morpho_resourceenumeratorclear(resourceenumerator *en) {
    for (int i=0; i<en->resources.count; i++) morpho_freeobject(en->resources.data[i]);
    varray_valueclear(&en->resources);
}

/** Enumerates resources
 @param[in] en - enumerator to use
 @param[out] out - next resource */
bool morpho_enumerateresources(resourceenumerator *en, value *out) {
    if (en->resources.count==0) return false;
    value next = en->resources.data[--en->resources.count];

    while (morpho_isdirectory(MORPHO_GETCSTRING(next))) {
        resources_searchfolder(en, MORPHO_GETCSTRING(next));
        morpho_freeobject(next);
        if (en->resources.count==0) return false;
        next = en->resources.data[--en->resources.count];
    }

    *out = next;
    return true;
}

/** Locates a resource
 @param[in] folder - folder specification to scan
 @param[in] fname - filename to match
 @param[in] ext - list of possible extensions, terminated by an empty string
 @param[in] recurse - search recursively
 @param[out] out - an objectstring that contains the resource file location */
bool morpho_findresource(char *folder, char *fname, char *ext[], bool recurse, value *out) {
    bool success=false;
    resourceenumerator en;
    morpho_resourceenumeratorinit(&en, folder, fname, ext, recurse);
    success=morpho_enumerateresources(&en, out);
    morpho_resourceenumeratorclear(&en);
    return success;
}

/** Loads a list of packages in ~/.morphopackages */
void resources_loadpackagelist(void) {
    varray_char line;
    varray_charinit(&line);

    char *home = getenv("HOME");
    varray_charadd(&line, home, (int) strlen(home));
    varray_charwrite(&line, MORPHO_SEPARATOR);
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
    value v = object_stringfromcstring(MORPHO_RESOURCESDIR, strlen(MORPHO_RESOURCESDIR));
    varray_valuewrite(&resourcelocations, v);

    resources_loadpackagelist();
    
    morpho_addfinalizefn(resources_finalize);
}

void resources_finalize(void) {
    for (int i=0; i<resourcelocations.count; i++) morpho_freeobject(resourcelocations.data[i]);
    varray_valueclear(&resourcelocations);
}
