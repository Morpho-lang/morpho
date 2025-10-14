/** @file help.c
 *  @author T J Atherton
 *
 *  @brief Morpho help
*/

#include <ctype.h>
#include <stdio.h>
#include <stdbool.h>
#include <varray.h>

#include "classes.h"
#include "help.h"
#include "resources.h"

/** The interactive help system uses a collection of Markdown files, located in
 *  MORPHO_HELPFOLDER, that define available topics. Help files are all
 *  valid Markdown, although only a subset is used, and the help system interprets
 *  Markdown syntax in special ways:
 *
 *  Headers defined with #, ##, etc are used to identify discrete topics.
 *  Successive levels of header are used to create subtopics.
 *
 *  Link definitions are used to include metadata:
 *
 *  [tag]: # (<TAG>)      is used to define additional synonyms for the topic.
 *
 *  The help system also recognizes code blocks etc. */


/* **********************************************************************
 * Morpho help files
 * ********************************************************************** */

/** Loads a help file
 *  @param file     file to load
 *  @returns true if any help entries were successfully loaded */
bool help_load(char *file) {
    
}

/** Searches for help files
 *  @returns true if any help files were successfully processed. */
void help_findfiles(void) {
    varray_value files;
    varray_valueinit(&files);
    
    if (morpho_listresources(MORPHO_RESOURCE_HELP, &files)) {
        for (int i=0; i<files.count; i++) {
            if (MORPHO_ISSTRING(files.data[i])) help_load(MORPHO_GETCSTRING(files.data[i]));
        }
    }
    
    varray_valueclear(&files);
}

/* **********************************************************************
 * Morpho help files
 * ********************************************************************** */

/** Interface to the morpho help system */
bool morpho_help(char *query, varray_char *result) {
    return false;
}

/* **********************************************************************
 * Initialization/finalization
 * ********************************************************************** */

/** @brief Initialization/finalization */
void help_initialize(void) {
    help_findfiles();
    
    morpho_addfinalizefn(help_finalize);
}

/** @brief Initialization/finalization */
void help_finalize(void) {
}
