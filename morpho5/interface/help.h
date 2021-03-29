/** @file help.h
 *  @author T J Atherton
 *
 *  @brief Interactive help system
*/

#ifndef help_h
#define help_h

#include <stdio.h>

#include "morpho.h"
#include "object.h"
#include "linedit.h"

typedef struct sobjecthelptopic {
    object obj;
    char *topic; // Topic name
    char *file; // File
    long int location; // Location in file
    struct sobjecthelptopic *parent; // Parent topic
    struct sobjecthelptopic *next; // Next topic (global linked list)
    dictionary subtopics; // Subtopics
} objecthelptopic;

#define HELP_INDEXPAGE "help" 

size_t help_querylength(char *query, char **s);
objecthelptopic *help_search(char *query);
void help_display(lineditor *edit, objecthelptopic *topic);

bool help_initialize(void);
void help_finalize(void); 

#endif /* help_h */
