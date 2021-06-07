/** @file help.c
 *  @author T J Atherton
 *
 *  @brief Interactive help system
*/

#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include "help.h"
#include "dictionary.h"
#include "parse.h"
#include "common.h"
#include "veneer.h"

/** The interactive help system uses a collection of Markdown files, located in
 *  MORPHO_HELPDIRECTORY, that define available topics. Help files are all
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
 *  The help system also recognizes code blocks etc.
 */

static dictionary helpdict;
static char *helpdir = MORPHO_HELPDIRECTORY;
static objecthelptopic *topics = NULL;

/* **********************************************************************
 * Help topics
 * ********************************************************************** */

/** Create a new help topic */
objecthelptopic *help_newtopic(char *topic, char *file, long int location, objecthelptopic *parent) {
    objecthelptopic *new = (objecthelptopic *) object_new(sizeof(objecthelptopic), OBJECT_EXTERN);
    
    if (new) {
        new->topic=morpho_strdup(topic);
        new->file=morpho_strdup(file);
        new->location=location;
        new->parent=parent;
        dictionary_init(&new->subtopics);
        /* Link into the list */
        new->next=topics;
        topics=new;
    }
    
    return new;
}

/** Free attached data from a help topic */
void help_cleartopic (objecthelptopic *topic) {
    if (topic) {
        dictionary_clear(&topic->subtopics);
        free(topic->file); topic->file=NULL;
        free(topic->topic); topic->topic=NULL;
    }
}

/* **********************************************************************
 * Search
 * ********************************************************************** */

#define HELP_LINELENGTH 2048

/** Determine starting point and length of a query
 * @param[in] query - the query to examine
 * @param[out] s - starting character of the query (optional)
 * @returns length of the query (0 indicates no query present) */
size_t help_querylength(char *query, char **s) {
    char *start = query;
    while (isspace(*start) || ispunct(*start)) start++;
    size_t length = 0;
    while (!iscntrl(start[length]) && !isspace(start[length]) && !ispunct(start[length])) length++;
    if (s) *s = start;
    return length;
}

/** Searches for a given query in the help system. If recurse is true, searches child dictionaries if nothing found here */
objecthelptopic *help_query(dictionary *dict, char *query, bool recurse) {
    objecthelptopic *topic = NULL;
    char *p;
    size_t length = help_querylength(query, &p);
    
    if (length>0) {
        /* Convert query to lower case */
        char q[length+1];
        for (unsigned int i=0; i<length; i++) q[i]=tolower(p[i]);
        
        objectstring key = MORPHO_STATICSTRINGWITHLENGTH(q, length);
        value result;
        
        if (dictionary_get(dict, MORPHO_OBJECT(&key), &result)) {
            if (MORPHO_ISOBJECT(result)) {
                topic = (objecthelptopic *) MORPHO_GETOBJECT(result);
            }
        } else if (recurse) {
            for (unsigned int i=0; !topic && i<dict->capacity; i++) {
                if (MORPHO_ISOBJECT(dict->contents[i].key)) {
                    objecthelptopic *child = (objecthelptopic *) MORPHO_GETOBJECT(dict->contents[i].val);
                    
                    topic = help_query(&child->subtopics, query, true);
                }
            }
        }
    }
    
    return topic;
}

/** Searches for a given query in the help system */
objecthelptopic *help_search(char *query) {
    objecthelptopic *topic = NULL;
    dictionary *dict = &helpdict;
    char *p;
    size_t length = help_querylength(query, &p);
    
    while (length>0) {
        topic=help_query(dict, p, true);
        length=help_querylength(p+length, &p);
        if (topic) dict=&topic->subtopics;
    }
    
    return topic;
}

/* **********************************************************************
 * Display help
 * ********************************************************************** */

/** Display a topic list */
void help_topiclist(dictionary *dict, lineditor *edit) {
    int width = linedit_getwidth(edit), max = 0;
    
    objectlist list = MORPHO_STATICLIST;
    varray_valueinit(&list.val);
    
    varray_char str;
    varray_charinit(&str);
    
    /* We search through the list of topics and find those that are in the dictionary */
    for (objecthelptopic *topic=topics; topic!=NULL; topic=topic->next) {
        objectstring key = MORPHO_STATICSTRING(topic->topic);
        
        if (dictionary_get(dict, MORPHO_OBJECT(&key), NULL)) {
            value str = dictionary_intern(dict, MORPHO_OBJECT(&key));
            varray_valuewrite(&list.val, str);
        }
    }
    
    list_sort(&list);
    
    for (unsigned int i=0; i<list.val.count; i++) {
        if (MORPHO_ISSTRING(list.val.data[i])) {
            int len = (int) MORPHO_GETSTRINGLENGTH(list.val.data[i]);
            if (len>max) max=len;
        }
    }
    int ncols = width/(max), k=0;
    bool single = list.val.count<ncols;
    
    for (unsigned int i=0; i<list.val.count; i++) {
        if (MORPHO_ISSTRING(list.val.data[i])) {
            objectstring *s = MORPHO_GETSTRING(list.val.data[i]);
            if (!isalpha(s->string[0])) continue;
            varray_charadd(&str, s->string, (int) s->length);
            if (single) {
                varray_charadd(&str, "  ", 2);
            } else {
                for (int k=(int) s->length; k<max+1; k++) varray_charwrite(&str, ' ');
            }
            k++;
        }
        if (k==ncols-1 || i==list.val.count-1) {
            varray_charadd(&str, "\n\0", 2);
            linedit_displaywithsyntaxcoloring(edit, str.data);
            str.count=0; k=0;
        }
    }
    
    varray_valueclear(&list.val);
    varray_charclear(&str);
}

/** Parse a 'show' command */
static void help_show(objecthelptopic *topic, lineditor *edit, char *command) {
    char *c=command;
    for (; isspace(*c); c++);
    if (strncmp(c, "topics", 6)==0) {
        help_topiclist(&helpdict, edit);
    } else if (strncmp(c, "subtopics", 9)==0) {
        help_topiclist(&topic->subtopics, edit);
    }
}

/** Parses a string looking for matching delimiters
 *  @param string   string to parse
 *  @param delim    delimiter character to use
 *  @returns the number of characters enclosed in the delimitor, or 0 if no final delimiter is found
 *  @warning the number of characters does *not* include the final delimitor */
static size_t help_parsesegment(char *string, char delim) {
    char *c=string;
    /* Now search for terminator */
    while (*c!=delim) {
        if (*c=='\0') return 0;
        c++;
    }
    return c-string;
}

/** Parses a section of inline code */
static char *help_parseinlinecode(lineditor *edit, char *string) {
    char *s = string;
    if (*s=='`') s++;
    size_t nchars = help_parsesegment(s, '`');
    if (nchars==0) return string;
    
    char str[nchars+1];
    strncpy(str, string+1, nchars);
    str[nchars]='\0';
    
    linedit_displaywithsyntaxcoloring(edit, str);
    return s + nchars;
}

/** Parses an emphasized section */
static char *help_parseemph(lineditor *edit, char *string, char delim, linedit_color col, linedit_emphasis emph) {
    char *s = string;
    if (*s==delim) s++;
    size_t nchars = help_parsesegment(s, delim);
    if (nchars==0) return string;
    
    char str[nchars+1];
    strncpy(str, string+1, nchars);
    str[nchars]='\0';
    
    linedit_displaywithstyle(edit, str, col, emph);
    return s + nchars;
}

/** Displays a single line of help text */
static bool help_displayline(objecthelptopic *topic, lineditor *edit, char *line, bool allowheader) {
    if (line[0]=='#') {
        if (allowheader) {
            char *s = line;
            while (*s=='#' || isspace(*s)) s++; // Skip leading space
            linedit_displaywithstyle(edit, s, LINEDIT_DEFAULTCOLOR, LINEDIT_UNDERLINE);
        } else {
            return true;
        }
    } else if (line[0]=='\t' || strncmp(line, "    ", 4)==0) {
        linedit_displaywithsyntaxcoloring(edit, line);
    } else if (line[0]=='[') {
        /*  Process commands */
        if (strncmp(line+1, "show", 4)==0) {
            for (char *c = line; *c!='\0'; c++) {
                if (*c=='#') {
                    help_show(topic, edit, c+1);
                    break;
                }
            }
        }
    } else {
        char *c = line;
        if (line[0]=='*') { printf("*"); c++; }
        
        /* Display the line, searching for inline markup */
        for (; *c!='\0'; c++) {
            char *next=NULL;
            switch (*c) {
                case '`':
                    next=help_parseinlinecode(edit, c); break;
                case '*':
                    next=help_parseemph(edit, c, '*', LINEDIT_DEFAULTCOLOR, LINEDIT_BOLD); break;
                case '_':
                    next=help_parseemph(edit, c, '_', LINEDIT_DEFAULTCOLOR, LINEDIT_UNDERLINE); break;
                default:
                    printf("%c", *c); break;
            }
            if (next) {
                /* If the matching delimiter wasn't found, just print the rest of the line and return */
                if (next==c) {
                    printf("%s", c);
                    return false;
                } else {
                    c=next;
                }
            }
        }
    }
    return false;
}

/** Displays a help topic */
void help_display(lineditor *edit, objecthelptopic *topic) {
    FILE *f = (topic ? fopen(topic->file, "r") : NULL);
    char line[HELP_LINELENGTH];
    
    if (f) {
        /* Jump to the topic */
        if (fseek(f, topic->location, SEEK_SET)==0) {
            /* Load and display the help line by line */
            for(unsigned int i=0; !feof(f); i++) {
                if (fgets(line, HELP_LINELENGTH, f)) {
                    if (help_displayline(topic, edit, line, (i==0))) break;
                }
            }
        }
        
        fclose(f);
    }
}

/* **********************************************************************
 * Process and load help files
 * ********************************************************************** */

/** Parses a topic name, returning it as a Morpho string converted to lower case.
 *  @warning the input line is modified in the process */
value help_parsetopicname(char *line) {
    char *start = line;
    size_t length = 0;
    while ((*start=='#' || isspace(*start)) && *start!='\0') start++;
    while (!iscntrl(start[length])) {
        start[length]=tolower(start[length]);
        length++;
    }
    
    return object_stringfromcstring(start, length);
}

/** Parses a tag, returning it as a Morpho string converted to lower case.
 *  @warning the input line is modified in the process */
value help_parsetag(char *line) {
    char *start = line;
    size_t length = 0;
    /* Skip past everything until the hash */
    while (*start!='\0' && *start!='#') start++;
    if (*start=='\0') return MORPHO_NIL;
    start++; /* Skip # */
    /* Now skip everything until the tag */
    while (isspace(*start) && *start!='\0') start++;
    
    /* Skip opening bracket if present */
    if (*start=='(') start++;
    
    while (!iscntrl(start[length]) &&
           !isspace(start[length]) &&
           start[length]!=')' // Skip closing bracket
           ) {
        start[length]=tolower(start[length]);
        length++;
    }
    
    return object_stringfromcstring(start, length);
}

#define HELP_MAXLEVEL 6

/** Determines the level of topic from the markdown header level */
int help_parsetopiclevel(char *line) {
    int level = 0;
    while (line[level]=='#') level++;
    
    return (level>=HELP_MAXLEVEL ? HELP_MAXLEVEL-1 : level-1);
}

/** Loads a help file
 *  @param file     file to load
 *  @returns true if any help entries were successfully loaded */
bool help_load(char *file) {
    char line[HELP_LINELENGTH];
    objecthelptopic *topic[HELP_MAXLEVEL];
    for (unsigned int i=0; i<HELP_MAXLEVEL; i++) topic[i]=NULL;
    int level = 0;
    bool toplevel = false;
    
#ifdef MORPHO_DEBUG_LOGHELPFILES
    printf("Loading help file '%s'\n",file);
#endif
    FILE *f = fopen(file, "r");
    
    if (f) {
        while (!feof(f)) {
            long int cloc = ftell(f); /* Store the current position */
            
            if (fgets(line, HELP_LINELENGTH, f)) {
                if (line[0]=='#') {
                    /* Headers define available topics */
                    value key = help_parsetopicname(line);
                    level = help_parsetopiclevel(line);
                    if (MORPHO_ISOBJECT(key)) {
                        topic[level]=help_newtopic(MORPHO_GETCSTRING(key), file, cloc,
                                                   (level>0 ? topic[level-1] : NULL) );
                        if (topic[level]) {
                            /* Insert the topic... */
                            dictionary *dict = &helpdict; /* ... either into the global dictionary */
                            if (level>0) dict=&topic[level-1]->subtopics; /* or into the parent's dictionary */
                            
                            dictionary_insert(dict, key, MORPHO_OBJECT(topic[level]));
#ifdef MORPHO_DEBUG_LOGHELPFILES
                            printf("Parsed topic '%s' level %i\n", MORPHO_GETCSTRING(key), level);
#endif
                        }
                    }
                } else if (strncmp(line, "[tag", 4)==0) {
                    /* Unused links that start with 'tag' define additional search terms */
                    value key = help_parsetag(line);
                    if (MORPHO_ISOBJECT(key)) {
                        /* Insert the topic... */
                        dictionary *dict = &helpdict; /* ... either into the global dictionary */
                        if (!toplevel && level>0) dict=&topic[level-1]->subtopics; /* or into the parent's dictionary */
                        dictionary_insert(dict, key, MORPHO_OBJECT(topic[level]));
#ifdef MORPHO_DEBUG_LOGHELPFILES
                        printf("Parsed tag '%s' level %i\n", MORPHO_GETCSTRING(key), level);
#endif
                    }
                } else if (strncmp(line, "[toplevel]", 10)==0) {
                    /* Toggles insertion into top level dictionary */
                    toplevel = !toplevel;
                }
            }
        }
        fclose(f);
    }
    
    return false;
}

/** Searches for help files
 *  @param path     directory to search (recursively)
 *  @returns true if any help files were successfully processed. */
bool help_searchpath(char *path) {
    DIR *d; /* Handle for the directory */
    struct dirent *entry; /* Entries in the directory */
    bool success=false; /* Did we load any entries? */
    
#ifdef MORPHO_DEBUG_LOGHELPFILES
    printf("Searching help directory '%s'\n", helpdir);
#endif
    
    d = opendir(helpdir);
    
    if (d) {
        while ((entry = readdir(d)) != NULL) {
            /* Contruct the file name */
            char file[strlen(path)+strlen(entry->d_name)+2];
            strcpy(file, path);
            strcat(file, "/");
            strcat(file, entry->d_name);
            
            /* If it's not a directory, try to open it */
            if (!morpho_isdirectory(file)) {
                bool res=help_load(file);
                if (res) success=true;
            }
        }
        closedir(d);
    }
    
    return success;
}

/* **********************************************************************
 * Public interface
 * ********************************************************************** */

/** Initializes the help system
 *  @returns true if help is available */
bool help_initialize(void) {
    dictionary_init(&helpdict);
    
    return help_searchpath(MORPHO_HELPDIRECTORY);
}

/** Finalizes the help system */
void help_finalize(void) {
    while (topics) {
        objecthelptopic *c = topics;
        help_cleartopic(c);
        topics = c->next;
        object_free((object *) c);
    }
    dictionary_clear(&helpdict);
}
