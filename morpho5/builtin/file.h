/** @file file.h
 *  @author T J Atherton
 *
 *  @brief Built in class to provide file input and output
 */

#ifndef file_h
#define file_h

#include <stdio.h>
#include "object.h"
#include "morpho.h"

/* -------------------------------------------------------
 * File objects
 * ------------------------------------------------------- */

extern objecttype objectfiletype;
#define OBJECT_FILE objectfiletype

typedef struct {
    object obj;
    value filename;
    FILE *f;
} objectfile;

/** Tests whether an object is a file */
#define MORPHO_ISFILE(val) object_istype(val, OBJECT_FILE)

/** Gets the object as an matrix */
#define MORPHO_GETFILE(val)   ((objectfile *) MORPHO_GETOBJECT(val))

/* -------------------------------------------------------
 * File class
 * ------------------------------------------------------- */

#define FILE_CLASSNAME    "File"

#define FILE_CLOSE        "close"
#define FILE_LINES        "lines"
#define FILE_READLINE     "readline"
#define FILE_READCHAR     "readchar"
#define FILE_WRITE        "write"
#define FILE_EOF          "eof"
#define FILE_RELATIVEPATH "relativepath"
#define FILE_FILENAME     "filename"

#define FILE_READMODE     "read"
#define FILE_WRITEMODE    "write"
#define FILE_APPENDMODE   "append"

/* -------------------------------------------------------
 * Folder class
 * ------------------------------------------------------- */

#define FOLDER_CLASSNAME  "Folder"

#define FOLDER_ISFOLDER   "isfolder"
#define FOLDER_CONTENTS   "contents"

/* -------------------------------------------------------
 * Error messages
 * ------------------------------------------------------- */

#define FILE_OPENFAILED                   "FlOpnFld"
#define FILE_OPENFAILED_MSG               "Couldn't open file '%s'."

#define FILE_FILENAMEARG                  "FlNmArgs"
#define FILE_FILENAMEARG_MSG              "First argument to File must be a filename."

#define FILE_NEEDSFILENAME                "FlNmMssng"
#define FILE_NEEDSFILENAME_MSG            "Filename missing."

#define FILE_MODE                         "FlMode"
#define FILE_MODE_MSG                     "Second argument to File should be 'read', 'write' or 'append'."

#define FILE_WRITEARGS                    "FlWrtArgs"
#define FILE_WRITEARGS_MSG                "Arguments to File.write must be strings."

#define FILE_WRITEFAIL                    "FlWrtFld"
#define FILE_WRITEFAIL_MSG                "Write to file failed."

#define FOLDER_EXPCTPATH                  "FldrExpctPth"
#define FOLDER_EXPCTPATH_MSG              "Folder methods expect a path as an argument."

#define FOLDER_NTFLDR                     "NtFldr"
#define FOLDER_NTFLDR_MSG                 "Not a folder."

bool file_getsize(FILE *f, size_t *s);

void file_setworkingdirectory(const char *script);

FILE *file_openrelative(const char *fname, const char *mode);

int file_readlineintovarray(FILE *f, varray_char *string);
bool file_readintovarray(FILE *f, varray_char *string);

void file_initialize(void);
void file_finalize(void);

#endif /* file_h */
