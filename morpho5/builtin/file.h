/** @file file.h
 *  @author T J Atherton
 *
 *  @brief Built in class to provide file input and output
 */

#ifndef file_h
#define file_h

#include <stdio.h>
#include "morpho.h"

#define FILE_CLASSNAME    "File"

#define FILE_FILEPROPERTY "@file"

#define FILE_CLOSE        "close"
#define FILE_LINES        "lines"
#define FILE_READLINE     "readline"
#define FILE_READCHAR     "readchar"
#define FILE_WRITE        "write"
#define FILE_EOF          "eof"

#define FILE_READMODE     "read"
#define FILE_WRITEMODE    "write"
#define FILE_APPENDMODE   "append"

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

bool file_getsize(FILE *f, size_t *s);

void file_setworkingdirectory(const char *script);

FILE *file_openrelative(const char *fname, const char *mode);

int file_readlineintovarray(FILE *f, varray_char *string);
bool file_readintovarray(FILE *f, varray_char *string);

void file_initialize(void);
void file_finalize(void);

#endif /* file_h */
