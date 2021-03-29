/** @file file.c
 *  @author T J Atherton
 *
 *  @brief Built in class to provide file input and output
 */

#include "file.h"
#include "builtin.h"
#include "error.h"
#include "object.h"
#include "morpho.h"
#include "common.h"
#include <stdio.h>
#include <limits.h>

static value file_fileproperty;

/** Store the current working directory (relative to the filing systems cwd) */
static varray_char workingdir;

/*
 * File handling utility functions
 */

#include <errno.h>

/* Determine the size of a file */
bool file_getsize(FILE *f, size_t *s) {
    long int curr, size;
    curr=ftell(f);
    if (fseek(f, 0L, SEEK_END)!=0) return false;
    size = ftell(f);
    if (fseek(f, curr, SEEK_SET)!=0) return false;
    if (s) *s = size;
    return true;
}


/* Gets the current file handle */
FILE *file_getfile(value obj) {
    value val;
    
    if (objectinstance_getproperty(MORPHO_GETINSTANCE(obj), file_fileproperty, &val)) {
        return (FILE *) MORPHO_GETOBJECT(val);
    }
    
    return NULL;
}

/** Sets the current file handle */
void file_setfile(value obj, FILE *f) {
    objectinstance_setproperty(MORPHO_GETINSTANCE(obj), file_fileproperty, MORPHO_OBJECT(f));
}

/** Sets the global current working directory (relative to the filing systems cwd.
 * @param[in] path - path to working directory. Any file name at the end is stripped */
void file_setworkingdirectory(const char *path) {
    int length = (int) strlen(path);
    int dirmarker = 0;
    
    /* Working backwards, find the last directory separator */
    for ( dirmarker=length; dirmarker>=0; dirmarker--) {
        if (path[dirmarker]=='\\' || path[dirmarker]=='/') break;
    }
    
    workingdir.count=0; /* Clear any prior working directory */
    if (dirmarker>0) {
        varray_charadd(&workingdir, (char *) path, dirmarker);
        varray_charwrite(&workingdir, '\0');
    }
}

/** Open a file relative to the current working directory (usually the script directory) */
FILE *file_openrelative(const char *fname, const char *mode) {
    varray_char name;
    varray_charinit(&name);

    /* Check the fname passed isn't a global file reference (i.e. starts with / or ~) */
    if (fname[0]!='~' && fname[0]!='/') {
        if (workingdir.count>0) {
            for (unsigned int i=0; i<workingdir.count && workingdir.data[i]!='\0'; i++) {
                varray_charwrite(&name, workingdir.data[i]);
            }
            varray_charwrite(&name, '/');
        }
    }
    varray_charadd(&name, (char *) fname, (int) strlen(fname));
    varray_charwrite(&name, '\0');
    
    FILE *out=fopen(name.data, mode);
    
    varray_charclear(&name);
    
    return out;
}

/** Initializer
 * In: 1. a file name
 *   2. (optional) a string giving the requested status, e.g. "wr+"
 */
value File_init(vm *v, int nargs, value *args) {
    char *fname=NULL;
    char *cmode = "r";
    
    if (nargs>0) {
        if (MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
            fname=MORPHO_GETCSTRING(MORPHO_GETARG(args, 0));
        } else MORPHO_RAISE(v, FILE_FILENAMEARG);
        
        if (nargs>1) {
            if (MORPHO_ISSTRING(MORPHO_GETARG(args, 1))) {
                char *mode=MORPHO_GETCSTRING(MORPHO_GETARG(args, 1));
                switch (mode[0]) {
                    case 'r': cmode="r"; break;
                    case 'w': cmode="w"; break;
                    case 'a': cmode="a"; break;
                    default: MORPHO_RAISE(v, FILE_MODE);
                }
            } else MORPHO_RAISE(v, FILE_MODE);
        }
    } else MORPHO_RAISE(v, FILE_NEEDSFILENAME);
    
    if (fname) {
        FILE *f = file_openrelative(fname, cmode);
        if (!f) MORPHO_RAISEVARGS(v, FILE_OPENFAILED, fname);
        
        file_setfile(MORPHO_SELF(args), f);
    }
    
    return MORPHO_NIL;
}

/** Close a file  */
value File_close(vm *v, int nargs, value *args) {
    FILE *f=file_getfile(MORPHO_SELF(args));
    if (f) {
        fclose(f);
        file_setfile(MORPHO_SELF(args), NULL);
    }
    
    return MORPHO_NIL;
}

/** Reads a line using a given buffer */
int file_readlineintovarray(FILE *f, varray_char *string) {
    int ic;
    
    for (ic=getc(f); ic!=EOF && ic!='\n'; ic=getc(f)) {
        varray_charwrite(string, (char) ic);
    }
    
    if (ic!=EOF || string->count>0) {
        varray_charwrite(string, '\0');
    }
    
    return ic;
}

/** Reads a whole file into a buffer */
bool file_readintovarray(FILE *f, varray_char *string) {
    size_t size;
    
    if (!file_getsize(f, &size)) return false;
    if (size>INT_MAX) return false;
    
    if (varray_charresize(string, (int) size+1)) {
        size_t nread=fread(string->data, sizeof(char), size, f);
        string->data[nread]='\0';
    }
    
    return true;
}


/** Reads a line using a given buffer */
value file_readlineusingvarray(FILE *f, varray_char *string) {
    int ic=file_readlineintovarray(f, string);
    
    if (ic!=EOF || string->count>0) {
        return object_stringfromvarraychar(string);
    }
    
    return MORPHO_NIL;
}

/** Read a line  */
value File_readline(vm *v, int nargs, value *args) {
    FILE *f=file_getfile(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (f) {
        varray_char string;
        varray_charinit(&string);
        
        out = file_readlineusingvarray(f, &string);
        varray_charclear(&string);
    }
    return out;
}

/** Get the contents of a file as an array */
value File_lines(vm *v, int nargs, value *args) {
    FILE *f=file_getfile(MORPHO_SELF(args));
    value out=MORPHO_NIL;

    if (f) {
        varray_value lines;
        varray_valueinit(&lines);
        
        varray_char string;
        varray_charinit(&string);
        
        do {
            value line = file_readlineusingvarray(f, &string);
            if (!MORPHO_ISNIL(line)) varray_valuewrite(&lines, line);
            string.count=0;
        } while (!feof(f));
        
        varray_charclear(&string);
        
        out=MORPHO_OBJECT(object_arrayfromvarrayvalue(&lines));
        varray_valueclear(&lines);
    }
    
    return out;
}

/** Reads a single character  */
value File_readchar(vm *v, int nargs, value *args) {
    FILE *f=file_getfile(MORPHO_SELF(args));
    if (f) {
        int ic=getc(f);
        if (ic!=EOF) {
            char c=(char) ic;
            return object_stringfromcstring(&c, 1);
        }
    }
    return MORPHO_NIL;
}

/** Write to a file  */
value File_write(vm *v, int nargs, value *args) {
    FILE *f=file_getfile(MORPHO_SELF(args));
    if (f) {
        for (unsigned int i=0; i<nargs; i++) {
            if (MORPHO_ISSTRING(MORPHO_GETARG(args, i))) {
                char *line = MORPHO_GETCSTRING(MORPHO_GETARG(args, i));
                if (fputs(line, f)==EOF) MORPHO_RAISE(v, FILE_WRITEFAIL);
                if (fputc('\n', f)==EOF) MORPHO_RAISE(v, FILE_WRITEFAIL);
            } else MORPHO_RAISE(v, FILE_WRITEARGS);
        }
    }
    
    return MORPHO_NIL;
}

/** Detects whether we're at the end of the file  */
value File_eof(vm *v, int nargs, value *args) {
    FILE *f=file_getfile(MORPHO_SELF(args));
    if (f && feof(f)) return MORPHO_TRUE;
    return MORPHO_FALSE;
}

/** Called when the file object is freed  */
value File_free(vm *v, int nargs, value *args) {
    File_close(v, nargs, args);
    return MORPHO_NIL;
}

MORPHO_BEGINCLASS(File)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, File_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_CLOSE, File_close, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_LINES, File_lines, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_READLINE, File_readline, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_READCHAR, File_readchar, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_WRITE, File_write, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_EOF, File_eof, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

void file_initialize(void) {
    varray_charinit(&workingdir);
    
    builtin_addclass(FILE_CLASSNAME, MORPHO_GETCLASSDEFINITION(File), MORPHO_NIL);
    file_fileproperty=builtin_internsymbolascstring(FILE_FILEPROPERTY);
    
    morpho_defineerror(FILE_OPENFAILED, ERROR_HALT, FILE_OPENFAILED_MSG);
    morpho_defineerror(FILE_NEEDSFILENAME, ERROR_HALT, FILE_NEEDSFILENAME_MSG);
    morpho_defineerror(FILE_FILENAMEARG, ERROR_HALT, FILE_FILENAMEARG_MSG);
    morpho_defineerror(FILE_MODE, ERROR_HALT, FILE_MODE_MSG);
    morpho_defineerror(FILE_WRITEARGS, ERROR_HALT, FILE_WRITEARGS_MSG);
    morpho_defineerror(FILE_WRITEFAIL, ERROR_HALT, FILE_WRITEFAIL_MSG);
}

void file_finalize(void) {
    varray_charclear(&workingdir);
}
