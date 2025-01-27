/** @file file.c
 *  @author T J Atherton
 *
 *  @brief Defines file object type as well as File and Folder classes
 */

#include <stdio.h>
#include <limits.h>

#include "morpho.h"
#include "classes.h"
#include "file.h"
#include "platform.h"

/** Store the current working directory (relative to the filing systems cwd) */
static varray_char workingdir;

/* **********************************************************************
 * File objects
 * ********************************************************************** */

objecttype objectfiletype;

/** File object definitions */
size_t objectfile_sizefn(object *obj) {
    return sizeof(objectfile);
}

void objectfile_markfn(object *obj, void *v) {
    objectfile *file = (objectfile *) obj;
    morpho_markvalue(v, file->filename);
}

void objectfile_freefn(object *obj) {
    objectfile *file = (objectfile *) obj;
    if (file->f) fclose(file->f);
}

void objectfile_printfn(object *obj, void *v) {
    objectfile *file = (objectfile *) obj;
    morpho_printf(v, "<File '");
    morpho_printvalue(v, file->filename);
    morpho_printf(v, "'>");
}

objecttypedefn objectfiledefn = {
    .printfn=objectfile_printfn,
    .markfn=objectfile_markfn,
    .freefn=objectfile_freefn,
    .sizefn=objectfile_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/** Creates a file object */
objectfile *object_newfile(value filename, FILE *f) {
    objectfile *new = (objectfile *) object_new(sizeof(objectfile), OBJECT_FILE);
    
    if (new) {
        new->filename=filename;
        new->f=f;
    }
    
    return new;
}

/* **********************************************************************
 * File handling utility functions
 * ********************************************************************** */

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
    if (MORPHO_ISFILE(obj)) return MORPHO_GETFILE(obj)->f;
    return NULL;
}

/** Sets the current file handle */
void file_setfile(value obj, FILE *f) {
    if (MORPHO_ISFILE(obj)) MORPHO_GETFILE(obj)->f=f;
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

/** Gets the relative path for a given file */
void file_relativepath(const char *fname, varray_char *name) {
    /* Check the fname passed isn't a global file reference (i.e. starts with / or ~) */
    if (fname[0]!='~' && fname[0]!='/' && 
        !(fname[0]!='\0' && fname[1]==':')) {
        if (workingdir.count>0) {
            for (unsigned int i=0; i<workingdir.count && workingdir.data[i]!='\0'; i++) {
                varray_charwrite(name, workingdir.data[i]);
            }
            varray_charwrite(name, '/');
        }
    }
    varray_charadd(name, (char *) fname, (int) strlen(fname));
    varray_charwrite(name, '\0'); // Ensure string is null terminated.
}

/** Open a file relative to the current working directory (usually the script directory) */
FILE *file_openrelative(const char *fname, const char *mode) {
    varray_char name;
    varray_charinit(&name);
    
    file_relativepath(fname, &name);
    
    FILE *out=fopen(name.data, mode);
    
    varray_charclear(&name);
    
    return out;
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

/* **********************************************************************
 * File class
 * ********************************************************************** */

/** File constructor
 * In: 1. a file name
 *   2. (optional) a string giving the requested status, e.g. "wr+"
 */
value file_constructor(vm *v, int nargs, value *args) {
    objectfile *new=NULL;
    value out=MORPHO_NIL;
    value filename=MORPHO_NIL;
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
        
        filename = object_stringfromcstring(fname, strlen(fname));
        new = object_newfile(filename, f);
    }
    
    if (new) {
        out=MORPHO_OBJECT(new);
        value bind[] = { filename, out };
        morpho_bindobjects(v, 2, bind);
    }
    
    return out;
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
        
        varray_valuewrite(&lines, out); // Tuck onto end of lines to bind all at once
        morpho_bindobjects(v, lines.count, lines.data);
        varray_valueclear(&lines);
    }
    
    return out;
}

/** Reads the whole file into a string */
value File_readall(vm *v, int nargs, value *args) {
    FILE *f=file_getfile(MORPHO_SELF(args));
    size_t size;
    value out = MORPHO_NIL;
    if (f && file_getsize(f, &size)) {
        objectstring *new=object_stringwithsize(size);
        if (new) {
            size_t nbytes = fread(new->string, sizeof(char), size, f);
            new->length=nbytes;
            out = MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        }
    }
    return out;
}

/** Read a line  */
value File_readline(vm *v, int nargs, value *args) {
    FILE *f=file_getfile(MORPHO_SELF(args));
    value out=MORPHO_NIL;
    
    if (f) {
        varray_char string;
        varray_charinit(&string);
        
        out = file_readlineusingvarray(f, &string);
        morpho_bindobjects(v, 1, &out);
        varray_charclear(&string);
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
            value out=object_stringfromcstring(&c, 1);
            morpho_bindobjects(v, 1, &out);
            return out;
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

/** Get the path of a file relative to the CWD */
value File_relativepath(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    value fname = MORPHO_GETFILE(MORPHO_SELF(args))->filename;
    varray_char path;
    varray_charinit(&path);
    
    if (MORPHO_ISSTRING(fname)) {
        file_relativepath(MORPHO_GETCSTRING(fname), &path);
        
        out = object_stringfromvarraychar(&path);
        morpho_bindobjects(v, 1, &out);
    }
    
    varray_charclear(&path);
    return out;
}

/** Get the filename */
value File_filename(vm *v, int nargs, value *args) {
    return MORPHO_GETFILE(MORPHO_SELF(args))->filename;
}

/** Detects whether we're at the end of the file  */
value File_eof(vm *v, int nargs, value *args) {
    FILE *f=file_getfile(MORPHO_SELF(args));
    if (f && feof(f)) return MORPHO_TRUE;
    return MORPHO_FALSE;
}

MORPHO_BEGINCLASS(File)
MORPHO_METHOD(FILE_CLOSE, File_close, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_LINES, File_lines, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_READALL, File_readall, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_READLINE, File_readline, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_READCHAR, File_readchar, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_WRITE, File_write, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_RELATIVEPATH, File_relativepath, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_FILENAME, File_filename, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FILE_EOF, File_eof, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Folder objects
 * ********************************************************************** */

/** Detect whether a resource is a folder  */
value Folder_isfolder(vm *v, int nargs, value *args) {
    value ret = MORPHO_FALSE;
    if (nargs==1 && MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        varray_char name;
        varray_charinit(&name);
        file_relativepath(MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)), &name);
        
        if (platform_isdirectory(name.data)) ret=MORPHO_TRUE;
        
        varray_charclear(&name);
    } else morpho_runtimeerror(v, FOLDER_EXPCTPATH);
    
    return ret;
}

/** Return the contents of a folder  */
value Folder_contents(vm *v, int nargs, value *args) {
    value ret = MORPHO_NIL;
    if (nargs==1 && MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        varray_char name;
        varray_charinit(&name);
        file_relativepath(MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)), &name);

        size_t size = platform_maxpathsize();
        char buffer[size];
        
        MorphoDirContents contents;
        if (platform_directorycontentsinit(&contents, name.data)) {
            varray_value list;
            varray_valueinit(&list);
            
            while (platform_directorycontents(&contents, buffer, size)) {
                value entry = object_stringfromcstring(buffer, strlen(buffer));
                if (MORPHO_ISSTRING(entry)) varray_valuewrite(&list, entry);
            };
            
            platform_directorycontentsclear(&contents);
            
            objectlist *clist = object_newlist(list.count, list.data);
            if (clist) {
                ret = MORPHO_OBJECT(clist);
                varray_valuewrite(&list, ret);
                morpho_bindobjects(v, list.count, list.data);
            } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
            
            varray_valueclear(&list);
        } else morpho_runtimeerror(v, FOLDER_NTFLDR);
        
        varray_charclear(&name);
    } else morpho_runtimeerror(v, FOLDER_EXPCTPATH);
    
    return ret;
}

MORPHO_BEGINCLASS(Folder)
MORPHO_METHOD(FOLDER_ISFOLDER, Folder_isfolder, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FOLDER_CONTENTS, Folder_contents, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void file_initialize(void) {
    varray_charinit(&workingdir);
    
    objectfiletype=object_addtype(&objectfiledefn);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    morpho_addfunction(FILE_CLASSNAME, FILE_CLASSNAME " (...)", file_constructor, MORPHO_FN_CONSTRUCTOR, NULL);
    
    value fileclass=builtin_addclass(FILE_CLASSNAME, MORPHO_GETCLASSDEFINITION(File), objclass);
    object_setveneerclass(OBJECT_FILE, fileclass);
    
    builtin_addclass(FOLDER_CLASSNAME, MORPHO_GETCLASSDEFINITION(Folder), MORPHO_NIL);
    
    morpho_defineerror(FILE_OPENFAILED, ERROR_HALT, FILE_OPENFAILED_MSG);
    morpho_defineerror(FILE_NEEDSFILENAME, ERROR_HALT, FILE_NEEDSFILENAME_MSG);
    morpho_defineerror(FILE_FILENAMEARG, ERROR_HALT, FILE_FILENAMEARG_MSG);
    morpho_defineerror(FILE_MODE, ERROR_HALT, FILE_MODE_MSG);
    morpho_defineerror(FILE_WRITEARGS, ERROR_HALT, FILE_WRITEARGS_MSG);
    morpho_defineerror(FILE_WRITEFAIL, ERROR_HALT, FILE_WRITEFAIL_MSG);
    
    morpho_defineerror(FOLDER_EXPCTPATH, ERROR_HALT, FOLDER_EXPCTPATH_MSG);
    morpho_defineerror(FOLDER_NTFLDR, ERROR_HALT, FOLDER_NTFLDR_MSG);
    
    morpho_addfinalizefn(file_finalize);
}

void file_finalize(void) {
    varray_charclear(&workingdir);
}
