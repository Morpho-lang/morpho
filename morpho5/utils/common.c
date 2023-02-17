/** @file common.c
 *  @author T J Atherton
 *
 *  @brief Common types, data structures and functions for the Morpho VM
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>
#include "common.h"
#include "object.h"
#include "sparse.h"
#include "cmplx.h"
#include "file.h"

/* **********************************************************************
* Utility functions
* ********************************************************************** */

/** @brief Compares two values
 * @param a value to compare
 * @param b value to compare
 * @returns 0 if a and b are equal, a positive number if b\>a and a negative number if a\<b
 * @warning does not work if a and b are not the same type (use MORPHO_CHECKCMPTYPE to promote types if ordering is important) */
#define EQUAL 0
#define NOTEQUAL 1
#define BIGGER 1
#define SMALLER -1
int morpho_comparevalue (value a, value b) {

    // if comparing a number to complex cast the number to complex
    // we don't need to bind here beacues the value never needs to exist beyond this scope
    // valgrin/check with tim to be sure
    if (MORPHO_ISCOMPLEX(a) && MORPHO_ISNUMBER(b)){
        // cast b to complex
        double val;
        morpho_valuetofloat(b,&val);
        return (MORPHO_GETCOMPLEX(a)->Z==val ? EQUAL: NOTEQUAL);
    }
    if (MORPHO_ISCOMPLEX(b) && MORPHO_ISNUMBER(a)){
        // cast b to complex
        double val;
        morpho_valuetofloat(a,&val);
        return (MORPHO_GETCOMPLEX(b)->Z==val ? EQUAL: NOTEQUAL);
    }


    if (!morpho_ofsametype(a, b)) return NOTEQUAL;

    if (MORPHO_ISFLOAT(a)) {
        double x = MORPHO_GETFLOATVALUE(b) - MORPHO_GETFLOATVALUE(a);
        if (x>DBL_EPSILON) return BIGGER; /* Fast way out for clear cut cases */
        if (x<-DBL_EPSILON) return SMALLER;
        /* Assumes absolute tolerance is the same as relative tolerance. */
        if (fabs(x)<=DBL_EPSILON*fmax(1.0, fmax(MORPHO_GETFLOATVALUE(a), MORPHO_GETFLOATVALUE(b)))) return EQUAL;
        return (x>0 ? BIGGER : SMALLER);
    } else {
        switch (MORPHO_GETTYPE(a)) {
            case VALUE_NIL:
                return EQUAL; /** Nones are always the same */
            case VALUE_INTEGER:
                return (MORPHO_GETINTEGERVALUE(b) - MORPHO_GETINTEGERVALUE(a));
            case VALUE_BOOL:
                return (MORPHO_GETBOOLVALUE(b) != MORPHO_GETBOOLVALUE(a));
            case VALUE_OBJECT:
                {
                    if (MORPHO_GETOBJECTTYPE(a)!=MORPHO_GETOBJECTTYPE(b)) {
                        return 1; /* Objects of different type are always different */
                    } else if (MORPHO_ISSTRING(a)) {
                        objectstring *astring = MORPHO_GETSTRING(a);
                        objectstring *bstring = MORPHO_GETSTRING(b);
                        size_t len = (astring->length > bstring->length ? astring->length : bstring->length);

                        return -strncmp(astring->string, bstring->string, len);
                    } else if (MORPHO_ISDOKKEY(a) && MORPHO_ISDOKKEY(b)) {
                        objectdokkey *akey = MORPHO_GETDOKKEY(a);
                        objectdokkey *bkey = MORPHO_GETDOKKEY(b);

                        return ((MORPHO_GETDOKKEYCOL(akey)==MORPHO_GETDOKKEYCOL(bkey) &&
                                 MORPHO_GETDOKKEYROW(akey)==MORPHO_GETDOKKEYROW(bkey)) ? EQUAL : NOTEQUAL);
                    } else if (MORPHO_ISCOMPLEX(a) && MORPHO_ISCOMPLEX(b)) {
                        objectcomplex *acomp = MORPHO_GETCOMPLEX(a);
                        objectcomplex *bcomp = MORPHO_GETCOMPLEX(b);
                        return (complex_equality(acomp,bcomp)? EQUAL: NOTEQUAL);
                    } else {
                        return (MORPHO_GETOBJECT(a) == MORPHO_GETOBJECT(b)? EQUAL: NOTEQUAL);
                    }
                }
            default:
                UNREACHABLE("unhandled value type for comparison [Check morpho_comparevalue]");
        }
    }
    return NOTEQUAL;
}
#undef EQUAL
#undef NOTEQUAL
#undef BIGGER
#undef SMALLER

/** @brief Prints a value
 * @param v The value to print */
void morpho_printvalue(value v) {
    if (MORPHO_ISFLOAT(v)) {
        printf("%g", MORPHO_GETFLOATVALUE(v) );
        return;
    } else {
        switch (MORPHO_GETTYPE(v)) {
            case VALUE_NIL:
                printf(COMMON_NILSTRING);
                return;
            case VALUE_BOOL:
                printf("%s", ( MORPHO_GETBOOLVALUE(v) ? COMMON_TRUESTRING : COMMON_FALSESTRING ));
                return;
            case VALUE_INTEGER:
                printf("%i", MORPHO_GETINTEGERVALUE(v) );
                return;
            case VALUE_OBJECT:
                object_print(v);
                return;
            default:
                return;
        }
    }
}

/** @brief Prints a value to a buffer */
#define MORPHO_TOSTRINGTMPBUFFERSIZE   64
void morpho_printtobuffer(vm *v, value val, varray_char *buffer) {
    char tmp[MORPHO_TOSTRINGTMPBUFFERSIZE];
    int nv;

    if (MORPHO_ISSTRING(val)) {
        objectstring *s = MORPHO_GETSTRING(val);
        varray_charadd(buffer, s->string, (int) s->length);
    } else if (MORPHO_ISOBJECT(val)) {
        objectclass *klass = morpho_lookupclass(val);

        if (klass) {
            objectstring str = MORPHO_STATICSTRING(MORPHO_TOSTRING_METHOD);
            value label = MORPHO_OBJECT(&str);
            value method, ret;

            if (morpho_lookupmethod(val, label, &method) &&
                morpho_invoke(v, val, method, 0, NULL, &ret)) {
                if (MORPHO_ISSTRING(ret)) {
                    varray_charadd(buffer, MORPHO_GETCSTRING(ret), (int) MORPHO_GETSTRINGLENGTH(ret));
                }
            } else {
                varray_charwrite(buffer, '<');
                morpho_printtobuffer(v, klass->name, buffer);
                varray_charwrite(buffer, '>');
            }
        } else if (MORPHO_ISBUILTINFUNCTION(val)) {
            objectbuiltinfunction *fn = MORPHO_GETBUILTINFUNCTION(val);
            varray_charadd(buffer, "<fn ", 4);
            morpho_printtobuffer(v, fn->name, buffer);
            varray_charwrite(buffer, '>');
        } else if (MORPHO_ISCLASS(val)) {
            objectclass *klass = MORPHO_GETCLASS(val);
            varray_charwrite(buffer, '@');
            morpho_printtobuffer(v, klass->name, buffer);
        }
    } else if (MORPHO_ISFLOAT(val)) {
        nv=sprintf(tmp, "%g", MORPHO_GETFLOATVALUE(val));
        varray_charadd(buffer, tmp, nv);
    } else if (MORPHO_ISINTEGER(val)) {
        nv=sprintf(tmp, "%i", MORPHO_GETINTEGERVALUE(val));
        varray_charadd(buffer, tmp, nv);
    } else if (MORPHO_ISBOOL(val)) {
        nv=sprintf(tmp, "%s", (MORPHO_ISTRUE(val) ? COMMON_TRUESTRING : COMMON_FALSESTRING));
        varray_charadd(buffer, tmp, nv);
    } else if (MORPHO_ISNIL(val)) {
        nv=sprintf(tmp, "%s", COMMON_NILSTRING);
        varray_charadd(buffer, tmp, nv);
    }
}

/** @brief Concatenates a sequence of values as a string */
value morpho_concatenate(vm *v, int nval, value *val) {
    varray_char buffer;
    varray_charinit(&buffer);

    for (unsigned int i=0; i<nval; i++) {
        morpho_printtobuffer(v, val[i], &buffer);
    }

    value out=object_stringfromcstring(buffer.data, buffer.count);

    varray_charclear(&buffer);

    return out;
}

/** @brief   Duplicates a string.
 *  @param   string String to duplicate
 *  @warning Caller should call MALLOC_FREE on the allocated string */
char *morpho_strdup(char *string) {
    size_t len = strlen(string) + 1;
    char* output = (char*) malloc ((len + 1) * sizeof(char));
    if (output) memcpy(output, string, len);

    return output;
}

/** @brief Returns the number of bytes in the next character of a given utf8 string
    @returns number of bytes */
int morpho_utf8numberofbytes(uint8_t *string) {
    uint8_t byte = * string;

    if ((byte & 0xc0) == 0x80) return 0; // In the middle of a utf8 string

    // Get the number of bytes from the first character
    if ((byte & 0xf8) == 0xf0) return 4;
    if ((byte & 0xf0) == 0xe0) return 3;
    if ((byte & 0xe0) == 0xc0) return 2;
    return 1;
}

/** @brief Computes the nearest power of 2 above an integer
 * @param   n An integer
 * @returns Nearest power of 2 above n
 * See: http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float */
unsigned int morpho_powerof2ceiling(unsigned int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;

    return n;
}

/* Tells if an object at path corresponds to a directory */
bool morpho_isdirectory(const char *path) {
   struct stat statbuf;
   if (stat(path, &statbuf) != 0)
       return 0;
   return (bool) S_ISDIR(statbuf.st_mode);
}

/** Determine weather the rest of a string is white space */
bool white_space_remainder(const char *s, int start){
	s += start;
	while (*s){
		if (!isspace(*s)){
			return false;
		}
		s++;
	}
	return true;
}

/** Count the number of fixed parameters in a callable object
 * @param[in] f - the function or callable object
 * @param[out] nparams - number of parameters; -1 if unknown
 * @returns true on success, false if f is not callable*/
bool morpho_countparameters(value f, int *nparams) {
    value g = f;
    bool success=false;

    if (MORPHO_ISINVOCATION(g)) { // Unpack invocation
        objectinvocation *inv = MORPHO_GETINVOCATION(g);
        g=inv->method;
    }

    if (MORPHO_ISCLOSURE(g)) { // Unpack closure
        objectclosure *cl = MORPHO_GETCLOSURE(g);
        g=MORPHO_OBJECT(cl->func);
    }

    if (MORPHO_ISFUNCTION(g)) {
        objectfunction *fun = MORPHO_GETFUNCTION(g);
        *nparams=fun->nargs;
        success=true;
    } else if (MORPHO_ISBUILTINFUNCTION(g)) {
        *nparams = -1;
        success=true;
    }

    return success;
}

/** Initialize tuple generator
 @param[in] nval - number of values
 @param[in] n - n-tuples to generate
 @param[in] c - workspace: supply an unsigned integer array of size 2xn  */
void morpho_tuplesinit(unsigned int nval, unsigned int n, unsigned int *c, tuplemode mode) {
    unsigned int *counter=c, *cmax=c+n; // Counters
    for (unsigned int i=0; i<n; i++) {
        counter[i]=(mode == MORPHO_SETMODE ? i : 0 );
        cmax[i]=(mode == MORPHO_SETMODE ? nval-n+i : nval-1);
    }
}

/** Generate n-tuples of unique elements indep of ordering from a list of values
 @param[in] nval - number of values
 @param[in] list - list of values
 @param[in] n - n-tuples to generate
 @param[in] c - workspace: supply an unsigned integer array of size 2xn;
 @param[out] tuple - generated tuple
 @returns true if we returned a valid tuple; false if we're done */
bool morpho_tuples(unsigned int nval, value *list, unsigned int n, unsigned int *c, tuplemode mode, value *tuple) {
    unsigned int *counter=c, *cmax=c+n; // Counters
    int k;

    if (counter[0]>cmax[0]) return false; // Done

    // Generate tuple from counter
    for (unsigned int i=0; i<n; i++) tuple[i]=list[counter[i]];

    // Increment counters
    counter[n-1]++; // Increment last counter
    for (k=n-1; k>0 && counter[k]>cmax[k]; k--) counter[k-1]++; // Carry

    if (k<n-1) {
        if (mode==MORPHO_TUPLEMODE) for (unsigned int i=k+1; i<n; i++) counter[i]=0;
        if (mode==MORPHO_SETMODE) for (unsigned int i=k+1; i<n; i++) counter[i]=counter[i-1]+1;
    }

    return true;
}

/* **********************************************************************
* Thread pools
* ********************************************************************** */

int threadpool_nthreads = MORPHO_DEFAULTTHREADNUMBER;

/** Sets the number of worker threads to use */
void morpho_setthreadnumber(int nthreads) {
    threadpool_nthreads = nthreads;
}

/** Returns the number of worker threads to use */
int morpho_threadnumber(void) {
    return threadpool_nthreads;
}

DEFINE_VARRAY(task, task);

/* Worker thread */
void *threadpool_worker(void *ref) {
    threadpool *pool = (threadpool *) ref;
    task t = { .func = NULL, .arg = NULL };

    while (true) {
        /* Await a task */
        pthread_mutex_lock(&pool->lock_mutex);
        while (pool->queue.count == 0 && !pool->stop)
            pthread_cond_wait(&pool->work_available_cond, &pool->lock_mutex);

        if (pool->stop) break; /* Terminate if asked to do so */

        varray_taskpop(&pool->queue, &t); /* Get the task */
        pool->nprocessing++;
        pthread_mutex_unlock(&pool->lock_mutex);

        if (t.func) { (t.func) (t.arg); }; /* Perform the assigned task */

        pthread_mutex_lock(&pool->lock_mutex);
        pool->nprocessing--;
        if (!pool->stop && pool->nprocessing == 0 && pool->queue.count == 0)
            pthread_cond_signal(&pool->work_halted_cond);
        pthread_mutex_unlock(&pool->lock_mutex);
    }

    /* No need to lock here as lock was already obtained */
    pool->nthreads--;
    pthread_cond_signal(&pool->work_halted_cond);
    pthread_mutex_unlock(&pool->lock_mutex);

    return NULL;
}

/* Interface */

/** Initialize a threadpool with n worker threads. */
bool threadpool_init(threadpool *pool, int nworkers) {
    if (nworkers<1) return false;

    varray_taskinit(&pool->queue);

    pthread_mutex_init(&pool->lock_mutex, NULL);
    pthread_cond_init(&pool->work_available_cond, NULL);
    pthread_cond_init(&pool->work_halted_cond, NULL);

    pool->nthreads=nworkers;
    pool->stop=false;
    pool->nprocessing=0;

    for (int i=0; i<pool->nthreads; i++) {
        pthread_t thread;
        pthread_create(&thread, NULL, threadpool_worker, pool);
        pthread_detach(thread);
    }

    return true;
}

/** Clears a threadpool. */
void threadpool_clear(threadpool *pool) {
    pthread_mutex_lock(&pool->lock_mutex);
    varray_taskclear(&pool->queue); /* Erase any remaining tasks */
    pool->stop = true; /* Tell workers to stop */
    pthread_cond_broadcast(&pool->work_available_cond); /* Signal to workers */
    pthread_mutex_unlock(&pool->lock_mutex);

    threadpool_fence(pool); /* Await workers to terminate */

    pthread_mutex_destroy(&pool->lock_mutex);
    pthread_cond_destroy(&pool->work_available_cond);
    pthread_cond_destroy(&pool->work_halted_cond);
}

/** Adds a task to the threadpool */
bool threadpool_add_task(threadpool *pool, workfn func, void *arg) {
    bool success=true;
    pthread_mutex_lock(&pool->lock_mutex);

    task t = { .func = func, .arg=arg };
    if (!varray_taskadd(&pool->queue, &t, 1)) success=false; /* Add the task to the queue */

    pthread_cond_broadcast(&pool->work_available_cond); /* Signal there is work to be done */
    pthread_mutex_unlock(&pool->lock_mutex);
    return true;
}

/** Blocks until all tasks in the thread pool are complete */
void threadpool_fence(threadpool *pool) {
    pthread_mutex_lock(&pool->lock_mutex);

    while (true) {
        if ((!pool->stop && (pool->queue.count > 0 || pool->nprocessing>0)) || // If we are simply waiting for tasks to finish
            (pool->stop && pool->nthreads > 0)) { // Or if we have been told to stop
            pthread_cond_wait(&pool->work_halted_cond, &pool->lock_mutex); // Block until working_cond is set
        } else break;
    }

    pthread_mutex_unlock(&pool->lock_mutex);
}

/*
bool worker(void *arg) {
    int *val = arg;
    int  old = *val;

    *val += 1000;
    printf("tid=%p, old=%d, val=%d\n", pthread_self(), old, *val);

   // if (*val%2)
   //     usleep(100000);

    return false;
}

void threadpool_test(void) {
    threadpool pool;
    int num_items = 100;
    int vals[num_items];

    threadpool_init(&pool, 4);

    for (int i=0; i<num_items; i++) {
        vals[i] = i;
        threadpool_add_task(&pool, worker, vals+i);
    }

    threadpool_fence(&pool);

    for (int i=0; i<num_items; i++) {
        printf("%d\n", vals[i]);
    }

    threadpool_clear(&pool);
}
*/

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

/** Checks if a filename matches all criteria in a resourceenumerator
 @param[in] en - initialized enumerator */
bool resources_matchfile(resourceenumerator *en, char *file) {
    char *c = file+strlen(file);

    while (c>=file && *c!='.') c--; // Skip past extension

    if (en->fname) { // Match filename if requested
        char *f = c;
        while (f>=file && *f!=MORPHO_SEPARATOR) f--;
        if (*f==MORPHO_SEPARATOR) f++;
        for (int i=0; f[i]!='\0' && f[i]!='.'; i++) { // Now compare filename against match
            if (f[i]!=en->fname[i]) return false;
        }
    }

    if (!en->ext) return true; // Match extension only if requested

    if (*c!='.') return false;
    for (int k=0; *en->ext[k]!='\0'; k++) { // Check extension against possible extensions
        if (strcmp(c+1, en->ext[k])==0) return true; // We have a match
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
}

void resources_finalize(void) {
    for (int i=0; i<resourcelocations.count; i++) morpho_freeobject(resourcelocations.data[i]);
    varray_valueclear(&resourcelocations);
}

/* **********************************************************************
* Extensions
* ********************************************************************** */

#include <dlfcn.h>

typedef struct {
    value name;
    void *handle;
} extension;

DECLARE_VARRAY(extension, extension)
DEFINE_VARRAY(extension, extension)

#define MORPHO_EXTENSIONINITIALIZE "initialize" // Function to call upon initialization
#define MORPHO_EXTENSIONFINALIZE "finalize"     // Function to call upon finalization

varray_extension extensions;

/** Trys to locate a function with NAME_FN in extension e, and calls it if found */
void extensions_call(extension *e, char *name, char *fn) {
    void (*fptr) (void);
    char fnname[strlen(name)+strlen(fn)+2];
    strcpy(fnname, name);
    strcat(fnname, "_");
    strcat(fnname, fn);
    
    fptr = dlsym(e->handle, fnname);
    if (fptr) (*fptr) ();
}

/** Attempts to load an extension with given name. Returns true if it was found and loaded successfully */
bool morpho_loadextension(char *name) {
    char *ext[] = { MORPHO_DYLIBEXTENSION, "dylib", "so", "" };
    value out = MORPHO_NIL;
    
    if (!morpho_findresource(MORPHO_EXTENSIONSDIR, name, ext, true, &out)) return false;
    
    extension e;
    e.handle = dlopen(MORPHO_GETCSTRING(out), RTLD_LAZY);
    if (e.handle) {
        e.name = object_stringfromcstring(name, strlen(name));
        varray_extensionwrite(&extensions, e);
        
        extensions_call(&e, name, MORPHO_EXTENSIONINITIALIZE);
    }
    
    morpho_freeobject(out);
    
    return e.handle;
}

void extensions_initialize(void) {
    varray_extensioninit(&extensions);
}

void extensions_finalize(void) {
    for (int i=0; i<extensions.count; i++) {
        /* Finalize and close each extension */
        value name = extensions.data[i].name;
        extensions_call(&extensions.data[i], MORPHO_GETCSTRING(name), MORPHO_EXTENSIONFINALIZE);
        morpho_freeobject(name);
        dlclose(extensions.data[i].handle);
    }
    varray_extensionclear(&extensions);
}

