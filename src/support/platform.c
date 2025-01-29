/** @file platform.c
 *  @author T J Atherton 
 *
 *  @brief Isolates platform dependent code in morpho */

/** Platform-dependent code in morpho arises in several ways: 
 *  - Complex numbers for platforms that do not fully implement C99
 *  - Navigating the file system
 *  - APIs for opening dynamic libraries
 *  - APIs for using threads 
 *  - Functions that involve time */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "build.h"
#include "platform.h"
#include "error.h"

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
#else 
#define _POSIX_C_SOURCE 199309L
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <pwd.h>
#include <time.h>
#include <dlfcn.h>
#endif

/* **********************************************************************
 * Platform name
 * ********************************************************************** */

const char *platform_name(void) {
#if __APPLE__
    return MORPHO_PLATFORM_MACOS;
#elif __linux__
    return MORPHO_PLATFORM_LINUX;
#elif __UNIX__
    return MORPHO_PLATFORM_UNIX;
#elif defined(_WIN32)
    return MORPHO_PLATFORM_WINDOWS;
#endif
    return NULL; // Unrecognized platform
}

/* **********************************************************************
 * Random numbers
 * ********************************************************************** */

/** Obtain a number of random bytes from the host platform */
bool platform_randombytes(char *buffer, size_t nbytes) {
    bool success=false;
#ifdef _WIN32
    HCRYPTPROV hProvider = 0;
    if (CryptAcquireContext(&hProvider, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT) &&
        CryptGenRandom(hProvider, nbytes, buffer)) {
        CryptReleaseContext(hProvider, 0);
        success=true; 
    }
#else 
    FILE *urandom;
    /* Initialize from OS random bits */
    urandom=fopen("/dev/urandom", "r");
    if (urandom) {
        for(int i=0; i<nbytes; i++) buffer[i]=(char) fgetc(urandom);
        fclose(urandom);
        success=true; 
    }
#endif 
    return success; 
}

/* **********************************************************************
 * Complex arithmetic
 * ********************************************************************** */

#ifdef _WIN32
MorphoComplex MCAdd(MorphoComplex a, MorphoComplex b) {
    return MCBuild(creal(a)+creal(b), cimag(a)+cimag(b)); 
}

MorphoComplex MCSub(MorphoComplex a, MorphoComplex b) {
    return MCBuild(creal(a)-creal(b), cimag(a)-cimag(b)); 
}

MorphoComplex MCDiv(MorphoComplex a, MorphoComplex b) {
    return _Cmulcr(MCMul(a, conj(b)), 1.0/norm(b));
}

bool MCSame(MorphoComplex a, MorphoComplex b) {
    return (creal(a)==creal(b) && cimag(a)==cimag(b));
}
#endif

/** Compare two complex numbers using both absolute and relative tolerance */
bool MCEq(MorphoComplex a, MorphoComplex b) {
    double diff = cabs(MCSub(a,b));
    double absa = cabs(a), absb = cabs(b);
    double absmax = (absa>absb ? absa : absb);
    return (diff == 0.0) || (absmax > DBL_MIN && diff/absmax <= MORPHO_RELATIVE_EPS);
}

/* **********************************************************************
 * File system functions
 * ********************************************************************** */

/* Tells if an object at path corresponds to a directory */
bool platform_isdirectory(const char *path) {
#ifdef _WIN32
    DWORD attributes = GetFileAttributes(path);
    if (attributes==INVALID_FILE_ATTRIBUTES) return false; 
    return (attributes & FILE_ATTRIBUTE_DIRECTORY);
#else
   struct stat statbuf;
   if (stat(path, &statbuf) != 0)
       return 0;
   return (bool) S_ISDIR(statbuf.st_mode);
#endif
}

/** Returns the maximum size of a file path */
size_t platform_maxpathsize(void) {
#ifdef _WIN32 
    return (size_t) MAX_PATH;
#else
    return pathconf("/", _PC_PATH_MAX);
#endif 
}

/** Sets the current working directory to path */
bool platform_setcurrentdirectory(const char *path) {
#ifdef _WIN32
    return SetCurrentDirectory(path);
#else
    return chdir(path)==0;
#endif
}

/** Gets the path for the current working directory */
bool platform_getcurrentdirectory(char *buffer, size_t size) {
#ifdef _WIN32 
    return GetCurrentDirectory(size, buffer);
#else 
    return getcwd(buffer, size);
#endif
}

/** Gets a path containing the user's home directory */
bool platform_gethomedirectory(char *buffer, size_t size) {
#ifdef _WIN32
    DWORD length = GetEnvironmentVariable("USERPROFILE", buffer, size);
    return (length!=0 && length<size);
#else
    const char *homedir=getenv("HOME"); 
    if (!homedir) {
        struct passwd *pwd = getpwuid(getuid());
        if (pwd) homedir = pwd->pw_dir;
    }
    if (homedir) strncpy(buffer, homedir, size);

    return homedir;
#endif
}

/** Initializes a MorphoDirContents structure with a given path */
bool platform_directorycontentsinit(MorphoDirContents *contents, const char *path) {
#ifdef _WIN32
    size_t len = strlen(path)+3;
    char srch[len];
    strcpy(srch,path);
    strcat(srch, "/*"); // Add required wildcard

    bool success=false;
    contents->handle = FindFirstFile(srch, &contents->finddata);
    if (contents->handle != INVALID_HANDLE_VALUE) {
        contents->isvalid=true; // Used by platform_directorcontentsnext to check that finddata contains valid information
        success=true; 
    }

    return success; 
#else
    contents->dir=opendir(path);
    return contents->dir;
#endif
}

/** Clears the contents of a MorphoDirContents structure */
void platform_directorycontentsclear(MorphoDirContents *contents) {
#ifdef _WIN32
    FindClose(contents->handle);
#else
    closedir(contents->dir);
#endif
}

/** Call this function repeatedly to extract the next file in the directory. Returns true if a file is found; the filename is in the buffer. */
bool platform_directorycontents(MorphoDirContents *contents, char *buffer, size_t size) {
#ifdef _WIN32
    if (!contents->isvalid) return false;

    while (strcmp(contents->finddata.cFileName, ".")==0 ||
           strcmp(contents->finddata.cFileName, "..")==0) {

        if (FindNextFile(contents->handle, &contents->finddata)==0) return false; 
    }

    strncpy(buffer, contents->finddata.cFileName, size);

    // Fetch next file and record whether it succeeded
    contents->isvalid=(FindNextFile(contents->handle, &contents->finddata)!=0);
    return true; 
#else
    struct dirent *entry;
    do {
        entry = readdir(contents->dir);
        if (!entry) return false;
    } while (strcmp(entry->d_name, ".")==0 ||
             strcmp(entry->d_name, "..")==0); // Skip links to this and parent folder
    
    strncpy(buffer, entry->d_name, size);
    return true;
#endif
}

/* **********************************************************************
 * Dynamic libraries
 * ********************************************************************** */

/** Opens a dynamic library, returning a handle for future use */
MorphoDLHandle platform_dlopen(const char *path) {
#ifdef _WIN32
    return LoadLibrary((LPCSTR) path);
#else
    return dlopen(path, RTLD_LAZY);
#endif
}

/** Closes a dynamic libary */
void platform_dlclose(MorphoDLHandle handle) {
#ifdef _WIN32
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
}

/** Looks up a symbol in a dynamic library */
void *platform_dlsym(MorphoDLHandle handle, const char *symbol) {
#ifdef _WIN32
    return (void *) GetProcAddress(handle, symbol);
#else 
    return dlsym(handle, symbol);
#endif
}

/* **********************************************************************
 * Threads
 * ********************************************************************** */

DEFINE_VARRAY(MorphoThread, MorphoThread);

/** Creates a thread */
bool MorphoThread_create(MorphoThread *thread, MorphoThreadFn threadfn, void *ref) {
#ifdef _WIN32
    DWORD threadId; 
    *thread = CreateThread(NULL, // Default security attributes
                           0, // Default stack size
                           threadfn, 
                           ref,
                           0, // Default creation flags
                           &threadId); 
    return (*thread!=NULL);
#else
    return (pthread_create(thread, NULL, threadfn, ref)==0);
#endif
}

/** Waits for a thread to finish */
void MorphoThread_join(MorphoThread thread) {
#ifdef _WIN32
    WaitForSingleObject(thread, INFINITE);
#else 
    pthread_join(thread, NULL);
#endif
}

/** Clears a thread, releasing any resources used */
void MorphoThread_clear(MorphoThread thread) {
#ifdef _WIN32
    CloseHandle(thread);
#endif
}

/** Exits a thread */
void MorphoThread_exit(void) {
#ifdef _WIN32
    ExitThread(0);
#else 
    pthread_exit(NULL);
#endif
}

/** Initializes a mutex */
bool MorphoMutex_init(MorphoMutex *mutex) {
#ifdef _WIN32
    InitializeCriticalSection(mutex);
    return true; 
#else 
    return pthread_mutex_init(mutex, NULL)==0;
#endif
}

/** Clears a mutex */
void MorphoMutex_clear(MorphoMutex *mutex) {
#ifdef _WIN32
    DeleteCriticalSection(mutex);
#else 
    pthread_mutex_destroy(mutex);
#endif
}

/** Locks a mutex */
void MorphoMutex_lock(MorphoMutex *mutex) {
#ifdef _WIN32
    EnterCriticalSection(mutex);
#else 
    pthread_mutex_lock(mutex);
#endif
}

/** Unlocks a mutex */
void MorphoMutex_unlock(MorphoMutex *mutex) {
#ifdef _WIN32
    LeaveCriticalSection(mutex);
#else 
    pthread_mutex_unlock(mutex);
#endif
}

/** Initializes a condition variable */
bool MorphoCond_init(MorphoCond *cond) {
#ifdef _WIN32
    InitializeConditionVariable(cond);
    return true;
#else 
    return (pthread_cond_init(cond, NULL)==0);
#endif
}

/** Clears a condition variable */
void MorphoCond_clear(MorphoCond *cond) {
#ifdef _WIN32
    /* Condition variables don't require cleanup on windows */
#else 
    pthread_cond_destroy(cond);
#endif
}

/** Signals condition variable, waking up a waiting thread */
void MorphoCond_signal(MorphoCond *cond) {
#ifdef _WIN32
    WakeConditionVariable(cond);
#else 
    pthread_cond_signal(cond);
#endif
}

/** Wake all threads waiting on a condition variable */
void MorphoCond_broadcast(MorphoCond *cond) {
#ifdef _WIN32
    WakeAllConditionVariable(cond);
#else 
    pthread_cond_broadcast(cond);
#endif
}

/** Waits for the condition variable to be signalled */
void MorphoCond_wait(MorphoCond *cond, MorphoMutex *mutex) {
#ifdef _WIN32
    SleepConditionVariableCS(cond, mutex, INFINITE);
#else 
    pthread_cond_wait(cond, mutex);
#endif
}

/* **********************************************************************
 * Time
 * ********************************************************************** */

/** Returns the system clock time in seconds. */
double platform_clock(void) {
#ifdef _WIN32
    SYSTEMTIME st;
    GetSystemTime(&st);
    double seconds = st.wMilliseconds*1e-3 +
                     st.wSecond +
                     st.wMinute*60 +
                     st.wHour*3600; 
    return seconds;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double) tv.tv_sec) + tv.tv_usec * 1e-6;
#endif
}

/** Sleep for a specified number of milliseconds */
void platform_sleep(int msecs) {
#ifdef _WIN32
    Sleep(msecs);
#else
    struct timespec t;
    t.tv_sec  =  msecs / 1000;
    t.tv_nsec = (msecs % 1000) * 1000000;
    nanosleep(&t, NULL);
#endif
}
