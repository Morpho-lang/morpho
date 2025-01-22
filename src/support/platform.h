/** @file platform.h
 *  @author T J Atherton 
 *
 *  @brief Isolates platform dependent code in morpho
*/

#ifndef platform_h
#define platform_h

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>
#include "varray.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <pthread.h>
#endif

/* -------------------------------------------------------
 * Detecting platform name
 * ------------------------------------------------------- */

#define MORPHO_PLATFORM_MACOS                  "macos"
#define MORPHO_PLATFORM_LINUX                  "linux"
#define MORPHO_PLATFORM_UNIX                   "unix"
#define MORPHO_PLATFORM_WINDOWS                "windows"

const char *platform_name(void);

/* -------------------------------------------------------
 * Random numbers
 * ------------------------------------------------------- */

bool platform_randombytes(char *buffer, size_t nbytes);

/* -------------------------------------------------------
 * Complex numbers
 * ------------------------------------------------------- */

/** The windows C library has only partial support for C99 complex numbers, 
 *  which are implemented as a struct rather than a native type. While
 *  C99 complex functions are provided, basic arithmetic operations don't
 *  work. Hence we provide a type, as well as several macros and functions 
 *  to fill in missing functionality: */

#ifdef _WIN32
typedef _Dcomplex MorphoComplex;
#define MCBuild(re,im) _Cbuild(re, im)

MorphoComplex MCAdd(MorphoComplex a, MorphoComplex b);
MorphoComplex MCSub(MorphoComplex a, MorphoComplex b);
#define MCMul(a,b) (_Cmulcc(a,b))
#define MCScale(a,b) (_Cmulcr(a,b))
MorphoComplex MCDiv(MorphoComplex a, MorphoComplex b);
bool MCSame(MorphoComplex a, MorphoComplex b);
#else
typedef double complex MorphoComplex;
#define MCBuild(re,im) (re + I * im)

#define MCAdd(a, b) (a + b)
#define MCSub(a, b) (a - b)
#define MCMul(a, b) (a * b)
#define MCScale(a, b) (a * b)
#define MCDiv(a, b) (a / b)
#define MCSame(a, b) (a == b)
#endif

bool MCEq(MorphoComplex a, MorphoComplex b);

/* -------------------------------------------------------
 * Navigating the file system
 * ------------------------------------------------------- */

size_t platform_maxpathsize(void);
bool platform_setcurrentdirectory(const char *path);
bool platform_getcurrentdirectory(char *buffer, size_t size);
bool platform_gethomedirectory(char *buffer, size_t size);
bool platform_isdirectory(const char *path);

typedef struct {
#ifdef _WIN32
    bool isvalid; 
    WIN32_FIND_DATA finddata;
    HANDLE handle;
#else
    DIR *dir;
#endif
} MorphoDirContents;

bool platform_directorycontentsinit(MorphoDirContents *contents, const char *path);
void platform_directorycontentsclear(MorphoDirContents *contents);
bool platform_directorycontents(MorphoDirContents *contents, char *buffer, size_t size);

/* -------------------------------------------------------
 * Dynamic libraries
 * ------------------------------------------------------- */

#ifdef _WIN32
typedef HMODULE MorphoDLHandle;
#else 
typedef void* MorphoDLHandle;
#endif

MorphoDLHandle platform_dlopen(const char *path);
void platform_dlclose(MorphoDLHandle handle);
void *platform_dlsym(MorphoDLHandle handle, const char *symbol);

bool morpho_isdirectory(const char *path);

/* -------------------------------------------------------
 * Threads
 * ------------------------------------------------------- */

#ifdef _WIN32
typedef HANDLE             MorphoThread;
typedef CRITICAL_SECTION   MorphoMutex;
typedef CONDITION_VARIABLE MorphoCond;
typedef DWORD              MorphoThreadFnReturnType; 
typedef DWORD (*MorphoThreadFn)(void *);
#else 
typedef pthread_t          MorphoThread;
typedef pthread_mutex_t    MorphoMutex;
typedef pthread_cond_t     MorphoCond;
typedef void*              MorphoThreadFnReturnType;
typedef void* (*MorphoThreadFn)(void *);
#endif

DECLARE_VARRAY(MorphoThread, MorphoThread);

bool MorphoThread_create(MorphoThread *thread, MorphoThreadFn threadfn, void *ref);
void MorphoThread_join(MorphoThread thread);
void MorphoThread_clear(MorphoThread thread);
void MorphoThread_exit(void);

bool MorphoMutex_init(MorphoMutex *mutex);
void MorphoMutex_clear(MorphoMutex *mutex);
void MorphoMutex_lock(MorphoMutex *mutex);
void MorphoMutex_unlock(MorphoMutex *mutex);

bool MorphoCond_init(MorphoCond *cond);
void MorphoCond_clear(MorphoCond *cond);
void MorphoCond_signal(MorphoCond *cond);
void MorphoCond_broadcast(MorphoCond *cond);
void MorphoCond_wait(MorphoCond *cond, MorphoMutex *mutex);

/* -------------------------------------------------------
 * Time
 * ------------------------------------------------------- */

double platform_clock(void);
void platform_sleep(int msecs);

#endif
