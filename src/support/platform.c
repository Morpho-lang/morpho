/** @file platform.c
 *  @author T J Atherton 
 *
 *  @brief Isolates platform dependent code in morpho */

/** Platform-dependent code in morpho arises in several ways: 
 *  - Navigating the file system 
 *  - APIs for opening dynamic libraries
 *  - APIs for using threads 
 *  - Functions that involve time */

#include "platform.h"

#ifndef _WIN32
#define _POSIX_C_SOURCE 199309L

#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include <dlfcn.h>
#endif

/* **********************************************************************
 * Platform name
 * ********************************************************************** */

const char *platform_name(void) {
#if __APPLE__
    return PLATFORM_MACOS;
#elif __linux__
    return PLATFORM_LINUX;
#elif __UNIX__
    return PLATFORM_UNIX;
#elif defined(_WIN32)
    return PLATFORM_WINDOWS;
#endif
    return NULL; // Unrecognized platform
}

/* **********************************************************************
 * File system functions
 * ********************************************************************** */

/* Tells if an object at path corresponds to a directory */
bool morpho_isdirectory(const char *path) {
   //struct stat statbuf;
   //if (stat(path, &statbuf) != 0)
   //    return 0;
   //return (bool) S_ISDIR(statbuf.st_mode);
   return false; 
}

/** Sets the current working directory to path */
bool platform_setcurrentdirectory(const char *path) {
#ifdef _WIN32
    return SetCurrentDirectory(path);
#else
    return chdir(path)==0;
#endif
}

size_t platform_sizepath(const char *path) {
#ifdef _WIN32 
    return (size_t) MAX_PATH;
#else
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
    gettimeofday (&tv, NULL);
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