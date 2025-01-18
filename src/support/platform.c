/** @file platform.c
 *  @author T J Atherton 
 *
 *  @brief Isolates platform dependent code in morpho */

/** Platform-dependent code in morpho arises in several ways: 
 *  - Navigating the file system 
 *  - APIs for opening dynamic libraries
 *  - APIs for using threads 
 * */

#include "platform.h"

#ifndef _WIN32
#include <dirent.h>
#include <sys/stat.h>
#include <dlfcn.h>
#endif

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
