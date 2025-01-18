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
   struct stat statbuf;
   if (stat(path, &statbuf) != 0)
       return 0;
   return (bool) S_ISDIR(statbuf.st_mode);
}

/* **********************************************************************
 * Dynamic libraries
 * ********************************************************************** */

MorphoDLHandle platform_dlopen(const char *path) {
#ifdef _WIN32
    return LoadLibrary((LPCSTR) path);
#else
    return dlopen(path, RTLD_LAZY);
#endif
}

void platform_dlclose(MorphoDLHandle handle) {
#ifdef _WIN32
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
}

void *platform_dlsym(MorphoDLHandle handle, const char *symbol) {
#ifdef _WIN32
    return (void *) GetProcAddress(handle, symbol);
#else 
    return dlsym(handle, symbol);
#endif
}
