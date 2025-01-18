/** @file platform.h
 *  @author T J Atherton 
 *
 *  @brief Isolates platform dependent code in morpho
*/

#ifndef platform_h
#define platform_h

#ifdef WIN32
#include <windows.h>
#endif

/* -------------------------------------------------------
 * Dynamic libraries
 * ------------------------------------------------------- */

#ifdef WIN32
typedef HMODULE MorphoDLHandle;
#else 
typedef void* MorphoDLHandle;
#endif

MorphoDLHandle platform_dlopen(const char *path);
void platform_dlclose(MorphoDLHandle handle);
void *platform_dlsym(MorphoDLHandle handle, const char *symbol)

bool morpho_isdirectory(const char *path);

#endif