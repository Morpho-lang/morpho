/** @file platform.h
 *  @author T J Atherton 
 *
 *  @brief Isolates platform dependent code in morpho
*/

#ifndef platform_h
#define platform_h

#include <stdbool.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* -------------------------------------------------------
 * Navigating the file system
 * ------------------------------------------------------- */

bool platform_setcurrentdirectory(const char *path);
bool platform_getcurrentdirectory(char *buffer, size_t size);

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
 * Time
 * ------------------------------------------------------- */

double platform_clock(void);
void platform_sleep(int msecs);

#endif