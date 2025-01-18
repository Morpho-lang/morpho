/** @file platform.h
 *  @author T J Atherton 
 *
 *  @brief Isolates platform dependent code in morpho
*/

#ifndef platform_h
#define platform_h

#include <stdbool.h>
#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* -------------------------------------------------------
 * Detecting platform 
 * ------------------------------------------------------- */

#define MORPHO_PLATFORM_MACOS                  "macos"
#define MORPHO_PLATFORM_LINUX                  "linux"
#define MORPHO_PLATFORM_UNIX                   "unix"
#define MORPHO_PLATFORM_WINDOWS                "windows"

const char *platform_name(void);

/* -------------------------------------------------------
 * Navigating the file system
 * ------------------------------------------------------- */

size_t platform_maxpathsize(void);
bool platform_setcurrentdirectory(const char *path);
bool platform_getcurrentdirectory(char *buffer, size_t size);
bool platform_gethomedirectory(char *buffer, size_t size);
bool platform_isdirectory(const char *path);

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
