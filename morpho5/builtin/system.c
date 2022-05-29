/** @file system.c
 *  @author T J Atherton
 *
 *  @brief Built in class to provide access to the runtime
 */

#include "build.h"
#include "system.h"
#include "builtin.h"

/** Returns a platform description */
value System_platform(vm *v, int nargs, value *args) {
    char *platform = NULL;
    value ret = MORPHO_NIL;
    
#if __APPLE__
    platform = SYSTEM_MACOS;
#elif __linux__
    platform = SYSTEM_LINUX;
#elif __UNIX__
    platform = SYSTEM_UNIX;
#elif defined(_WIN32)
    platform = SYSTEM_WINDOWS;
#endif
    
    if (platform) {
        ret = object_stringfromcstring(platform, strlen(platform));
        morpho_bindobjects(v, 1, &ret);
    }
    
    return ret;
}

/** Returns the version descriptor */
value System_version(vm *v, int nargs, value *args) {
    value ret = object_stringfromcstring(MORPHO_VERSIONSTRING, strlen(MORPHO_VERSIONSTRING));
    morpho_bindobjects(v, 1, &ret);
    
    return ret;
}

MORPHO_BEGINCLASS(System)
MORPHO_METHOD(SYSTEM_PLATFORM, System_platform, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_VERSION, System_version, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void system_initialize(void) {
    builtin_addclass(SYSTEM_CLASSNAME, MORPHO_GETCLASSDEFINITION(System), MORPHO_NIL);
}

void system_finalize(void) {
}
