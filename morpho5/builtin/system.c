/** @file system.c
 *  @author T J Atherton
 *
 *  @brief Built in class to provide access to the runtime
 */

#include <time.h>
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

/** Clock */
value System_clock(vm *v, int nargs, value *args) {
    clock_t time;
    time = clock();
    return MORPHO_FLOAT( ((double) time)/((double) CLOCKS_PER_SEC) );
}

/** Exit */
value System_exit(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, VM_EXIT);
    return MORPHO_NIL;
}

MORPHO_BEGINCLASS(System)
MORPHO_METHOD(SYSTEM_PLATFORM_METHOD, System_platform, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_VERSION_METHOD, System_version, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_CLOCK_METHOD, System_clock, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_EXIT_METHOD, System_exit, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void system_initialize(void) {
    builtin_addclass(SYSTEM_CLASSNAME, MORPHO_GETCLASSDEFINITION(System), MORPHO_NIL);
    
    morpho_defineerror(VM_EXIT, ERROR_EXIT, VM_EXIT_MSG);
}

void system_finalize(void) {
}
