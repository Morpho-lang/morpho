/** @file system.c
 *  @author T J Atherton
 *
 *  @brief Built in class to provide access to the runtime
 */

#include <time.h>
#include "build.h"
#include "system.h"
#include "builtin.h"
#include "veneer.h"

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

/** Print */
value System_print(vm *v, int nargs, value *args) {
    for (int i=0; i<nargs; i++) morpho_printvalue(MORPHO_GETARG(args, i));
    return MORPHO_NIL;
}

/** Sleep for a specified number of milliseconds */
void system_sleep(int msecs) {
#ifdef WIN32
    Sleep (msecs);
#else
    struct timespec t;
    t.tv_sec  =  msecs / 1000;
    t.tv_nsec = (msecs % 1000) * 1000000;
    nanosleep (&t, NULL);
#endif
}

/** Sleep for a specified number of seconds */
value System_sleep(vm *v, int nargs, value *args) {
    if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double t;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &t)) {
            system_sleep((int) (1000*t));
        }
    } else morpho_runtimeerror(v, SLEEP_ARGS);
    
    return MORPHO_NIL;
}

/** Readline */
value System_readline(vm *v, int nargs, value *args) {
    char buffer[MORPHO_INPUTBUFFERDEFAULTSIZE];
    value out = MORPHO_NIL;
     
    if (fgets(buffer, sizeof(buffer), stdin)) {
        char *p = strchr(buffer, '\n');
        if (p) *p = '\0';
        
        out = object_stringfromcstring(buffer, strlen(buffer));
        if (MORPHO_ISSTRING(out)) morpho_bindobjects(v, 1, &out);
    }
    
    return out;
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
MORPHO_METHOD(MORPHO_PRINT_METHOD, System_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_SLEEP_METHOD, System_sleep, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_READLINE_METHOD, System_readline, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_EXIT_METHOD, System_exit, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void system_initialize(void) {
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    builtin_addclass(SYSTEM_CLASSNAME, MORPHO_GETCLASSDEFINITION(System), objclass);
    
    morpho_defineerror(SLEEP_ARGS, ERROR_HALT, SLEEP_ARGS_MSG);
    morpho_defineerror(VM_EXIT, ERROR_EXIT, VM_EXIT_MSG);
}

void system_finalize(void) {
}
