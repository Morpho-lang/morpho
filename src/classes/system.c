/** @file system.c
 *  @author T J Atherton
 *
 *  @brief Defines System class to provide access to the runtime and system
 */

#include <stdio.h>

#include "morpho.h"
#include "classes.h"
#include "system.h"
#include "platform.h"

/* **********************************************************************
 * System utility functions
 * ********************************************************************** */

/** Set arguments passed to morpho program */
static value arglist;

/** Set arguments with which the host program was called with */
void morpho_setargs(int argc, const char * argv[]) {
    if (!MORPHO_ISLIST(arglist)) return;
    objectlist *alist = MORPHO_GETLIST(arglist);
    for (int i=0; i<argc; i++) {
        value arg = object_stringfromcstring(argv[i], strlen(argv[i]));
        if (MORPHO_ISSTRING(arg)) list_append(alist, arg);
    }
}

/** Free arguments */
void system_freeargs(void) {
    if (!MORPHO_ISLIST(arglist)) return;
    objectlist *alist = MORPHO_GETLIST(arglist);
    
    for (int i=0; i<list_length(alist); i++) {
        value el;
        if (!list_getelement(alist, i, &el)) continue;
        morpho_freeobject(el);
    }
    morpho_freeobject(arglist);
}

/* **********************************************************************
 * System class
 * ********************************************************************* */

/** Returns a platform description */
value System_platform(vm *v, int nargs, value *args) {
    const char *platform = platform_name();
    value ret = MORPHO_NIL;
    
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
    return MORPHO_FLOAT(platform_clock());
}

/** Print */
value System_print(vm *v, int nargs, value *args) {
    for (int i=0; i<nargs; i++) morpho_printvalue(v, MORPHO_GETARG(args, i));
    return MORPHO_NIL;
}

/** Sleep for a specified number of seconds */
value System_sleep(vm *v, int nargs, value *args) {
    if (nargs==1 && MORPHO_ISNUMBER(MORPHO_GETARG(args, 0))) {
        double t;
        if (morpho_valuetofloat(MORPHO_GETARG(args, 0), &t)) {
            platform_sleep((int) (1000*t));
        }
    } else morpho_runtimeerror(v, SLEEP_ARGS);
    
    return MORPHO_NIL;
}

/** Readline */
value System_readline(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    varray_char buffer;
    varray_charinit(&buffer);
    
    if (morpho_readline(v, &buffer)) {
        out = object_stringfromvarraychar(&buffer);
        if (MORPHO_ISSTRING(out)) morpho_bindobjects(v, 1, &out);
    }

    varray_charclear(&buffer);
    
    return out;
}

/** Arguments passed to the process */
value System_arguments(vm *v, int nargs, value *args) {
    return arglist;
}

/** Exit */
value System_exit(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, VM_EXIT);
    return MORPHO_NIL;
}

/** Set working folder */
value System_setworkingfolder(vm *v, int nargs, value *args) {
    if (nargs==1 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        char *path = MORPHO_GETCSTRING(MORPHO_GETARG(args, 0));
        
        if (platform_setcurrentdirectory(path)) morpho_runtimeerror(v, SYS_STWRKDR);
    } else morpho_runtimeerror(v, STWRKDR_ARGS);
    
    return MORPHO_NIL;
}

/** Get working folder */
value System_workingfolder(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    
    size_t size = platform_maxpathsize();
    char str[size];

    if (platform_getcurrentdirectory(str, size)) {
        out = object_stringfromcstring(str, strlen(str));
        if (MORPHO_ISOBJECT(out)) {
            morpho_bindobjects(v, 1, &out);
        } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

/** Get current user's home folder */
value System_homefolder(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;

    size_t size = platform_maxpathsize();
    char str[size];

    if (platform_gethomedirectory(str, size)) {
        out = object_stringfromcstring(str, strlen(str));
        if (MORPHO_ISOBJECT(out)) {
            morpho_bindobjects(v, 1, &out);
        } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

MORPHO_BEGINCLASS(System)
MORPHO_METHOD(SYSTEM_PLATFORM_METHOD, System_platform, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_VERSION_METHOD, System_version, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_CLOCK_METHOD, System_clock, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, System_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_SLEEP_METHOD, System_sleep, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_READLINE_METHOD, System_readline, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_ARGUMENTS_METHOD, System_arguments, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_EXIT_METHOD, System_exit, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_SETWORKINGFOLDER_METHOD, System_setworkingfolder, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_WORKINGFOLDER_METHOD, System_workingfolder, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SYSTEM_HOMEFOLDER_METHOD, System_homefolder, BUILTIN_FLAGSEMPTY)
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
    morpho_defineerror(SYS_STWRKDR, ERROR_EXIT, SYS_STWRKDR_MSG);
    morpho_defineerror(STWRKDR_ARGS, ERROR_EXIT, STWRKDR_ARGS_MSG);
    
    objectlist *alist = object_newlist(0, NULL);
    if (alist) arglist = MORPHO_OBJECT(alist);
    
    morpho_addfinalizefn(system_finalize);
}

void system_finalize(void) {
    system_freeargs();
}
