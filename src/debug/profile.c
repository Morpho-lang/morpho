/** @file profile.c
 *  @author T J Atherton
 *
 *  @brief Profiler
 */

#include "profile.h"
#include "strng.h"
#include "platform.h"

/* **********************************************************************
* Profiler
* ********************************************************************** */

#ifdef MORPHO_PROFILER

#define PROFILER_NVALUES 2
#define PROFILER_VALUEPOSN 1

#define PROFILER_SAMPLINGINTERVAL (0.0001*CLOCKS_PER_SEC)

#define PROFILER_GLOBAL "(global)"
#define PROFILER_ANON   "(anonymous)"
#define PROFILER_GC     "(garbage collector)"

/** Record a sample */
void profiler_sample(profiler *profile, value func) {
    value v=MORPHO_INTEGER(1);
    
    if (dictionary_get(&profile->profile_dict, func, &v)) {
        v=MORPHO_INTEGER(MORPHO_GETINTEGERVALUE(v)+1);
    }
    
    dictionary_insert(&profile->profile_dict, func, v);
}

/** Profiler monitor thread */
MorphoThreadFnReturnType profiler_thread(void *arg) {
    vm *v = (vm *) arg;
    profiler *profile = v->profiler;
    if (!v->profiler) MorphoThread_exit();
    clock_t last = clock();
    clock_t time = last;
    
    while (true) {
        MorphoMutex_lock(&profile->profile_lock);
        bool quit=profile->profiler_quit;
        MorphoMutex_unlock(&profile->profile_lock);
        
        if (quit) MorphoThread_exit(); 
        
        while (time-last<PROFILER_SAMPLINGINTERVAL) time = clock();
        last = time;
        
        objectbuiltinfunction *infunction=v->fp->inbuiltinfunction;
        
        if (v->status==VM_INGC) {
            profiler_sample(profile, MORPHO_INTEGER(1));
        } else if (infunction) {
            profiler_sample(profile, MORPHO_OBJECT(infunction));
        } else {
            profiler_sample(profile, MORPHO_OBJECT(v->fp->function));
        }
    }
}

/** Initialize profiler data structure */
void profiler_init(profiler *profile, program *p) {
    dictionary_init(&profile->profile_dict);
    MorphoMutex_init(&profile->profile_lock);
    profile->profiler_quit=false;
    profile->program=p;
}

/** Clear profiler data structure */
void profiler_clear(profiler *profile) {
    MorphoMutex_clear(&profile->profile_lock);
    dictionary_clear(&profile->profile_dict);
}

/** Kills a profiler thread */
void profiler_kill(profiler *profile) {
    MorphoMutex_lock(&profile->profile_lock);
    profile->profiler_quit=true;
    MorphoMutex_unlock(&profile->profile_lock);
    MorphoThread_join(profile->profiler);
    MorphoThread_clear(profile->profiler);
}

/** Sorting function */
int profiler_sort(const void *a, const void *b) {
    value *aa = (value *) a;
    value *bb = (value *) b;
    return morpho_comparevalue(aa[PROFILER_VALUEPOSN], bb[PROFILER_VALUEPOSN]);
}

/** Returns the function name */
bool profiler_getname(value func, value *name, value *klass) {
    bool success=false;
    objectclass *k = NULL;
    
    if (MORPHO_ISINTEGER(func)) {
        *name = MORPHO_NIL; // In the Garbage collector
        success=true;
    } else if (MORPHO_ISBUILTINFUNCTION(func)) {
        *name = MORPHO_GETBUILTINFUNCTION(func)->name;
        k=MORPHO_GETBUILTINFUNCTION(func)->klass;
        success=true;
    } else if (MORPHO_ISFUNCTION(func)) {
        *name = MORPHO_GETFUNCTION(func)->name;
        k=MORPHO_GETFUNCTION(func)->klass;
        success=true;
    }
    
    *klass = (k ? k->name : MORPHO_NIL);
    return success;
}

/** Calculates the length to display */
size_t profiler_calculatelength(profiler *p, value func) {
    value name, klass;
    size_t length=0;
    
    if (profiler_getname(func, &name, &klass)) {
        if (MORPHO_ISSTRING(name)) {
            length+=MORPHO_GETSTRINGLENGTH(name);
        } else if (MORPHO_ISINTEGER(func)) {
            length+=strlen(PROFILER_GC);
        } else if (MORPHO_ISNIL(name)) {
            if (MORPHO_ISSAME(func, MORPHO_OBJECT(p->program->global))) length+=strlen(PROFILER_GLOBAL);
            else length+=strlen(PROFILER_ANON);
        }
                 
        if (MORPHO_ISSTRING(klass)) length+=MORPHO_GETSTRINGLENGTH(klass)+1; // extra length for the '.'
    }
    
    return length;
}

/** Display the function name and its sampling count */
void profiler_display(profiler *p, value func, vm *v) {
    value name, klass;
    if (profiler_getname(func, &name, &klass)) {
        if (MORPHO_ISSTRING(klass)) {
            morpho_printvalue(v, klass);
            morpho_printf(v, ".");
        }
        
        if (MORPHO_ISSTRING(name)) {
            morpho_printvalue(v, name);
        } else if (MORPHO_ISINTEGER(func)) {
            morpho_printf(v, PROFILER_GC);
        } else if (MORPHO_ISNIL(name)) {
            if (MORPHO_ISSAME(func, MORPHO_OBJECT(p->program->global))) morpho_printf(v, PROFILER_GLOBAL);
            else morpho_printf(v, PROFILER_ANON);
        }
    }
}

/** Report the outcome of profiling */
void profiler_report(profiler *profile, vm *v) {
    varray_value samples;
    varray_valueinit(&samples);
    
    for (unsigned int i=0; i<profile->profile_dict.capacity; i++) {
        if (!MORPHO_ISNIL(profile->profile_dict.contents[i].key)) {
            varray_valuewrite(&samples, profile->profile_dict.contents[i].key);
            varray_valuewrite(&samples, profile->profile_dict.contents[i].val);
        }
    }
    
    qsort(samples.data, samples.count/PROFILER_NVALUES, sizeof(value)*PROFILER_NVALUES, profiler_sort);
    
    /* Calculate the length of the names to display */
    long nsamples = 0;
    size_t maxlength = 0;
    for (unsigned int i=0; i<samples.count; i+=PROFILER_NVALUES) {
        size_t length = profiler_calculatelength(profile, samples.data[i]);
        if (length>maxlength) maxlength = length;
        
        nsamples += (long) MORPHO_GETINTEGERVALUE(samples.data[i+PROFILER_VALUEPOSN]);
    }
    
    // Now report the output
    morpho_printf(v, "===Profiler output: Execution took %.3f seconds with %ld samples===\n", ((double) profile->end - profile->start)/((double) CLOCKS_PER_SEC), nsamples);
    for (unsigned int i=0; i<samples.count; i+=PROFILER_NVALUES) {
        profiler_display(profile, samples.data[i], v); // Display the function or method
        size_t length = profiler_calculatelength(profile, samples.data[i]);
        for (int j=0; j<maxlength-length; j++) morpho_printf(v, " ");
        
        int fsamples = MORPHO_GETINTEGERVALUE(samples.data[i+PROFILER_VALUEPOSN]); // Display the count
        morpho_printf(v, " %.2f%% [%i samples]\n", 100.0*((float) fsamples)/nsamples, fsamples);
    }
    morpho_printf(v, "===\n");
    
    varray_valueclear(&samples);
}

/** Profile the execution of a program
 * @param[in] v - the virtual machine to use
 * @param[in] p - program to run
 * @returns true on success, false otherwise */
bool morpho_profile(vm *v, program *p) {
    profiler profile;

    profiler_init(&profile, p);
    
    if (MorphoThread_create(&profile.profiler, profiler_thread, v)) {
        UNREACHABLE("Unable to run profiler.");
    }
    
    v->profiler=&profile;
    
    profile.start=clock();
    bool success=morpho_run(v, p);
    profile.end=clock();
    
    profiler_kill(&profile);
    
    profiler_report(&profile, v);
    profiler_clear(&profile);
    
    return success;
}

#else

bool morpho_profile(vm *v, program *p) {
    return morpho_run(v, p);
}

#endif
