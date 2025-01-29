/** @file build.h
 *  @author T J Atherton
 *
 *  @brief Define constants that choose how Morpho is built
 */

#include <float.h>

/* **********************************************************************
 * Version
 * ********************************************************************** */

#define MORPHO_VERSIONSTRING "0.6.2"

#define MORPHO_VERSION_MAJOR 0
#define MORPHO_VERSION_MINOR 6
#define MORPHO_VERSION_PATCH 2

/* **********************************************************************
 * Paths and file system
 * ********************************************************************** */

#ifndef MORPHO_HELP_BASEDIR
    #define MORPHO_HELP_BASEDIR "/usr/local/share/morpho/help"
#endif

#ifndef MORPHO_MODULE_BASEDIR
    #define MORPHO_MODULE_BASEDIR "/usr/local/share/morpho/modules"
#endif

#define MORPHO_HELPDIR "share/help"           // Package subdir. where help files are found
#define MORPHO_MODULEDIR "share/modules"      // Package subdir. where modules are found
#define MORPHO_EXTENSIONDIR "lib"             // Package subdir. where extensions are found

#define MORPHO_EXTENSION "morpho"             // File extension for morpho files
#define MORPHO_HELPEXTENSION "md"             // File extension for help files
#ifndef MORPHO_DYLIBEXTENSION
    #define MORPHO_DYLIBEXTENSION "dylib"     // File extension for extensions
#endif

#define MORPHO_DIRSEPARATOR '/'               // File directory separator

#define MORPHO_PACKAGELIST ".morphopackages"  // File in $HOME that contains package locations

/* **********************************************************************
 * Numeric tolerances
 * ********************************************************************** */

/** Value used to detect zero */
#define MORPHO_EPS DBL_EPSILON

/** Relative tolerance used to compare double precision equality */
#define MORPHO_RELATIVE_EPS DBL_EPSILON

/* **********************************************************************
 * Size limits
 * ********************************************************************** */

/** @brief Maximum length of a Morpho error string. */
#define MORPHO_ERRORSTRINGSIZE 255

/** @brief Default size of input buffer. */
#define MORPHO_INPUTBUFFERDEFAULTSIZE 1024

/** @brief Maximum file name length. */
#define MORPHO_MAXIMUMFILENAMELENGTH 255

/** @brief Size of the call frame stack. */
#define MORPHO_CALLFRAMESTACKSIZE 255

/** @brief Size of the error handler stack. */
#define MORPHO_ERRORHANDLERSTACKSIZE 64

/** @brief Maximum number of object types */
#define MORPHO_MAXIMUMOBJECTDEFNS 64

/** @brief Type numbers in a value must be less than this value */
#define MORPHO_MAXIMUMVALUETYPES 8

/** @brief Maximum number of arguments */
#define MORPHO_MAXARGS 255 /** @warning Note that this cannot easily be adjusted >255 without changing the instruction encoding */

/** @brief Maximum number of constants */
#define MORPHO_MAXCONSTANTS 65536 /** @warning Note that this cannot easily be adjusted >65536 without changing the instruction encoding */

/* **********************************************************************
* Performance
* ********************************************************************** */

/** @brief Build Morpho VM with computed gotos */
#define MORPHO_COMPUTED_GOTO

/** @brief Build Morpho VM with small but hacky value type [NaN boxing] */
#ifndef _NO_NAN_BOXING
#define MORPHO_NAN_BOXING
#endif
/** @brief Number of bytes to bind before GC first runs */
#define MORPHO_GCINITIAL 1024;
/** It seems that DeltaBlue benefits strongly from garbage collecting while the heap is still fairly small */

/** @brief Controls how rapidly the GC tries to collect garbage */
#define MORPHO_GCGROWTHFACTOR 2

/** @brief Initial size of the stack */
#define MORPHO_STACKINITIALSIZE 256

/** @brief Controls how rapidly the stack grows */
#define MORPHO_STACKGROWTHFACTOR 2

/** @brief Limits size of statically allocated arrays on the C stack */
#define MORPHO_MAXIMUMSTACKALLOC 256

/** @brief Default number of threads */
#define MORPHO_DEFAULTTHREADNUMBER 0

/** @brief Size of L1 cache line */
#define _MORPHO_L1CACHELINESIZE 128 // M1/M2 is 128; most intel are 64

/** @brief Pad data structures involved in multiprocessing */
#define _MORPHO_PADDING char __padding[_MORPHO_L1CACHELINESIZE]

/* **********************************************************************
 * Core library [options set in CMake]
 * ********************************************************************** */

/** Build with Matrix class using BLAS/LAPACK */
//#define MORPHO_INCLUDE_LINALG

/** Build with Sparse class */
//#define MORPHO_INCLUDE_SPARSE

/** Build with geometry classes */
//#define MORPHO_INCLUDE_GEOMETRY

/* **********************************************************************
 * Libraries
 * ********************************************************************** */

/** Use Apple's accelerate library for dense linear algebra */
#define MORPHO_LINALG_USE_ACCELERATE

/** Use the LAPACKE library for dense linear algebra */
//#define MORPHO_LINALG_USE_LAPACKE

/** Use CSparse for sparse matrix */
#define MORPHO_LINALG_USE_CSPARSE

/* **********************************************************************
* Debugging
* ********************************************************************** */

/** @brief Include debugging features */
#define MORPHO_DEBUG

/** @brief Print each instruction executed by the VM. */
//#define MORPHO_DEBUG_PRINT_INSTRUCTIONS

/** @brief Display syntax tree after parsing */
//#define MORPHO_DEBUG_DISPLAYSYNTAXTREE

/** @brief Display register allocations during compilation */
//#define MORPHO_DEBUG_DISPLAYREGISTERALLOCATION

/** @brief Disables garbage collector */
//#define MORPHO_DEBUG_DISABLEGARBAGECOLLECTOR

/** @brief Stress test garbage collector */
#ifdef _DEBUG_STRESSGARBAGECOLLECTOR
    #define MORPHO_DEBUG_STRESSGARBAGECOLLECTOR
#endif

/** @brief Log garbage collector */
//#define MORPHO_DEBUG_LOGGARBAGECOLLECTOR

/** @brief Check GC size tracking */
//#define MORPHO_DEBUG_GCSIZETRACKING

/** @brief Fill global constant table */
//#define MORPHO_DEBUG_FILLGLOBALCONSTANTTABLE

/** @brief Debug symbol table */
//#define MORPHO_DEBUG_SYMBOLTABLE

/** @brief Diagnose opcode usage */
//#define MORPHO_OPCODE_USAGE

/** @brief Buiild with profile support */
#define MORPHO_PROFILER
