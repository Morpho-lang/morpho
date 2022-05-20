/** @file build.h
 *  @author T J Atherton
 *
 *  @brief Define constants that choose how Morpho is built
 */

/* **********************************************************************
 * Paths and file system
 * ********************************************************************** */

#define MORPHO_HELPDIRECTORY "/usr/local/share/morpho/help"
#define MORPHO_MODULEDIRECTORY "/usr/local/share/morpho/modules"

#define MORPHO_SEPARATOR "/"

#define MORPHO_EXTENSION ".morpho"

/* **********************************************************************
 * Features
 * ********************************************************************** */

#ifndef DEBUG
/** @brief Use coloring in output */
#define MORPHO_COLORTERMINAL
#endif

/* **********************************************************************
 * Language features
 * ********************************************************************** */

/** @brief Support string interpolation */
#define MORPHO_STRINGINTERPOLATION

/** @brief Newlines as statement terminators */
#define MORPHO_NEWLINETERMINATORS

/** @brief Enable compatibility with Lox language */
//#define MORPHO_LOXCOMPATIBILITY

/** Turn off features incompatible with lox */
#ifdef MORPHO_LOXCOMPATIBILITY
#undef MORPHO_NEWLINETERMINATORS
#endif

/* **********************************************************************
 * Numeric tolerances
 * ********************************************************************** */

/** Value used to detect zero */
#define MORPHO_EPS 1e-16

/* **********************************************************************
 * Size limits
 * ********************************************************************** */

/** @brief Maximum length of a Morpho error string. */
#define MORPHO_ERRORSTRINGSIZE 255

/** @brief Default size of input buffer. */
#define MORPHO_INPUTBUFFERDEFAULTSIZE 255

/** @brief Maximum file name length. */
#define MORPHO_MAXIMUMFILENAMELENGTH 255

/** @brief Size of the call frame stack. */
#define MORPHO_CALLFRAMESTACKSIZE 64

/** @brief Size of the error handler stack. */
#define MORPHO_ERRORHANDLERSTACKSIZE 64

/** @brief Maximum number of arguments */
#define MORPHO_MAXARGS 255

/** @brief Maximum number of constants */
#define MORPHO_MAXCONSTANTS 65536

/** @brief Maximum number of object types */
#define MORPHO_MAXIMUMOBJECTDEFNS 64

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

/** @brief Avoid using global variables (suitable for small programs only) */
//#define MORPHO_NOGLOBALS

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

/** @brief Log help file parsing */
//#define MORPHO_DEBUG_LOGHELPFILES

/** @brief Debug symbol table */
//#define MORPHO_DEBUG_SYMBOLTABLE

/** @brief Diagnose opcode usage */
//#define MORPHO_OPCODE_USAGE

/* **********************************************************************
* UI
* ********************************************************************** */

/** @brief Full welcome message in CLI */
// #define MORPHO_LONG_BANNER

/*************************************
 * GPU flags                         *
 ************************************/
#define CUDA_ACC
#ifdef CUDA_ACC
    #define GPU_ACC
#endif
#ifdef OPENCL_ACC
    #define GPU_ACC
#endif
