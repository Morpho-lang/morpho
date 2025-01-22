/** @file core.h
 *  @author T J Atherton
 *
 *  @brief Data types for core Morpho components
*/

#ifndef core_h
#define core_h

#include <stdio.h>
#include <stddef.h>

typedef struct svm vm;

#include "error.h"
#include "random.h"
#include "varray.h"
#include "value.h"
#include "object.h"
#include "common.h"
#include "dictionary.h"
#include "builtin.h"
#include "platform.h"

#include "program.h"

/* **********************************************************************
 * Instruction format
 * ********************************************************************** */

/** @brief A Morpho instruction
 *
 *  @details Each instruction fits into a 32 bit unsigned int with the following arrangement
 *  <pre>
 *         24      16      8       0
 *  .......|.......|.......|.......|
 *                          ***op*** Opcode (gives 256 instructions)
 *                  ***A****         Operand A
 *          ***B****                 } Operands B & C
 *  ***C****                         }
 *                                      OR
 *  **Bx************                 } Operand Bx as unsigned short
 *  *sBx************                 } Signed operand Bx
 *                                      OR
 *  ***********LBx**********         } 24 bit unsigned integer
 * </pre>
 */

/* ---------------------------------------------
 * Macros for encoding and decoding instructions
 * --------------------------------------------- */

/** Encodes an instruction with operands A, B and C. */
#define ENCODE(op, A, B, C) ( (((unsigned) op) & 0xff) | ((A & 0xff) << 8) | ((B & 0xff) << 16) | ((C & 0xff) << 24) )

/** Encodes an instruction with no operands */
#define ENCODE_BYTE(op) (((unsigned) op) & 0xff)

/** Encodes an instruction with operand A */
#define ENCODE_SINGLE(op, A) ( (((unsigned) op) & 0xff)  | ((A & 0xff) << 8))

/** Encodes an instruction with two operands A and B */
#define ENCODE_DOUBLE(op, A, B) ( (((unsigned) op) & 0xff)  | ((A & 0xff) << 8) | ((B & 0xff) << 16))

/** Encodes an instruction with operand A and long operand Bx */
#define ENCODE_LONG(op, A, Bx) ( (instruction) (((unsigned) op) & 0xff) | (instruction) ((A & 0xff) << 8) | (instruction) ((Bx & 0xffff) << 16) )

/** Encodes an instruction with operand Ax and extended operand Bl */
#define ENCODE_EXTENDED(op, Ax) ( (((unsigned) op) & 0xff) | ((A & 0xffffff) << 8) )

#define MASK_OP     (0xff)
#define MASK_A      (0xff << 8)
#define MASK_B      (0xff << 16)
#define MASK_C      (0xff << 24)

/** Encodes an empty operand */
#define ENCODE_EMPTYOPERAND 0

/** Decode the opcode */
#define DECODE_OP(x) (x & 0xff)

/** Decode operand A */
#define DECODE_A(x) ((x>>8) & (0xff))
/** Decode operand B */
#define DECODE_B(x) ((x>>16) & (0xff))
/** Decode operand C */
#define DECODE_C(x) ((x>>24) & (0xff))

/** Decode long operand Bx */
#define DECODE_Bx(x) ((x>>16) & (0xffff))

/** Decode signed long operand Bx */
#define DECODE_sBx(x) ((short) ((x>>16) & (0xffff)))

/* -----------------------------
 * Opcodes (built automatically)
 * ----------------------------- */

typedef enum {
  #define OPCODE(name) OP_##name,
  #include "opcodes.h"
  #undef OPCODE
} opcode;

/* **********************************************************************
 * Virtual machine
 * ********************************************************************** */

/** @brief Maximum number of registers per call frame. */
#define MORPHO_MAXREGISTERS 255

typedef struct {
    objectfunction *function;
    objectclosure *closure;
    ptrdiff_t roffset; // Offset of register from base
    instruction *pc;
    unsigned int stackcount;
    unsigned int returnreg; // Stores where any return value should be placed
    bool ret; // Should the interpreter return from this frame?
    int nopt; // Number of optional arguments called 
#ifdef MORPHO_PROFILER
    objectbuiltinfunction *inbuiltinfunction; // Keep track if we're in a built in function
#endif
} callframe;

/* **********************************************************************
 * Error handlers
 * ********************************************************************** */

typedef struct {
    callframe *fp;
    value dict;
} errorhandler;

/* **********************************************************************
 * Debugger backend
 * ********************************************************************** */

typedef struct {
    bool singlestep; /** Is single step mode on? */
    
    struct svm *currentvm; /** Current virtual machine on entry */
    int currentline; /** Record current line */
    objectfunction *currentfunc; /** Record current function */
    value currentmodule; /** Current module */
    instructionindx iindx; /** Record current instruction */
    
    error *err; /** Report errors */
    
    int nbreakpoints; /** Number of active breakpoints */
    varray_char breakpoints; /** Keep track of breakpoints */
} debugger;

/* **********************************************************************
 * Profiler
 * ********************************************************************** */

#ifdef MORPHO_PROFILER
/** @brief Morpho profiler */
typedef struct {
    MorphoThread profiler;
    MorphoMutex profile_lock;
    bool profiler_quit;
    dictionary profile_dict;
    clock_t start;
    clock_t end;
    program *program; 
} profiler;
#endif

/* **********************************************************************
 * Virtual machines
 * ********************************************************************** */

/** Gray list for garbage collection */
typedef struct {
    unsigned int graycount;
    unsigned int graycapacity;
    object **list;
} graylist;

/** @brief Highest register addressable in a window. */
#define VM_MAXIMUMREGISTERNUMBER 255

/** Varrays of vms */
DECLARE_VARRAY(vm, vm*)

/** @brief A Morpho virtual machine and its current state */
struct svm {
    program *current; /** The current program being executed */

    varray_value globals; /** Global variables */
    varray_value tlvars; /** Thread-local variables */
    varray_value stack; /** The stack */
    varray_value retain; /** List of values to retain across GC */
    callframe frame[MORPHO_CALLFRAMESTACKSIZE]; /** The call frame stack */
    errorhandler errorhandlers[MORPHO_ERRORHANDLERSTACKSIZE]; /** Error handler stack */

    instruction *instructions; /* Base of instructions */
    value *konst; /* Current constant table */
    callframe *fp; /* Frame pointer saved on exit */
    callframe *fpmax; /* Maximum value of the frame pointer */
    errorhandler *ehp; /* Error handler pointer */

    error err; /** An error struct that will be filled out when an error occurs */
    callframe *errfp; /** Record frame pointer when an error occured */

    object *objects; /** Linked list of objects */
    graylist gray; /** Graylist for garbage collection */
    size_t bound; /** Estimated size of bound bytes */
    size_t nextgc; /** Next garbage collection threshold */

    debugger *debug; 

#ifdef MORPHO_PROFILER
    profiler *profiler;
    enum { VM_RUNNING, VM_INGC } status; 
#endif
    
    objectupvalue *openupvalues; /** Linked list of open upvalues */
    
    vm *parent; /** Parent vm */
    varray_vm subkernels; /** Subkernels */
    
    morphoprintfn printfn; /** Print callback */
    void *printref; /** Print callback reference */
    varray_char buffer; /** Buffer for printing */
    
    morphoinputfn inputfn; /** Input callback */
    void *inputref; /** Input callback reference */
    
    morphowarningfn warningfn; /** Warning callback */
    void *warningref; /** Warning callback reference */
    
    morphodebuggerfn debuggerfn; /** Debugger callback */
    void *debuggerref; /** Debugger callback reference */
    
    _MORPHO_PADDING; /** Ensure subkernels do not cause false sharing */
};

/* **********************************************************************
* Initializers
* ********************************************************************** */

void compile_initialize(void);
void compile_finalize(void);

#endif /* core_h */
