/** @file core.h
 *  @author T J Atherton
 *
 *  @brief Data types for core Morpho components
*/

#ifndef core_h
#define core_h

#include <stdio.h>
#include <stddef.h>

typedef struct sprogram program;
typedef struct svm vm;

#include "error.h"
#include "random.h"
#include "varray.h"
#include "value.h"
#include "object.h"
#include "common.h"
#include "dictionary.h"
#include "builtin.h"

/* **********************************************************************
* Virtual machine instructions
* ********************************************************************** */

/** @brief A Morpho instruction
 *
 *  @details Each instruction fits into a 32 bit unsigned int with the following arrangement
 *  <pre>
 *         24      16      8       0
 *  .......|.......|.......|.......|
 *  **op**                           Opcode (gives 64 instructions)
 *        ***A****                   Operand A
 *                *        *         Select Constant (set) or Register for operands B & C
 *                 ***B**** ***C**** } Operands B & C
 *                fg**Bx************ } Operand Bx with two boolean flags f & g
 *                fg*sBx************ } Signed operand Bx with two boolean flags f & g
 * </pre>
 */

/** @brief Morpho instructions */
typedef unsigned int instruction;

/* ---------------------------------------------
 * Macros for encoding and decoding instructions
 * --------------------------------------------- */

/** Encodes an instruction with operands A, B and C. */
#define ENCODE(op, A, B, C) ( ((((unsigned) op) & 0xff) << 26) | ((A & 0xff) << 18) | ((B & 0xff) << 9) | (C & 0xff) )

/** Encodes an instruction with operands A, B and C also setting the constant flags Bc and Cc */
#define ENCODEC(op, A, Bc, B, Cc, C) ( ((((unsigned) op) & 0xff) << 26) | ((A & 0xff) << 18) | ((Bc & 0x1) << 17) | ((B & 0xff) << 9) | ((Cc & 0x1) << 8) | (C & 0xff) )

/** Encodes an instruction with operand A and long operand Bx */
#define ENCODE_LONG(op, A, Bx) ( ((((unsigned) op) & 0xff) << 26) | ((A & 0xff) << 18) | (Bx & 0xffff) )

/** Encodes an instruction with operand A and long operand Bx */
#define ENCODE_LONGFLAGS(op, A, f, g, Bx) ( ((((unsigned) op) & 0xff) << 26) | ((A & 0xff) << 18) | ((f & 0x1) << 17) | ((g & 0x1) << 16) | (Bx & 0xffff) )

/** Encodes an instruction with operand A */
#define ENCODE_SINGLE(op, A) ( ((((unsigned) op) & 0xff) << 26) | ((A & 0xff) << 18))

/** Encodes an instruction with operands A and B */
#define ENCODE_DOUBLE(op, A, Bc, B) ( ((((unsigned) op) & 0xff) << 26) | ((A & 0xff) << 18) | ((Bc & 0x1) << 17) | ((B & 0xff) << 9) )

/** Encodes an instruction with no operands */
#define ENCODE_BYTE(op) ((((unsigned) op) & 0xff) << 26)

/** Encodes an empty operand */
#define ENCODE_EMPTYOPERAND 0

/** Decode the opcode */
#define DECODE_OP(x) (x >> 26)

/** Decode operand A */
#define DECODE_A(x) ((x>>18) & (0xff))
/** Decode operand B */
#define DECODE_B(x) ((x>>9) & (0xff))
/** Decode operand C */
#define DECODE_C(x) (x & 0xff)

/** Is Operand B a constant or a register? */
#define DECODE_ISBCONSTANT(x) ((x) & (1<<17))
/** Is Operand C a constant or a register? */
#define DECODE_ISCCONSTANT(x) ((x) & (1<<8))

/** Decode long operand Bx */
#define DECODE_Bx(x) (x & 0xffff)

/** Decode signed long operand Bx */
#define DECODE_sBx(x) ((short) (x & 0xffff))

/** Decode flags F and G */
#define DECODE_F(x) ((bool) (x & (1<<17)))
#define DECODE_G(x) ((bool) (x & (1<<16)))


/* -----------------------------
 * Opcodes (built automatically)
 * ----------------------------- */

typedef enum {
  #define OPCODE(name) OP_##name,
  #include "opcodes.h"
  #undef OPCODE
} opcode;

/* **********************************************************************
 * Call frames
 * ********************************************************************** */

DECLARE_VARRAY(instruction, instruction);

/** @brief Maximum number of registers per call frame. */
#define MORPHO_MAXREGISTERS 255

/** @brief Index into instructions */
typedef indx instructionindx;

typedef struct {
    objectfunction *function;
    objectclosure *closure;
    ptrdiff_t roffset; // Offset of register from base
    instruction *pc;
    unsigned int stackcount;
    bool ret; // Should the interpreter return from this frame? 
} callframe;

/* **********************************************************************
 * Error handlers
 * ********************************************************************** */

typedef struct {
    callframe *fp;
    value dict;
} errorhandler;

/* **********************************************************************
 * Debug info
 * ********************************************************************** */

/** Annotations for the compiled code to link back to the source */
typedef struct {
    enum {
        DEBUG_FUNCTION, // Set the current function
        DEBUG_CLASS, // Set the current class
        DEBUG_MODULE, // Set the current module
        DEBUG_REGISTER, // Associates a symbol with a register
        DEBUG_ELEMENT, // Associates a sequence of instructions with a code element
        DEBUG_PUSHERR, // Push an error handler
        DEBUG_POPERR // Pop an error handler
    } type;
    union {
        struct {
            objectdictionary *handler;
        } errorhandler;
        struct {
            objectfunction *function;
        } function;
        struct {
            objectclass *klass;
        } klass;
        struct {
            value module;
        } module;
        struct {
            indx reg;
            value symbol;
        } reg;
        struct {
            int ninstr;
            int line;
            int posn;
        } element;
    } content;
} debugannotation;

DECLARE_VARRAY(debugannotation, debugannotation)

/* **********************************************************************
 * Programs
 * ********************************************************************** */

/** @brief Morpho code program and associated data */
struct sprogram {
    varray_instruction code; /** Compiled instructions */
    varray_debugannotation annotations; /** Information about how the code connects to the source */
    objectfunction *global;  /** Pseudofunction containing global data */
    unsigned int nglobals;
    object *boundlist; /** Linked list of static objects bound to this program */
    dictionary symboltable; /** The symbol table */
};

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

/** @brief A Morpho virtual machine and its current state */
struct svm {
    program *current; /** The current program being executed */
    
    varray_value globals; /** Global variables */
    varray_value stack; /** The stack */
    callframe frame[MORPHO_CALLFRAMESTACKSIZE]; /** The call frame stack */
    errorhandler errorhandlers[MORPHO_ERRORHANDLERSTACKSIZE]; /** Error handler stack */
    
    instruction *instructions; /* Base of instructions */
    value *konst; /* Current constant table */
    callframe *fp; /* Frame pointer saved on exit */
    errorhandler *ehp; /* Error handler pointer */
    
    error err; /** An error struct that will be filled out when an error occurs */
    
    object *objects; /** Linked list of objects */
    graylist gray; /** Graylist for garbage collection */
    size_t bound; /** Estimated size of bound bytes */
    size_t nextgc; /** Next garbage collection threshold */
    
    bool debug; /** Is the debugger active or not */
    
    objectupvalue *openupvalues; /** Linked list of open upvalues */
};

/* **********************************************************************
* Initializers
* ********************************************************************** */

void compile_initialize(void);
void compile_finalize(void);

#endif /* core_h */
