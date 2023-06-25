/** @file optimize.h
 *  @author T J Atherton
 *
 *  @brief Optimizer for compiled code
*/

#ifndef optimize_h
#define optimize_h

#define MORPHO_CORE
#include "core.h"
#include "compile.h"
#include "morpho.h"

#define INSTRUCTIONINDX_EMPTY -1
#define CODEBLOCKDEST_EMPTY -1
typedef int codeblockindx;

/** Keep track of register contents */
typedef struct {
    returntype contains;  // What does the register contain?
    indx id;              // index of global, register, constant etc.
    int used;             // Count how many times this has been used in the block
    instructionindx iix;  // Instruction that last wrote to this register
    codeblockindx block;  // Which block was responsible for the write?
    value type;           // Type checking
} reginfo;

/** Keep track of global contents */
typedef struct {
    returntype contains;  // What does the register contain?
    indx id;              // index of global, register, constant etc.
    int used;             // Count how many times this has been used in the block
    value type;           // Type checking
    value contents;       // Values written to the register
} globalinfo;

#define OPTIMIZER_AMBIGUOUS (MORPHO_OBJECT(NULL))
#define OPTIMIZER_ISAMBIGUOUS(a) ((MORPHO_ISOBJECT(a)) && (MORPHO_GETOBJECT(a)==NULL))

DECLARE_VARRAY(codeblockindx, codeblockindx)

/** Basic blocks for control flow graph */
typedef struct scodeblock {
    instructionindx start; /** First instruction in the block */
    instructionindx end; /** Last instruction in the block */
    
    codeblockindx dest[2]; /** Indices of destination blocks */
    varray_codeblockindx src; /** Indices of source blocks */
    dictionary retain; /** Registers retained by other blocks */
    
    int inbound; /** Count inbound */
    int visited; /** Count how many times optimizer has visited the block */
    
    int nreg; /** Size of register file */
    reginfo *reg; /** Register state at end */
    
    objectfunction *func; /** Function for this block */
    
    instructionindx ostart; /** First instruction in output */
    instructionindx oend; /** Last instruction in output */
    
    bool isroot; /** Is this a root block (i.e. a function call or similar) */
} codeblock;

DECLARE_VARRAY(codeblock, codeblock)

/** Optimizer data structure */
typedef struct {
    program *out;
    instructionindx iindx;    // Index to current instruction
    
    codeblockindx currentblock; // Current code block
    instruction current;     // Current instruction
    int op;                  // Current opcode
    registerindx overwrites; // Keep check of any register overwritten
    reginfo overwriteprev;   // Keep track of register contents before overwrite
    int nchanged;            // Count number of instructions changed in this block
    
    int maxreg;              // Maximum number of registers to track
    objectfunction *func;    // Current function
    
    dictionary functions;  // Functions and methods to process
    varray_codeblock cfgraph; // Control flow graph
    
    reginfo reg[MORPHO_MAXARGS];
    globalinfo *globals;
    
    vm *v;                   // We keep a VM to do things like constant folding etc.
    program *temp;           // Temporary program
    
    indx a;                  // Current annotation
    instructionindx aindx;   // Instruction counter for annotations
    indx aoffset;                 // Instruction offset for current annotation
    varray_debugannotation aout; // Annotations out
} optimizer;

bool optimize(program *prog);

void optimize_initialize(void);
void optimize_finalize(void);

#endif /* optimize_h */
