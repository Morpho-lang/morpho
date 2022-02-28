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

/** Keep track of register contents */
typedef struct {
    returntype contains;  // What does the register contain?
    indx id;              // index of global, register, constant etc.
    int used;             // Count how many times this has been used
    instructionindx iix;  // Instruction that last wrote to this register
} reginfo;

#define CODEBLOCKDEST_EMPTY -1
typedef int codeblockindx;

/** Basic blocks for control flow graph */
typedef struct scodeblock {
    instructionindx start; /** First instruction in the block */
    instructionindx end; /** Last instruction in the block */
    codeblockindx dest[2]; /** Indices of destination blocks */
    int inbound; /** Count inbound */
    int visited; /** Count how many times optimizer has visited the block */
    int nreg; /** Size of register state */
    reginfo *reg; /** Register state at end */
    instructionindx ostart; /** First instruction in output */
    instructionindx oend; /** Last instruction in output */
} codeblock;

DECLARE_VARRAY(codeblock, codeblock)
DECLARE_VARRAY(codeblockindx, codeblockindx)

/** Optimizer data structure */
typedef struct {
    program *out;
    instructionindx iindx;    // Index to current instruction
    
    instruction current;     // Current instruction
    int op;                  // Current opcode
    registerindx overwrites; // Keep check of any register overwritten
    reginfo overwriteprev;   // Keep track of register contents before overwrite
    
    int maxreg;              // Maximum number of registers to track
    objectfunction *func;    // Current function
    
    varray_codeblock cfgraph; // Control flow graph
    
    reginfo reg[MORPHO_MAXARGS];
    reginfo *globals;
    
    vm *v;                   // We keep a VM to do things like constant folding etc.
    program *temp;           // Temporary program
    
    debugannotation *a;      // Current annotation
    debugannotation *amax;   // Last annotation
    unsigned int ai;         // Counter for current annotation
    unsigned int adel;       // Count number of deletions
} optimizer;

bool optimize(program *prog);

void optimize_initialize(void);
void optimize_finalize(void);

#endif /* optimize_h */
