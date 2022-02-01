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

/** Optimizer data structure */
typedef struct {
    program *out;
    instructionindx next;    // Index to next instruction
    
    instruction current;     // Current instruction
    int op;                  // Current opcode
    registerindx overwrites; // Keep check of any register overwritten
    reginfo overwriteprev;   // Keep track of register contents before overwrite
    
    int maxreg;              // Maximum number of registers to track
    objectfunction *func;    // Current function
    
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
