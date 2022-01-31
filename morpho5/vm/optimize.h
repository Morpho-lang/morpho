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

/** Keep track of register contents */
typedef struct {
    returntype contains; // What does the register contain?
    indx id;             // Which global,
    int used;            // Count how many times this has been used
} reginfo;

/** Optimizer data structure */
typedef struct {
    program *out;
    instructionindx next;    // Index to next instruction
    
    instruction current;     // Current instruction
    int op;                  // Current opcode
    registerindx overwrites; // Keep check of any register overwritten
    reginfo overwriteprev;   // Keep track of register contents before overwrite
    
    reginfo reg[MORPHO_MAXARGS];
    
    
    reginfo *globals;
} optimizer;

bool optimize(program *prog);

void optimize_initialize(void);
void optimize_finalize(void);

#endif /* optimize_h */
