/** @file optimize.c
 *  @author T J Atherton
 *
 *  @brief Optimizer for compiled code
*/

#include "optimize.h"
#include "debug.h"

/* **********************************************************************
* Advance
* ********************************************************************** */

/** Checks if we're at the end of the program */
bool optimize_atend(optimizer *opt) {
    return (opt->next >= opt->out->code.count);
}

/** Sets the contents of a register */
static inline void optimize_regcontents(optimizer *opt, registerindx reg, returntype type, indx id) {
    opt->reg[reg].contains=type;
    opt->reg[reg].id=id;
}

/** Indicates an instruction uses a register */
static inline void optimize_reguse(optimizer *opt, registerindx reg) {
    opt->reg[reg].used++;
}

/** Fetches the next instruction and fills out appropriate data structures */
void optimize_advance(optimizer *opt) {
    opt->overwrites=REGISTER_UNALLOCATED;
    
    instruction instr=opt->out->code.data[opt->next];
    
    debug_disassembleinstruction(instr, opt->next, NULL, NULL);
    printf("\n");
    
    int op=DECODE_OP(instr);
    switch (op) {
        case OP_LCT:
            optimize_regcontents(opt, DECODE_A(instr), CONSTANT, DECODE_Bx(instr));
            break;
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
            if (!DECODE_ISBCONSTANT(instr)) optimize_reguse(opt, DECODE_B(instr));
            if (!DECODE_ISCCONSTANT(instr)) optimize_reguse(opt, DECODE_C(instr));
            optimize_regcontents(opt, DECODE_A(instr), VALUE, NOTHING);
            break;
        case OP_CALL:
            optimize_regcontents(opt, DECODE_A(instr), VALUE, NOTHING);
            break;
        case OP_LGL:
            optimize_regcontents(opt, DECODE_A(instr), GLOBAL, DECODE_Bx(instr));
            break;
        case OP_SGL:
            optimize_reguse(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), GLOBAL, DECODE_Bx(instr));
            break;
        case OP_PRINT:
            break;
        case OP_END:
            break;
        default:
            UNREACHABLE("Opcode not supported in optimizer.");
    }
    
    opt->current=instr;
    opt->next++;
}

void optimize_regshow(optimizer *opt) {
    unsigned int regmax = 0;
    for (unsigned int i=0; i<MORPHO_MAXARGS; i++) if (opt->reg[i].contains!=NOTHING) regmax=i;
    
    printf("----register contents:\n");
    for (unsigned int i=0; i<=regmax; i++) {
        printf("r%u : ", i);
        switch (opt->reg[i].contains) {
            case NOTHING: break;
            case REGISTER: printf("r%td", opt->reg[i].id); break;
            case GLOBAL: printf("g%td", opt->reg[i].id); break;
            case CONSTANT: printf("c%td", opt->reg[i].id); break;
            case UPVALUE: printf("u%td", opt->reg[i].id); break;
            case VALUE: printf("value"); break;
        }
        if (opt->reg[i].contains!=NOTHING) printf(" : %u", opt->reg[i].used);
        printf("\n");
    }
    printf("----\n");
}

/* **********************************************************************
* Data structures
* ********************************************************************** */

/** Clears the reginfo structure */
void optimize_regclear(optimizer *opt) {
    for (unsigned int i=0; i<MORPHO_MAXARGS; i++) {
        opt->reg[i].contains=NOTHING;
        opt->reg[i].id=0;
        opt->reg[i].used=0;
    }
}

/** Initializes optimizer data structure */
void optimize_init(optimizer *opt, program *prog) {
    opt->out=prog;
    optimize_regclear(opt);
    opt->next=0;
}

/** Clears optimizer data structure */
void optimize_clear(optimizer *opt) {
    
}

/* **********************************************************************
* Public interface
* ********************************************************************** */

/** Public interface to optimizer */
bool optimize(program *prog) {
    optimizer opt;
    
    optimize_init(&opt, prog);
    
    while (!optimize_atend(&opt)) {
        optimize_advance(&opt);
        optimize_regshow(&opt);
    }
    
    optimize_clear(&opt);
    
    return true;
}

/* **********************************************************************
* Initialization/Finalization
* ********************************************************************** */

/** Initializes the optimizer */
void optimize_initialize(void) {
}

/** Finalizes the optimizer */
void optimize_finalize(void) {
}
