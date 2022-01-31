/** @file optimize.c
 *  @author T J Atherton
 *
 *  @brief Optimizer for compiled code
*/

#include "optimize.h"
#include "debug.h"

/* **********************************************************************
 * Display
 * ********************************************************************** */

/** Show the contents of a register */
void optimize_regshow(optimizer *opt) {
    unsigned int regmax = 0;
    for (unsigned int i=0; i<MORPHO_MAXARGS; i++) if (opt->reg[i].contains!=NOTHING) regmax=i;
    
    for (unsigned int i=0; i<=regmax; i++) {
        printf("|\tr%u : ", i);
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
    printf("\n");
}

/* **********************************************************************
* Utility functions
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

/** Indicates an instruction overwrites a register */
static inline void optimize_regoverwrite(optimizer *opt, registerindx reg) {
    opt->overwrites=reg;
}

/** Indicates no overwrite takes place */
static inline void optimize_nooverwrite(optimizer *opt) {
    opt->overwrites=REGISTER_UNALLOCATED;
}

/** Replaces the current instruction */
void optimize_replaceinstruction(optimizer *opt, instruction inst) {
    opt->out->code.data[opt->next]=inst;
    opt->current=inst;
    opt->op=DECODE_OP(inst);
}

/* **********************************************************************
* Optimization strategies
* ********************************************************************** */

typedef bool (*optimizationstrategyfn) (optimizer *opt);

typedef struct {
    int match;
    optimizationstrategyfn fn;
} optizationstrategy;

/** Identifies duplicate load instructions */
bool optimize_duplicate_load(optimizer *opt) {
    registerindx out = DECODE_A(opt->current);
    indx global = DECODE_Bx(opt->current);
    
    // Find if another register contains this global
    for (registerindx i=0; i<MORPHO_MAXARGS; i++) {
        if (opt->reg[i].contains==GLOBAL &&
            opt->reg[i].id==global) {
            
            if (i!=out) { // Replace with a move instruction and note the duplication
                optimize_replaceinstruction(opt, ENCODE_DOUBLE(OP_MOV, out, false, i));
            } else { // Register already contains this global
                optimize_replaceinstruction(opt, ENCODE_BYTE(OP_NOP));
            }
            return true;
        }
    }
    
    return false;
}

/** Trace back through duplicate registers */
registerindx optimize_findoriginalregister(optimizer *opt, registerindx reg) {
    registerindx out=reg;
    while (opt->reg[out].contains==REGISTER) out=(registerindx) opt->reg[out].id;
    return out;
}

/** Replaces duplicate registers  */
bool optimize_register_replacement(optimizer *opt) {
    if (opt->op<OP_ADD || opt->op>OP_LE) return false; // Quickly eliminate non-arithmetic instructions
    
    instruction instr=opt->current;
    registerindx a=DECODE_A(instr),
                 b=DECODE_B(instr),
                 c=DECODE_C(instr);
    bool Bc=DECODE_ISBCONSTANT(instr), Cc=DECODE_ISCCONSTANT(instr);
    
    if (!Bc) b=optimize_findoriginalregister(opt, b);
    if (!Cc) c=optimize_findoriginalregister(opt, c);
    
    optimize_replaceinstruction(opt, ENCODEC(opt->op, a, Bc, b, Cc, c));
    
    return false;
}

/* --------------------------
 * Table of strategies
 * -------------------------- */

#define OP_ANY -1

optizationstrategy strategies[] = {
    { OP_ANY, optimize_register_replacement },
    { OP_LGL, optimize_duplicate_load },
    { OP_END, NULL }
};

void optimize_optimizeinstruction(optimizer *opt) {
    for (optizationstrategy *s = strategies; s->match!=OP_END; s++) {
        if (s->match==OP_ANY || s->match==opt->op) {
            if ((*s->fn) (opt)) return;
        }
    }
}

/* **********************************************************************
* Decode instructions
* ********************************************************************** */

/** Fetches the instruction  */
void optimize_fetch(optimizer *opt) {
    optimize_nooverwrite(opt);
    
    opt->current=opt->out->code.data[opt->next];
    opt->op=DECODE_OP(opt->current);
}

/** Advance to next instruction */
void optimize_advance(optimizer *opt) {
    opt->next++;
}

/** Track contents of registers etc*/
void optimize_track(optimizer *opt) {
    instruction instr=opt->current;
    
    debug_disassembleinstruction(instr, opt->next, NULL, NULL);
    printf("\n");
    
    int op=DECODE_OP(instr);
    switch (op) {
        case OP_NOP: // Opcodes to ignore
        case OP_PUSHERR:
        case OP_POPERR:
        case OP_BREAK:
        case OP_END:
            break;
        case OP_MOV:
            optimize_reguse(opt, DECODE_B(instr));
            optimize_regcontents(opt, DECODE_A(instr), REGISTER, DECODE_B(instr));
            optimize_regoverwrite(opt, DECODE_A(instr));
            break;
        case OP_LCT:
            optimize_regcontents(opt, DECODE_A(instr), CONSTANT, DECODE_Bx(instr));
            optimize_regoverwrite(opt, DECODE_A(instr));
            break;
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
            if (!DECODE_ISBCONSTANT(instr)) optimize_reguse(opt, DECODE_B(instr));
            if (!DECODE_ISCCONSTANT(instr)) optimize_reguse(opt, DECODE_C(instr));
            optimize_regcontents(opt, DECODE_A(instr), VALUE, NOTHING);
            optimize_regoverwrite(opt, DECODE_A(instr));
            break;
        case OP_CALL:
            optimize_regcontents(opt, DECODE_A(instr), VALUE, NOTHING);
            optimize_regoverwrite(opt, DECODE_A(instr));
            break;
        case OP_LGL:
            optimize_regcontents(opt, DECODE_A(instr), GLOBAL, DECODE_Bx(instr));
            optimize_regoverwrite(opt, DECODE_A(instr));
            break;
        case OP_SGL:
            optimize_reguse(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), GLOBAL, DECODE_Bx(instr));
            break;
        case OP_PRINT:
            optimize_reguse(opt, DECODE_A(instr));
            break;
        default:
            UNREACHABLE("Opcode not supported in optimizer.");
    }
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
        optimize_fetch(&opt);
        optimize_optimizeinstruction(&opt);
        optimize_track(&opt);
        optimize_regshow(&opt);
        if (opt.overwrites!=REGISTER_UNALLOCATED) opt.reg[opt.overwrites].used=0;
        optimize_advance(&opt);
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
