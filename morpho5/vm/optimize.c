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
    for (unsigned int i=0; i<opt->maxreg; i++) if (opt->reg[i].contains!=NOTHING) regmax=i;
    
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
    return (opt->iindx >= opt->out->code.count);
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
    opt->overwriteprev=opt->reg[reg];
    opt->overwrites=reg;
}

/** Indicates an instruction uses a global */
static inline void optimize_loadglobal(optimizer *opt, indx ix) {
    if (opt->globals) opt->globals[ix].used++;
}

/** Indicates no overwrite takes place */
static inline void optimize_nooverwrite(optimizer *opt) {
    opt->overwrites=REGISTER_UNALLOCATED;
}

/** Fetches an instruction at a given indx */
instruction optimize_fetchinstructionat(optimizer *opt, indx ix) {
    return opt->out->code.data[ix];
}

/** Replaces an instruction at a given indx */
void optimize_replaceinstructionat(optimizer *opt, indx ix, instruction inst) {
    opt->out->code.data[ix]=inst;
}

/** Replaces the current instruction */
void optimize_replaceinstruction(optimizer *opt, instruction inst) {
    opt->out->code.data[opt->iindx]=inst;
    opt->current=inst;
    opt->op=DECODE_OP(inst);
}

/** Trace back through duplicate registers */
registerindx optimize_findoriginalregister(optimizer *opt, registerindx reg) {
    registerindx out=reg;
    while (opt->reg[out].contains==REGISTER) out=(registerindx) opt->reg[out].id;
    return out;
}

/** Finds if a given register, or one that it duplicates, contains a constant. If so returns the constant indx and returns true */
bool optimize_findconstant(optimizer *opt, registerindx reg, indx *out) {
    registerindx r = optimize_findoriginalregister(opt, reg);
    
    if (opt->reg[r].contains==CONSTANT) {
        *out = opt->reg[r].id;
        return true;
    }
    return false;
}

/** Adds a constant to the current constant table*/
bool optimize_addconstant(optimizer *opt, value val, indx *out) {
    unsigned int k;
    // Does the constant already exist?
    if (varray_valuefindsame(&opt->func->konst, val, &k)) {
        *out=k; return true;
    }
    varray_valuewrite(&opt->func->konst, val);
    *out=opt->func->konst.count-1;
    
    return true;
}

/* **********************************************************************
 * Evaluation using the VM
 * ********************************************************************** */

bool optimize_evaluateprogram(optimizer *opt, instruction *list, registerindx dest, value *out) {
    bool success=false;
    objectfunction *storeglobal=opt->temp->global; // Retain the old global function
    opt->temp->global=opt->func; // Patch in function
    
    varray_instruction *code = &opt->temp->code;
    code->count=0; // Clear the program
    for (instruction *ins = list; ; ins++) { // Load the list of instructions into the program
        varray_instructionwrite(code, *ins);
        if (DECODE_OP(*ins)==OP_END) break;
    }
    
    if (morpho_run(opt->v, opt->temp)) { // Run the program and extract output
        if (out && dest< opt->v->stack.count) *out = opt->v->stack.data[dest];
        success=true;
    }
    opt->temp->global=storeglobal; // Restore the global function
    
    return success;
}

/* **********************************************************************
* Optimization strategies
* ********************************************************************** */

typedef bool (*optimizationstrategyfn) (optimizer *opt);

typedef struct {
    int match;
    optimizationstrategyfn fn;
} optimizationstrategy;

/** Identifies duplicate load instructions */
bool optimize_duplicate_load(optimizer *opt) {
    registerindx out = DECODE_A(opt->current);
    indx global = DECODE_Bx(opt->current);
    
    // Find if another register contains this global
    for (registerindx i=0; i<opt->maxreg; i++) {
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
    
    return false; // This allows other optimization strategies to intervene after
}

/** Searches to see if an expression has already been calculated  */
bool optimize_subexpression_elimination(optimizer *opt) {
    if (opt->op<OP_ADD || opt->op>OP_LE) return false; // Quickly eliminate non-arithmetic instructions
    static instruction mask = (MASK_OP | MASK_B | MASK_Bc | MASK_C | MASK_Cc);
    
    // Find if another register contains the same calculated value.
    for (registerindx i=0; i<opt->maxreg; i++) {
        if (opt->reg[i].contains==VALUE) {
            instruction comp = optimize_fetchinstructionat(opt, opt->reg[i].iix);
            
            if ((comp & mask)==(opt->current & mask)) {
                optimize_replaceinstruction(opt, ENCODE_DOUBLE(OP_MOV, DECODE_A(opt->current), false, i));
                return true;
            }
        }
    }
    return false;
}

/** Optimize unconditional branches */
bool optimize_branch_optimization(optimizer *opt) {
    int sbx=DECODE_sBx(opt->current);
    
    if (sbx==0) {
        optimize_replaceinstruction(opt, ENCODE_BYTE(OP_NOP));
        return true;
    }
    
    return false;
}

/** Replaces duplicate registers  */
bool optimize_constant_folding(optimizer *opt) {
    if (opt->op<OP_ADD || opt->op>OP_LE) return false; // Quickly eliminate non-arithmetic instructions
    
    instruction instr=opt->current;
    bool Bc=DECODE_ISBCONSTANT(instr), Cc=DECODE_ISCCONSTANT(instr);
    indx left=DECODE_B(instr), right=DECODE_C(instr);
    
    if (!Bc) Bc=optimize_findconstant(opt, DECODE_B(instr), &left);
    if (!Bc) return false;
    if (!Cc) Cc=optimize_findconstant(opt, DECODE_C(instr), &right);
    
    if (Cc) {
        // A program that evaluates the required op with the selected constants.
        instruction ilist[] = {
            ENCODE_LONG(OP_LCT, 0, left),
            ENCODE_LONG(OP_LCT, 1, right),
            ENCODE(opt->op, 0, 0, 1),
            ENCODE_BYTE(OP_END)
        };
        
        value out;
        if (optimize_evaluateprogram(opt, ilist, 0, &out)) {
            indx nkonst;
            if (MORPHO_ISOBJECT(out)) UNREACHABLE("Optimizer encountered object while constant folding");
            if (optimize_addconstant(opt, out, &nkonst)) {
                optimize_replaceinstruction(opt, ENCODE_LONG(OP_LCT, DECODE_A(instr), nkonst));
                return true;
            }
        }
    }
    
    return false;
}

/** Deletes unused globals  */
bool optimize_unused_global(optimizer *opt) {
    indx gid=DECODE_Bx(opt->current);
    if (!opt->globals[gid].used) { // If the global is unused, do not store to it
        optimize_replaceinstruction(opt, ENCODE_BYTE(OP_NOP));
    }
    
    return false;
}

/** Removes expressions generating an unused expression at the end */
bool optimize_eliminate_unused_at_end(optimizer *opt) {
    for (registerindx i=0; i<opt->maxreg; i++) {
        if (opt->reg[i].contains!=NOTHING &&
            opt->reg[i].used==0) {
            // Should check for side effects!
            optimize_replaceinstructionat(opt, opt->reg[i].iix, ENCODE_BYTE(OP_NOP));
        }
    }
    return false;
}

/* --------------------------
 * Table of strategies
 * -------------------------- */

#define OP_ANY -1
#define OP_LAST OP_END+1

optimizationstrategy firstpass[] = {
    { OP_ANY, optimize_register_replacement },
    { OP_ANY, optimize_subexpression_elimination },
    { OP_ANY, optimize_constant_folding },
    { OP_LGL, optimize_duplicate_load },
    { OP_B, optimize_branch_optimization },
    { OP_END, optimize_eliminate_unused_at_end },
    { OP_LAST, NULL }
};

optimizationstrategy secondpass[] = {
    { OP_ANY, optimize_register_replacement },
    { OP_ANY, optimize_subexpression_elimination },
    { OP_ANY, optimize_constant_folding },
    { OP_LGL, optimize_duplicate_load },
    { OP_SGL, optimize_unused_global },
    { OP_END, optimize_eliminate_unused_at_end },
    { OP_LAST, NULL }
};

/** Apply optimization strategies to the current instruction */
void optimize_optimizeinstruction(optimizer *opt, optimizationstrategy *strategies) {
    if (opt->op==OP_NOP) return;
    for (optimizationstrategy *s = strategies; s->match!=OP_LAST; s++) {
        if (s->match==OP_ANY || s->match==opt->op) {
            if ((*s->fn) (opt)) return;
        }
    }
}

/** Process overwrite */
void optimize_overwrite(optimizer *opt) {
    if (opt->overwrites==REGISTER_UNALLOCATED) return;
    
    // Detect unused expression
    if (opt->overwriteprev.contains!=NOTHING &&
        opt->overwriteprev.used==0) {
        
        optimize_replaceinstructionat(opt, opt->reg[opt->overwrites].iix, ENCODE_BYTE(OP_NOP));
    }
    
    opt->reg[opt->overwrites].used=0;
    opt->reg[opt->overwrites].iix=opt->iindx;
}

/* **********************************************************************
* Handling code annotations
* ********************************************************************** */

void optimize_advanceannotationtoelement(optimizer *opt) {
    while (opt->a<opt->amax && opt->a->type!=DEBUG_ELEMENT) opt->a++;
    if (opt->a>=opt->amax) opt->a=NULL;
}

/** Start annotation */
void optimize_restartannotation(optimizer *opt) {
    opt->a=opt->out->annotations.data;
    opt->amax=opt->out->annotations.data+opt->out->annotations.count;
    opt->ai=0;
    opt->adel=0;
    optimize_advanceannotationtoelement(opt);
}

/** Reset annotation instruction counter */
void optimize_resetannotationinstructioncounter(optimizer *opt) {
    opt->ai=0;
    opt->adel=0;
}

/** Mark instruction for deletion */
void optimize_annotationdeleteinstruction(optimizer *opt) {
    opt->adel+=1;
}

/** Advance annotation by one instruction */
void optimize_advanceannotation(optimizer *opt) {
    if (!opt->a) return;

    opt->ai++; // Advance instruction counter for annotations
    
    /* Check if we need to advance to the next annotation */
    if (opt->ai>=opt->a->content.element.ninstr) {
        opt->a->content.element.ninstr-=opt->adel;
        opt->a++;
        
        optimize_resetannotationinstructioncounter(opt);
        
        optimize_advanceannotationtoelement(opt);
    }
    
    return;
}

/* **********************************************************************
* Decode instructions
* ********************************************************************** */

/** Fetches the instruction  */
void optimize_fetch(optimizer *opt) {
    optimize_nooverwrite(opt);
    
    opt->current=opt->out->code.data[opt->iindx];
    opt->op=DECODE_OP(opt->current);
}

/** Returns the index of the instruction currently decoded */
instructionindx optimizer_currentindx(optimizer *opt) {
    return opt->iindx;
}

/** Advance to next instruction */
void optimize_advance(optimizer *opt) {
    optimize_advanceannotation(opt);
    opt->iindx++;
}

/** Move to next instruction; ignore annotations */
void optimize_next(optimizer *opt) {
    opt->iindx++;
}

/** Move to a different instruction. */
void optimize_moveto(optimizer *opt, instructionindx indx) {
    opt->iindx=indx;
    optimize_fetch(opt);
}

/** Track contents of registers etc*/
void optimize_track(optimizer *opt) {
    instruction instr=opt->current;

#ifdef MORPHO_DEBUG_LOGOPTIMIZER
    debug_disassembleinstruction(instr, opt->iindx, NULL, NULL);
    printf("\n");
#endif
    
    int op=DECODE_OP(instr);
    switch (op) {
        case OP_NOP: // Opcodes to ignore
        case OP_PUSHERR:
        case OP_POPERR:
        case OP_BREAK:
        case OP_END:
        case OP_B:
            break;
        case OP_MOV:
            optimize_reguse(opt, DECODE_B(instr));
            optimize_regoverwrite(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), REGISTER, DECODE_B(instr));
            break;
        case OP_LCT:
            optimize_regoverwrite(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), CONSTANT, DECODE_Bx(instr));
            break;
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
        case OP_POW:
        case OP_EQ:
        case OP_NEQ:
        case OP_LT:
        case OP_LE:
            if (!DECODE_ISBCONSTANT(instr)) optimize_reguse(opt, DECODE_B(instr));
            if (!DECODE_ISCCONSTANT(instr)) optimize_reguse(opt, DECODE_C(instr));
            optimize_regoverwrite(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), VALUE, NOTHING);
            break;
        case OP_NOT:
            if (!DECODE_ISBCONSTANT(instr)) optimize_reguse(opt, DECODE_B(instr));
            optimize_regoverwrite(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), VALUE, NOTHING);
            break;
        case OP_BIF:
            optimize_reguse(opt, DECODE_A(instr));
            break;
        case OP_CALL:
        {
            registerindx a = DECODE_A(instr);
            registerindx b = DECODE_B(instr);
            optimize_reguse(opt, a);
            for (unsigned int i=0; i<b; i++) opt->reg[a+i+1].contains=NOTHING; // call uses and overwrites arguments.
            optimize_regoverwrite(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), VALUE, NOTHING);
        }
            break;
        case OP_RETURN:
            if (DECODE_A(instr)>0) optimize_reguse(opt, DECODE_B(instr));
            break;
        case OP_LGL:
            optimize_regoverwrite(opt, DECODE_A(instr));
            optimize_loadglobal(opt, DECODE_Bx(instr));
            optimize_regcontents(opt, DECODE_A(instr), GLOBAL, DECODE_Bx(instr));
            break;
        case OP_SGL:
            optimize_reguse(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), GLOBAL, DECODE_Bx(instr));
            break;
        case OP_PRINT:
            if (!DECODE_ISBCONSTANT(instr)) optimize_reguse(opt, DECODE_B(instr));
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
        opt->reg[i].iix=0;
    }
}

/** Restart from a designated instruction */
void optimize_restart(optimizer *opt, instructionindx start) {
    optimize_regclear(opt);
    opt->iindx=start;
    
    opt->a=NULL;
    opt->amax=NULL;
}

/** Sets the current function */
void optimize_setfunction(optimizer *opt, objectfunction *func) {
    opt->maxreg=func->nregs;
    opt->func=func;
}

/** Initializes optimizer data structure */
void optimize_init(optimizer *opt, program *prog) {
    opt->out=prog;
    opt->globals=MORPHO_MALLOC(sizeof(reginfo)*(opt->out->nglobals+1));
    for (unsigned int i=0; i<opt->out->nglobals; i++) {
        opt->globals[i].contains=NOTHING;
        opt->globals[i].used=0;
    }
    opt->maxreg=MORPHO_MAXARGS;
    optimize_setfunction(opt, prog->global);
    optimize_restart(opt, 0);
    
    varray_codeblockinit(&opt->cfgraph);
    
    opt->a=NULL;
    opt->amax=NULL;
    
    opt->v=morpho_newvm();
    opt->temp=morpho_newprogram();
}

/** Clears optimizer data structure */
void optimize_clear(optimizer *opt) {
    if (opt->globals) MORPHO_FREE(opt->globals);
    
    varray_codeblockclear(&opt->cfgraph);
    
    if (opt->v) morpho_freevm(opt->v);
    if (opt->temp) morpho_freeprogram(opt->temp);
}

/* **********************************************************************
* Control Flow graph
* ********************************************************************** */

DEFINE_VARRAY(codeblock, codeblock);
DEFINE_VARRAY(codeblockindx, codeblockindx);

/** Initialize a code block */
void optimize_initcodeblock(codeblock *block, instructionindx start) {
    block->start=start;
    block->end=start;
    block->inbound=0;
    block->dest[0]=CODEBLOCKDEST_EMPTY;
    block->dest[1]=CODEBLOCKDEST_EMPTY;
    block->visited=0;
}

/** Create a new block that starts at a given index and add it to the control flow graph; returns a handle to this block */
codeblockindx optimize_newblock(optimizer *opt, instructionindx start) {
    codeblock new;
    optimize_initcodeblock(&new, start);
    varray_codeblockwrite(&opt->cfgraph, new);
    return opt->cfgraph.count-1;
}

/** Get a block's starting point */
instructionindx optimize_getstart(optimizer *opt, codeblockindx handle) {
    return opt->cfgraph.data[handle].start;
}

/** Get a block's end point */
instructionindx optimize_getend(optimizer *opt, codeblockindx handle) {
    return opt->cfgraph.data[handle].end;
}

/** Set a block's end point */
void optimize_setend(optimizer *opt, codeblockindx handle, instructionindx end) {
    opt->cfgraph.data[handle].end=end;
}

/** Get a block's visited index */
int optimize_getvisited(optimizer *opt, codeblockindx handle) {
    return opt->cfgraph.data[handle].visited;
}

/** Increment the inbound counter on a block */
void optimize_incinbound(optimizer *opt, codeblockindx handle) {
    opt->cfgraph.data[handle].inbound+=1;
}

/** Get how many blocks link to this one */
int optimize_getinbound(optimizer *opt, codeblockindx handle) {
    return opt->cfgraph.data[handle].inbound;
}

/** Mark the code block as visited */
void optimize_visit(optimizer *opt, codeblockindx handle) {
    opt->cfgraph.data[handle].visited+=1;
}

/** Adds a destination
 * @param[in] opt - optimizer
 * @param[in] handle - block to add the destination to
 * @param[in] dest - destination block to add */
void optimize_adddest(optimizer *opt, codeblockindx handle, codeblockindx dest) {
    codeblock *block = &opt->cfgraph.data[handle];
    int i;
    for (i=0; i<2; i++) {
        if (block->dest[i]==CODEBLOCKDEST_EMPTY) {
            block->dest[i]=dest;
            optimize_incinbound(opt, dest);
            return;
        }
    }
    
    UNREACHABLE("Too many destinations in code block.");
}

/** Clear block destination */
void optimize_cleardest(optimizer *opt, codeblockindx handle) {
    opt->cfgraph.data[handle].dest[0]=CODEBLOCKDEST_EMPTY;
    opt->cfgraph.data[handle].dest[1]=CODEBLOCKDEST_EMPTY;
}

/** Copy block destination */
void optimize_copydest(optimizer *opt, codeblockindx src, codeblockindx dest) {
    opt->cfgraph.data[dest].dest[0]=opt->cfgraph.data[src].dest[0];
    opt->cfgraph.data[dest].dest[1]=opt->cfgraph.data[src].dest[1];
}

/** Copy block destinations to work list */
void optimize_desttoworklist(optimizer *opt, codeblockindx src, varray_codeblockindx *worklist) {
    for (int i=0; i<2; i++) {
        codeblockindx dest = opt->cfgraph.data[src].dest[i];
        if (dest!=CODEBLOCKDEST_EMPTY) varray_codeblockindxwrite(worklist, dest);
    }
}

/** Splits a block into two
 * @param[in] opt - optimizer
 * @param[in] handle - block to split
 * @param[in] split - instruction index to split at
 * @returns handle of new block */
codeblockindx optimize_splitblock(optimizer *opt, codeblockindx handle, instructionindx split) {
    instructionindx start = optimize_getstart(opt, handle),
                    end = optimize_getend(opt, handle);
    
    if (split==start) return handle;
    if (split<start || split>end) UNREACHABLE("Splitting an invalid block");
    
    codeblockindx new = optimize_newblock(opt, split);
    optimize_setend(opt, new, end);
    optimize_setend(opt, handle, split-1);
    optimize_copydest(opt, handle, new); // New block carries over destinations
    optimize_cleardest(opt, handle);    // } Old block points to new block
    optimize_adddest(opt, handle, new); // }
    
    return new;
}

/** Finds a block with instruction indx inside
 * @param[in] opt - optimizer
 * @param[in] indx - index to find
 * @param[out] handle - block handle if found
 * @returns true if found, false otherwise */
bool optimize_findblock(optimizer *opt, instructionindx indx, codeblockindx *handle) {
    for (codeblockindx i=0; i<opt->cfgraph.count; i++) {
        if (indx>=opt->cfgraph.data[i].start &&
            indx<=opt->cfgraph.data[i].end) {
            *handle=i;
            return true;
        }
    }
    return false;
}

/** Processes a branch instruction.
 * @details Finds whether the branch points to or wthin an existing block and either splits it as necessary or creates a new block
 * @param[in] opt - optimizer
 * @param[in] handle - handle of block where the branch is from.
 * @param[out] dest - block handle if found
 * @returns handle for destination branch */
codeblockindx optimize_branchto(optimizer *opt, codeblockindx handle, instructionindx dest, varray_codeblockindx *worklist) {
    codeblockindx existing = CODEBLOCKDEST_EMPTY;
    codeblockindx out;
    
    if (optimize_findblock(opt, dest, &existing)) {
        out = optimize_splitblock(opt, existing, dest);
    } else {
        out = optimize_newblock(opt, dest);
        varray_codeblockindxwrite(worklist, out); // Add to worklist
    }
    optimize_adddest(opt, handle, out);
    
    return out;
}

/** Build a code block from the current starting point
 * @param[in] opt - optimizer
 * @param[in] block - block to process
 * @param[out] worklist - worklist of blocks to process; updated */
void optimize_buildblock(optimizer *opt, codeblockindx block, varray_codeblockindx *worklist) {
    optimize_moveto(opt, optimize_getstart(opt, block));
    
    while (!optimize_atend(opt)) {
        optimize_fetch(opt);
        
        // If we have come upon an existing block terminate this one
        codeblockindx next;
        if (optimize_findblock(opt, optimizer_currentindx(opt), &next) && next!=block) {
            optimize_adddest(opt, block, next); // and link this block to the existing one
            return; // Terminate block
        }
        
        optimize_setend(opt, block, optimizer_currentindx(opt));
        
        switch (opt->op) {
            case OP_B:
            {
                int branchby = DECODE_sBx(opt->current);
                optimize_branchto(opt, block, optimizer_currentindx(opt)+1+branchby, worklist);
            }
                return; // Terminate current block
            case OP_BIF:
            {
                int branchby = DECODE_sBx(opt->current);
                
                // Create two new blocks, one for each possible destination
                optimize_branchto(opt, block, optimizer_currentindx(opt)+1, worklist);
                optimize_branchto(opt, block, optimizer_currentindx(opt)+1+branchby, worklist);
            }
                return; // Terminate current block
            case OP_END:
                return; // Terminate current block
            default:
                break;
        }
        
        optimize_next(opt);
    }
}

/** Builds the control flow graph from the source */
void optimize_buildcontrolflowgraph(optimizer *opt) {
    varray_codeblockindx worklist; // Worklist of blocks to analyze
    varray_codeblockindxinit(&worklist);
    
    codeblockindx first = optimize_newblock(opt, opt->func->entry); // Start at the entry point of the program
    varray_codeblockindxwrite(&worklist, first);
    optimize_incinbound(opt, first);
    
    while (worklist.count>0) {
        codeblockindx current;
        if (!varray_codeblockindxpop(&worklist, &current)) UNREACHABLE("Unexpectedly empty worklist in control flow graph");
        
        optimize_buildblock(opt, current, &worklist);
    }
    varray_codeblockindxclear(&worklist);
    
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
    for (codeblockindx i=0; i<opt->cfgraph.count; i++) {
        codeblock *block = opt->cfgraph.data+i;
        printf("Block %u [%td, %td]", i, block->start, block->end);
        if (block->dest[0]>=0) printf(" -> %u", block->dest[0]);
        if (block->dest[1]>=0) printf(" -> %u", block->dest[1]);
        printf(" (inbound: %u)\n", block->inbound);
    }
#endif
}

/* **********************************************************************
* Finalize, clearing nops and fixing debug info
* ********************************************************************** */

void optimize_compactify(optimizer *opt) {
    unsigned int write=0; // Keep track of where we're writing to
    
    optimize_restart(opt, 0);
    optimize_restartannotation(opt);
    
    while (!optimize_atend(opt)) {
        optimize_fetch(opt);
        optimize_replaceinstructionat(opt, write, opt->current); // Copy this instruction down
        
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
        debug_disassembleinstruction(opt->current, opt->iindx, NULL, NULL);
        printf("\n");
#endif
        
        if (opt->op==OP_NOP) {
            optimize_annotationdeleteinstruction(opt); // Mark this instruction for deletion
        } else write++; // otherwise continue
        optimize_advance(opt);
    }
    opt->out->code.count=write; // Set length of code
}

/* **********************************************************************
* Optimize a block
* ********************************************************************** */

void optimize_optimizeblock(optimizer *opt, codeblockindx block, optimizationstrategy *strategies) {
    instructionindx start=optimize_getstart(opt, block),
                    end=optimize_getend(opt, block);
            
    for (optimize_restart(opt, start);
         opt->iindx<=end;
         optimize_next(opt)) {
        
        optimize_fetch(opt);
        optimize_optimizeinstruction(opt, strategies);
        optimize_track(opt); // Track contents of registers
        optimize_overwrite(opt); //
        
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
        optimize_regshow(opt);
#endif
    }
}

/* **********************************************************************
* Public interface
* ********************************************************************** */

/** Public interface to optimizer */
bool optimize(program *prog) {
    optimizer opt;
    //optimizationstrategy *pass[2] = { firstpass, secondpass};
    
    optimize_init(&opt, prog);
    
    optimize_buildcontrolflowgraph(&opt);
    
    // Now optimize blocks
    varray_codeblockindx worklist;
    varray_codeblockindxinit(&worklist);
    varray_codeblockindxwrite(&worklist, 0);
    
    while (worklist.count>0) {
        codeblockindx current;
        if (!varray_codeblockindxpop(&worklist, &current)) UNREACHABLE("Unexpectedly empty worklist in optimizer");
        
        if (optimize_getvisited(&opt, current)>=optimize_getinbound(&opt, current)) continue;
        
        optimize_optimizeblock(&opt, current, firstpass);
        optimize_visit(&opt, current);
        optimize_desttoworklist(&opt, current, &worklist);
    }
    
    varray_codeblockindxclear(&worklist);
    optimize_clear(&opt);
    
    /*
    optimize_compactify(&opt);
    */
    
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
