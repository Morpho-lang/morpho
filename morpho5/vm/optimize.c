/** @file optimize.c
 *  @author T J Atherton
 *
 *  @brief Optimizer for compiled code
*/

#include "optimize.h"
#include "debug.h"

DEFINE_VARRAY(codeblock, codeblock);
DEFINE_VARRAY(codeblockindx, codeblockindx);

/* **********************************************************************
 * Data structures
 * ********************************************************************** */

/* -----------
 * Code blocks
 * ----------- */

/** Initialize a code block */
void optimize_initcodeblock(codeblock *block, instructionindx start) {
    block->start=start;
    block->end=start;
    block->inbound=0;
    block->dest[0]=CODEBLOCKDEST_EMPTY;
    block->dest[1]=CODEBLOCKDEST_EMPTY;
    varray_codeblockindxinit(&block->src);
    dictionary_init(&block->retain);
    block->visited=0;
    block->nreg=0;
    block->reg=NULL;
    block->ostart=0;
    block->oend=0;
}

/** Clear a code block */
void optimize_clearcodeblock(codeblock *block) {
    varray_codeblockindxclear(&block->src);
    dictionary_clear(&block->retain);
}

/** Sets the current block */
void optimize_setcurrentblock(optimizer *opt, codeblockindx handle) {
    opt->currentblock=handle;
}

/** Gets the current block */
codeblockindx optimize_getcurrentblock(optimizer *opt) {
    return opt->currentblock;
}

/** Gets the code block from the handle */
codeblock *optimize_getblock(optimizer *opt, codeblockindx handle) {
    return &opt->cfgraph.data[handle];
}

/** Indicate that register `reg` in block `handle` is used by a subsequent block */
void optimize_retain(optimizer *opt, codeblockindx handle, registerindx reg) {
    codeblock *block=optimize_getblock(opt, handle);
    dictionary_insert(&block->retain, MORPHO_INTEGER(reg), MORPHO_TRUE);
}

/** Check if a register  `reg` in block `handle` has been marked as retained */
bool optimize_isretained(optimizer *opt, codeblockindx handle, registerindx reg) {
    value val;
    codeblock *block=optimize_getblock(opt, handle);
    return dictionary_get(&block->retain, MORPHO_INTEGER(reg), &val);
}

/** Marks reg as retained in parents of the current code block */
void optimize_retaininparents(optimizer *opt, registerindx reg) {
    codeblock *block=optimize_getblock(opt, optimize_getcurrentblock(opt));
    
    for (unsigned int i=0; i<block->src.count; i++) {
        codeblockindx dest = block->src.data[i];
        if (dest!=CODEBLOCKDEST_EMPTY) optimize_retain(opt, dest, reg);
    }
}

/* -----------
 * Reginfo
 * ----------- */

/** Clears the reginfo structure */
void optimize_regclear(optimizer *opt) {
    for (unsigned int i=0; i<MORPHO_MAXARGS; i++) {
        opt->reg[i].contains=NOTHING;
        opt->reg[i].id=0;
        opt->reg[i].used=0;
        opt->reg[i].iix=0;
        opt->reg[i].block=CODEBLOCKDEST_EMPTY;
    }
}

/** Sets the contents of a register */
static inline void optimize_regcontents(optimizer *opt, registerindx reg, returntype type, indx id) {
    opt->reg[reg].contains=type;
    opt->reg[reg].id=id;
}

/** Indicates an instruction uses a register */
void optimize_reguse(optimizer *opt, registerindx reg) {
    if (opt->reg[reg].block!=CODEBLOCKDEST_EMPTY && opt->reg[reg].block!=optimize_getcurrentblock(opt)) {
        optimize_retaininparents(opt, reg);
    }
    opt->reg[reg].used++;
}

/** Indicates an instruction overwrites a register */
static inline void optimize_regoverwrite(optimizer *opt, registerindx reg) {
    opt->overwriteprev=opt->reg[reg];
    opt->overwrites=reg;
}

/** Show the contents of a register */
void optimize_showreginfo(unsigned int regmax, reginfo *reg) {
    for (unsigned int i=0; i<regmax; i++) {
        printf("|\tr%u : ", i);
        switch (reg[i].contains) {
            case NOTHING: break;
            case REGISTER: printf("r%td", reg[i].id); break;
            case GLOBAL: printf("g%td", reg[i].id); break;
            case CONSTANT: printf("c%td", reg[i].id); break;
            case UPVALUE: printf("u%td", reg[i].id); break;
            case VALUE: printf("value"); break;
        }
        if (reg[i].contains!=NOTHING) printf(" [%u] : %u", reg[i].block, reg[i].used);
        printf("\n");
    }
    printf("\n");
}

/** Show register contents */
void optimize_regshow(optimizer *opt) {
    unsigned int regmax = opt->maxreg;
    for (unsigned int i=0; i<opt->maxreg; i++) if (opt->reg[i].contains!=NOTHING) regmax=i;
    
    optimize_showreginfo(opt->maxreg, opt->reg);
}

/* ------------
 * Instructions
 * ------------ */

/** Fetches an instruction at a given indx */
instruction optimize_fetchinstructionat(optimizer *opt, indx ix) {
    return opt->out->code.data[ix];
}

/** Replaces an instruction at a given indx */
void optimize_replaceinstructionat(optimizer *opt, indx ix, instruction inst) {
    opt->nchanged+=1;
    opt->out->code.data[ix]=inst;
}

/** Replaces the current instruction */
void optimize_replaceinstruction(optimizer *opt, instruction inst) {
    opt->out->code.data[opt->iindx]=inst;
    opt->current=inst;
    opt->op=DECODE_OP(inst);
}

/* ------------
 * Search
 * ------------ */

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
* Optimizer data structure
* ********************************************************************** */

/** Restart from a designated instruction */
void optimize_restart(optimizer *opt, instructionindx start) {
    optimize_regclear(opt);
    opt->iindx=start;
}

/** Checks if we're at the end of the program */
bool optimize_atend(optimizer *opt) {
    return (opt->iindx >= opt->out->code.count);
}

/** Sets the current function */
void optimize_setfunction(optimizer *opt, objectfunction *func) {
    opt->maxreg=func->nregs;
    opt->func=func;
}

/** Indicates an instruction uses a global */
static inline void optimize_loadglobal(optimizer *opt, indx ix) {
    if (opt->globals) opt->globals[ix].used++;
}

/** Indicates no overwrite takes place */
static inline void optimize_nooverwrite(optimizer *opt) {
    opt->overwrites=REGISTER_UNALLOCATED;
}

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
    opt->iindx++;
}

/** Move to a different instruction. Does not move annotations */
void optimize_moveto(optimizer *opt, instructionindx indx) {
    opt->iindx=indx;
    optimize_fetch(opt);
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
    varray_debugannotationinit(&opt->aout);
    
    opt->v=morpho_newvm();
    opt->temp=morpho_newprogram();
}

/** Clears optimizer data structure */
void optimize_clear(optimizer *opt) {
    if (opt->globals) MORPHO_FREE(opt->globals);
    
    for (int i=0; i<opt->cfgraph.count; i++) {
        optimize_clearcodeblock(opt->cfgraph.data+i);
    }
    
    varray_codeblockclear(&opt->cfgraph);
    varray_debugannotationclear(&opt->aout);
    
    if (opt->v) morpho_freevm(opt->v);
    if (opt->temp) morpho_freeprogram(opt->temp);
}

/* **********************************************************************
 * Handling code annotations
 * ********************************************************************** */

/** Gets the current annotation */
debugannotation *optimize_currentannotation(optimizer *opt) {
    return &opt->out->annotations.data[opt->a];
}

/** Are we at the end of annotations */
bool optimize_annotationatend(optimizer *opt) {
    return !(opt->a < opt->out->annotations.count);
}

/** Reset element counters */
void optimize_annotationresetelementcounters(optimizer *opt) {
    opt->ai=0;
    opt->acopied=0;
}

/** Processes a DEBUG_ELEMENT record */
void optimize_annotationprocesselement(optimizer *opt, bool copy) {
    debugannotation *ann = optimize_currentannotation(opt);
    opt->ai++;
    if (copy) opt->acopied++;
    if (opt->ai==ann->content.element.ninstr) {
        if (opt->acopied>0) {
            varray_debugannotationadd(&opt->aout, ann, 1);
            // Fix the number of instructions
            opt->aout.data[opt->aout.count-1].content.element.ninstr=opt->acopied;
        }
        opt->a++; // Move to next record
        optimize_annotationresetelementcounters(opt);
    }
}

/** Advance annotation system by one instruction. Copies annotations associated with the instruction if copy is set */
void optimize_annotationadvance(optimizer *opt, bool copy) {
    // Advance through the annotations
    while (!optimize_annotationatend(opt)) {
        debugannotation *ann = optimize_currentannotation(opt);
        if (ann->type==DEBUG_ELEMENT) { // Until we get to a DEBUG_ELEMENT record
            optimize_annotationprocesselement(opt, copy);
            break;
        }
        if (copy) varray_debugannotationadd(&opt->aout, ann, 1);
        opt->a++;
    }
    opt->aindx++;
}

/** Advances annotation system to current instruction */
void optimize_annotationmoveto(optimizer *opt) {
    while (opt->aindx<opt->iindx) {
        optimize_annotationadvance(opt, false);
    }
}

/** Restarts annotations */
void optimize_restartannotation(optimizer *opt) {
    opt->a=0;
    opt->aindx=0;
    optimize_annotationresetelementcounters(opt);
}

/* **********************************************************************
* Decode instructions
* ********************************************************************** */

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

/** Process overwrite */
void optimize_overwrite(optimizer *opt, bool detectunused) {
    if (opt->overwrites==REGISTER_UNALLOCATED) return;
    
    // Detect unused expression
    if (detectunused && opt->overwriteprev.contains!=NOTHING &&
        opt->overwriteprev.used==0 &&
        opt->overwriteprev.block==optimize_getcurrentblock(opt)) {
        
        optimize_replaceinstructionat(opt, opt->reg[opt->overwrites].iix, ENCODE_BYTE(OP_NOP));
    }
    
    opt->reg[opt->overwrites].used=0;
    opt->reg[opt->overwrites].iix=opt->iindx;
    opt->reg[opt->overwrites].block=optimize_getcurrentblock(opt);
}

/* **********************************************************************
* Control Flow graph
* ********************************************************************** */

/** Show the current code blocks*/
void optimize_showcodeblocks(optimizer *opt) {
    for (codeblockindx i=0; i<opt->cfgraph.count; i++) {
        codeblock *block = opt->cfgraph.data+i;
        printf("Block %u [%td, %td]", i, block->start, block->end);
        if (block->dest[0]>=0) printf(" -> %u", block->dest[0]);
        if (block->dest[1]>=0) printf(" -> %u", block->dest[1]);
        printf(" (inbound: %u)\n", block->inbound);
    }
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

/** Has the block been visited? */
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

/** Save reginfo to the block */
void optimize_saveregisterstatetoblock(optimizer *opt, codeblockindx handle) {
    codeblock *block = &opt->cfgraph.data[handle];
    
    if (!block->reg || opt->maxreg>block->nreg) block->reg=MORPHO_REALLOC(block->reg, opt->maxreg*sizeof(reginfo));
    block->nreg=opt->maxreg;
    
    if (block->reg) for (unsigned int i=0; i<opt->maxreg; i++) block->reg[i]=opt->reg[i];
}

/** Adds a source to a block */
void optimize_addsrc(optimizer *opt, codeblockindx dest, codeblockindx src) {
    codeblock *block = &opt->cfgraph.data[dest];
    
    for (int i=0; i<block->src.count; i++) { // Is the src already part of the block?
        if (block->src.data[i]==src) return;
    }
    varray_codeblockindxwrite(&block->src, src);
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
            case OP_CALL:
            {
                
            }
                break;
            case OP_RETURN:
            case OP_END:
                return; // Terminate current block
            default:
                break;
        }
        
        optimize_track(opt); // Track register contents
        optimize_overwrite(opt, false);
        optimize_advance(opt);
    }
}

/** Add cross references to the sources of each block; done at the end of building the CF graph */
void optimize_addsrcrefs(optimizer *opt) {
    for (codeblockindx i=0; i<opt->cfgraph.count; i++) {
        codeblock *block = optimize_getblock(opt, i);
        for (int j=0; j<2; j++) if (block->dest[j]!=CODEBLOCKDEST_EMPTY) optimize_addsrc(opt, block->dest[j], i);
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
    
    optimize_addsrcrefs(opt);
    
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
    optimize_showcodeblocks(opt);
#endif
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
    { OP_LAST, NULL }
};

optimizationstrategy secondpass[] = {
    { OP_ANY, optimize_register_replacement },
    { OP_ANY, optimize_subexpression_elimination },
    { OP_ANY, optimize_constant_folding },
    { OP_LGL, optimize_duplicate_load },
    { OP_SGL, optimize_unused_global },
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

/* **********************************************************************
 * Optimize a block
 * ********************************************************************** */

void optimize_showregisterstateforblock(optimizer *opt, codeblockindx handle) {
    codeblock *block = optimize_getblock(opt, handle);
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
    printf("Register state from block %u\n", handle);
    optimize_showreginfo(block->nreg, block->reg);
#endif
}

/** Restores register info from the parents of a block */
void optimize_restoreregisterstate(optimizer *opt, codeblockindx handle) {
    codeblock *block = optimize_getblock(opt, handle);
    
    // Check if all parents have been visited.
    for (unsigned int i=0; i<block->src.count; i++) if (!optimize_getvisited(opt, block->src.data[i])) return;
    
    // Copy across the first block
    if (block->src.count>0) {
        codeblock *src = optimize_getblock(opt, block->src.data[0]);
        for (unsigned int j=0; j<src->nreg; j++) opt->reg[j]=src->reg[j];
        optimize_showregisterstateforblock(opt, block->src.data[0]);
    }
    
    /** Now update based on subsequent blocks */
    for (unsigned int i=1; i<block->src.count; i++) {
        codeblock *src = optimize_getblock(opt, block->src.data[i]);
        optimize_showregisterstateforblock(opt, block->src.data[i]);
        
        for (unsigned int j=0; j<src->nreg; j++) {
            // If it doesn't match the type, we mark it as a VALUE
            if (opt->reg[j].contains!=src->reg[j].contains ||
                opt->reg[j].id!=src->reg[j].id) {
                
                opt->reg[j].contains=VALUE;
                
                // Mark block creator
                if (opt->reg[j].contains==NOTHING || opt->reg[j].block==CODEBLOCKDEST_EMPTY) {
                    opt->reg[j].block=src->reg[j].block;
                }
                // Copy usage information
                if (src->reg[j].used>opt->reg[j].used)  opt->reg[j].used=src->reg[j].used;
            }
        }
        
        if (src->nreg>opt->maxreg) opt->maxreg=src->nreg;
    }
    
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
        printf("Combined register state:\n");
        optimize_regshow(opt);
#endif
}

/** Optimize a block */
void optimize_optimizeblock(optimizer *opt, codeblockindx block, optimizationstrategy *strategies) {
    instructionindx start=optimize_getstart(opt, block),
                    end=optimize_getend(opt, block);
    
    optimize_setcurrentblock(opt, block);
    
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
    printf("Optimizing block %u.\n", block);
#endif
    
    do {
        opt->nchanged=0;
        optimize_restart(opt, start);
        optimize_restoreregisterstate(opt, block); // Load registers
        
        for (;
            opt->iindx<=end;
            optimize_advance(opt)) {
            
            optimize_fetch(opt);
            optimize_optimizeinstruction(opt, strategies);
            optimize_track(opt); // Track contents of registers
            optimize_overwrite(opt, true);
            
    #ifdef MORPHO_DEBUG_LOGOPTIMIZER
            optimize_regshow(opt);
    #endif
        }
    } while (opt->nchanged>0);
    
    optimize_saveregisterstatetoblock(opt, block);
    optimize_showreginfo(opt->maxreg, opt->reg);
}

/* **********************************************************************
 * Check for unused instructions
 * ********************************************************************** */

/** Check all blocks for unused instructions */
void optimize_checkunused(optimizer *opt) {
    //return;
    for (codeblockindx i=0; i<opt->cfgraph.count; i++) {
        codeblock *block=optimize_getblock(opt, i);
        
        for (registerindx j=0; j<block->nreg; j++) {
            if (block->reg[j].contains!=NOTHING &&
                block->reg[j].used==0 &&
                !optimize_isretained(opt, i, j)) {
                // Should check for side effects!
                optimize_replaceinstructionat(opt, block->reg[j].iix, ENCODE_BYTE(OP_NOP));
            }
        }
    }
}

/* **********************************************************************
 * Final processing and layout of final program
 * ********************************************************************** */

codeblock *blocklist;

/** Sort blocks */
int optimize_blocksortfn(const void *a, const void *b) {
    codeblockindx ia = *(codeblockindx *) a,
                  ib = *(codeblockindx *) b;
    
    instructionindx starta = blocklist[ia].start,
                    startb = blocklist[ib].start;

    return (startb>starta ? -1 : (startb==starta ? 0 : 1 ) );
}

/** Construct and sort a list of block indices */
void optimize_sortblocks(optimizer *opt, varray_codeblockindx *out) {
    codeblockindx nblocks = opt->cfgraph.count;
    
    // Sort blocks by position
    for (codeblockindx i=0; i<nblocks; i++) {
        varray_codeblockindxwrite(out, i);
    }
    
    blocklist = opt->cfgraph.data;
    qsort(out->data, nblocks, sizeof(codeblockindx), optimize_blocksortfn);
}

/** Compactifies a block, writing the results to dest*/
int optimize_compactifyblock(optimizer *opt, codeblock *block, varray_instruction *dest) {
    int count=0; // Count number of copied instructions
    optimize_moveto(opt, block->start);
    optimize_annotationmoveto(opt);
    
    do {
        optimize_fetch(opt);
        
        if (opt->op!=OP_NOP) {
            varray_instructionwrite(dest, opt->current);
            count++;
        }
        
        optimize_advance(opt);
        optimize_annotationadvance(opt, opt->op!=OP_NOP);
    } while (optimizer_currentindx(opt)<=block->end);
    
    return count;
}

/** Fix branch instructions */
void optimize_fixbranch(optimizer *opt, codeblock *block, varray_instruction *dest) {
    instruction last = dest->data[block->oend];
    
    if (DECODE_OP(last)==OP_B) {
        codeblock *destblock = optimize_getblock(opt, block->dest[0]);
        dest->data[block->oend] = ENCODE_LONG(OP_B, REGISTER_UNALLOCATED, destblock->ostart-block->oend-1);
    } else if (DECODE_OP(last)==OP_BIF) {
        codeblock *destblock = optimize_getblock(opt, block->dest[1]);
        dest->data[block->oend] = ENCODE_LONGFLAGS(OP_BIF, DECODE_A(last), DECODE_F(last), false, destblock->ostart-block->oend-1);
    }
}

/** Layout blocks */
void optimize_layoutblocks(optimizer *opt) {
    codeblockindx nblocks = opt->cfgraph.count;
    varray_codeblockindx sorted; // Sorted block indices
    varray_instruction out; // Destination program
    
    varray_codeblockindxinit(&sorted);
    varray_instructioninit(&out);
    
    optimize_sortblocks(opt,&sorted);
    optimize_restart(opt, 0);
    
    instructionindx iout=0; // Track instruction count
    
    /** Copy and compactify blocks */
    for (unsigned int i=0; i<nblocks; i++) {
        codeblock *block = optimize_getblock(opt, sorted.data[i]);
        
        int ninstructions=optimize_compactifyblock(opt, block, &out);
        
        block->ostart=iout; // Record block's new start and end point
        block->oend=iout+ninstructions-1;
        
        iout+=ninstructions;
    }
    
    /** Fix branch instructions */
    for (unsigned int i=0; i<nblocks; i++) {
        codeblock *block = optimize_getblock(opt, sorted.data[i]);
        optimize_fixbranch(opt, block, &out);
    }
    
    /** Patch instructions into program */
    varray_instructionclear(&opt->out->code);
    opt->out->code=out;
    /** Patch new annotations into program */
    varray_debugannotationclear(&opt->out->annotations);
    opt->out->annotations=opt->aout;
    varray_debugannotationinit(&opt->aout); // Reinitialize optimizers annotation record
    
    varray_codeblockindxclear(&sorted);
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
    varray_codeblockindxwrite(&worklist, 0); // Start with first block
    
    while (worklist.count>0) {
        codeblockindx current;
        if (!varray_codeblockindxpop(&worklist, &current)) UNREACHABLE("Unexpectedly empty worklist in optimizer");
        
        // Make sure we didn't already finalize this block
        if (optimize_getvisited(&opt, current)>=optimize_getinbound(&opt, current)) continue;
        
        optimize_optimizeblock(&opt, current, firstpass);
        optimize_visit(&opt, current);
        optimize_desttoworklist(&opt, current, &worklist);
    }
    
    optimize_checkunused(&opt);
    
    varray_codeblockindxclear(&worklist);
    optimize_layoutblocks(&opt);
    
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
