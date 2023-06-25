/** @file optimize.c
 *  @author T J Atherton
 *
 *  @brief Optimizer for compiled code
*/

#include "optimize.h"
#include "debug.h"
#include "vm.h"

DEFINE_VARRAY(codeblock, codeblock);
DEFINE_VARRAY(codeblockindx, codeblockindx);

/* **********************************************************************
 * Data structures
 * ********************************************************************** */

/* -----------
 * Code blocks
 * ----------- */

/** Initialize a code block */
void optimize_initcodeblock(codeblock *block, objectfunction *func, instructionindx start) {
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
    block->func=func;
    block->isroot=false;
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
    
    codeblockindx dest=opt->reg[reg].block;
    
    if (dest!=CODEBLOCKDEST_EMPTY) {
        optimize_retain(opt, dest, reg);
    }
}

/* -----------
 * Globals
 * ----------- */

/** Initialize globals */
void optimize_initglobals(optimizer *opt) {
    if (!opt->globals) return;
    for (unsigned int i=0; i<opt->out->nglobals; i++) {
        opt->globals[i].contains=NOTHING;
        opt->globals[i].used=0;
        opt->globals[i].type=MORPHO_NIL;
        opt->globals[i].contents=MORPHO_NIL;
    }
}

/** Indicates an instruction uses a global */
void optimize_useglobal(optimizer *opt, indx ix) {
    if (opt->globals) opt->globals[ix].used++;
}

/** Remove a reference to a global */
void optimize_unuseglobal(optimizer *opt, indx ix) {
    if (opt->globals) opt->globals[ix].used--;
}

/** Updates the contents of a global */
void optimize_globalcontents(optimizer *opt, indx ix, returntype type, indx id) {
    if (opt->globals) {
        if (opt->globals[ix].contains==NOTHING) {
            opt->globals[ix].contains = type;
            opt->globals[ix].id = id;
        } else if (opt->globals[ix].contains==type) {
            if (opt->globals[ix].id!=id) opt->globals[ix].id = GLOBAL_UNALLOCATED;
        } else {
            opt->globals[ix].contains = VALUE;
            opt->globals[ix].id = GLOBAL_UNALLOCATED;
        }
    }
}

/** Decides whether two types match */
bool optimize_matchtype(value t1, value t2) {
    return MORPHO_ISSAME(t1, t2);
}

/** Updates a globals type  */
void optimize_updateglobaltype(optimizer *opt, indx ix, value type) {
    if (!opt->globals) return;
    
    if (MORPHO_ISNIL(type) || MORPHO_ISNIL(opt->globals[ix].type)) {
        opt->globals[ix].type = type;
    } else if (MORPHO_ISSAME(opt->globals[ix].contents, OPTIMIZER_AMBIGUOUS)) {
        return;
    } else if (!optimize_matchtype(opt->globals[ix].type, type)) opt->globals[ix].type = OPTIMIZER_AMBIGUOUS;
}

/** Updates a globals value  */
void optimize_updateglobalcontents(optimizer *opt, indx ix, value val) {
    if (!opt->globals) return;
    
    if (MORPHO_ISNIL(opt->globals[ix].contents)) {
        opt->globals[ix].contents = val;
    } else if (MORPHO_ISSAME(opt->globals[ix].contents, OPTIMIZER_AMBIGUOUS)) {
        return; 
    } else if (!MORPHO_ISEQUAL(val, opt->globals[ix].contents)) opt->globals[ix].contents = OPTIMIZER_AMBIGUOUS;
}

/** Gets the type of a global */
value optimize_getglobaltype(optimizer *opt, indx ix) {
    if (opt->globals) return opt->globals[ix].type;
    return MORPHO_NIL;
}

/** Gets the contents of a global */
bool optimize_getglobalcontents(optimizer *opt, indx ix, returntype *contains, indx *id, value *contents) {
    if (opt->globals) {
        if (contains) *contains = opt->globals[ix].contains;
        if (id) *id = opt->globals[ix].id;
        if (contents) *contents = opt->globals[ix].contents;
        return true;
    }
    return false;
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
        opt->reg[i].iix=INSTRUCTIONINDX_EMPTY;
        opt->reg[i].block=CODEBLOCKDEST_EMPTY;
        opt->reg[i].type=MORPHO_NIL;
    }
}

/** Sets the contents of a register */
static inline void optimize_regcontents(optimizer *opt, registerindx reg, returntype type, indx id) {
    opt->reg[reg].contains=type;
    opt->reg[reg].id=id;
}

/** Indicates an instruction uses a register */
void optimize_reguse(optimizer *opt, registerindx reg) {
    //if (opt->reg[reg].contains==NOTHING) printf("Unresolved reference in reg %u.\n", reg);
    if (opt->reg[reg].block!=CODEBLOCKDEST_EMPTY && opt->reg[reg].block!=optimize_getcurrentblock(opt)) {
        optimize_retaininparents(opt, reg);
    }
    opt->reg[reg].used++;
}

/** Invalidates any old copies of a stored quantity */
void optimize_reginvalidate(optimizer *opt, returntype type, indx id) {
    for (unsigned int i=0; i<opt->maxreg; i++) {
        if (opt->reg[i].contains==type && opt->reg[i].id==id) {
            opt->reg[i].contains=VALUE;
        }
    }
}

/** Sets the type of value in a register */
void optimize_regsettype(optimizer *opt, registerindx reg, value value) {
    opt->reg[reg].type=value;
}

/** Resolves the type of value produced by an arithmetic instruction */
void optimize_resolvearithmetictype(optimizer *opt) {
    registerindx a=DECODE_A(opt->current);
    registerindx b=DECODE_B(opt->current);
    registerindx c=DECODE_C(opt->current);
    value ta = MORPHO_NIL, tb = opt->reg[b].type, tc = opt->reg[c].type;
    
    if (MORPHO_ISINTEGER(tb) && MORPHO_ISINTEGER(tc)) {
        ta = MORPHO_INTEGER(1);
    } else if ((MORPHO_ISINTEGER(tb) && MORPHO_ISFLOAT(tc)) ||
               (MORPHO_ISFLOAT(tb) && MORPHO_ISINTEGER(tc)) ||
               (MORPHO_ISFLOAT(tb) && MORPHO_ISFLOAT(tc))) {
        ta = MORPHO_FLOAT(1.0);
    }
    
    optimize_regsettype(opt, a, ta);
}

/** Gets the type of value in a register if known */
value optimize_getregtype(optimizer *opt, registerindx reg) {
    return opt->reg[reg].type;
}

/** Indicates an instruction overwrites a register */
void optimize_regoverwrite(optimizer *opt, registerindx reg) {
    optimize_regsettype(opt, reg, MORPHO_NIL);
    optimize_reginvalidate(opt, REGISTER, reg); // Invalidate any aliases of this register
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
        if (reg[i].contains!=NOTHING) {
            printf(" [%u] : %u", reg[i].block, reg[i].used);
            if (!MORPHO_ISNIL(reg[i].type)) {
                printf(" (");
                if (OPTIMIZER_ISAMBIGUOUS(reg[i].type)) printf("ambiguous");
                else morpho_printvalue(reg[i].type);
                printf(")");
            }
        }
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
    if (ix==INSTRUCTIONINDX_EMPTY) UNREACHABLE("Trying to fetch an undefined instruction.");
    return opt->out->code.data[ix];
}

/** Replaces an instruction at a given indx */
void optimize_replaceinstructionat(optimizer *opt, instructionindx ix, instruction inst) {
    instruction old = opt->out->code.data[ix];
    if (ix==INSTRUCTIONINDX_EMPTY) UNREACHABLE("Trying to replace an undefined instruction.");
    if (opt->out->code.data[ix]!=inst) {
        /* Update usage etc. here (should be expanded on) */
        if (DECODE_OP(old)==OP_LGL) {
            optimize_unuseglobal(opt, DECODE_Bx(old));
        }
        
        opt->nchanged+=1;
        opt->out->code.data[ix]=inst;
    }
}

/** Replaces the current instruction */
void optimize_replaceinstruction(optimizer *opt, instruction inst) {
    optimize_replaceinstructionat(opt, opt->iindx, inst);
    opt->current=inst;
    opt->op=DECODE_OP(inst);
}

/* ------------
 * Search
 * ------------ */

/** Trace back through duplicate registers */
registerindx optimize_findoriginalregister(optimizer *opt, registerindx reg) {
    registerindx out=reg;
    while (opt->reg[out].contains==REGISTER) {
        out=(registerindx) opt->reg[out].id;
        if (out==reg) return out;
    }
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
    
    if (MORPHO_ISOBJECT(val)) {
        // Bind the object to the program
        program_bindobject(opt->out, MORPHO_GETOBJECT(val));
    }
    
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

/** Indicates no overwrite takes place */
static inline void optimize_nooverwrite(optimizer *opt) {
    opt->overwrites=REGISTER_UNALLOCATED;
}

/** Fetches the instruction  */
void optimize_fetch(optimizer *opt) {
    optimize_nooverwrite(opt);
    if (opt->iindx>=opt->out->code.count) return; 
    
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

void optimize_restartannotation(optimizer *opt);


/** Initializes optimizer data structure */
void optimize_init(optimizer *opt, program *prog) {
    opt->out=prog;
    opt->globals=MORPHO_MALLOC(sizeof(globalinfo)*(opt->out->nglobals+1));
    optimize_initglobals(opt);
    opt->maxreg=MORPHO_MAXARGS;
    optimize_setfunction(opt, prog->global);
    optimize_restart(opt, 0);
    
    dictionary_init(&opt->functions);
    varray_codeblockinit(&opt->cfgraph);
    varray_debugannotationinit(&opt->aout);
    optimize_restartannotation(opt);
    
    opt->v=morpho_newvm();
    opt->temp=morpho_newprogram();
}

/** Clears optimizer data structure */
void optimize_clear(optimizer *opt) {
    if (opt->globals) MORPHO_FREE(opt->globals);
    
    for (int i=0; i<opt->cfgraph.count; i++) {
        optimize_clearcodeblock(opt->cfgraph.data+i);
    }
    
    dictionary_clear(&opt->functions);
    varray_codeblockclear(&opt->cfgraph);
    varray_debugannotationclear(&opt->aout);
    
    if (opt->v) morpho_freevm(opt->v);
    if (opt->temp) morpho_freeprogram(opt->temp);
}

/* **********************************************************************
 * Handling code annotations
 * ********************************************************************** */

/** Restarts annotations */
void optimize_restartannotation(optimizer *opt) {
    opt->a=0;
    opt->aindx=0;
    opt->aoffset=0;
}

/** Gets the current annotation */
debugannotation *optimize_currentannotation(optimizer *opt) {
    return &opt->out->annotations.data[opt->a];
}

/** Get next annotation and update annotation counters */
void optimize_annotationadvance(optimizer *opt) {
    debugannotation *ann=optimize_currentannotation(opt);
    
    opt->aoffset=0;
    if (ann->type==DEBUG_ELEMENT) opt->aindx+=ann->content.element.ninstr;
    
    opt->a++;
}

/** Are we at the end of annotations */
bool optimize_annotationatend(optimizer *opt) {
    return !(opt->a < opt->out->annotations.count);
}

/** Moves the annotation  system to a specified instruction */
void optimize_annotationmoveto(optimizer *opt, instructionindx ix) {
    indx lastelement = 0;
    if (ix<opt->aindx) optimize_restartannotation(opt);
    for (;
         !optimize_annotationatend(opt);
         optimize_annotationadvance(opt)) {
        
        debugannotation *ann=optimize_currentannotation(opt);
        if (ann->type==DEBUG_ELEMENT) {
            if (opt->aindx+ann->content.element.ninstr>ix) {
                opt->aoffset=ix-opt->aindx;
                // If we are at the start of an element, instead return us to just after the last element record
                if (opt->aoffset==0 && lastelement<opt->a) opt->a=lastelement+1;
                return;
            }
            
            lastelement=opt->a;
        }
    }
}

/** Copies across annotations for a specific code block */
void optimize_annotationcopyforblock(optimizer *opt, codeblock *block) {
    optimize_annotationmoveto(opt, block->start);
    instructionindx iindx = block->start;
    
    for (;
         !optimize_annotationatend(opt);
         optimize_annotationadvance(opt)) {
        debugannotation *ann=optimize_currentannotation(opt);
        
        if (ann->type==DEBUG_ELEMENT) {
            // Figure out how many instructions are left in the block and in the annotation
            instructionindx remaininginblock = block->end-iindx+1;
            instructionindx remaininginann = ann->content.element.ninstr-opt->aoffset;
            
            int nnops=0; // Count NOPs which will be deleted by compactify
            
            instructionindx ninstr=(remaininginann<remaininginblock ? remaininginann : remaininginblock);
            
            for (int i=0; i<ninstr; i++) {
                instruction instruction=optimize_fetchinstructionat(opt, i+iindx);
                if (DECODE_OP(instruction)==OP_NOP) nnops++;
            }
            
            if (ninstr>nnops) {
                debugannotation new = { .type=DEBUG_ELEMENT,
                    .content.element.ninstr = (int) ninstr - nnops,
                    .content.element.line = ann->content.element.line,
                    .content.element.posn = ann->content.element.posn
                };
                varray_debugannotationwrite(&opt->aout, new);
            }
            
            iindx+=ninstr;
            
            // Break if we are done with the block
            if (remaininginblock<=remaininginann) break;
        } else {
            // Copy across non element records
            varray_debugannotationadd(&opt->aout, ann, 1);
        }
    }
}

/** Copies across and fixes annotations */
void optimize_fixannotations(optimizer *opt, varray_codeblockindx *blocks) {
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
    debug_showannotations(&opt->out->annotations);
#endif
    optimize_restartannotation(opt);
    for (unsigned int i=0; i<blocks->count; i++) {
//        printf("Fixing annotations for block %i\n", blocks->data[i]);
        codeblock *block = optimize_getblock(opt, blocks->data[i]);
        optimize_annotationcopyforblock(opt, block);
    }
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
    debug_showannotations(&opt->aout);
#endif
}

/* **********************************************************************
* Decode instructions
* ********************************************************************** */

/** Track contents of registers etc*/
void optimize_track(optimizer *opt) {
    instruction instr=opt->current;
    
    int op=DECODE_OP(instr);
    switch (op) {
        case OP_NOP: // Opcodes to ignore
        case OP_PUSHERR:
        case OP_POPERR:
        case OP_BREAK:
        case OP_END:
        case OP_B:
        case OP_CLOSEUP:
            break;
        case OP_MOV:
            optimize_reguse(opt, DECODE_B(instr));
            optimize_regoverwrite(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), REGISTER, DECODE_B(instr));
            optimize_regsettype(opt, DECODE_A(instr), optimize_getregtype(opt, DECODE_B(instr)));
            break;
        case OP_LCT:
            optimize_regoverwrite(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), CONSTANT, DECODE_Bx(instr));
            if (opt->func && DECODE_Bx(instr)<opt->func->konst.count) {
                value k = opt->func->konst.data[DECODE_Bx(instr)];
                optimize_regsettype(opt, DECODE_A(instr), k);
            }
            break;
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
        case OP_POW:
            optimize_reguse(opt, DECODE_B(instr));
            optimize_reguse(opt, DECODE_C(instr));
            optimize_regoverwrite(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), VALUE, REGISTER_UNALLOCATED);
            optimize_resolvearithmetictype(opt);
            break;
        case OP_EQ:
        case OP_NEQ:
        case OP_LT:
        case OP_LE:
            optimize_reguse(opt, DECODE_B(instr));
            optimize_reguse(opt, DECODE_C(instr));
            optimize_regoverwrite(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), VALUE, REGISTER_UNALLOCATED);
            optimize_regsettype(opt, DECODE_A(instr), MORPHO_TRUE);
            break;
        case OP_NOT:
            optimize_reguse(opt, DECODE_B(instr));
            optimize_regoverwrite(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), VALUE, REGISTER_UNALLOCATED);
            optimize_regsettype(opt, DECODE_A(instr), MORPHO_TRUE);
            break;
        case OP_BIF:
        case OP_BIFF:
            optimize_reguse(opt, DECODE_A(instr));
            break;
        case OP_CALL:
        {
            registerindx a = DECODE_A(instr);
            registerindx b = DECODE_B(instr);
            optimize_reguse(opt, a);
            for (unsigned int i=0; i<b; i++) {
                optimize_reguse(opt, a+i+1);
                opt->reg[a+i+1].contains=NOTHING; // call uses and overwrites arguments.
            }
            optimize_regoverwrite(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), VALUE, REGISTER_UNALLOCATED);
        }
            break;
        case OP_INVOKE:
        {
            registerindx a = DECODE_A(instr);
            registerindx b = DECODE_B(instr);
            registerindx c = DECODE_C(instr);
            optimize_reguse(opt, a);
            optimize_reguse(opt, b);
            for (unsigned int i=0; i<c; i++) {
                optimize_reguse(opt, a+i+1);
                opt->reg[a+i+1].contains=NOTHING; // invoke uses and overwrites arguments.
            }
            optimize_regoverwrite(opt, a);
            optimize_regcontents(opt, a, VALUE, REGISTER_UNALLOCATED);
        }
            break;
        case OP_RETURN:
            if (DECODE_A(instr)>0) optimize_reguse(opt, DECODE_B(instr));
            break;
        case OP_LGL:
        {
            registerindx a = DECODE_A(instr);
            optimize_regoverwrite(opt, a);
            optimize_useglobal(opt, DECODE_Bx(instr));
            optimize_regcontents(opt, a, GLOBAL, DECODE_Bx(instr));
            optimize_regsettype(opt, DECODE_A(instr), optimize_getglobaltype(opt, DECODE_Bx(instr)));
        }
            break;
        case OP_SGL:
            optimize_reginvalidate(opt, GLOBAL, DECODE_Bx(instr));
            optimize_reguse(opt, DECODE_A(instr));
            optimize_regcontents(opt, DECODE_A(instr), GLOBAL, DECODE_Bx(instr));
            break;
        case OP_LPR:
        {
            registerindx a = DECODE_A(instr);
            optimize_reguse(opt, DECODE_B(instr));
            optimize_reguse(opt, DECODE_C(instr));
            optimize_regoverwrite(opt, a);
            optimize_regcontents(opt, a, VALUE, REGISTER_UNALLOCATED);
        }
            break;
        case OP_SPR:
            optimize_reguse(opt, DECODE_A(instr));
            optimize_reguse(opt, DECODE_B(instr));
            optimize_reguse(opt, DECODE_C(instr));
            break;
        case OP_CLOSURE:
        {
            optimize_reguse(opt, DECODE_A(instr));
            registerindx b = DECODE_B(instr); // Get which registers are used from the upvalue prototype
            varray_upvalue *v = &opt->func->prototype.data[b];
            for (unsigned int i=0; i<v->count; i++) optimize_reguse(opt, (registerindx) v->data[i].reg);
            optimize_regoverwrite(opt, DECODE_A(instr)); // Generates a closure in register
            optimize_regcontents(opt, DECODE_A(instr), VALUE, REGISTER_UNALLOCATED);
        }
            break;
        case OP_LUP:
            optimize_regoverwrite(opt, DECODE_A(instr));
            //optimize_regcontents(opt, DECODE_A(instr), UPVALUE, DECODE_B(instr));
            optimize_regcontents(opt, DECODE_A(instr), VALUE, REGISTER_UNALLOCATED);
            break;
        case OP_SUP:
            optimize_reguse(opt, DECODE_B(instr));
            break;
        case OP_LIX:
        {
            registerindx a=DECODE_A(instr);
            registerindx b=DECODE_B(instr);
            registerindx c=DECODE_C(instr);
            optimize_reguse(opt, a);
            for (unsigned int i=b; i<=c; i++) optimize_reguse(opt, i);
            optimize_regoverwrite(opt, b);
            optimize_regcontents(opt, b, VALUE, REGISTER_UNALLOCATED);
        }
            break;
        case OP_SIX:
        {
            registerindx a=DECODE_A(instr);
            registerindx b=DECODE_B(instr);
            registerindx c=DECODE_C(instr);
            optimize_reguse(opt, a);
            for (unsigned int i=b; i<=c; i++) optimize_reguse(opt, i);
        }
            break;
        case OP_PRINT:
            optimize_reguse(opt, DECODE_A(instr));
            break;
        case OP_CAT:
        {
            registerindx a=DECODE_A(instr);
            registerindx b=DECODE_B(instr);
            registerindx c=DECODE_C(instr);
            for (unsigned int i=b; i<=c; i++) optimize_reguse(opt, i);
            optimize_regoverwrite(opt, a);
            optimize_regcontents(opt, a, VALUE, REGISTER_UNALLOCATED);
        }
            break;
        default:
            UNREACHABLE("Opcode not supported in optimizer.");
    }
}

/* Replaces an unused instruction */
void optimize_replaceunused(optimizer *opt, reginfo *reg) {
    if (reg->iix!=INSTRUCTIONINDX_EMPTY) {
        instruction op = DECODE_OP(optimize_fetchinstructionat(opt, reg->iix));
        if (op==OP_INVOKE || op==OP_CALL) return;
        
        optimize_replaceinstructionat(opt, reg->iix, ENCODE_BYTE(OP_NOP));
    }
}

/** Process overwrite */
void optimize_overwrite(optimizer *opt, bool detectunused) {
    if (opt->overwrites==REGISTER_UNALLOCATED) return;
    
    // Detect unused expression
    if (detectunused && opt->overwriteprev.contains==CONSTANT &&
        opt->overwriteprev.used==0 &&
        opt->overwriteprev.block==optimize_getcurrentblock(opt)) {
        
        // We need to type check this!
        optimize_replaceunused(opt, &opt->reg[opt->overwrites]);
    }
    
    opt->reg[opt->overwrites].used=0;
    opt->reg[opt->overwrites].iix=optimizer_currentindx(opt);
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
    optimize_initcodeblock(&new, opt->func, start);
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

/** Get the function associated with a block */
objectfunction *optimize_getfunction(optimizer *opt, codeblockindx handle) {
    return opt->cfgraph.data[handle].func;
}

/** Set a block's end point */
void optimize_setend(optimizer *opt, codeblockindx handle, instructionindx end) {
    opt->cfgraph.data[handle].end=end;
}

/** Indicate that a block is root */
void optimize_setroot(optimizer *opt, codeblockindx handle) {
    opt->cfgraph.data[handle].isroot=true;
}

/** Check if a block is listed as root */
bool optimize_isroot(optimizer *opt, codeblockindx handle) {
    return opt->cfgraph.data[handle].isroot;
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
    
    codeblockindx new = optimize_newblock(opt, split); // Inherits function
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
    codeblockindx srcblock, destblock=CODEBLOCKDEST_EMPTY;
    
    if (optimize_findblock(opt, dest, &destblock)) {
        optimize_splitblock(opt, destblock, dest);
    } else {
        codeblockindx out = optimize_newblock(opt, dest);
        varray_codeblockindxwrite(worklist, out); // Add to worklist
    }
    
    if (optimize_findblock(opt, optimizer_currentindx(opt), &srcblock) &&
        optimize_findblock(opt, dest, &destblock)) {
        optimize_adddest(opt, srcblock, destblock);
    } else {
        UNREACHABLE("Couldn't find block.");
    }
    
    return destblock;
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
            case OP_POPERR:
            {
                int branchby = DECODE_sBx(opt->current);
                optimize_branchto(opt, block, optimizer_currentindx(opt)+1+branchby, worklist);
            }
                return; // Terminate current block
            case OP_BIF:
            case OP_BIFF:
            {
                int branchby = DECODE_sBx(opt->current);
                
                // Create two new blocks, one for each possible destination
                optimize_branchto(opt, block, optimizer_currentindx(opt)+1, worklist);
                optimize_branchto(opt, block, optimizer_currentindx(opt)+1+branchby, worklist);
            }
                return; // Terminate current block
            case OP_PUSHERR:
            {
                int ix = DECODE_Bx(opt->current);
                if (MORPHO_ISDICTIONARY(opt->func->konst.data[ix])) {
                    objectdictionary *dict = MORPHO_GETDICTIONARY(opt->func->konst.data[ix]);
                    
                    for (unsigned int i=0; i<dict->dict.capacity; i++) {
                        if (MORPHO_ISTRUE(dict->dict.contents[i].key)) {
                            instructionindx hindx=MORPHO_GETINTEGERVALUE(dict->dict.contents[i].val);
                            codeblockindx errhandler = optimize_newblock(opt, hindx); // Start at the entry point of the program
                            optimize_setroot(opt, errhandler);
                            varray_codeblockindxwrite(worklist, errhandler);
                        }
                    }
                }
                optimize_branchto(opt, block, optimizer_currentindx(opt)+1, worklist);
            }
                return; // Terminate current block
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

void optimize_addfunction(optimizer *opt, value func);

/** Searches a list of values for entry points */
void optimize_searchlist(optimizer *opt, varray_value *list) {
    for (unsigned int i=0; i<list->count; i++) {
        value entry = list->data[i];
        
        if (MORPHO_ISFUNCTION(entry)) optimize_addfunction(opt, entry);
    }
}

/** Adds a function to the control flow graph */
void optimize_addfunction(optimizer *opt, value func) {
    if (!MORPHO_ISFUNCTION(func)) return;
    dictionary_insert(&opt->functions, func, MORPHO_TRUE);
    optimize_searchlist(opt, &MORPHO_GETFUNCTION(func)->konst); // Search constant table
}

/** Builds all blocks starting from the current function */
void optimize_rootblock(optimizer *opt, varray_codeblockindx *worklist) {
    codeblockindx first = optimize_newblock(opt, opt->func->entry); // Start at the entry point of the program
    
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
    printf("Building root block [%u] for function '", first);
    morpho_printvalue(MORPHO_OBJECT(opt->func));
    printf("'\n");
#endif
    
    optimize_setroot(opt, first);
    varray_codeblockindxwrite(worklist, first);
    optimize_incinbound(opt, first);

    while (worklist->count>0) {
        codeblockindx current;
        if (!varray_codeblockindxpop(worklist, &current)) UNREACHABLE("Unexpectedly empty worklist in control flow graph");
        
        optimize_buildblock(opt, current, worklist);
    }
}

/** Builds the control flow graph from the source */
void optimize_buildcontrolflowgraph(optimizer *opt) {
    varray_codeblockindx worklist; // Worklist of blocks to analyze
    varray_codeblockindxinit(&worklist);
    
    optimize_addfunction(opt, MORPHO_OBJECT(opt->out->global)); // Add the global function
    
    // Now build the blocks for each function in the table
    for (unsigned int i=0; i<opt->functions.capacity; i++) {
        value func=opt->functions.contents[i].key;
        if (MORPHO_ISFUNCTION(func)) {
            optimize_setfunction(opt, MORPHO_GETFUNCTION(func));
            optimize_rootblock(opt, &worklist);
        }
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
    
    objectfunction temp=*opt->func; // Keep all the function's info, e.g. constant table
    temp.entry=0;
    
    opt->temp->global=&temp; // Patch in our function
    
    varray_instruction *code = &opt->temp->code;
    code->count=0; // Clear the program
    for (instruction *ins = list; ; ins++) { // Load the list of instructions into the program
        varray_instructionwrite(code, *ins);
        if (DECODE_OP(*ins)==OP_END) break;
    }
    
    if (morpho_run(opt->v, opt->temp)) { // Run the program and extract output
        if (out && dest< opt->v->stack.count) {
            *out = opt->v->stack.data[dest];
            if (MORPHO_ISOBJECT(*out)) vm_unbindobject(opt->v, *out);
        }
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

/** Identifies duplicate constants instructions */
bool optimize_duplicate_loadconst(optimizer *opt) {
    registerindx out = DECODE_A(opt->current);
    indx cindx = DECODE_Bx(opt->current);
    
    // Find if another register contains this constant
    for (registerindx i=0; i<opt->maxreg; i++) {
        if (opt->reg[i].contains==CONSTANT &&
            opt->reg[i].id==cindx &&
            opt->reg[i].block==opt->currentblock &&
            opt->reg[i].iix<optimizer_currentindx(opt)) {
            
            if (i!=out) { // Replace with a move instruction and note the duplication
                optimize_replaceinstruction(opt, ENCODE_DOUBLE(OP_MOV, out, i));
            } else { // Register already contains this constant
                optimize_replaceinstruction(opt, ENCODE_BYTE(OP_NOP));
            }
            
            return true;
        }
    }
    
    return false;
}

/** Identifies duplicate load instructions */
bool optimize_duplicate_loadglobal(optimizer *opt) {
    registerindx out = DECODE_A(opt->current);
    indx global = DECODE_Bx(opt->current);
    
    // Find if another register contains this global
    for (registerindx i=0; i<opt->maxreg; i++) {
        if (opt->reg[i].contains==GLOBAL &&
            opt->reg[i].id==global &&
            opt->reg[i].block==opt->currentblock &&
            opt->reg[i].iix<optimizer_currentindx(opt)) { // Nonlocal eliminations require understanding the call graph to check for SGL. 
            
            if (i!=out) { // Replace with a move instruction and note the duplication
                optimize_replaceinstruction(opt, ENCODE_DOUBLE(OP_MOV, out, i));
            } else { // Register already contains this global
                optimize_replaceinstruction(opt, ENCODE_BYTE(OP_NOP));
            }
            
            return true;
        }
    }
    
    return false;
}

/** Reduces powers to multiplies */
bool optimize_power_reduction(optimizer *opt) {
    indx kindx;
    if (optimize_findconstant(opt, DECODE_C(opt->current), &kindx)) {
        value konst = opt->func->konst.data[kindx];
        if (MORPHO_ISINTEGER(konst) && MORPHO_GETINTEGERVALUE(konst)==2) {
            optimize_replaceinstruction(opt, ENCODE(OP_MUL, DECODE_A(opt->current), DECODE_B(opt->current), DECODE_B(opt->current)));
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
    
    b=optimize_findoriginalregister(opt, b);
    c=optimize_findoriginalregister(opt, c);
    
    optimize_replaceinstruction(opt, ENCODE(opt->op, a, b, c));
    
    return false; // This allows other optimization strategies to intervene after
}

/* Check if a register is overwritten between two instructions */
bool optimize_checkoverwites(optimizer *opt, instructionindx start, instructionindx end, int nregisters, registerindx *reg) {
    bool result=true;
    optimizer temp = *opt; /* Preserve the optimizer state */
    
    optimize_moveto(opt, start);
    while (optimizer_currentindx(opt)<end && result) {
        optimize_advance(opt);
        optimize_fetch(opt);
        optimize_track(opt);
        for (unsigned int i=0; i<nregisters; i++) if (reg[i]==opt->overwrites) {
            result=false; break;
        }
    }
    
    *opt = temp; /* Restore the optimizer state */
    return result;
}

/** Searches to see if an expression has already been calculated  */
bool optimize_subexpression_elimination(optimizer *opt) {
    if (opt->op<OP_ADD || opt->op>OP_LE) return false; // Quickly eliminate non-arithmetic instructions
    static instruction mask = ( MASK_OP | MASK_B | MASK_C );
    registerindx reg[] = { DECODE_B(opt->current), DECODE_C(opt->current) } ;
    
    // Find if another register contains the same calculated value.
    for (registerindx i=0; i<opt->maxreg; i++) {
        if (opt->reg[i].contains==VALUE) {
            if (opt->reg[i].block!=opt->currentblock || // Only look within this block
                opt->reg[i].iix==INSTRUCTIONINDX_EMPTY) continue;
            instruction comp = optimize_fetchinstructionat(opt, opt->reg[i].iix);
            
            if ((comp & mask)==(opt->current & mask)) {
                /* Need to check if an instruction between the previous one and the
                   current one overwrites any operands */
                
                if (!optimize_checkoverwites(opt, opt->reg[i].iix, optimizer_currentindx(opt), (opt->op==OP_NOT ? 1 : 2), reg)) return false;
                
                optimize_replaceinstruction(opt, ENCODE_DOUBLE(OP_MOV, DECODE_A(opt->current), i));
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

/** Folds constants  */
bool optimize_constant_folding(optimizer *opt) {
    if (opt->op<OP_ADD || opt->op>OP_LE) return false; // Quickly eliminate non-arithmetic instructions
    
    instruction instr=opt->current;
    indx left, right;
    
    bool Bc=optimize_findconstant(opt, DECODE_B(instr), &left);
    if (!Bc) return false;
    bool Cc=optimize_findconstant(opt, DECODE_C(instr), &right);
    
    if (Cc) {
        // A program that evaluates the required op with the selected constants.
        instruction ilist[] = {
            ENCODE_LONG(OP_LCT, 0, (instruction) left),
            ENCODE_LONG(OP_LCT, 1, (instruction) right),
            ENCODE(opt->op, 0, 0, 1),
            ENCODE_BYTE(OP_END)
        };
        
        value out;
        
        if (optimize_evaluateprogram(opt, ilist, 0, &out)) {
            indx nkonst;
            //if (MORPHO_ISOBJECT(out)) UNREACHABLE("Optimizer encountered object while constant folding");
            if (optimize_addconstant(opt, out, &nkonst)) {
                if (nkonst==19) {
                    
                }
                optimize_replaceinstruction(opt, ENCODE_LONG(OP_LCT, DECODE_A(instr), (unsigned int) nkonst));
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

/** Identifies globals that just contain a constant */
bool optimize_constant_global(optimizer *opt) {
    indx global = DECODE_Bx(opt->current);
    returntype contents;
    value val;
    
    if (optimize_getglobalcontents(opt, global, &contents, NULL, &val) &&
        contents==CONSTANT &&
        !OPTIMIZER_ISAMBIGUOUS(val)) {
        indx kindx;
        
        if (optimize_addconstant(opt, val, &kindx)) {
            optimize_replaceinstruction(opt, ENCODE_LONG(OP_LCT, DECODE_A(opt->current), kindx));
            optimize_unuseglobal(opt, global);
            return true;
        }
    }
    
    return false;
}


/** Tracks information written to a global */
bool optimize_storeglobal_trackcontents(optimizer *opt) {
    registerindx rix = DECODE_A(opt->current);
    value type = optimize_getregtype(opt, rix);
    
    optimize_updateglobaltype(opt, DECODE_Bx(opt->current), (MORPHO_ISNIL(type) ? OPTIMIZER_AMBIGUOUS: type));
    
    if (opt->reg[rix].contains!=NOTHING) {
        optimize_globalcontents(opt, DECODE_Bx(opt->current), opt->reg[rix].contains, opt->reg[rix].id);
        optimize_updateglobalcontents(opt, DECODE_Bx(opt->current), opt->reg[rix].type);
    }
    
    return false;
}

/* --------------------------
 * Table of strategies
 * -------------------------- */

#define OP_ANY -1
#define OP_LAST OP_END+1

// The first pass establishes the data flow from block-block
// Only put things that can act on incomplete data flow here
optimizationstrategy firstpass[] = {
    { OP_LCT, optimize_duplicate_loadconst },
    { OP_LGL, optimize_duplicate_loadglobal },
    { OP_SGL, optimize_storeglobal_trackcontents },
    { OP_POW, optimize_power_reduction },
    { OP_LAST, NULL }
};

// The second pass is for arbitrary transformations
optimizationstrategy secondpass[] = {
    { OP_ANY, optimize_register_replacement },
    { OP_ANY, optimize_subexpression_elimination },
    { OP_ANY, optimize_constant_folding },          // Must be in second pass for correct data flow
    { OP_LCT, optimize_duplicate_loadconst },
    { OP_LGL, optimize_duplicate_loadglobal },
    { OP_LGL, optimize_constant_global },           // Second pass to ensure all sgls have been seen
    { OP_B,   optimize_branch_optimization },
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
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
    codeblock *block = optimize_getblock(opt, handle);
    printf("Register state from block %u\n", handle);
    optimize_showreginfo(block->nreg, block->reg);
#endif
}

/** Restores register info from the parents of a block */
void optimize_restoreregisterstate(optimizer *opt, codeblockindx handle) {
    codeblock *block = optimize_getblock(opt, handle);
    
    // Check if all parents have been visited.
    //for (unsigned int i=0; i<block->src.count; i++) if (!optimize_getvisited(opt, block->src.data[i])) return;
    
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
            if (src->reg[j].contains==NOTHING) continue;
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

/** Prints the code in a block */
void optimize_printblock(optimizer *opt, codeblockindx block) {
    instructionindx start=optimize_getstart(opt, block),
                    end=optimize_getend(opt, block);
    optimize_restart(opt, start);
    for (;
        optimizer_currentindx(opt)<=end;
        optimize_advance(opt)) {
        optimize_fetch(opt);
        debug_disassembleinstruction(opt->current, optimizer_currentindx(opt), NULL, NULL);
        printf("\n");
    }
}

/** Optimize a block */
void optimize_optimizeblock(optimizer *opt, codeblockindx block, optimizationstrategy *strategies) {
    instructionindx start=optimize_getstart(opt, block),
                    end=optimize_getend(opt, block);
    
    optimize_setcurrentblock(opt, block);
    optimize_setfunction(opt, optimize_getfunction(opt, block));
    
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
    printf("Optimizing block %u.\n", block);
#endif
    
    do {
        opt->nchanged=0;
        optimize_restart(opt, start);
        optimize_restoreregisterstate(opt, block); // Load registers
        
        for (;
            optimizer_currentindx(opt)<=end;
            optimize_advance(opt)) {
            
            optimize_fetch(opt);
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
            debug_disassembleinstruction(opt->current, optimizer_currentindx(opt), NULL, NULL);
            printf("\n");
#endif
            optimize_optimizeinstruction(opt, strategies);
            optimize_track(opt); // Track contents of registers
            optimize_overwrite(opt, true);
            
    #ifdef MORPHO_DEBUG_LOGOPTIMIZER
            optimize_regshow(opt);
    #endif
        }
    } while (opt->nchanged>0);
    
    optimize_saveregisterstatetoblock(opt, block);
    
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
    printf("Optimized block %u:\n", block);
    optimize_printblock(opt, block);
#endif
}

/* **********************************************************************
 * Check for unused instructions
 * ********************************************************************** */

/** Check all blocks for unused instructions */
/*void optimize_checkunused(optimizer *opt) {
    //return;
    for (codeblockindx i=0; i<opt->cfgraph.count; i++) {
        codeblock *block=optimize_getblock(opt, i);
        
        for (registerindx j=0; j<block->nreg; j++) {
            
            // This needs to check for side effects to be more general 
            if (block->reg[j].contains==CONSTANT && // More general check needed!
                block->reg[j].used==0 &&
                block->reg[j].block==i &&
                !optimize_isretained(opt, i, j)) {
                optimize_replaceunused(opt, &block->reg[j]);
            }
        }
    }
}*/

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
    
    do {
        optimize_fetch(opt);
        
        if (opt->op!=OP_NOP) {
            varray_instructionwrite(dest, opt->current);
            count++;
        }
        
        optimize_advance(opt);
    } while (optimizer_currentindx(opt)<=block->end);
    
    return count;
}

/** Fixes a pusherr dictionary */
void optimize_fixpusherr(optimizer *opt, codeblock *block, varray_instruction *dest) {
    instruction last = dest->data[block->oend];
    int ix = DECODE_Bx(last);
    if (MORPHO_ISDICTIONARY(block->func->konst.data[ix])) {
        objectdictionary *dict = MORPHO_GETDICTIONARY(block->func->konst.data[ix]);
        
        // Loop over error handler dictionary, repairing indices into code.
        for (unsigned int i=0; i<dict->dict.capacity; i++) {
            if (MORPHO_ISFALSE(dict->dict.contents[i].key)) continue;
            instructionindx hindx=MORPHO_GETINTEGERVALUE(dict->dict.contents[i].val);
            
            codeblockindx errhandler;
            if (optimize_findblock(opt, hindx, &errhandler)) {
                codeblock *h = optimize_getblock(opt, errhandler);
                if (h) dict->dict.contents[i].val=MORPHO_INTEGER(h->ostart);
                else UNREACHABLE("Couldn't find block for error handler");
            } else UNREACHABLE("Couldn't find error handler");
        }
    }
}

/** Fix branch instructions */
void optimize_fixbranch(optimizer *opt, codeblock *block, varray_instruction *dest) {
    if (block->oend<block->ostart) return;
    instruction last = dest->data[block->oend];
    
    if (DECODE_OP(last)==OP_B || DECODE_OP(last)==OP_POPERR) {
        codeblock *destblock = optimize_getblock(opt, block->dest[0]);
        dest->data[block->oend] = ENCODE_LONG(DECODE_OP(last), REGISTER_UNALLOCATED, destblock->ostart - block->oend - 1);
    } else if (DECODE_OP(last)==OP_BIF || DECODE_OP(last)==OP_BIFF) {
        if (block->dest[1]==INSTRUCTIONINDX_EMPTY) UNREACHABLE("Branch to unknown block");
        codeblock *destblock = optimize_getblock(opt, block->dest[1]);
        dest->data[block->oend] = ENCODE_LONG(DECODE_OP(last), DECODE_A(last), destblock->ostart - block->oend-1);
    } else if (DECODE_OP(last)==OP_PUSHERR) {
        optimize_fixpusherr(opt, block, dest);
    }
}

/** Fix function starting point */
void optimize_fixfunction(optimizer *opt, codeblock *block) {
    if (block->isroot && block->func->entry==block->start) {
        block->func->entry=block->ostart;
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
    
    /** Fix annotations */
    optimize_fixannotations(opt, &sorted);
    
    /** Copy and compactify blocks */
    for (unsigned int i=0; i<nblocks; i++) {
        codeblock *block = optimize_getblock(opt, sorted.data[i]);
        
#ifdef MORPHO_DEBUG_LOGOPTIMIZER
        printf("Compacting block %u.\n", sorted.data[i]);
        optimize_printblock(opt, sorted.data[i]);
#endif
        
        int ninstructions=optimize_compactifyblock(opt, block, &out);
        
        block->ostart=iout; // Record block's new start and end point
        block->oend=iout+ninstructions-1;
        
        optimize_fixfunction(opt, block);
        
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

/* Perform an optimization pass */
bool optimization_pass(optimizer *opt, optimizationstrategy *strategylist) {
    // Now optimize blocks
    varray_codeblockindx worklist;
    varray_codeblockindxinit(&worklist);
    
    for (unsigned int i=0; i<opt->cfgraph.count; i++) {
        opt->cfgraph.data[i].visited=0; // Clear visit flag
        // Ensure all root blocks are on the worklist
        if (opt->cfgraph.data[i].isroot) varray_codeblockindxwrite(&worklist, i); // Add to the worklist
    }
    
    while (worklist.count>0) {
        codeblockindx current;
        if (!varray_codeblockindxpop(&worklist, &current)) UNREACHABLE("Unexpectedly empty worklist in optimizer");
        
        // Make sure we didn't already finalize this block
        if (optimize_getvisited(opt, current)>=optimize_getinbound(opt, current)) continue;
        
        optimize_optimizeblock(opt, current, strategylist);
        optimize_visit(opt, current);
        optimize_desttoworklist(opt, current, &worklist);
    }
    
    //optimize_checkunused(&opt);
    
    varray_codeblockindxclear(&worklist);
    
    return true;
}


/** Public interface to optimizer */
bool optimize(program *prog) {
    optimizer opt;
    optimizationstrategy *pass[2] = { firstpass, secondpass};
    
    optimize_init(&opt, prog);
    
    optimize_buildcontrolflowgraph(&opt);
    for (int i=0; i<2; i++) {
        optimization_pass(&opt, pass[i]);
    }
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
