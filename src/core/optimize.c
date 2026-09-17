/** @file optimize.c
 *  @brief Dependency queries on Morpho callables.
 */

#include "vm.h"
#include "classes.h"
#include "program.h"
#include "optimize.h"

/** Extract a function from a value. */
static objectfunction *_getfunction(value f) {
    if (MORPHO_ISINVOCATION(f)) f=MORPHO_GETINVOCATION(f)->method;
    if (MORPHO_ISCLOSURE(f)) return MORPHO_GETCLOSUREFUNCTION(f);
    if (MORPHO_ISFUNCTION(f)) return MORPHO_GETFUNCTION(f);
    return NULL;
}

/** True if konst is needle, or either is a metafunction containing the other. */
static bool _constmatches(value konst, value needle) {
    if (MORPHO_ISSAME(konst, needle)) return true;
    if (MORPHO_ISMETAFUNCTION(needle) &&
        metafunction_matchfn(MORPHO_GETMETAFUNCTION(needle), konst)) return true;
    if (MORPHO_ISMETAFUNCTION(konst) &&
        metafunction_matchfn(MORPHO_GETMETAFUNCTION(konst), needle)) return true;
    return false;
}

/** Checks if a register is read by the instruction. */
static bool _regread(instruction bc, int r) {
    opcode op=DECODE_OP(bc);
    int a=DECODE_A(bc), b=DECODE_B(bc), c=DECODE_C(bc);
    switch (op) {
        case OP_NOP: case OP_LCT: case OP_LGL: case OP_LUP: case OP_B:
        case OP_PUSHERR: case OP_POPERR: case OP_BREAK: case OP_END: case OP_CLOSURE:
            return false;
        case OP_MOV: case OP_NOT: return b==r;
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_POW:
        case OP_EQ: case OP_NEQ: case OP_LT: case OP_LE: case OP_CAT: case OP_LPR:
            return b==r || c==r;
        case OP_PRINT: case OP_TYPECHECK: case OP_BIF: case OP_BIFF: case OP_SGL:
            return a==r;
        case OP_CALL: return r>=a && r<=a+b+2*c;
        case OP_METHOD: case OP_INVOKE: return r>=a && r<=a+1+b+2*c;
        case OP_RETURN: return a>0 && b==r;
        case OP_SUP: return b==r;
        case OP_LIX: case OP_LIXL: case OP_SIX: return a==r || (r>=b && r<=c);
        case OP_SPR: return a==r || b==r || c==r;
        default: return true;
    }
}

/** Mark every queried constant as loaded. */
static void _hitall(int nvals, bool *hit) {
    for (int i=0; i<nvals; i++) hit[i]=true;
}

/** Returns true if the function body reads argument arg.
 *  True also when the callable or program cannot be inspected. */
bool optimize_fnaccessesarg(vm *v, value f, int arg) {
    if (arg<0) return false;

    objectfunction *func=_getfunction(f);
    if (!func || arg>=func->nargs) return true;

    program *p=v ? v->current : NULL;
    if (!p) return true;
    if (func->end<=func->entry) return false;

    int r=arg+1;
    for (indx i=func->entry; i<func->end; i++) {
        instruction bc;
        if (!program_getinstruction(p, i, &bc)) return true;
        if (_regread(bc, r)) return true;
    }
    return false;
}

/** Sets hit[i] if the body loads konsts[i]. One bytecode pass.
 *  If the callable or program cannot be inspected, every hit[i] is true. */
void optimize_fnloadsconstants(vm *v, value f, int nvals, value *konsts, bool *hit) {
    if (!hit) return;
    for (int i=0; i<nvals; i++) hit[i]=false;
    if (!konsts || nvals<=0) return;

    objectfunction *func=_getfunction(f);
    if (!func) { _hitall(nvals, hit); return; }

    program *p=v ? v->current : NULL;
    if (!p) { _hitall(nvals, hit); return; }
    if (func->end<=func->entry) return;

    for (indx i=func->entry; i<func->end; i++) {
        instruction bc;
        if (!program_getinstruction(p, i, &bc)) { _hitall(nvals, hit); return; }
        if (DECODE_OP(bc)!=OP_LCT) continue;

        int k=DECODE_Bx(bc);
        if (k<0 || k>=(int) func->konst.count) continue;

        value konst=func->konst.data[k];
        for (int j=0; j<nvals; j++) {
            if (!hit[j] && _constmatches(konst, konsts[j])) hit[j]=true;
        }
    }
}
