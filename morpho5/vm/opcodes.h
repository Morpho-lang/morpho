/** @file opcodes.h
 *  @author T J Atherton
 *
 *  @brief Morpho opcodes
 */

#ifdef OPCODE

/** No operation */
OPCODE(NOP)

/** Moves contents between registers */
OPCODE(MOV)

/** Moves a constant into a register */
OPCODE(LCT)

/** Add the contents of two registers */
OPCODE(ADD)

/** Subtract the contents of two registers */
OPCODE(SUB)

/** Multiply the contents of two registers */
OPCODE(MUL)

/** Divide the contents of two registers */
OPCODE(DIV)

/** Raise to a power */
OPCODE(POW)

/** Logical NOT of a register */
OPCODE(NOT)

/** Comparison test */
OPCODE(EQ)

/** Comparison test */
OPCODE(NEQ)

/** Comparison test */
OPCODE(LT)

/** Comparison test */
OPCODE(LE)

/** Branching */
OPCODE(B)

/** Branch if true */
OPCODE(BIF)

/** Branch if false */
OPCODE(BIFF)

/** Call */
OPCODE(CALL)

/** Invoke */
OPCODE(INVOKE)

/** Return */
OPCODE(RETURN)

/** Create a closure */
OPCODE(CLOSURE)

/** Load upvalue */
OPCODE(LUP)

/** Store upvalue */
OPCODE(SUP)

/** Close upvalues */
OPCODE(CLOSEUP)

/** Load property */
OPCODE(LPR)

/** Store property */
OPCODE(SPR)

/** Load index */
OPCODE(LIX)

/** Store index */
OPCODE(SIX)

/** Load global */
OPCODE(LGL)

/** Store global */
OPCODE(SGL)

/** Push error handler */
OPCODE(PUSHERR)

/** Pop error handler */
OPCODE(POPERR)

/** Creates an array */
//OPCODE(ARRAY)

/** Converts a sequence of registers to strings if necessary and concatenates them */
OPCODE(CAT)

/** Print the cotents of a register */
OPCODE(PRINT)

/** Raise error */
//OPCODE(RAISE)

/** Breakpoint */
OPCODE(BREAK)

/** End program */
OPCODE(END)

#endif
