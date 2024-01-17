/** @file compile.h
 *  @author T J Atherton
 *
 *  @brief Compiles raw input to Morpho instructions
*/

#ifndef compile_h
#define compile_h

#define MORPHO_CORE

#include "core.h"
#include "syntaxtree.h"
#include "parse.h"
#include "debug.h"

/* **********************************************************************
 * Compiler error messages
 * ********************************************************************** */

#define COMPILE_SYMBOLNOTDEFINED          "SymblUndf"
#define COMPILE_SYMBOLNOTDEFINED_MSG      "Symbol '%s' not defined."

#define COMPILE_SYMBOLNOTDEFINEDNMSPC     "SymblUndfNmSpc"
#define COMPILE_SYMBOLNOTDEFINEDNMSPC_MSG "Symbol '%s' not defined in namespace '%s'."

#define COMPILE_TOOMANYCONSTANTS          "TooMnyCnst"
#define COMPILE_TOOMANYCONSTANTS_MSG      "Too many constants."

#define COMPILE_ARGSNOTSYMBOLS            "FnPrmSymb"
#define COMPILE_ARGSNOTSYMBOLS_MSG        "Function parameters must be symbols."

#define COMPILE_PROPERTYNAMERQD           "PptyNmRqd"
#define COMPILE_PROPERTYNAMERQD_MSG       "Property name required."

#define COMPILE_SELFOUTSIDECLASS          "SlfOtsdClss"
#define COMPILE_SELFOUTSIDECLASS_MSG      "Cannot use 'self' outside of a class."

#define COMPILE_RETURNININITIALIZER       "InitRtn"
#define COMPILE_RETURNININITIALIZER_MSG   "Cannot return a value in an initializer."

#define COMPILE_SUPERCLASSNOTFOUND        "SprNtFnd"
#define COMPILE_SUPERCLASSNOTFOUND_MSG    "Superclass '%s' not found."

#define COMPILE_SUPEROUTSIDECLASS         "SprOtsdClss"
#define COMPILE_SUPEROUTSIDECLASS_MSG     "Cannot use 'super' outside of a class."

#define COMPILE_NOSUPER                   "SprSelMthd"
#define COMPILE_NOSUPER_MSG               "Can only use 'super' to select a method."

#define COMPILE_INVALIDASSIGNMENT         "InvldAssgn"
#define COMPILE_INVALIDASSIGNMENT_MSG     "Invalid assignment target."

#define COMPILE_CLASSINHERITSELF          "ClssCrcRf"
#define COMPILE_CLASSINHERITSELF_MSG      "A class cannot inherit from itself."

#define COMPILE_TOOMANYARGS               "TooMnyArg"
#define COMPILE_TOOMANYARGS_MSG           "Too many arguments."

#define COMPILE_TOOMANYPARAMS             "TooMnyPrm"
#define COMPILE_TOOMANYPARAMS_MSG         "Too many parameters."

#define COMPILE_ISOLATEDSUPER             "IsoSpr"
#define COMPILE_ISOLATEDSUPER_MSG         "Expect '.' after 'super'."

#define COMPILE_VARALREADYDECLARED        "VblDcl"
#define COMPILE_VARALREADYDECLARED_MSG    "Variable with this name already declared in this scope."

#define COMPILE_FILENOTFOUND              "FlNtFnd"
#define COMPILE_FILENOTFOUND_MSG          "File '%s' not found."

#define COMPILE_MODULENOTFOUND            "MdlNtFnd"
#define COMPILE_MODULENOTFOUND_MSG        "Module '%s' not found."

#define COMPILE_IMPORTFLD                 "ImprtFld"
#define COMPILE_IMPORTFLD_MSG             "Import of file '%s' failed."

#define COMPILE_BRKOTSDLP                 "BrkOtsdLp"
#define COMPILE_BRKOTSDLP_MSG             "Break encountered outside a loop."

#define COMPILE_CNTOTSDLP                 "CntOtsdLp"
#define COMPILE_CNTOTSDLP_MSG             "Continue encountered outside a loop."

#define COMPILE_OPTPRMDFLT                "OptPrmDflt"
#define COMPILE_OPTPRMDFLT_MSG            "Optional parameter default values must be constants."

#define COMPILE_FORWARDREF                "UnrslvdFrwdRf"
#define COMPILE_FORWARDREF_MSG            "Function '%s' is called but not defined in the same scope."

#define COMPILE_MLTVARPRMTR               "MltVarPrmtr"
#define COMPILE_MLTVARPRMTR_MSG           "Functions can have at most one variadic parameter."

#define COMPILE_VARPRMLST                 "VarPrLst"
#define COMPILE_VARPRMLST_MSG             "Cannot have fixed parameters after a variadic parameter."

#define COMPILE_MSSNGLOOPBDY              "MssngLoopBdy"
#define COMPILE_MSSNGLOOPBDY_MSG          "Missing loop body."

#define COMPILE_NSTDCLSS                  "NstdClss"
#define COMPILE_NSTDCLSS_MSG              "Cannot define a class within another class."

#define COMPILE_INVLDLBL                  "InvldLbl"
#define COMPILE_INVLDLBL_MSG              "Invalid label in catch statment."

/* **********************************************************************
 * Compiler typedefs
 * ********************************************************************** */

/* -------------------------------------------------------
 * Track globals
 * ------------------------------------------------------- */

/** @brief Index of a global */
typedef int globalindx;

/** @brief Value to indicate a global has not been allocated */
#define GLOBAL_UNALLOCATED -1

/* -------------------------------------------------------
 * Track register allocation
 * ------------------------------------------------------- */

/** @brief Index of a register */
typedef int registerindx;

/** @brief Value to indicate a register has not been allocated */
#define REGISTER_UNALLOCATED -1

/** This structure tracks the contents of each register as the function
    is being compiled. */
typedef struct {
    bool isallocated; /** Whether the register has been allocated */
    bool iscaptured; /** Whether the register becomes an upvalue */
    unsigned int scopedepth; /** Scope depth at which the register was allocated */
    value symbol; /** Symbol associated with the register */
} registeralloc;

DECLARE_VARRAY(registeralloc, registeralloc)

/* -------------------------------------------------------
 * Codeinfo
 * ------------------------------------------------------- */

typedef enum {
    REGISTER,
    CONSTANT,
    UPVALUE,
    GLOBAL,
    VALUE,  // Used by the optimizer
    NOTHING //
} returntype;

typedef struct {
    returntype returntype;
    registerindx dest;
    unsigned int ninstructions;
} codeinfo;

#define CODEINFO_ISREGISTER(info) (info.returntype==REGISTER)
#define CODEINFO_ISCONSTANT(info) (info.returntype==CONSTANT)
#define CODEINFO_ISSHORTCONSTANT(info) (info.returntype==CONSTANT && info.dest<MORPHO_MAXREGISTERS)
#define CODEINFO_ISUPVALUE(info) (info.returntype==UPVALUE)
#define CODEINFO_ISGLOBAL(info) (info.returntype==GLOBAL)

#define CODEINFO(c, d, n) ((codeinfo) { .returntype=(c), .dest=(d), .ninstructions=(n)})
#define CODEINFO_EMPTY CODEINFO(REGISTER, REGISTER_UNALLOCATED, 0)

/* -------------------------------------------------------
 * Forward function references
 * ------------------------------------------------------- */

typedef struct {
    value symbol; /** Symbol associated with the reference */
    syntaxtreenode *node; /** Syntax tree node associated with the forward reference */
    returntype returntype; /** Return type */
    registerindx dest; /** Index into constant table */
    unsigned int scopedepth; /** Scope depth at which the function with the forward reference occurred */
} forwardreference;

DECLARE_VARRAY(forwardreference, forwardreference)

/* -------------------------------------------------------
 * Function types
 * ------------------------------------------------------- */

/** The type of the current function */
typedef enum {
    FUNCTION,
    METHOD, /* For methods */
    INITIALIZER /* For initializer methods */
} functiontype;

#define FUNCTIONTYPE_ISMETHOD(f) (f==FUNCTION ? false : true)

#define FUNCTIONTYPE_ISINITIALIZER(f) (f==INITIALIZER)

/** This structure tracks compiler information for the current function. */
typedef struct {
    objectfunction *func;
    functiontype type;
    varray_registeralloc registers;
    varray_upvalue upvalues;
    varray_forwardreference forwardref;
    registerindx varg;
    unsigned int nreg; /* Largest number of registers used */
    unsigned int scopedepth;
    unsigned int loopdepth; /* Count number of nesting depths of a loop */
    bool inargs; /* Set while compiling function calls to ensure allocations are at the top of the stack */
} functionstate;

/* -------------------------------------------------------
 * Lists
 * ------------------------------------------------------- */

/** This structure holds a list as it is being created */
typedef struct scompilerlist {
    varray_value entries;
    struct scompilerlist *next;
} compilerlist;

/* -------------------------------------------------------
 * Namespaces
 * ------------------------------------------------------- */

typedef struct _namespc {
    value label; /** Label  */
    dictionary symbols; /** Symbol table */
    
    struct _namespc *next; /** Make a linked list of namespaces */
} namespc;

/* -------------------------------------------------------
 * Overall state of the compiler
 * ------------------------------------------------------- */

/** @brief A structure that stores the state of a compiler */
typedef struct scompiler {
    lexer lex;
    parser parse;
    
    syntaxtree tree;
    
    error err;
    
    /* Line number */
    int line; 
    
    /* Globals */
    dictionary globals;
    
    /* Function state stack */
    functionstate fstack[MORPHO_CALLFRAMESTACKSIZE];
    indx fstackp;
    
    /* Most recently completed function declaration; used to bind method definitions */
    objectfunction *prevfunction;
    
    /* Current class being compiled */
    objectclass *currentclass;
    
    /* Syntax tree node of the current method being compiled */
    syntaxtreenode *currentmethod;
    
    /* Namespaces */
    namespc *namespaces;
    
    /* Current module */
    value currentmodule;
    
    /* Program we're compiling to */
    program *out;
    
    /* Modules included */
    dictionary modules;
    
    /* The parent compiler */
    struct scompiler *parent;
} compiler;

/* -------------------------------------------------------
 * AST nodes are now compiled by the bytecode compiler
 * ------------------------------------------------------- */

/** A compiler_nodefn takes a syntax tree node and compiles it to bytecode */
typedef codeinfo (*compiler_nodefn) (compiler *c, syntaxtreenode *node, registerindx reg);

/** @brief A compilenoderule rule will be defined for each syntax tree node type,
 *  providing a function to compile the node. */
typedef struct {
    compiler_nodefn nodefn;
} compilenoderule;

void compiler_init(const char *source, program *out, compiler *c);
void compiler_clear(compiler *c);

#endif /* compile_h */
