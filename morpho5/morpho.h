/** @file morpho.h
 *  @author T J Atherton
 *
 *  @brief Define public interface to Morpho
 */

#ifndef morpho_h
#define morpho_h

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "build.h"
#include "value.h"
#include "error.h"

/* **********************************************************************
* VM types
* ********************************************************************** */

#ifndef MORPHO_CORE
typedef void vm;
typedef void program;
typedef void compiler;
#endif

/* **********************************************************************
* Standard methods
* ********************************************************************** */

#define MORPHO_INITIALIZER_METHOD "init"

#define MORPHO_GETINDEX_METHOD "index"
#define MORPHO_SETINDEX_METHOD "setindex"

#define MORPHO_ADD_METHOD "add"
#define MORPHO_ADDR_METHOD "addr" 
#define MORPHO_SUB_METHOD "sub"
#define MORPHO_SUBR_METHOD "subr"
#define MORPHO_MUL_METHOD "mul"
#define MORPHO_MULR_METHOD "mulr"
#define MORPHO_DIV_METHOD "div"
#define MORPHO_DIVR_METHOD "divr"
#define MORPHO_ACC_METHOD "acc"
#define MORPHO_SUM_METHOD "sum"

#define MORPHO_UNION_METHOD "union"
#define MORPHO_INTERSECTION_METHOD "intersection"
#define MORPHO_DIFFERENCE_METHOD "difference"

#define MORPHO_CLASS_METHOD "clss"
#define MORPHO_SUPER_METHOD "superclass"
#define MORPHO_SERIALIZE_METHOD "serialize"
#define MORPHO_RESPONDSTO_METHOD "respondsto"
#define MORPHO_INVOKE_METHOD "invoke"
#define MORPHO_CLONE_METHOD "clone"

#define MORPHO_ENUMERATE_METHOD "enumerate"
#define MORPHO_COUNT_METHOD "count"
#define MORPHO_CLONE_METHOD "clone"
#define MORPHO_PRINT_METHOD "prnt"
#define MORPHO_SAVE_METHOD "save"

/* Non-standard methods */
#define MORPHO_APPEND_METHOD "append"

extern value initselector;
extern value indexselector;
extern value setindexselector;
extern value addselector;
extern value subselector;
extern value mulselector;
extern value divselector;
extern value printselector;
extern value enumerateselector;
extern value countselector;
extern value cloneselector;

/* **********************************************************************
* Public interfaces
* ********************************************************************** */

/* Error handling */
void morpho_writeerrorwithid(error *err, errorid id, int line, int position, ...);
void morpho_defineerror(errorid id, errorcategory cat, char *message);
errorid morpho_geterrorid(error *err);

/* Programs */
program *morpho_newprogram(void);
void morpho_freeprogram(program *p);

/* Virtual machine */
vm *morpho_newvm(void);
void morpho_freevm(vm *v);

/* Bind new objects to the virtual machine */
void morpho_bindobjects(vm *v, int nobj, value *obj);

/* Raise runtime errors */
void morpho_runtimeerror(vm *v, errorid id, ...);

/* Activate/deactivate the debugger */
void morpho_setdebug(vm *v, bool active);

/* Compilation */
compiler *morpho_newcompiler(program *out);
void morpho_freecompiler(compiler *c);
bool morpho_compile(char *in, compiler *c, error *err);
const char *morpho_compilerrestartpoint(compiler *c);
void morpho_resetentry(program *p);

/* Interpreting */
bool morpho_run(vm *v, program *p);
bool morpho_lookupmethod(vm *v, value obj, value label, value *method);
bool morpho_call(vm *v, value fn, int nargs, value *args, value *ret);
bool morpho_invoke(vm *v, value obj, value method, int nargs, value *args, value *ret);
error *morpho_geterror(vm *v);

void morpho_printvalue(value v);

/* Disassembly */
void morpho_disassemble(program *code, int *matchline);
void morpho_stacktrace(vm *v);

/* Initialization and finalization */
void morpho_setbaseclass(value clss);
void morpho_initialize(void);
void morpho_finalize(void);

#endif /* morpho_h */
