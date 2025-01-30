/** @file morpho.h
 *  @author T J Atherton
 *
 *  @brief Define public interface to Morpho
 */

#ifndef morpho_h
#define morpho_h

#include "build.h"
#include "value.h"
#include "error.h"
#include "dictionary.h"
#include "version.h"

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
#define MORPHO_TOSTRING_METHOD "tostring"
#define MORPHO_FORMAT_METHOD "format"

#define MORPHO_ASSIGN_METHOD "assign"
#define MORPHO_ADD_METHOD "add"
#define MORPHO_ADDR_METHOD "addr" 
#define MORPHO_SUB_METHOD "sub"
#define MORPHO_SUBR_METHOD "subr"
#define MORPHO_MUL_METHOD "mul"
#define MORPHO_MULR_METHOD "mulr"
#define MORPHO_DIV_METHOD "div"
#define MORPHO_DIVR_METHOD "divr"
#define MORPHO_POW_METHOD "pow"
#define MORPHO_POWR_METHOD "powr"
#define MORPHO_ACC_METHOD "acc"
#define MORPHO_SUM_METHOD "sum"

#define MORPHO_CONTAINS_METHOD "contains"

#define MORPHO_UNION_METHOD "union"
#define MORPHO_INTERSECTION_METHOD "intersection"
#define MORPHO_DIFFERENCE_METHOD "difference"

#define MORPHO_ROLL_METHOD "roll"
#define MORPHO_JOIN_METHOD "join"

#define MORPHO_CLASS_METHOD "clss"
#define MORPHO_SUPER_METHOD "superclass"
#define MORPHO_SERIALIZE_METHOD "serialize"
#define MORPHO_HAS_METHOD "has"
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
#define MORPHO_LINEARIZATION_METHOD "linearization"

#define MORPHO_THROW_METHOD "throw"
#define MORPHO_WARNING_METHOD "warning"

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

/* Version checking */
void morpho_version(version *v);

/* Error handling */
void morpho_writeerrorwithid(error *err, errorid id, char *file, int line, int posn, ...);
void morpho_defineerror(errorid id, errorcategory cat, char *message);
errorid morpho_geterrorid(error *err);

/* Programs */
program *morpho_newprogram(void);
void morpho_freeprogram(program *p);

/* Optimizers */
typedef bool (optimizerfn) (program *in);
void morpho_setoptimizer(optimizerfn *optimizer);

/* Virtual machine */
vm *morpho_newvm(void);
void morpho_freevm(vm *v);

/* Bind new objects to the virtual machine */
void morpho_bindobjects(vm *v, int nobj, value *obj);
value morpho_wrapandbind(vm *v, object *obj);

/* Interact with the garbage collector in an object definition */
void morpho_markobject(void *v, object *obj);
void morpho_markvalue(void *v, value val);
void morpho_markvarrayvalue(void *v, varray_value *array);
void morpho_markdictionary(void *v, dictionary *dict);
void morpho_searchunmanagedobject(void *v, object *obj);
bool morpho_ismanagedobject(object *obj); 

/* Tell the VM that the size of an object has changed */
void morpho_resizeobject(vm *v, object *obj, size_t oldsize, size_t newsize);

/* Temporarily retain objects across multiple calls into the VM */
int morpho_retainobjects(vm *v, int nobj, value *obj);
void morpho_releaseobjects(vm *v, int handle);

/* Raise runtime errors and warnings */
void morpho_warning(vm *v, error *err);
void morpho_error(vm *v, error *err);

void morpho_runtimeerror(vm *v, errorid id, ...);
void morpho_runtimewarning(vm *v, errorid id, ...);

/* Compilation */
compiler *morpho_newcompiler(program *out);
void morpho_freecompiler(compiler *c);
bool morpho_compile(char *in, compiler *c, bool optimize, error *err);
const char *morpho_compilerrestartpoint(compiler *c);
void morpho_resetentry(program *p);

/* Interpreting */
bool morpho_run(vm *v, program *p);
bool morpho_profile(vm *v, program *p);
bool morpho_debug(vm *v, program *p);
bool morpho_lookupmethod(value obj, value label, value *method);
bool morpho_countparameters(value f, int *nparams);
bool morpho_call(vm *v, value fn, int nargs, value *args, value *ret);
bool morpho_invoke(vm *v, value obj, value method, int nargs, value *args, value *ret);
error *morpho_geterror(vm *v);

/* I/O */
int morpho_printf(vm *v, char *format, ...);
void morpho_printvalue(vm *v, value val);
int morpho_readline(vm *v, varray_char *buffer);

/* Stack trace */
void morpho_stacktrace(vm *v);

/* Disassembler */
void morpho_disassemble(vm *v, program *code, int *matchline);

/* Multithreading */
void morpho_setthreadnumber(int nthreads);
int morpho_threadnumber(void);

/* Initialization and finalization */
typedef void (*morpho_finalizefn) (void);
void morpho_addfinalizefn(morpho_finalizefn finalizefn);
void morpho_setbaseclass(value clss);
void morpho_initialize(void);
void morpho_finalize(void);
void morpho_setargs(int argc, const char * argv[]); // Pass arguments to morpho

/* Obtain and use subkernels [for internal use only] */
bool vm_subkernels(vm *v, int nkernels, vm **subkernels);
void vm_releasesubkernel(vm *subkernel);
void vm_cleansubkernel(vm *subkernel);

/* Thread local storage [for internal use only] */
int vm_addtlvar(void);
bool vm_settlvar(vm *v, int handle, value val);
bool vm_gettlvar(vm *v, int handle, value *out);

#endif /* morpho_h */
