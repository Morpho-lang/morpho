/** @file function.c
 *  @author T J Atherton
 *
 *  @brief Implement objectfunctions and the Function veneer class
 */

#include "morpho.h"
#include "classes.h"
#include "common.h"

/* **********************************************************************
 * objectfunction definitions
 * ********************************************************************** */

void objectfunction_freefn(object *obj) {
    objectfunction *func = (objectfunction *) obj;
    morpho_freeobject(func->name);
    varray_optionalparamclear(&func->opt);
    object_functionclear(func);
    signature_clear(&func->sig);
}

void objectfunction_markfn(object *obj, void *v) {
    objectfunction *f = (objectfunction *) obj;
    morpho_markvalue(v, f->name);
    morpho_markvarrayvalue(v, &f->konst);
}

size_t objectfunction_sizefn(object *obj) {
    return sizeof(objectfunction);
}

void objectfunction_printfn(object *obj, void *v) {
    objectfunction *f = (objectfunction *) obj;
    if (f) morpho_printf(v, "<fn %s>", (MORPHO_ISNIL(f->name) ? "" : MORPHO_GETCSTRING(f->name)));
}

objecttypedefn objectfunctiondefn = {
    .printfn=objectfunction_printfn,
    .markfn=objectfunction_markfn,
    .freefn=objectfunction_freefn,
    .sizefn=objectfunction_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/* **********************************************************************
 * objectfunction utility functions
 * ********************************************************************** */

/** @brief Initializes a new function */
void object_functioninit(objectfunction *func) {
    func->entry=0;
    func->name=MORPHO_NIL;
    func->nargs=0;
    func->nopt=0;
    func->parent=NULL;
    func->creg=-1;
    func->nregs=0;
    varray_valueinit(&func->konst);
    varray_varray_upvalueinit(&func->prototype);
    signature_init(&func->sig);
}

/** @brief Clears a function */
void object_functionclear(objectfunction *func) {
    varray_valueclear(&func->konst);
    /** Clear the upvalue prototypes */
    for (unsigned int i=0; i<func->prototype.count; i++) {
        varray_upvalueclear(&func->prototype.data[i]);
    }
    varray_varray_upvalueclear(&func->prototype);
    signature_clear(&func->sig);
}

/** @brief Creates a new function */
objectfunction *object_newfunction(indx entry, value name, objectfunction *parent, unsigned int nargs) {
    objectfunction *new = (objectfunction *) object_new(sizeof(objectfunction), OBJECT_FUNCTION);

    if (new) {
        object_functioninit(new);
        new->entry=entry;
        new->name=object_clonestring(name);
        new->nargs=nargs;
        new->varg=-1; // No vargs
        new->parent=parent;
        new->klass=NULL;
        varray_optionalparaminit(&new->opt);
    }

    return new;
}

/** Gets the parent of a function */
objectfunction *object_getfunctionparent(objectfunction *func) {
    return func->parent;
}

/** Gets the name of a function */
value object_getfunctionname(objectfunction *func) {
    return func->name;
}

/** Gets the constant table associated with a function */
varray_value *object_functiongetconstanttable(objectfunction *func) {
    if (func) {
        return &func->konst;
    }
    return NULL;
}

/** Adds an upvalue prototype to a function
 * @param[in]  func   function object to add to
 * @param[in]  v      a varray of upvalues that will be copied into the function
 *                    definition.
 * @param[out] ix     index of the closure created
 * @returns true on success */
bool object_functionaddprototype(objectfunction *func, varray_upvalue *v, indx *ix) {
    bool success=false;
    varray_upvalue new;
    varray_upvalueinit(&new);
    success=varray_upvalueadd(&new, v->data, v->count);
    if (success) varray_varray_upvalueadd(&func->prototype, &new, 1);
    if (success && ix) *ix = (indx) func->prototype.count-1;
    return success;
}

/** Returns the number of positional arguments (including a variadic arg if any) */
int function_countpositionalargs(objectfunction *func) {
    return func->nargs + (func->varg>=0 ? 1 : 0);
}

/** Returns the number of optional arguments */
int function_countoptionalargs(objectfunction *func) {
    return func->nopt;
}

/** Does a function have variadic args? */
bool function_hasvargs(objectfunction *func) {
    return (func->varg>=0);
}

/** Sets the parameter number of a variadic argument */
void function_setvarg(objectfunction *func, int varg) {
    func->varg=varg;
}

/** Sets that a function must be enclosed */
void function_setclosure(objectfunction *func, int creg) {
    func->creg=creg;
}

/** Checks if a function is enclosed */
bool function_isclosure(objectfunction *func) {
    return (func->creg>=0);
}

/** Sets the signature of a function
 * @param[in]  func   function object
 * @param[in]  signature list of types for each parameter (length from func->nargs) */
void function_setsignature(objectfunction *func, value *signature) {
    signature_set(&func->sig, function_countpositionalargs(func), signature);
}

/** Returns true if any of the parameters are typed */
bool function_hastypedparameters(objectfunction *func) {
    return signature_istyped(&func->sig);
}

/* **********************************************************************
 * Function veneer class
 * ********************************************************************** */

value Function_tostring(vm *v, int nargs, value *args) {
    objectfunction *func=MORPHO_GETFUNCTION(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    varray_char buffer;
    varray_charinit(&buffer);

    varray_charadd(&buffer, "<fn ", 4);
    morpho_printtobuffer(v, func->name, &buffer);
    varray_charwrite(&buffer, '>');

    out = object_stringfromvarraychar(&buffer);
    if (MORPHO_ISSTRING(out)) {
        morpho_bindobjects(v, 1, &out);
    }
    varray_charclear(&buffer);

    return out;
}

MORPHO_BEGINCLASS(Function)
MORPHO_METHOD(MORPHO_TOSTRING_METHOD, Function_tostring, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Object_print, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization and finalization
 * ********************************************************************** */

objecttype objectfunctiontype;

void function_initialize(void) {
    // Create function object type
    objectfunctiontype=object_addtype(&objectfunctiondefn);
    
    // Locate the Object class to use as the parent class of Function
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // Create function veneer class
    value functionclass=builtin_addclass(FUNCTION_CLASSNAME, MORPHO_GETCLASSDEFINITION(Function), objclass);
    object_setveneerclass(OBJECT_FUNCTION, functionclass);
    
    // No constructor as objectfunctions are generated by the compiler
    
    // Function error messages
}
