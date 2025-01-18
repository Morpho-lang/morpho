/** @file signature.c
 *  @author T J Atherton
 *
 *  @brief Function signatures and their declarations
*/

#include "classes.h"

#include "signature.h"
#include "parse.h"

/* **********************************************************************
 * Manage signature structures
 * ********************************************************************** */

void signature_init(signature *s) {
    varray_valueinit(&s->types);
    s->ret=MORPHO_NIL;
    s->varg=false;
}

void signature_clear(signature *s) {
    varray_valueclear(&s->types);
}

/** @brief Sets the contents of a signature
 *  @param[in] s - the signature structure
 *  @param[in] nparam - number of fixed parameters
 *  @param[in] types - list of types of each parameter */
void signature_set(signature *s, int nparam, value *types) {
    s->types.count=0; // Reset
    varray_valueadd(&s->types, types, nparam);
}

/** @brief Sets whether a signature contains variadic arguments */
void signature_setvarg(signature *s, bool varg) {
    s->varg=varg;
}

/** @brief Sets whether a signature contains variadic arguments */
bool signature_isvarg(signature *s) {
    return s->varg;
}

/** @brief Returns true if any entries in the signature are typed*/
bool signature_istyped(signature *s) {
    for (int i=0; i<s->types.count; i++) if (!MORPHO_ISNIL(s->types.data[i])) return true;
    return false;
}

/** @brief Check if two signatures are equal */
bool signature_isequal(signature *a, signature *b) {
    if (a->types.count!=b->types.count) return false;
    for (int i=0; i<a->types.count; i++) if (!MORPHO_ISEQUAL(a->types.data[i], b->types.data[i])) return false;
    return true; 
}

/** @brief Return list of types */
bool signature_paramlist(signature *s, int *nparams, value **ptypes) {
    if (nparams) *nparams = s->types.count;
    if (ptypes) *ptypes = s->types.data;
    return s->types.data;
}

/** @brief Returns the type of the i'th parameter, if it exists */
bool signature_getparamtype(signature *s, int i, value *type) {
    if (i>=s->types.count) return false;
    if (type) *type = s->types.data[i];
    return true; 
}

/** @brief Returns the return type from the signature if defined */
value signature_getreturntype(signature *s) {
    return s->ret;
}

/** @brief Count the number of parameters in a signature */
int signature_countparams(signature *s) {
    return s->types.count;
}

/* **********************************************************************
 * Parse signatures
 * ********************************************************************** */

enum {
    SIGNATURE_LEFTBRACE,
    SIGNATURE_RIGHTBRACE,
    SIGNATURE_COMMA,
    SIGNATURE_DOTDOTDOT,
    SIGNATURE_SYMBOL,
    SIGNATURE_EOF
};

tokendefn sigtokens[] = {
    { "(",          SIGNATURE_LEFTBRACE         , NULL },
    { ")",          SIGNATURE_RIGHTBRACE        , NULL },
    { ",",          SIGNATURE_COMMA             , NULL },
    { "...",        SIGNATURE_DOTDOTDOT         , NULL },
    { "",           SIGNATURE_EOF               , NULL }
};

/** @brief Initializes a lexer for parsing signatures */
void signature_initializelexer(lexer *l, char *signature) {
    lex_init(l, signature, 0);
    lex_settokendefns(l, sigtokens);
    lex_seteof(l, SIGNATURE_EOF);
    lex_setsymboltype(l, SIGNATURE_SYMBOL);
}

/** @brief Parses a type name held in the parser's previous type */
bool signature_parsetype(parser *p, value *type) {
    value symbol;
    if (!parse_stringfromtoken(p, 0, p->previous.length, &symbol)) return false;
    value klass = builtin_findclass(symbol);
    morpho_freeobject(symbol);
    
    if (MORPHO_ISCLASS(klass)) *type=klass;
    return MORPHO_ISCLASS(klass);
}

/** @brief Parser function to process a symbol held in p->previous */
bool signature_parsesymbol(parser *p, void *out) {
    signature *sig = (signature *) out;
    bool success=false;
    
    if (p->previous.length==1 && *p->previous.start=='_') {
        value blank = MORPHO_NIL;
        success=varray_valueadd(&sig->types, &blank, 1);
    } else {
        value type;
        if (signature_parsetype(p, &type)) success=varray_valueadd(&sig->types, &type, 1);
    }
    return success;
}

/** @brief Parser function to process varg */
bool signature_parsevarg(parser *p, void *out) {
    signature *sig = (signature *) out;
    
    value blank = MORPHO_NIL;
    bool success=varray_valueadd(&sig->types, &blank, 1);
    sig->varg=true;
    
    return success;
}

/** @brief Main parser function for signatures */
bool signature_parsesignature(parser *p, void *out) {
    signature *sig = (signature *) out;
    
    if (parse_checktokenadvance(p, SIGNATURE_SYMBOL)) {
        value type;
        if (signature_parsetype(p, &type)) sig->ret=type;
    }
    
    if (!parse_checktokenadvance(p, SIGNATURE_LEFTBRACE)) return false;
    
    while (!parse_checktoken(p, SIGNATURE_RIGHTBRACE) &&
           !parse_checktoken(p, SIGNATURE_EOF)) {
        if (parse_checktokenadvance(p, SIGNATURE_SYMBOL)) {
            if (!signature_parsesymbol(p, out)) return false;
        } else if (parse_checktokenadvance(p, SIGNATURE_DOTDOTDOT)) {
            if (!signature_parsevarg(p, out)) return false; 
        } else return false;
        
        parse_checktokenadvance(p, SIGNATURE_COMMA);
    }
    
    if (!parse_checktokenadvance(p, SIGNATURE_RIGHTBRACE)) return false;
    
    return true;
}

/** Parses a signature */
bool signature_parse(char *sig, signature *out) {
    error err;
    error_init(&err);
    
    lexer l;
    signature_initializelexer(&l, sig);
    
    parser p;
    parse_init(&p, &l, &err, out);
    parse_setbaseparsefn(&p, signature_parsesignature);
    parse_setskipnewline(&p, false, TOKEN_NONE);
    
    bool success=parse(&p);
    
    parse_clear(&p);
    lex_clear(&l);
    
    return success;
}

/** Print a signature for debugging purposes */
void signature_print(signature *s) {
    printf("(");
    for (int i=0; i<s->types.count; i++) {
        value type=s->types.data[i];
        
        if (s->varg && i==s->types.count-1) printf("...");
        else if (MORPHO_ISNIL(type)) printf("_");
        else if (MORPHO_ISCLASS(type)) morpho_printvalue(NULL, MORPHO_GETCLASS(type)->name);
        
        if (i<s->types.count-1) printf(",");
    }
    printf(")");
}
