/** @file signature.c
 *  @author T J Atherton
 *
 *  @brief Function signatures and their declarations
*/

#include "morpho.h"
#include "classes.h"

#include "signature.h"
#include "parse.h"

/* **********************************************************************
 * Manage signature structures
 * ********************************************************************** */

void signature_init(signature *s) {
    varray_valueinit(&s->types);
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

/** @brief Returns true if any entries in the signature are typed*/
bool signature_istyped(signature *s) {
    for (int i=0; i<s->types.count; i++) if (!MORPHO_ISNIL(s->types.data[i])) return true;
    return false;
}

/** @brief Return list of types */
bool signature_paramlist(signature *s, int *nparams, value **ptypes) {
    if (nparams) *nparams = s->types.count;
    if (ptypes) *ptypes = s->types.data;
    return s->types.data;
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

/** @brief Parser function to process a symbol held in p->previous */
bool signature_parsesymbol(parser *p, void *out) {
    signature *sig = (signature *) out;
    bool success=false;
    
    if (p->previous.length==1 && *p->previous.start=='_') {
        value blank = MORPHO_NIL;
        success=varray_valueadd(&sig->types, &blank, 1);
    } else {
        value symbol;
        if (!parse_stringfromtoken(p, 0, p->previous.length, &symbol)) return false;
        value klass = builtin_findclass(symbol);
        morpho_freeobject(symbol);
        
        if (MORPHO_ISCLASS(klass)) success=varray_valueadd(&sig->types, &klass, 1);
    }
    return success;
}

/** @brief Main parser function for signatures */
bool signature_parsesignature(parser *p, void *out) {
    if (parse_checktokenadvance(p, SIGNATURE_SYMBOL)) {
        // Return type
    }
    
    if (!parse_checktokenadvance(p, SIGNATURE_LEFTBRACE)) return false;
    
    while (!parse_checktoken(p, SIGNATURE_RIGHTBRACE) &&
           !parse_checktoken(p, SIGNATURE_EOF)) {
        if (parse_checktokenadvance(p, SIGNATURE_SYMBOL)) {
            if (!signature_parsesymbol(p, out)) return false;
        } else if (parse_checktokenadvance(p, SIGNATURE_DOTDOTDOT)) {
            
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
    return success;
}

/** Print a signature for debugging purposes */
void signature_print(signature *s) {
    printf("(");
    for (int i=0; i<s->types.count; i++) {
        value type=s->types.data[i];
        if (MORPHO_ISNIL(type)) printf("_");
        else if (MORPHO_ISCLASS(type)) morpho_printvalue(NULL, MORPHO_GETCLASS(type)->name);
        
        if (i<s->types.count-1) printf(",");
    }
    printf(")\n");
}
