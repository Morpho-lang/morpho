/** @file json.c
 *  @author T J Atherton
 *
 *  @brief JSON parser
 */

#include "morpho.h"
#include "classes.h"

#include "parse.h"
#include "dictionary.h"
#include "json.h"

/* **********************************************************************
 * JSON lexer
 * ********************************************************************** */

/* -------------------------------------------------------
 * JSON token types and process functions
 * ------------------------------------------------------- */

bool json_lexstring(lexer *l, token *tok, error *err);
bool json_lexnumber(lexer *l, token *tok, error *err);

enum {
    JSON_LEFTCURLYBRACE,
    JSON_RIGHTCURLYBRACE,
    JSON_LEFTSQUAREBRACE,
    JSON_RIGHTSQUAREBRACE,
    JSON_COMMA,
    JSON_COLON,
    JSON_MINUS,
    JSON_QUOTE,
    JSON_TRUE,
    JSON_FALSE,
    JSON_NULL,
    JSON_STRING,
    JSON_NUMBER,
    JSON_FLOAT,
    JSON_EOF
};

tokendefn jsontokens[] = {
    { "{",          JSON_LEFTCURLYBRACE         , NULL },
    { "}",          JSON_RIGHTCURLYBRACE        , NULL },
    { "[",          JSON_LEFTSQUAREBRACE        , NULL },
    { "]",          JSON_RIGHTSQUAREBRACE       , NULL },
    { ",",          JSON_COMMA                  , NULL },
    { ":",          JSON_COLON                  , NULL },
    { "-",          JSON_MINUS                  , json_lexnumber },
    { "\"",         JSON_QUOTE                  , json_lexstring },
    { "true",       JSON_TRUE                   , NULL },
    { "false",      JSON_FALSE                  , NULL },
    { "null",       JSON_NULL                   , NULL },
    { "",           TOKEN_NONE                  , NULL }
};

/** Skip over JSON whitespace */
bool json_lexwhitespace(lexer *l, token *tok, error *err) {
    for (;;) {
        char c = lex_peek(l);
        
        switch (c) {
            case '\n':
                lex_newline(l); // V Intentional fallthrough
            case ' ':
            case '\t':
            case '\r':
                lex_advance(l);
                break;
            default:
                return true;
        }
    }
    return true;
}

/** Record JSON strings as a token */
bool json_lexstring(lexer *l, token *tok, error *err) {
    char c;
    
    while (lex_peek(l) != '"' && !lex_isatend(l)) {
        if (lex_peek(l)=='\\') lex_advance(l); // Detect an escaped character

        lex_advance(l);
    }
    
    if (lex_isatend(l)) {
        morpho_writeerrorwithid(err, LEXER_UNTERMINATEDSTRING, tok->line, tok->posn);
        return false;
    }
    
    lex_advance(l); // Advance over final quote
    lex_recordtoken(l, JSON_STRING, tok);
    
    return true;
}

/** Record JSON numbers as a token */
bool json_lexnumber(lexer *l, token *tok, error *err) {
    tokentype type = JSON_NUMBER;
    while (lex_isdigit(lex_peek(l)) && !lex_isatend(l)) lex_advance(l);
    
    if (lex_peek(l)=='.') { lex_advance(l); type = JSON_FLOAT; };
    
    while (lex_isdigit(lex_peek(l)) && !lex_isatend(l)) lex_advance(l);
    
    if (lex_peek(l)=='e' || lex_peek(l)=='E') { lex_advance(l); type = JSON_FLOAT; };
    if (lex_peek(l)=='+' || lex_peek(l)=='-') lex_advance(l);
    
    while (lex_isdigit(lex_peek(l)) && !lex_isatend(l)) lex_advance(l);
    
    lex_recordtoken(l, type, tok);
    
    return true;
}

/** Lexer token preprocessor function */
bool json_lexpreprocess(lexer *l, token *tok, error *err) {
    char c = lex_peek(l);
    if (lex_isdigit(c)) return json_lexnumber(l, tok, err);
    return false;
}

/* -------------------------------------------------------
 * Initialize a JSON lexer
 * ------------------------------------------------------- */

void json_initializelexer(lexer *l, char *src) {
    lex_init(l, src, 0);
    lex_settokendefns(l, jsontokens);
    lex_setprefn(l, json_lexpreprocess);
    lex_setwhitespacefn(l, json_lexwhitespace);
    lex_seteof(l, JSON_EOF);
}

/* **********************************************************************
 * JSON parser
 * ********************************************************************** */

/* -------------------------------------------------------
 * JSON parse functions
 * ------------------------------------------------------- */

bool json_parsevalue(parser *p, void *out);

/** Convenience function to place a value in the opaque out pointer */
void json_output(void *out, value v) {
    * ((value *) out) = v;
}

bool json_parsestring(parser *p, void *out) {
    bool success=false;
    varray_char str;
    varray_charinit(&str);
    
    const char *input = p->previous.start;
    unsigned int length = p->previous.length;
    
    for (unsigned int i=1; i<length-1; i++) {
        if (iscntrl(input[i])) {
            parse_error(p, true, JSON_UNESCPDCTRL);
            goto json_parsestring_cleanup;
        } else if (input[i]!='\\') {
            varray_charwrite(&str, input[i]);
        } else {
            i++;
            switch (input[i]) {
                case '\"': varray_charwrite(&str, '\"'); break;
                case '\\': varray_charwrite(&str, '\\'); break;
                case '/': varray_charwrite(&str, '/'); break;
                case 'b': varray_charwrite(&str, '\b'); break;
                case 'f': varray_charwrite(&str, '\f'); break;
                case 'n': varray_charwrite(&str, '\n'); break;
                case 'r': varray_charwrite(&str, '\r'); break;
                case 't': varray_charwrite(&str, '\t'); break;
                case 'u':
                {
                    long c = strtol(&input[i+2], NULL, 16);
                    
                }
                    break;
                default:
                    parse_error(p, true, PARSE_STRESC);
                    goto json_parsestring_cleanup;
            }
        }
    }
    
    objectstring *new = object_stringfromvarraychar(&str);
    if (new) {
        success=true;
        json_output(out, MORPHO_OBJECT(new));
    } else {
        parse_error(p, true, ERROR_ALLOCATIONFAILED);
    }
    
json_parsestring_cleanup:
    varray_charclear(&str);
    return success;
}

/** Parses an integer */
bool json_parsenumber(parser *p, void *out) {
    long f = strtol(p->previous.start, NULL, 10);
    
    if ( (errno==ERANGE && (f==LONG_MAX || f==LONG_MIN)) || // Check for underflow or overflow
        f>INT_MAX || f<INT_MIN) {
        parse_error(p, true, PARSE_VALRANGE);
        return false;
    }
    
    json_output(out, MORPHO_INTEGER((int) f));
    
    return true;
}

/** Parses an floating point value */
bool json_parsefloat(parser *p, void *out) {
    double f = strtod(p->previous.start, NULL);
    
    if ( errno==ERANGE && (f==HUGE_VAL || f==-HUGE_VAL || f<=DBL_MIN) ) {
        parse_error(p, true, PARSE_VALRANGE);
        return false;
    }
    
    json_output(out, MORPHO_FLOAT((double) f));
    
    return true;
}

/** Parses the true identifier */
bool json_parsetrue(parser *p, void *out) {
    json_output(out, MORPHO_TRUE);
    return true;
}

/** Parses the false identifier */
bool json_parsefalse(parser *p, void *out) {
    json_output(out, MORPHO_FALSE);
    return true;
}

/** Parses the null identifier */
bool json_parsenull(parser *p, void *out) {
    json_output(out, MORPHO_NIL);
    return true;
}

/** Parse an 'object', which is really a dictionary */
bool json_parseobject(parser *p, void *out) {
    objectdictionary *new = object_newdictionary();
    if (!new) {
        parse_error(p, true, ERROR_ALLOCATIONFAILED);
        return false;
    }
    
    while (!parse_checktoken(p, JSON_RIGHTCURLYBRACE) &&
           !parse_checktoken(p, JSON_EOF)) {
        value key=MORPHO_NIL, val=MORPHO_NIL;
        
        /* Parse the key/value pair */
        if (parse_checktoken(p, JSON_STRING)) {
            if (!json_parsevalue(p, &key)) goto json_parseobjectcleanup;
        } else {
            parse_error(p, true, JSON_OBJCTKEY);
            goto json_parseobjectcleanup;
        }
        
        if (!parse_checkrequiredtoken(p, JSON_COLON, PARSE_DCTSPRTR)) goto json_parseobjectcleanup;
        if (!json_parsevalue(p, &val)) goto json_parseobjectcleanup;
        
        dictionary_insert(&new->dict, key, val);
        
        if (!parse_checktoken(p, JSON_RIGHTCURLYBRACE)) {
            if (!parse_checkrequiredtoken(p, JSON_COMMA, PARSE_DCTTRMNTR)) goto json_parseobjectcleanup;
        }
    }
    
    if (!parse_checkrequiredtoken(p, JSON_RIGHTCURLYBRACE, PARSE_DCTTRMNTR)) goto json_parseobjectcleanup;
    
    json_output(out, MORPHO_OBJECT(new));
    
    return true;
    
json_parseobjectcleanup:
    if (new) object_free((object *) new);
    return false;
}

/** Parses an array, e.g. [ 1, 2, 3 ] */
bool json_parsearray(parser *p, void *out) {
    objectlist *new = object_newlist(0, NULL);
    if (!new) {
        parse_error(p, true, ERROR_ALLOCATIONFAILED);
        return false;
    }

    while (!parse_checktoken(p, JSON_RIGHTSQUAREBRACE) &&
           !parse_checktoken(p, JSON_EOF)) {
        value v=MORPHO_NIL;
        if (json_parsevalue(p, &v)) {
            list_append(new, v);
        } else goto json_parsearraycleanup;
        
        if (!parse_checktoken(p, JSON_RIGHTSQUAREBRACE)) {
            if (!parse_checkrequiredtoken(p, JSON_COMMA, PARSE_MSSNGCOMMA)) goto json_parsearraycleanup;
        }
    }
    
    if (!parse_checkrequiredtoken(p, JSON_RIGHTSQUAREBRACE, PARSE_MSSNGSQBRC)) goto json_parsearraycleanup;
    
    json_output(out, MORPHO_OBJECT(new));
    return true;
    
json_parsearraycleanup:
    if (new) object_free((object *) new);
    return false;
}

/** Parses a json value using the parse table */
bool json_parsevalue(parser *p, void *out) {
    return parse_precedence(p, PREC_ASSIGN, out);
}

/** Base JSON parse type */
bool json_parseelement(parser *p, void *out) {
    bool success=json_parsevalue(p, out);
    if (success && p->current.type!=JSON_EOF) {
        parse_error(p, false, JSON_EXTRNSTOK);
        return false;
    }
    return success;
}

/* -------------------------------------------------------
 * JSON parse table
 * ------------------------------------------------------- */

parserule json_rules[] = {
    PARSERULE_PREFIX(JSON_LEFTCURLYBRACE, json_parseobject),
    PARSERULE_PREFIX(JSON_LEFTSQUAREBRACE, json_parsearray),
    PARSERULE_PREFIX(JSON_STRING, json_parsestring),
    PARSERULE_PREFIX(JSON_NUMBER, json_parsenumber),
    PARSERULE_PREFIX(JSON_FLOAT, json_parsefloat),
    PARSERULE_PREFIX(JSON_TRUE, json_parsetrue),
    PARSERULE_PREFIX(JSON_FALSE, json_parsefalse),
    PARSERULE_PREFIX(JSON_NULL, json_parsenull),
    PARSERULE_UNUSED(TOKEN_NONE)
};

/* -------------------------------------------------------
 * Initialize a JSON parser
 * ------------------------------------------------------- */

/** Initializes a parser to parse JSON */
void json_initializeparser(parser *p, lexer *l, error *err, void *out) {
    parse_init(p, l, err, out);
    parse_setbaseparsefn(p, json_parseelement);
    parse_setparsetable(p, json_rules);
    parse_setskipnewline(p, false, TOKEN_NONE);
}

/* **********************************************************************
 * Interface to JSON parser
 * ********************************************************************** */

/** @brief Parses JSON into a value.
    @param[in] in - source string
    @param[in] err - error block to fill out on failure
    @param[out] out - value on succes
    @param[out] objects - [optional] a varray filled out with all objects generated in parsing
    @returns true on success, false otherwise */
bool json_parse(char *in, error *err, value *out, varray_value *objects) {
    lexer l;
    json_initializelexer(&l, in);

    parser p;
    json_initializeparser(&p, &l, err, out);
    
    bool success=parse(&p);
    
    parse_clear(&p);
    lex_clear(&l);
    
    return success;
}

/* **********************************************************************
 * JSON class
 * ********************************************************************** */

value JSON_parse(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    if (nargs==1 && MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        char *src = MORPHO_GETCSTRING(MORPHO_GETARG(args, 0));
        error err;
        error_init(&err);
        if (!json_parse(src, &err, &out, NULL)) {
            morpho_runtimeerror(v, err.id);
        }
    }
    return out;
}

MORPHO_BEGINCLASS(JSON)
MORPHO_METHOD(JSON_PARSEMETHOD, JSON_parse, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization/finalization
 * ********************************************************************** */

void json_initialize(void) {
    // Locate the Object class to use as the parent class of JSON
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // JSON class
    value jsonclass=builtin_addclass(JSON_CLASSNAME, MORPHO_GETCLASSDEFINITION(JSON), objclass);
    
    morpho_defineerror(JSON_OBJCTKEY, ERROR_PARSE, JSON_OBJCTKEY_MSG);
    morpho_defineerror(JSON_PRSARGS, ERROR_PARSE, JSON_PRSARGS_MSG);
    morpho_defineerror(JSON_EXTRNSTOK, ERROR_PARSE, JSON_EXTRNSTOK_MSG);
    morpho_defineerror(JSON_UNESCPDCTRL, ERROR_PARSE, JSON_UNESCPDCTRL_MSG);
}

void json_finalize(void) {
    
}
