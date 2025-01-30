/** @file json.c
 *  @author T J Atherton
 *
 *  @brief JSON class
 *  @details Aims to be compliant with RFC 8259, tested against https://github.com/nst/JSONTestSuite
 *           Currently passes all except "n_multidigit_number_then_00.json"
 */

#include <ctype.h>
#include "morpho.h"
#include "classes.h"

#include "common.h"
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
    while (lex_peek(l) != '"' && !lex_isatend(l)) {
        if (lex_peek(l)=='\\') lex_advance(l); // Detect an escaped character

        lex_advance(l);
    }
    
    if (lex_isatend(l)) {
        morpho_writeerrorwithid(err, LEXER_UNTERMINATEDSTRING, NULL, tok->line, tok->posn);
        return false;
    }
    
    lex_advance(l); // Advance over final quote
    lex_recordtoken(l, JSON_STRING, tok);
    
    return true;
}

/** Record JSON numbers as a token */
bool json_lexnumber(lexer *l, token *tok, error *err) {
    bool hasexp=false;
    tokentype type = JSON_NUMBER;
    
    // Detect if we are missing digits (ie an isolated '-')
    char c = lex_peek(l);
    if (c=='0') {
        lex_advance(l);
        if (lex_isdigit(lex_peek(l))) goto json_lexnumberinvld; // Cannot follow '0' by digits.
    }
    
    if (lex_isdigit(c)) {
        // Advance through initial digits
        while (lex_isdigit(lex_peek(l)) && !lex_isatend(l)) lex_advance(l);
    } else goto json_lexnumberinvld;
    
    // Detect fractional separator
    if (lex_peek(l)=='.') {
        lex_advance(l);
        type = JSON_FLOAT;
        
        // Digits are required after fractional separator
        if (!lex_isdigit(lex_peek(l))) goto json_lexnumberinvld;
        while (lex_isdigit(lex_peek(l)) && !lex_isatend(l)) lex_advance(l);
    };
    
    if (lex_peek(l)=='e' || lex_peek(l)=='E') {
        lex_advance(l); hasexp=true; type = JSON_FLOAT;
    }
    if (lex_peek(l)=='+' || lex_peek(l)=='-') {
        if (hasexp) lex_advance(l); else goto json_lexnumberinvld; // Only allow +/- after exp
    }
    
    // Digits are required after exponent
    if (hasexp && !lex_isdigit(lex_peek(l))) goto json_lexnumberinvld;
    while (lex_isdigit(lex_peek(l)) && !lex_isatend(l)) lex_advance(l);
    
    lex_recordtoken(l, type, tok);
    
    return true;

json_lexnumberinvld:
    morpho_writeerrorwithid(err, JSON_NMBRFRMT, NULL, tok->line, tok->posn);
    return false;
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
 * Output structure
 * ------------------------------------------------------- */

/** Type for output of the type function */
typedef struct {
    value out;
    varray_value *objects;
} jsonoutput;

/** Place a value into the opaque output structure */
void json_setoutput(void *out, value v) {
    jsonoutput *output = (jsonoutput *) out;
    output->out=v;
    if (output->objects && MORPHO_ISOBJECT(v)) varray_valuewrite(output->objects, v);
}

/** Retrieve output value from opaque output structure */
value json_getoutput(jsonoutput *out) {
    return out->out;
}

/** Initializes the output structure with an optional output varray.
 @warning: The output array must have been previously initialized */
void json_outputinit(jsonoutput *out, varray_value *output) {
    out->out=MORPHO_NIL;
    out->objects=output;
    if (output) out->objects->count=0;
}

/* -------------------------------------------------------
 * JSON parse functions
 * ------------------------------------------------------- */

bool json_parsevalue(parser *p, void *out);

/** Parses a string */
bool json_parsestring(parser *p, void *out) {
    bool success=false;
    varray_char str;
    varray_charinit(&str);
    
    const char *input = p->previous.start;
    unsigned int length = p->previous.length;
    
    for (unsigned int i=1; i<length-1; i++) {
        if (iscntrl((unsigned char) input[i]) && input[i]<='\x1f') { // RFC 8259 mandates that ctrl characters are 0x00 - 0x1f
            parse_error(p, true, PARSE_UNESCPDCTRL);
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
                    if (!parse_codepointfromhex(p, &input[i+1], 4, false, &str)) goto json_parsestring_cleanup;
                    i+=4;
                }
                    break;
                default:
                    parse_error(p, true, PARSE_STRESC);
                    goto json_parsestring_cleanup;
            }
        }
    }
    
    value new = object_stringfromvarraychar(&str);
    if (MORPHO_ISOBJECT(new)) {
        success=true;
        json_setoutput(out, new);
    } else {
        parse_error(p, true, ERROR_ALLOCATIONFAILED);
    }
    
json_parsestring_cleanup:
    varray_charclear(&str);
    return success;
}

/** Parses an integer */
bool json_parsenumber(parser *p, void *out) {
    long f;
    if (!parse_tokentointeger(p, &f)) return false;
    
    json_setoutput(out, MORPHO_INTEGER((int) f));
    
    return true;
}

/** Parses an floating point value */
bool json_parsefloat(parser *p, void *out) {
    double f;
    if (!parse_tokentodouble(p, &f)) return false;
    
    json_setoutput(out, MORPHO_FLOAT((double) f));
    
    return true;
}

/** Parses the true identifier */
bool json_parsetrue(parser *p, void *out) {
    json_setoutput(out, MORPHO_TRUE);
    return true;
}

/** Parses the false identifier */
bool json_parsefalse(parser *p, void *out) {
    json_setoutput(out, MORPHO_FALSE);
    return true;
}

/** Parses the null identifier */
bool json_parsenull(parser *p, void *out) {
    json_setoutput(out, MORPHO_NIL);
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
        //value key=MORPHO_NIL, val=MORPHO_NIL;
        jsonoutput key = *(jsonoutput *) out, val = *(jsonoutput *) out;
        
        /* Parse the key/value pair */
        if (parse_checktoken(p, JSON_STRING)) {
            if (!json_parsevalue(p, &key)) goto json_parseobjectcleanup;
        } else {
            parse_error(p, true, JSON_OBJCTKEY);
            goto json_parseobjectcleanup;
        }
        
        if (!parse_checkrequiredtoken(p, JSON_COLON, PARSE_DCTSPRTR)) goto json_parseobjectcleanup;
        if (!json_parsevalue(p, &val)) goto json_parseobjectcleanup;
        
        dictionary_insert(&new->dict, json_getoutput(&key), json_getoutput(&val));
        
        if (!parse_checktoken(p, JSON_RIGHTCURLYBRACE)) {
            if (!parse_checkrequiredtoken(p, JSON_COMMA, PARSE_DCTSPRTR)) goto json_parseobjectcleanup;
            if (parse_checkdisallowedtoken(p, JSON_RIGHTCURLYBRACE, JSON_BLNKELMNT)) goto json_parseobjectcleanup;
        }
    }
    
    if (!parse_checkrequiredtoken(p, JSON_RIGHTCURLYBRACE, PARSE_DCTTRMNTR)) goto json_parseobjectcleanup;
    
    json_setoutput(out, MORPHO_OBJECT(new));
    
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
        if (parse_checkdisallowedtoken(p, JSON_COMMA, JSON_BLNKELMNT)) goto json_parsearraycleanup;
        jsonoutput v = *(jsonoutput *) out;
        
        if (json_parsevalue(p, &v)) {
            list_append(new, json_getoutput(&v));
        } else goto json_parsearraycleanup;
        
        if (!parse_checktoken(p, JSON_RIGHTSQUAREBRACE)) {
            if (!parse_checkrequiredtoken(p, JSON_COMMA, PARSE_MSSNGCOMMA)) goto json_parsearraycleanup;
            if (parse_checkdisallowedtoken(p, JSON_RIGHTSQUAREBRACE, JSON_BLNKELMNT)) goto json_parsearraycleanup;
        }
    }
    
    if (!parse_checkrequiredtoken(p, JSON_RIGHTSQUAREBRACE, PARSE_MSSNGSQBRC)) goto json_parsearraycleanup;
    
    json_setoutput(out, MORPHO_OBJECT(new));
    return true;
    
json_parsearraycleanup:
    if (new) object_free((object *) new);
    return false;
}

/** Parses a json value using the parse table */
bool json_parsevalue(parser *p, void *out) {
    if (!parse_incrementrecursiondepth(p)) return false; // Increment and check
    
    bool success=parse_precedence(p, PREC_ASSIGN, out);
    
    parse_decrementrecursiondepth(p);
    return success;
}

/** Base JSON parse type */
bool json_parseelement(parser *p, void *out) {
    if (parse_checkdisallowedtoken(p, JSON_EOF, JSON_BLNKELMNT)) return false;
    
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
    varray_value obj;
    varray_valueinit(&obj);
    
    jsonoutput output;
    json_outputinit(&output, &obj);
    
    lexer l;
    json_initializelexer(&l, in);

    parser p;
    json_initializeparser(&p, &l, err, &output);
    
    bool success=parse(&p);
    
    parse_clear(&p);
    lex_clear(&l);
    
    if (success) {
        *out = json_getoutput(&output);
        if (objects) varray_valueadd(objects, obj.data, obj.count);
    } else { // Free any objects that were already allocated
        for (unsigned int i=0; i<obj.count; i++) morpho_freeobject(obj.data[i]);
    }
    
    varray_valueclear(&obj);
    
    return success;
}

/* **********************************************************************
 * JSON output
 * ********************************************************************** */

bool json_valuetovarraychar(vm *v, value in, varray_char *out) {
    bool success=false;
    if (MORPHO_ISINTEGER(in) ||
        MORPHO_ISFLOAT(in) ||
        MORPHO_ISBOOL(in)) {
        success=morpho_printtobuffer(v, in, out);
    } else if (MORPHO_ISNIL(in)) {
        success=varray_charadd(out, JSON_NULL_LABEL, strlen(JSON_NULL_LABEL));
    } else if (MORPHO_ISSTRING(in)) {
        objectstring *str = MORPHO_GETSTRING(in);
        
        success=varray_charadd(out, "\"", 1);
        
        for (char *c = str->string; c<str->string+str->length; ) {
            int nbytes = morpho_utf8numberofbytes(c);
            if (!nbytes) return false;
            
            if (nbytes==1 && iscntrl((unsigned char) *c)) {
                switch (*c) {
                    case '\b': success=varray_charadd(out, "\\b", 2); break;
                    case '\f': success=varray_charadd(out, "\\f", 2); break;
                    case '\n': success=varray_charadd(out, "\\n", 2); break;
                    case '\r': success=varray_charadd(out, "\\r", 2); break;
                    case '\t': success=varray_charadd(out, "\\t", 2); break;
                    case '\\': success=varray_charadd(out, "\\\\", 2); break;
                    default: {
                        char temp[128];
                        int n = snprintf(temp, 128, "\\u%04x", (int) *c);
                        success=varray_charadd(out, temp, n);
                    }
                }
            } else {
                success=varray_charadd(out, c, nbytes);
            }
            
            c+=nbytes;
        }
        
        success=varray_charadd(out, "\"", 1);
        
    } else if (MORPHO_ISLIST(in)) {
        objectlist *lst = MORPHO_GETLIST(in);
        varray_charadd(out, "[", 1);
        
        for (int i=0; i<lst->val.count; i++)  {
            if (!json_valuetovarraychar(v, lst->val.data[i], out)) return false;
            if (i<lst->val.count-1) varray_charadd(out, ",", 1);
        }
        
        success=varray_charadd(out, "]", 1);
    } else if (MORPHO_ISDICTIONARY(in)) {
        objectdictionary *dict = MORPHO_GETDICTIONARY(in);
        varray_charadd(out, "{", 1);
        
        for (unsigned int i=0, k=0; i<dict->dict.capacity; i++) {
            value key = dict->dict.contents[i].key;
            if (MORPHO_ISNIL(key)) continue;
            
            if (!MORPHO_ISSTRING(key)) success=varray_charadd(out, "\"", 1);
            if (!json_valuetovarraychar(v, key, out)) return false;
            if (!MORPHO_ISSTRING(key)) success=varray_charadd(out, "\"", 1);
            
            success=varray_charadd(out, ":", 1);
            
            if (!json_valuetovarraychar(v, dict->dict.contents[i].val, out)) return false;
            
            if (k<dict->dict.count-1) varray_charadd(out, ",", 1);
            
            k++;
        }
        
        success=varray_charadd(out, "}", 1);
    }
    return success;
}

bool json_tostring(vm *v, value in, value *out) {
    bool success=false;
    varray_char str;
    varray_charinit(&str);
    
    success=json_valuetovarraychar(v, in, &str);
    
    if (success) {
        *out=object_stringfromvarraychar(&str);
        if (!MORPHO_ISSTRING(*out)) {
            morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
            success=false;
        }
    }
    
    varray_charclear(&str);
    return success;
}

/* **********************************************************************
 * JSON class
 * ********************************************************************** */

value JSON_parse(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    if (nargs==1 && MORPHO_ISSTRING(MORPHO_GETARG(args, 0))) {
        char *src = MORPHO_GETCSTRING(MORPHO_GETARG(args, 0));
        varray_value objects;
        varray_valueinit(&objects);
        error err;
        error_init(&err);
        
        if (json_parse(src, &err, &out, &objects)) {
            morpho_bindobjects(v, objects.count, objects.data);
        } else {
            morpho_runtimeerror(v, err.id);
        }
        varray_valueclear(&objects);
    } else morpho_runtimeerror(v, JSON_PRSARGS);
    
    return out;
}

value JSON_tostring(vm *v, int nargs, value *args) {
    value out = MORPHO_NIL;
    
    if (nargs==1) {
        if (json_tostring(v, MORPHO_GETARG(args, 0), &out)) {
            morpho_bindobjects(v, 1, &out);
        }
    }
    
    return out;
}

MORPHO_BEGINCLASS(JSON)
MORPHO_METHOD(JSON_PARSEMETHOD, JSON_parse, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_TOSTRING_METHOD, JSON_tostring, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization/finalization
 * ********************************************************************** */

void json_initialize(void) {
    // Locate the Object class to use as the parent class of JSON
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    // JSON class
    builtin_addclass(JSON_CLASSNAME, MORPHO_GETCLASSDEFINITION(JSON), objclass);
    
    morpho_defineerror(JSON_OBJCTKEY, ERROR_PARSE, JSON_OBJCTKEY_MSG);
    morpho_defineerror(JSON_PRSARGS, ERROR_PARSE, JSON_PRSARGS_MSG);
    morpho_defineerror(JSON_EXTRNSTOK, ERROR_PARSE, JSON_EXTRNSTOK_MSG);
    morpho_defineerror(JSON_NMBRFRMT, ERROR_PARSE, JSON_NMBRFRMT_MSG);
    morpho_defineerror(JSON_BLNKELMNT, ERROR_PARSE, JSON_BLNKELMNT_MSG);
}
