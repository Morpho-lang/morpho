/** @file parse.c
 *  @author T J Atherton 
 *
 *  @brief Parser
*/

#include <string.h>
#include <float.h>
#include <limits.h>
#include <errno.h>
#include <ctype.h>
#include "parse.h"
#include "object.h"
#include "common.h"
#include "cmplx.h"
#include "syntaxtree.h"

/** Varrays of parse rules */
DEFINE_VARRAY(parserule, parserule)

/** Macro to check return of a bool function */
#define PARSE_CHECK(f) if (!(f)) return false;

/* **********************************************************************
 * Parser utility functions
 * ********************************************************************** */

/** @brief Fills out the error record
 *  @param p        the parser
 *  @param use_prev use the previous token? [this is the more typical usage]
 *  @param id       error id
 *  @param ...      additional data for sprintf. */
void parse_error(parser *p, bool use_prev, errorid id, ... ) {
    va_list args;
    token *tok = (use_prev ? &p->previous : &p->current);
    
    /** Only return the first error that occurs */
    if (ERROR_FAILED(*p->err)) return;
    
    va_start(args, id);
    morpho_writeerrorwithid(p->err, id, NULL, tok->line, tok->posn, args);
    va_end(args);
}

/** @brief Advance the parser by one token
 *  @param   p the parser in use.
 *  @returns true on success, false otherwise */
bool parse_advance(parser *p) {
    lexer *l = p->lex;
    
    p->previous=p->current;
    p->nl=false;
    
    for (;;) {
        if (!lex(l, &p->current, p->err)) return false;
        
        /* Skip any newlines encountered */
        if (p->skipnewline && p->current.type==p->toknewline) {
            p->nl=true;
            continue;
        } else break;
    }
    
    return ERROR_SUCCEEDED(*p->err);
}

/** Saves the state of the parser and attached lexer */
void parse_savestate(parser *p, parser *op, lexer *ol) {
    *ol = *p->lex; // Save the state of the parser and lexer
    *op = *p;
}

/** Restores the parser from a saved position.
 @warning: You must take care to ensure no new objects have been allocated prior to calling this. */
void parse_restorestate(parser *op, lexer *ol, parser *out) {
    *out = *op;
    *out->lex = *ol;
}

/** @brief Continues parsing while tokens have a lower or equal precendece than a specified value.
 *  @param   p    the parser in use
 *  @param   precendence precedence value to keep below or equal to
 *  @returns syntaxtreeindx for the expression parsed */
bool parse_precedence(parser *p, precedence prec, void *out) {
    parsefunction prefixrule=NULL, infixrule=NULL;
    
    PARSE_CHECK(parse_advance(p));
    
    parserule *rule = parse_getrule(p, p->previous.type);
    if (rule) prefixrule = rule->prefix;
    
    if (!rule || !prefixrule) {
        parse_error(p, true, PARSE_EXPECTEXPRESSION);
        return false;
    }
    
    PARSE_CHECK(prefixrule(p, out));
    
    /* Now keep parsing while the tokens have lower precedence */
    rule=parse_getrule(p, p->current.type);
    while (rule!=NULL && prec <= rule->precedence) {
        /* Break if a newline is encountered before a function call */
        if (p->current.type==TOKEN_LEFTPAREN && p->nl) break;
        
        PARSE_CHECK(parse_advance(p));
        
        infixrule = parse_getrule(p, p->previous.type)->infix;
        if (infixrule) {
            PARSE_CHECK(infixrule(p, out))
        } else UNREACHABLE("No infix rule defined for this token type [check parser definition table].");
        
        rule=parse_getrule(p, p->current.type);
    }

    return true;
}

/** Checks whether the current token matches a specified tokentype */
bool parse_checktoken(parser *p, tokentype type) {
    return p->current.type==type;
}

/** Checks whether the current token matches any of the specified tokentypes */
bool parse_checktokenmulti(parser *p, int n, tokentype *type) {
    for (int i=0; i<n; i++) {
        if (p->current.type==type[i]) return true;
    }
    
    return false;
}

/** Checks whether the current token matches a given type and advances if so. */
bool parse_checktokenadvance(parser *p, tokentype type) {
    PARSE_CHECK(parse_checktoken(p, type));
    PARSE_CHECK(parse_advance(p));
    return true;
}

/** Checks whether the current token is a keyword */
bool parse_checktokeniskeywordadvance(parser *p) {
    PARSE_CHECK(lex_tokeniskeyword(p->lex, &p->current));
    PARSE_CHECK(parse_advance(p));
    return true;
}

/** @brief Checks if the next token has the required type and advance if it does, otherwise generates an error.
 *  @param   p    the parser in use
 *  @param   type type to check
 *  @param   id   error id to generate if the token doesn't match
 *  @returns true on success */
bool parse_checkrequiredtoken(parser *p, tokentype type, errorid id) {
    if (parse_checktoken(p, type)) {
        PARSE_CHECK(parse_advance(p));
        return true;
    }
        
    if (id!=ERROR_NONE) parse_error(p, false, id);
    return false;
}

/** @brief Checks if the next token has a specific type and if it does generates an error.
 *  @param   p    the parser in use
 *  @param   type type to check
 *  @param   id   error id to generate if the token is found doesn't match
 *  @returns true if the disallowed token was found */
bool parse_checkdisallowedtoken(parser *p, tokentype type, errorid id) {
    if (parse_checktoken(p, type)) {
        parse_error(p, true, id);
        return true;
    }
    return false;
}

/** Converts a hex string to a character code, outputting it into a varray
    @param[in] p - the current parser
    @param[in] codestr - code string to parse
    @param[in] nhex - number of hex characters to parse
    @param[in] raw - returns a raw ascii byte, rather than the utf8 encoded character
    @param[out] out - characters are added
    @returns true on success, false on failure */
bool parse_codepointfromhex(parser *p, const char *codestr, int nhex, bool raw, varray_char *out) {
    char in[nhex+1];
    
    for (int j=0; j<nhex; j++) {
        if (isxdigit(codestr[j])) {
            in[j]=codestr[j];
        } else {
            parse_error(p, true, PARSE_INVLDUNCD);
            return false;
        }
    }
    in[nhex]='\0';
    
    long codept = strtol(in, NULL, 16);
    char buffer[4];
    int nchars=1;
    if (!raw) {
        nchars = morpho_encodeutf8((int) codept, buffer);
    } else {
        buffer[0] = (char) codept;
    }
    
    if (!varray_charadd(out, buffer, nchars)) {
        parse_error(p, true, ERROR_ALLOCATIONFAILED);
        return false;
    }
    return true;
}

/** Turn a token into a string, parsing escape characters */
bool parse_stringfromtoken(parser *p, unsigned int start, unsigned int length, value *out) {
    bool success=false;
    const char *input=p->previous.start;
    varray_char str;
    varray_charinit(&str);
    
    for (unsigned int i=start, nbytes; i<length; i+=nbytes) {
        nbytes=morpho_utf8numberofbytes(&input[i]);
        if (!nbytes) return false;
        
        if (nbytes>1) { // Unicode literals
            varray_charadd(&str, (char *) &input[i], nbytes);
        } else if (input[i]=='\n') { // Newlines are ok 
            varray_charwrite(&str, input[i]);
        } else if (iscntrl((unsigned char) input[i])) { // Unescaped control codes are not
            parse_error(p, true, PARSE_UNESCPDCTRL);
            goto parse_stringfromtokencleanup;
        } else if (nbytes==1 && input[i]=='\\') { // Escape sequence
            i++;
            switch (input[i]) {
                case 'b': varray_charwrite(&str, '\b'); break;
                case 'f': varray_charwrite(&str, '\f'); break;
                case 'n': varray_charwrite(&str, '\n'); break;
                case 'r': varray_charwrite(&str, '\r'); break;
                case 't': varray_charwrite(&str, '\t'); break;
                case 'u':
                    if (!parse_codepointfromhex(p, &input[i+1], 4, false, &str)) goto parse_stringfromtokencleanup;
                    i+=4;
                    break;
                case 'U':
                    if (!parse_codepointfromhex(p, &input[i+1], 8, false, &str)) goto parse_stringfromtokencleanup;
                    i+=8;
                    break;
                case 'x':
                    if (!parse_codepointfromhex(p, &input[i+1], 2, true, &str)) goto parse_stringfromtokencleanup;
                    i+=2;
                    break;
                default:
                    varray_charwrite(&str, input[i]); break;
            }
        } else varray_charwrite(&str, input[i]); // Any other single character
    }
    
    success=true;
    if (out) {
        *out = object_stringfromvarraychar(&str);
        if (!(MORPHO_ISSTRING(*out))) parse_error(p, true, ERROR_ALLOCATIONFAILED);
    }
    
parse_stringfromtokencleanup:
    varray_charclear(&str);
    
    return success;
}

/** Parses the previous token into a value with no processing. */
value parse_tokenasstring(parser *p) {
    value s = object_stringfromcstring(p->previous.start, p->previous.length);
    if (MORPHO_ISNIL(s)) parse_error(p, true, ERROR_ALLOCATIONFAILED, OBJECT_SYMBOLLABEL);

    return s;
}

/** Parses the next token as a symbol regardless of whether it is a keyword; returns true on success */
bool parse_tokenassymbol(parser *p) {
    bool oldmatch = lex_matchkeywords(p->lex);
    
    lex_setmatchkeywords(p->lex, false);
    bool success=parse_checktokenadvance(p, lex_symboltype(p->lex));
    lex_setmatchkeywords(p->lex, oldmatch); // Restore state of lexer
    
    return success;
}

/** Adds a node to the syntax tree. */
bool parse_addnode(parser *p, syntaxtreenodetype type, value content, token *tok, syntaxtreeindx left, syntaxtreeindx right, syntaxtreeindx *out) {
    syntaxtree *tree = (syntaxtree *) p->out;
    
    if (!syntaxtree_addnode(tree, type, content, tok->line, tok->posn, left, right, out)) {
        parse_error(p, true, ERROR_ALLOCATIONFAILED);
        return false;
    }
    
    p->left=*out; /* Record this for a future infix operator to catch */
    
    return true;
}

/** Retrieve a syntaxtree node from an index. */
syntaxtreenode *parse_lookupnode(parser *p, syntaxtreeindx i) {
    return syntaxtree_nodefromindx((syntaxtree *) p->out, i);
}

/** Checks whether a long created by strtol is within range. Raises a parse error if not and returns false. */
bool parse_validatestrtol(parser *p, long f) {
    if ( ((f==LONG_MAX || f==LONG_MIN) && errno==ERANGE) || // Check for underflow or overflow
        f>INT_MAX || f<INT_MIN) {
        parse_error(p, true, PARSE_VALRANGE);
        return false;
    }
    return true;
}

/** Checks whether a double created by strtod is within range. Raises a parse error if not and returns false. */
bool parse_validatestrtod(parser *p, double f) {
    if ( errno==ERANGE && (f==HUGE_VAL || f==-HUGE_VAL || f==DBL_MIN) ) {
        parse_error(p, true, PARSE_VALRANGE);
        return false;
    }
    return true;
}

/** Converts a token to an integer, returning true on success */
bool parse_tokentointeger(parser *p, long *i) {
    *i = strtol(p->previous.start, NULL, 10);
    return parse_validatestrtol(p, *i);
}

/** Converts an token to a double, returning true on success */
bool parse_tokentodouble(parser *p, double *x) {
    *x = strtod(p->previous.start, NULL);
    return parse_validatestrtod(p, *x);
}

/** Increments the recursion depth counter. If it exceeds PARSE_RECURSIONLIMIT an error is generated */
bool parse_incrementrecursiondepth(parser *p) {
    if (!(p->recursiondepth<p->maxrecursiondepth)) {
        parse_error(p, false, PARSE_RCRSNLMT);
        return false;
    }
        
    p->recursiondepth++;
    return true;
}

/** Decrements the recursion depth counter. */
bool parse_decrementrecursiondepth(parser *p) {
    if (p->recursiondepth>0) p->recursiondepth--;
    return false;
}

/** Adds an object to the parser */
void parse_addobject(parser *p, value obj) {
    varray_valuewrite(&p->objects, obj);
}

/** Frees objects generated by the parser */
void parse_freeobjects(parser *p) {
    for (unsigned int i=0; i<p->objects.count; i++) morpho_freeobject(p->objects.data[i]);
}

/** Clears the object list */
void parse_clearobjects(parser *p) {
    varray_valueclear(&p->objects);
}

/* ------------------------------------------
 * Parser implementation functions (parselets)
 * ------------------------------------------- */

bool parse_arglist(parser *p, tokentype rightdelimiter, unsigned int *nargs, void *out);
bool parse_variable(parser *p, errorid id, void *out);
bool parse_statementterminator(parser *p);
bool parse_checkstatementterminator(parser *p);
bool parse_synchronize(parser *p);

/* ------------------------------------------
 * Utility functions for this parser
 * ------------------------------------------- */

/** @brief Parses a list of expressions
 * @param[in]  p     the parser
 * @param[in]  rightdelimiter  token type that denotes the end of the arguments list
 * @param[out] nel the number of elements
 * @details Note that the arguments are output in reverse order, i.e. the
 *          first argument is deepest in the tree. */
bool parse_expressionlist(parser *p, tokentype rightdelimiter, unsigned int *nel, void *out) {
    syntaxtreeindx prev=SYNTAXTREE_UNCONNECTED, current=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    unsigned int n=0;
    
    if (!parse_checktoken(p, rightdelimiter)) {
        do {
            PARSE_CHECK(parse_pseudoexpression(p, &current));
            PARSE_CHECK(parse_addnode(p, NODE_ARGLIST, MORPHO_NIL, &start, prev, current, &current));
            prev = current;
            n++;
        } while (parse_checktokenadvance(p, TOKEN_COMMA));
    }
    
    /* Output the number of args */
    if (nel) *nel=n;
    
    *((syntaxtreeindx *) out)=current;
    
    return true;
}

/** @brief Parses an argument list
 * @param[in]  p     the parser
 * @param[in]  rightdelimiter  token type that denotes the end of the arguments list
 * @param[out] nargs the number of arguments
 * @param[out] out the syntaxtreeindex, updated
 * @returns true on success
 * @details Note that the arguments are output in reverse order, i.e. the
 *          first argument is deepest in the tree. */
bool parse_arglist(parser *p, tokentype rightdelimiter, unsigned int *nargs, void *out) {
    syntaxtreeindx prev=SYNTAXTREE_UNCONNECTED, current=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    unsigned int n=0;
    bool varg=false;
    
    if (!parse_checktoken(p, rightdelimiter)) {
        do {
            bool vargthis = false;
            if (parse_checktokenadvance(p, TOKEN_DOTDOTDOT)) {
                // If we are trying to index something
                // then ... represents an open range
                if (rightdelimiter == TOKEN_RIGHTSQBRACKET) {
                    
                } else if (varg) {
                    parse_error(p, true, PARSE_ONEVARPR);
                    return false;
                }
                varg = true; vargthis = true;
            }
            
            if (parse_checktokenadvance(p, TOKEN_SYMBOL)) {
                PARSE_CHECK(parse_symbol(p, &current));
                
                if (parse_checktokenadvance(p, TOKEN_SYMBOL)) { // If two symbols in a row, then the first is the type
                    syntaxtreeindx label;
                    PARSE_CHECK(parse_symbol(p, &label));
                    
                    PARSE_CHECK(parse_addnode(p, NODE_TYPE, MORPHO_NIL, &start, current, label, &current));
                } else if (parse_checktokenadvance(p, TOKEN_DOT)) { // Symbol followed by dot is a type in a namespace
                    syntaxtreeindx type, label;
                    
                    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_SYMBOL, PARSE_SYMBLEXPECTED));
                    PARSE_CHECK(parse_symbol(p, &type));
                    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_SYMBOL, PARSE_SYMBLEXPECTED));
                    PARSE_CHECK(parse_symbol(p, &label));
                    
                    PARSE_CHECK(parse_addnode(p, NODE_DOT, MORPHO_NIL, &start, current, type, &current));
                    PARSE_CHECK(parse_addnode(p, NODE_TYPE, MORPHO_NIL, &start, current, label, &current));
                } else if (parse_checktokenadvance(p, TOKEN_EQUAL)) { // Symbol followed by equals is an optional argument
                    syntaxtreeindx val;
                    PARSE_CHECK(parse_pseudoexpression(p, &val));
                    
                    PARSE_CHECK(parse_addnode(p, NODE_ASSIGN, MORPHO_NIL, &start, current, val, &current));
                }
            } else PARSE_CHECK(parse_pseudoexpression(p, &current));

            if (vargthis) PARSE_CHECK(parse_addnode(p, NODE_RANGE, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, current, &current));
            
            n++;
            
            PARSE_CHECK(parse_addnode(p, NODE_ARGLIST, MORPHO_NIL, &start, prev, current, &current));
            
            prev = current;
        } while (parse_checktokenadvance(p, TOKEN_COMMA));
    }
    
    /* Output the number of args */
    if (nargs) *nargs=n;
    
    *((syntaxtreeindx *) out)=current;
    
    return true;
}

/** Parses a variable name, or raises and error if a symbol isn't found */
bool parse_variable(parser *p, errorid id, void *out) {
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_SYMBOL, id));
    return parse_symbol(p, out);
}

/** Parses a reference that could be a symbol or a namespace.symbol reference */
bool parse_reference(parser *p, errorid errid, void *out) {
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_SYMBOL, errid));
    
    syntaxtreeindx symbol, selector=SYNTAXTREE_UNCONNECTED;
    PARSE_CHECK(parse_symbol(p, &symbol));
    
    if (parse_checktokenadvance(p, TOKEN_DOT)) {
        PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_SYMBOL, errid));
        PARSE_CHECK(parse_symbol(p, &selector));
    }
    
    if (selector!=SYNTAXTREE_UNCONNECTED) {
        PARSE_CHECK(parse_addnode(p, NODE_DOT, MORPHO_NIL, &p->previous, symbol, selector, &symbol));
    }
    
    *((syntaxtreeindx *) out)=symbol;
    
    return true;
}

/** Parse a statement terminator  */
bool parse_statementterminator(parser *p) {
    if (parse_checktoken(p, TOKEN_SEMICOLON)) {
        PARSE_CHECK(parse_advance(p));
    } else if (p->nl || parse_checktoken(p, TOKEN_EOF) || parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET)) {
    } else if (parse_checktoken(p, TOKEN_IN) || parse_checktoken(p, TOKEN_ELSE)) {
    } else {
        parse_error(p, true, PARSE_MISSINGSEMICOLONEXP);
        return false;
    }
    return true;
}

/** Checks whether a possible statement terminator is next */
bool parse_checkstatementterminator(parser *p) {
    return (parse_checktoken(p, TOKEN_SEMICOLON)
            || (p->nl)
            || parse_checktoken(p, TOKEN_EOF)
            || parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET)
            || parse_checktoken(p, TOKEN_IN)
            ) ;
}

/** @brief Keep parsing til the end of a statement boundary. */
bool parse_synchronize(parser *p) {
    while (p->current.type!=TOKEN_EOF) {
        /** Align */
        if (p->previous.type == TOKEN_SEMICOLON) return true;
        switch (p->current.type) {
            case TOKEN_PRINT:
            case TOKEN_IF:
            case TOKEN_WHILE:
            case TOKEN_FOR:
            case TOKEN_DO:
            case TOKEN_BREAK:
            case TOKEN_CONTINUE:
            case TOKEN_RETURN:
            case TOKEN_TRY:
                
            case TOKEN_CLASS:
            case TOKEN_FUNCTION:
            case TOKEN_VAR:
                return true;
            default:
                ;
        }
        
        PARSE_CHECK(parse_advance(p));
    }

    return true;
}

/* ------------------------------------------
 * Basic literals
 * ------------------------------------------- */

/** Parses nil */
bool parse_nil(parser *p, void *out) {
    return parse_addnode(p, NODE_NIL,
        MORPHO_NIL, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parses an integer */
bool parse_integer(parser *p, void *out) {
    long f;
    PARSE_CHECK(parse_tokentointeger(p, &f));
    
    return parse_addnode(p, NODE_INTEGER, MORPHO_INTEGER(f), &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parses a number */
bool parse_number(parser *p, void *out) {
    double f;
    PARSE_CHECK(parse_tokentodouble(p, &f));
    
    return parse_addnode(p, NODE_FLOAT, MORPHO_FLOAT(f), &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parse a complex number */
bool parse_complex(parser *p, void *out) {
    double f;
    if (p->previous.length==2) { // just a bare im symbol
        f = 1;
    } else {
        PARSE_CHECK(parse_tokentodouble(p, &f));
    }
    value c = MORPHO_OBJECT(object_newcomplex(0,f));
    parse_addobject(p, c);
    return parse_addnode(p, NODE_IMAG, c, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parses a bool */
bool parse_bool(parser *p, void *out) {
    return parse_addnode(p, NODE_BOOL,
        MORPHO_BOOL((p->previous.type==TOKEN_TRUE ? true : false)), &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parses a self token */
bool parse_self(parser *p, void *out) {
    return parse_addnode(p, NODE_SELF, MORPHO_NIL, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parses a super token */
bool parse_super(parser *p, void *out) {
    if (!parse_checktoken(p, TOKEN_DOT)) {
        parse_error(p, false, PARSE_EXPECTDOTAFTERSUPER);
        return false;
    }

    return parse_addnode(p, NODE_SUPER, MORPHO_NIL, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parses a symbol */
bool parse_symbol(parser *p, void *out) {
    value s = object_stringfromcstring(p->previous.start, p->previous.length);
    parse_addobject(p, s);
    if (MORPHO_ISNIL(s)) {
        parse_error(p, true, ERROR_ALLOCATIONFAILED, OBJECT_SYMBOLLABEL);
        return false;
    }

    return parse_addnode(p, NODE_SYMBOL, s, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parses a string */
bool parse_string(parser *p, void *out) {
    value s;
    PARSE_CHECK(parse_stringfromtoken(p, 1, p->previous.length-1, &s));
    parse_addobject(p, s);
    
    return parse_addnode(p, NODE_STRING, s, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** @brief: Parses a dictionary.
 * @details Dictionaries are a list of key/value pairs,  { key : value, key: value } */
bool parse_dictionary(parser *p, void *out) {
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED;
    parse_addnode(p, NODE_DICTIONARY, MORPHO_NIL, &p->current, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, &last);
    
    while (!parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET) &&
           !parse_checktoken(p, TOKEN_EOF)) {
        syntaxtreeindx key, val, pair;
        token tok=p->current; // Keep track of the token that corresponds to each key/value pair
        
        /* Parse the key/value pair */
        PARSE_CHECK(parse_expression(p, &key));
        PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_COLON, PARSE_DCTSPRTR));
        PARSE_CHECK(parse_expression(p, &val));
        
        /* Create an entry node */
        PARSE_CHECK(parse_addnode(p, NODE_DICTENTRY, MORPHO_NIL, &tok, key, val, &pair));
        
        /* These are linked into a chain of dictionary nodes */
        PARSE_CHECK(parse_addnode(p, NODE_DICTIONARY, MORPHO_NIL, &tok, last, pair, &last));
        
        if (!parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET)) {
            PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_COMMA, PARSE_MSSNGCOMMA));
        }
    };
    
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTCURLYBRACKET, PARSE_DCTTRMNTR));
    
    *((syntaxtreeindx *) out) = last;
    
    return true;
}

/** Parses a string interpolation. */
bool parse_interpolation(parser *p, void *out) {
    token tok = p->previous;
    
    /* First copy the string */
    value s;
    PARSE_CHECK(parse_stringfromtoken(p, 1, tok.length-2, &s));
    parse_addobject(p, s);
    
    syntaxtreeindx left=SYNTAXTREE_UNCONNECTED, right=SYNTAXTREE_UNCONNECTED;
    
    if (!(parse_checktoken(p, TOKEN_STRING) &&
        *p->current.start=='}')) {
        PARSE_CHECK(parse_expression(p, &left));
    }
    
    if (parse_checktokenadvance(p, TOKEN_STRING)) {
        if (!parse_string(p, &right)) return false;
    } else if (parse_checktokenadvance(p, TOKEN_INTERPOLATION)) {
        if (!parse_interpolation(p, &right)) return false;
    } else {
        parse_error(p, false, PARSE_INCOMPLETESTRINGINT);
        return false;
    }
    
    return parse_addnode(p, NODE_INTERPOLATION, s, &tok, left, right, (syntaxtreeindx *) out);
}

/** Helper function to parse a tuple
 @param[in] p - current parser
 @param[in] start - token for starting '(' of the tuple
 @param[in] first - syntax tree index of first expression, already parsed
 @param[out] out - syntax tree entry of the output tuple node
 @returns true on success */
bool parse_tuple(parser *p, token *start, syntaxtreeindx first, void *out) {
    syntaxtreeindx prev=first, current=SYNTAXTREE_UNCONNECTED;
    // First entry was already parsed by calling function
    PARSE_CHECK(parse_addnode(p, NODE_ARGLIST, MORPHO_NIL, &p->previous, prev, current, &current));
    
    if (!parse_checktoken(p, TOKEN_RIGHTPAREN)) {
        do {
            PARSE_CHECK(parse_pseudoexpression(p, &current));
            PARSE_CHECK(parse_addnode(p, NODE_ARGLIST, MORPHO_NIL, &p->previous, prev, current, &current));
            prev = current;
        } while (parse_checktokenadvance(p, TOKEN_COMMA));
    }
    
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_MSSNGSQBRC));

    return parse_addnode(p, NODE_TUPLE, MORPHO_NIL, start, SYNTAXTREE_UNCONNECTED, current, out);
}

/** Parses an expression in parentheses */
bool parse_grouping(parser *p, void *out) {
    token start = p->previous;
    syntaxtreeindx new;
    PARSE_CHECK(parse_pseudoexpression(p, &new));
    
    syntaxtreenode *node = parse_lookupnode(p, new);
    
    // Detect a tuple from a comma after the first expression or if the
    // grouping encloses a tuple.
    if (parse_checktokenadvance(p, TOKEN_COMMA) ||
        node->type==NODE_TUPLE) {
        return parse_tuple(p, &start, new, out);
    }
    
    PARSE_CHECK(parse_addnode(p, NODE_GROUPING, MORPHO_NIL, &p->previous, new, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out));
    
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_MISSINGPARENTHESIS));
    return true;
}

/** Parse a unary operator */
bool parse_unary(parser *p, void *out) {
    token start = p->previous;
    syntaxtreenodetype nodetype=NODE_LEAF;
    
    /* Determine which operator */
    switch (start.type) {
        case TOKEN_MINUS: nodetype = NODE_NEGATE; break;
        case TOKEN_EXCLAMATION: nodetype = NODE_NOT; break;
        case TOKEN_AT: nodetype = NODE_BREAKPOINT; break;
        default:
            UNREACHABLE("unhandled unary operator [Check parser definition table]");
    }
    
    /* Now add this node */
    syntaxtreeindx right;
    PARSE_CHECK(parse_precedence(p, PREC_UNARY, &right));
    return parse_addnode(p, nodetype, MORPHO_NIL, &start, right, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parse a binary operator */
bool parse_binary(parser *p, void *out) {
    token start = p->previous;
    syntaxtreenodetype nodetype=NODE_LEAF;
    enum {LEFT, RIGHT} assoc = LEFT; /* for left associative operators */
    
    /* Determine which operator */
    switch (start.type) {
        case TOKEN_EQUAL:
            nodetype = NODE_ASSIGN;
            assoc = RIGHT;
            break;
            
        case TOKEN_PLUS:        nodetype = NODE_ADD; break;
        case TOKEN_MINUS:       nodetype = NODE_SUBTRACT; break;
        case TOKEN_STAR:        nodetype = NODE_MULTIPLY; break;
        case TOKEN_SLASH:       nodetype = NODE_DIVIDE; break;
        case TOKEN_CIRCUMFLEX:
            nodetype = NODE_POW;
            assoc = RIGHT; 
            break;
        
        case TOKEN_EQ:          nodetype = NODE_EQ; break;
        case TOKEN_NEQ:         nodetype = NODE_NEQ; break;
        case TOKEN_LT:          nodetype = NODE_LT; break;
        case TOKEN_GT:          nodetype = NODE_GT; break;
        case TOKEN_LTEQ:        nodetype = NODE_LTEQ; break;
        case TOKEN_GTEQ:        nodetype = NODE_GTEQ; break;
            
        case TOKEN_DOT:         nodetype = NODE_DOT; break;
            
        case TOKEN_DBLAMP:      nodetype = NODE_AND; break;
        case TOKEN_DBLVBAR:     nodetype = NODE_OR; break;
        default:
            UNREACHABLE("unhandled binary operator [Check parser definition table]");
    }
    
    parserule *rule=parse_getrule(p, start.type);
    syntaxtreeindx left=p->left;
    syntaxtreeindx right=SYNTAXTREE_UNCONNECTED;
    
    /* Check if we have a right hand side. */
    if (parse_checktoken(p, TOKEN_EOF)) {
        parse_error(p, true, PARSE_INCOMPLETEEXPRESSION);
        return false;
    } else {
        if (nodetype==NODE_ASSIGN &&
            parse_checktokenadvance(p, TOKEN_FUNCTION)) {
            PARSE_CHECK(parse_anonymousfunction(p, &right));
        } else if (nodetype==NODE_DOT &&
                   parse_checktokeniskeywordadvance(p)) {
            PARSE_CHECK(parse_symbol(p, &right));
        } else {
            PARSE_CHECK(parse_precedence(p, rule->precedence + (assoc == LEFT ? 1 : 0), &right));
        }
    }
    
    /* Now add this node */
    return parse_addnode(p, nodetype, MORPHO_NIL, &start, left, right, (syntaxtreeindx *) out);
}

/** Parse ternary operator */
bool parse_ternary(parser *p, void *out) {
    token start = p->previous;
    syntaxtreeindx cond=p->left;
    syntaxtreeindx left, right, outcomes;
    PARSE_CHECK(parse_expression(p, &left));
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_COLON, PARSE_TRNRYMSSNGCOLON));
    PARSE_CHECK(parse_expression(p, &right));
    
    PARSE_CHECK(parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, left, right, &outcomes));
    return parse_addnode(p, NODE_TERNARY, MORPHO_NIL, &start, cond, outcomes, (syntaxtreeindx *) out);
}

/** Parse operators like +=, -=, *= etc. */
bool parse_assignby(parser *p, void *out) {
    token start = p->previous;
    syntaxtreenodetype nodetype=NODE_LEAF;
    
    /* Determine which operator */
    switch (start.type) {
        case TOKEN_PLUSEQ:      nodetype = NODE_ADD; break;
        case TOKEN_MINUSEQ:     nodetype = NODE_SUBTRACT; break;
        case TOKEN_STAREQ:      nodetype = NODE_MULTIPLY; break;
        case TOKEN_SLASHEQ:     nodetype = NODE_DIVIDE; break;
        default:
            UNREACHABLE("unhandled assignment operator [Check parser definition table]");
    }
    
    parserule *rule=parse_getrule(p, start.type);
    syntaxtreeindx left=p->left;
    syntaxtreeindx right=SYNTAXTREE_UNCONNECTED;
    
    /* Check if we have a right hand side. */
    if (parse_checktoken(p, TOKEN_EOF)) {
        parse_error(p, true, PARSE_INCOMPLETEEXPRESSION);
        return false;
    } else {
        PARSE_CHECK(parse_precedence(p, rule->precedence, &right));
    }
    
    if (!parse_addnode(p, nodetype, MORPHO_NIL, &start, left, right, &right)) return false;
    
    /* Now add this node */
    return parse_addnode(p, NODE_ASSIGN, MORPHO_NIL, &start, left, right, (syntaxtreeindx *) out);
}

/** Parses a range */
bool parse_range(parser *p, void *out) {
    token start = p->previous;
    bool inclusive = (start.type==TOKEN_DOTDOT);
    
    syntaxtreeindx left=p->left;
    syntaxtreeindx right;
    
    PARSE_CHECK(parse_expression(p, &right));
    syntaxtreeindx new;
    PARSE_CHECK(parse_addnode(p, (inclusive ? NODE_INCLUSIVERANGE : NODE_RANGE), MORPHO_NIL, &start, left, right, &new));
    
    if (parse_checktokenadvance(p, TOKEN_COLON)) { // Wrap in an outer NODE_RANGE
        syntaxtreeindx step;
        PARSE_CHECK(parse_expression(p, &step));
        
        PARSE_CHECK(parse_addnode(p, NODE_RANGE, MORPHO_NIL, &start, new, step, &new));
    }
    
    *((syntaxtreeindx *) out) = new;
    return true;
}

/** Parse a function call */
bool parse_call(parser *p, void *out) {
    token start = p->previous;
    syntaxtreeindx left=p->left;
    syntaxtreeindx right;
    unsigned int nargs;
    
    PARSE_CHECK(parse_expressionlist(p, TOKEN_RIGHTPAREN, &nargs, &right));
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_CALLRGHTPARENMISSING));
    
    return parse_addnode(p, NODE_CALL, MORPHO_NIL, &start, left, right, out);
}

/** Parse index
        index
       /          \
  symbol         indices
 */
bool parse_index(parser *p, void *out) {
    token start = p->previous;
    syntaxtreeindx left=p->left;
    syntaxtreeindx right;
    unsigned int nindx;
    
    PARSE_CHECK(parse_expressionlist(p, TOKEN_RIGHTSQBRACKET, &nindx, &right));
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTSQBRACKET, PARSE_CALLRGHTPARENMISSING));
    
    return parse_addnode(p, NODE_INDEX, MORPHO_NIL, &start, left, right, (syntaxtreeindx *) out);
}

/** Parse list  */
bool parse_list(parser *p, void *out) {
    token start = p->previous;
    syntaxtreeindx right;
    PARSE_CHECK(parse_expressionlist(p, TOKEN_RIGHTSQBRACKET, NULL, &right));
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTSQBRACKET, PARSE_MSSNGSQBRC));

    return parse_addnode(p, NODE_LIST, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, right, out);
}

/** Parses an anonymous function */
bool parse_anonymousfunction(parser *p, void *out) {
    token start = p->previous;
    syntaxtreeindx args=SYNTAXTREE_UNCONNECTED,
                   body=SYNTAXTREE_UNCONNECTED;
    
    /* Parameter list */
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_LEFTPAREN, PARSE_FNLEFTPARENMISSING));
    PARSE_CHECK(parse_arglist(p, TOKEN_RIGHTPAREN, NULL, &args));
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_FNRGHTPARENMISSING));
    
    /* Function body */
    if (parse_checktokenadvance(p, TOKEN_LEFTCURLYBRACKET)) { // fn (x) { ... }
        PARSE_CHECK(parse_blockstatement(p, &body));
    } else {
        PARSE_CHECK(parse_expression(p, &body)); // Short form: fn (x) x
        PARSE_CHECK(parse_addnode(p, NODE_RETURN, MORPHO_NIL, &start, body, SYNTAXTREE_UNCONNECTED, &body));
    }
    
    return parse_addnode(p, NODE_FUNCTION, MORPHO_NIL, &start, args, body, out);
}

/** @brief: Parses a switch block
 * @details Switch blocks are key/statement pairs. Each pair is stored in a NODE_DICTIONARY list */
bool parse_switch(parser *p, void *out) {
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED;
    
    while(!parse_checktokenadvance(p, TOKEN_RIGHTCURLYBRACKET) && !parse_checktoken(p, TOKEN_EOF)) {
        syntaxtreeindx key, statements, pair;
        token tok=p->current; // Keep track of the token that corresponds to each key/value pair
        
        /* Parse the key/value pair */
        PARSE_CHECK(parse_expression(p, &key));
        PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_COLON, PARSE_SWTCHSPRTR));
        tokentype terminators[] = { TOKEN_STRING, TOKEN_INTEGER, TOKEN_NUMBER, TOKEN_TRUE, TOKEN_FALSE, TOKEN_NIL, TOKEN_RIGHTCURLYBRACKET };
        PARSE_CHECK(parse_declarationmulti(p, 7, terminators, &statements));
        
        /* Create an entry node */
        PARSE_CHECK(parse_addnode(p, NODE_DICTENTRY, MORPHO_NIL, &tok, key, statements, &pair));
        
        /* These are linked into a chain of dictionary nodes */
        PARSE_CHECK(parse_addnode(p, NODE_DICTIONARY, MORPHO_NIL, &tok, last, pair, &last));
    };
    
    *((syntaxtreeindx *) out) = last;
    
    return true;
}

/* -------------------------------
 * Declarations
 * ------------------------------- */

/** Parses a variable declaration */
bool parse_vardeclaration(parser *p, void *out) {
    syntaxtreeindx symbol, initializer, new=SYNTAXTREE_UNCONNECTED, last=SYNTAXTREE_UNCONNECTED;
    
    do {
        token start = p->previous;
        
        PARSE_CHECK(parse_variable(p, PARSE_VAREXPECTED, &symbol));
        
        if (parse_checktokenadvance(p, TOKEN_LEFTSQBRACKET)) {
            if (!parse_index(p, &symbol)) return false;
        }
        
        if (parse_checktokenadvance(p, TOKEN_EQUAL)) {
            PARSE_CHECK(parse_pseudoexpression(p, &initializer));
        } else initializer=SYNTAXTREE_UNCONNECTED;
        
        PARSE_CHECK(parse_addnode(p, NODE_DECLARATION, MORPHO_NIL, &start, symbol, initializer, &new));
        
        if (last!=SYNTAXTREE_UNCONNECTED) {
            PARSE_CHECK(parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, last, new, &new));
        }
        
        last=new;
    } while (parse_checktokenadvance(p, TOKEN_COMMA));
    
    PARSE_CHECK(parse_statementterminator(p));
    
    *((syntaxtreeindx *) out) = new;
    
    return true;
}

/** Parses a possible typed var declaration */
bool parse_typedvardeclaration(parser *p, void *out) {
    syntaxtreeindx new=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    lexer ol; // Store the state of the parser
    parser op;
    parse_savestate(p, &op, &ol);
    parse_advance(p);
    
    if (parse_checktoken(p, TOKEN_SYMBOL)) { // It is a typed variable declaration
        syntaxtreeindx type=SYNTAXTREE_UNCONNECTED, var=SYNTAXTREE_UNCONNECTED;
        PARSE_CHECK(parse_symbol(p, &type));
        PARSE_CHECK(parse_vardeclaration(p, &var));
        PARSE_CHECK(parse_addnode(p, NODE_TYPE, MORPHO_NIL, &start, type, var, &new));
    } else if (parse_checktokenadvance(p, TOKEN_DOT) && // Check that we actually match a var declaration
               parse_checktokenadvance(p, TOKEN_SYMBOL) &&
               parse_checktokenadvance(p, TOKEN_SYMBOL)) {
        parse_restorestate(&op, &ol, p); // Match successful, so rewind the parser
        parse_advance(p);
        
        syntaxtreeindx namespace=SYNTAXTREE_UNCONNECTED, type=SYNTAXTREE_UNCONNECTED, var=SYNTAXTREE_UNCONNECTED;
        
        PARSE_CHECK(parse_symbol(p, &namespace));
        parse_advance(p);
        parse_advance(p); // Advance over TOKEN_DOT
        
        PARSE_CHECK(parse_symbol(p, &type));
        PARSE_CHECK(parse_vardeclaration(p, &var));
        
        PARSE_CHECK(parse_addnode(p, NODE_DOT, MORPHO_NIL, &start, namespace, type, &type));
        PARSE_CHECK(parse_addnode(p, NODE_TYPE, MORPHO_NIL, &start, type, var, &new));
    } else { // Perhaps it was really an expression statement
        parse_restorestate(&op, &ol, p);
        PARSE_CHECK(parse_statement(p, &new));
    }
    
    *((syntaxtreeindx *) out) = new;
    
    return true;
}

/** Parses a function declaration */
bool parse_functiondeclaration(parser *p, void *out) {
    value name=MORPHO_NIL;
    token start = p->previous;
    syntaxtreeindx args=SYNTAXTREE_UNCONNECTED,
                   body=SYNTAXTREE_UNCONNECTED;
    
    /* Function name */
    if (parse_checktokenadvance(p, TOKEN_SYMBOL) ||
        parse_checktokeniskeywordadvance(p)) {
        name=parse_tokenasstring(p);
        parse_addobject(p, name);
    } else {
        parse_error(p, false, PARSE_FNNAMEMISSING);
        return false;
    }
    
    /* Parameter list */
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_LEFTPAREN, PARSE_FNLEFTPARENMISSING));
    PARSE_CHECK(parse_arglist(p, TOKEN_RIGHTPAREN, NULL, &args));
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_FNRGHTPARENMISSING));
    
    /* Function body */
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_LEFTCURLYBRACKET, PARSE_FNLEFTCURLYMISSING));
    PARSE_CHECK(parse_blockstatement(p, &body));
    
    return parse_addnode(p, NODE_FUNCTION, name, &start, args, body, (syntaxtreeindx *) out);
}

/* Parses a class declaration */
bool parse_classdeclaration(parser *p, void *out) {
    value name=MORPHO_NIL;
    syntaxtreeindx sclass=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    /* Class name */
    if (parse_checktokenadvance(p, TOKEN_SYMBOL)) {
        name=parse_tokenasstring(p);
        parse_addobject(p, name);
    } else {
        parse_error(p, false, PARSE_EXPECTCLASSNAME);
        return false;
    }
    
    /* Extract a superclass name */
    if (parse_checktokenadvance(p, TOKEN_LT) || parse_checktokenadvance(p, TOKEN_IS)) {
        PARSE_CHECK(parse_reference(p, PARSE_EXPECTSUPER, &sclass));
    }
    
    if (parse_checktokenadvance(p, TOKEN_WITH)) {
        do {
            syntaxtreeindx smixin;
            PARSE_CHECK(parse_reference(p, PARSE_EXPECTMIXIN, &smixin));
            PARSE_CHECK(parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &p->previous, smixin, sclass, &sclass)); // Mixins end up being recorded in reverse order
            
        } while (parse_checktokenadvance(p, TOKEN_COMMA));
    }
    
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_LEFTCURLYBRACKET, PARSE_CLASSLEFTCURLYMISSING));
    /* Method declarations */
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED, current=SYNTAXTREE_UNCONNECTED;
    
    while (!parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET) && !parse_checktoken(p, TOKEN_EOF)) {
        PARSE_CHECK(parse_functiondeclaration(p, &current));
        
        /* If we now have more than one node, insert a sequence node */
        if (last!=SYNTAXTREE_UNCONNECTED) {
            PARSE_CHECK(parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, last, current, &current));
        }
        
        last = current;
    }
    
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTCURLYBRACKET, PARSE_CLASSRGHTCURLYMISSING));
    
    return parse_addnode(p, NODE_CLASS, name, &start, sclass, current, (syntaxtreeindx *) out);
}

/** Parse an import qualifier list. This list consists of elements introduced by for and as in any order.
 */
bool parse_importqualifierlist(parser *p, void *out) {
    syntaxtreeindx list=SYNTAXTREE_UNCONNECTED;
    bool asincluded=false; // Only one as is allowed
    
    while (!parse_checkstatementterminator(p) && !parse_checktoken(p, TOKEN_COMMA)) {
        if (parse_checktokenadvance(p, TOKEN_AS)) {
            if (asincluded) { // Only one 'as' allowed per import
                parse_error(p, true, PARSE_IMPORTMLTPLAS);
                return false;
            }
            
            if (parse_checktokenadvance(p, TOKEN_SYMBOL)) {
                syntaxtreeindx symbl;
                PARSE_CHECK(parse_symbol(p, &symbl));
                PARSE_CHECK(parse_addnode(p, NODE_AS, MORPHO_NIL, &p->previous, symbl, list, &list));
            } else {
                parse_error(p, true, PARSE_IMPORTASSYMBL);
                return false;
            }
            
            asincluded=true;
        } else if (parse_checktokenadvance(p, TOKEN_FOR)) {
            do {
                if (parse_checktokenadvance(p, TOKEN_SYMBOL)) {
                    syntaxtreeindx symbl;
                    PARSE_CHECK(parse_symbol(p, &symbl));
                    PARSE_CHECK(parse_addnode(p, NODE_FOR, MORPHO_NIL, &p->previous, symbl, list, &list));
                } else {
                    parse_error(p, true, PARSE_IMPORTFORSYMBL);
                    return false;
                }
            } while (parse_checktokenadvance(p, TOKEN_COMMA));
        } else {
            parse_error(p, true, PARSE_IMPORTUNEXPCTDTOK);
            return false;
        }
    }
        
    *((syntaxtreeindx *) out)=list;
    
    return true;
}

/** Parse an import declaration. Each import has the following structure:
 *          IMPORT
 *         /              \
 *     module           import qualifier list
 * These are chained together as sequence nodes if there's more than one
 */
bool parse_importdeclaration(parser *p, void *out) {
    syntaxtreeindx prev=SYNTAXTREE_UNCONNECTED, // Use to construct list
                   current=SYNTAXTREE_UNCONNECTED;
    
    do {
        syntaxtreeindx modulename=SYNTAXTREE_UNCONNECTED,
                       qualifier=SYNTAXTREE_UNCONNECTED;
        token start = p->previous;
        
        if (parse_checktokenadvance(p, TOKEN_STRING)) {
            PARSE_CHECK(parse_string(p, &modulename));
        } else if (parse_checktokenadvance(p, TOKEN_SYMBOL)){
            PARSE_CHECK(parse_symbol(p, &modulename));
        } else {
            parse_error(p, true, PARSE_IMPORTMISSINGNAME);
            return false;
        }
        
        PARSE_CHECK(parse_importqualifierlist(p, &qualifier));
        
        PARSE_CHECK(parse_addnode(p, NODE_IMPORT, MORPHO_NIL, &start, modulename, qualifier, &current));
        
        PARSE_CHECK(parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, prev, current, &current));
        prev = current;
    } while (parse_checktokenadvance(p, TOKEN_COMMA));
    
    PARSE_CHECK(parse_statementterminator(p));
    
    *((syntaxtreeindx *) out)=current;
    
    return true;
}

/* -------------------------------
 * Statements
 * ------------------------------- */

/** Parse a print statement */
bool parse_printstatement(parser *p, void *out) {
    token start = p->previous;
    syntaxtreeindx left;
    PARSE_CHECK(parse_pseudoexpression(p, &left));
    PARSE_CHECK(parse_statementterminator(p));
    return parse_addnode(p, NODE_PRINT, MORPHO_NIL, &start, left, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parse an expression statement */
bool parse_expressionstatement(parser *p, void *out) {
    syntaxtreeindx new;
    PARSE_CHECK(parse_expression(p, &new));
    PARSE_CHECK(parse_statementterminator(p));
    *((syntaxtreeindx *) out) = new;
    return true;
}

/** @brief Parse a block statement.
 *  @details This wraps up a sequence of statements in a SCOPE node:
 *                     SCOPE
 *                    /     \
 *                   -       body
 **/
bool parse_blockstatement(parser *p, void *out) {
    syntaxtreeindx body = SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    tokentype terminator[] = { TOKEN_RIGHTCURLYBRACKET };
    
    PARSE_CHECK(parse_declarationmulti(p, 1, terminator, &body));
    if (parse_checktoken(p, TOKEN_EOF)) {
        parse_error(p, false, PARSE_INCOMPLETEEXPRESSION);
        return false;
    } else {
        PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTCURLYBRACKET, PARSE_MISSINGSEMICOLONEXP));
    }
    
    return parse_addnode(p, NODE_SCOPE, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, body, out);
}

/** Parse an if statement */
bool parse_ifstatement(parser *p, void *out) {
    syntaxtreeindx  cond=SYNTAXTREE_UNCONNECTED,
                    then=SYNTAXTREE_UNCONNECTED,
                    els=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_LEFTPAREN, PARSE_IFLFTPARENMISSING));
    PARSE_CHECK(parse_expression(p, &cond));
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_IFRGHTPARENMISSING));
    
    token thentok = p->current;
    PARSE_CHECK(parse_statement(p, &then));
    
    if (parse_checktoken(p, TOKEN_ELSE)) {
        PARSE_CHECK(parse_advance(p));
        PARSE_CHECK(parse_statement(p, &els));
        
        /* Create an additional node that contains both statements */
        PARSE_CHECK(parse_addnode(p, NODE_THEN, MORPHO_NIL, &thentok, then, els, &then));
    }
    
    return parse_addnode(p, NODE_IF, MORPHO_NIL, &start, cond, then, out);
}

/** Parse a while statement */
bool parse_whilestatement(parser *p, void *out) {
    syntaxtreeindx  cond=SYNTAXTREE_UNCONNECTED,
                    body=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_LEFTPAREN, PARSE_WHILELFTPARENMISSING));
    PARSE_CHECK(parse_expression(p, &cond));
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_IFRGHTPARENMISSING));
    PARSE_CHECK(parse_statement(p, &body));
    
    return parse_addnode(p, NODE_WHILE, MORPHO_NIL, &start, cond, body, out);
}

/** Parse a for statement. */
bool parse_forstatement(parser *p, void *out) {
    syntaxtreeindx init=SYNTAXTREE_UNCONNECTED, // Initializer
                   cond=SYNTAXTREE_UNCONNECTED, // Condition
                   body=SYNTAXTREE_UNCONNECTED, // Loop body
                   final=SYNTAXTREE_UNCONNECTED; // Final statement
    token start = p->current;
    bool forin=false, ret=false;
 
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_LEFTPAREN, PARSE_FORLFTPARENMISSING));
    if (parse_checktokenadvance(p, TOKEN_SEMICOLON)) {
        
    } else if (parse_checktokenadvance(p, TOKEN_VAR)) {
        PARSE_CHECK(parse_vardeclaration(p, &init));
    } else {
        PARSE_CHECK(parse_expression(p, &init));
        while (parse_checktokenadvance(p, TOKEN_COMMA)) {
            syntaxtreeindx new;
            PARSE_CHECK(parse_expressionstatement(p, &new));
            parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &p->current, init, new, &init);
        }
        parse_checktokenadvance(p, TOKEN_SEMICOLON);
    }
    
    if (parse_checktokenadvance(p, TOKEN_IN)) {
        /* If its an for..in loop, parse the collection */
        PARSE_CHECK(parse_expression(p, &cond));
        forin=true;
    } else {
        /* Otherwise, parse the condition and final clause in a traditional for loop. */
        if (!parse_checktokenadvance(p, TOKEN_SEMICOLON)) {
            PARSE_CHECK(parse_expressionstatement(p, &cond));
        }
        
        if (!parse_checktoken(p, TOKEN_RIGHTPAREN)) {
            PARSE_CHECK(parse_expression(p, &final));
        }
    }
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_FORRGHTPARENMISSING));
    
    if (!parse_checkstatementterminator(p)) {
        PARSE_CHECK(parse_statement(p, &body));
    }
    
    if (forin) {
        /* A for..in loop is parsed into the syntax tree as follows:
         *
         *                 forin
         *                /     \
         *               in      body
         *              /  \
         *          init    collection
         */
        syntaxtreeindx innode;
        PARSE_CHECK(parse_addnode(p, NODE_IN, MORPHO_NIL, &start, init, cond, &innode));
        ret=parse_addnode(p, NODE_FOR, MORPHO_NIL, &start, innode, body, (syntaxtreeindx *) out);
    } else {
        /* A traditional for loop is parsed into an equivalent while loop:
         * -> for (init; cond; inc) body;
         *
         * becomes
         *              scope
         *                   \
         *                    ;
         *                   / \
         *               init   while
         *                     /     \
         *                 cond       ; // The presence of the seq. indicates a for loop
         *                           / \
         *                       body   inc
         * */
        syntaxtreeindx loop,whil,seq;
        
        PARSE_CHECK(parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, body, final, &loop));
        PARSE_CHECK(parse_addnode(p, NODE_WHILE, MORPHO_NIL, &start, cond, loop, &whil));
        PARSE_CHECK(parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, init, whil, &seq));
        ret=parse_addnode(p, NODE_SCOPE, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, seq, (syntaxtreeindx *) out);
    }
    
    return ret;
}

/** Parses a do...while loop */
bool parse_dostatement(parser *p, void *out) {
    syntaxtreeindx body=SYNTAXTREE_UNCONNECTED, // Loop body
                   cond=SYNTAXTREE_UNCONNECTED; // Condition
    token start = p->current;
    
    if (!parse_statement(p, &body)) return false;
    
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_WHILE, PARSE_EXPCTWHL));
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_LEFTPAREN, PARSE_WHILELFTPARENMISSING));
    PARSE_CHECK(parse_expression(p, &cond));
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_IFRGHTPARENMISSING));
    
    /* Optional statement terminator */
    if (parse_checkstatementterminator(p)) {
        PARSE_CHECK(parse_statementterminator(p));
    }
    
    return parse_addnode(p, NODE_DO, MORPHO_NIL, &start, body, cond, (syntaxtreeindx *) out);
}

/** Parses a break or continue statement */
bool parse_breakstatement(parser *p, void *out) {
    token start = p->previous;
    
    PARSE_CHECK(parse_statementterminator(p));
    
    return parse_addnode(p, (start.type==TOKEN_BREAK ? NODE_BREAK: NODE_CONTINUE), MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parse a return statement */
bool parse_returnstatement(parser *p, void *out) {
    token start = p->previous;
    syntaxtreeindx left = SYNTAXTREE_UNCONNECTED;
    
    if (!parse_checkstatementterminator(p)) {
        PARSE_CHECK(parse_pseudoexpression(p, &left));
    }
    
    PARSE_CHECK(parse_statementterminator(p));
    
    return parse_addnode(p, NODE_RETURN, MORPHO_NIL, &start, left, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parse a try/catch statement
        try
      /          \
    body        catch block */
bool parse_trystatement(parser *p, void *out) {
    syntaxtreeindx try=SYNTAXTREE_UNCONNECTED, // Try block
                   catch=SYNTAXTREE_UNCONNECTED; // Catch dictionary
    token start = p->current;
    
    PARSE_CHECK(parse_statement(p, &try));
    
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_CATCH, PARSE_EXPCTCTCH));
    PARSE_CHECK(parse_checkrequiredtoken(p, TOKEN_LEFTCURLYBRACKET, PARSE_CATCHLEFTCURLYMISSING));
    
    PARSE_CHECK(parse_switch(p, &catch));
    
    /* Optional statement terminator */
    if (parse_checkstatementterminator(p)) {
        PARSE_CHECK(parse_statementterminator(p));
    }
    
    return parse_addnode(p, NODE_TRY, MORPHO_NIL, &start, try, catch, (syntaxtreeindx *) out);
}

/** Parse a breakpoint statement */
bool parse_breakpointstatement(parser *p, void *out) {
    token start = p->previous;
    
    if (parse_checkstatementterminator(p)) {
        PARSE_CHECK(parse_statementterminator(p));
    }
    
    return parse_addnode(p, NODE_BREAKPOINT, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/* -------------------------------------------------------
 * Parsers for different statement types
 * ------------------------------------------------------- */

/** Parses an expression */
bool parse_expression(parser *p, void *out) {
    return parse_precedence(p, PREC_ASSIGN, out);
}

/** Parses an expression that may include an anonymous function */
bool parse_pseudoexpression(parser *p, void *out) {
    if (parse_checktokenadvance(p, TOKEN_FUNCTION)) {
        return parse_anonymousfunction(p, out);
    } else {
        return parse_expression(p, out);
    }
}

/** @brief Parse statements
 *  @details Statements are things that are allowed inside control flow statements */
bool parse_statement(parser *p, void *out) {
    if (parse_checktokenadvance(p, TOKEN_PRINT)) {
        return parse_printstatement(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_IF)) {
        return parse_ifstatement(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_WHILE)) {
        return parse_whilestatement(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_FOR)) {
        return parse_forstatement(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_DO)) {
        return parse_dostatement(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_BREAK)) {
        return parse_breakstatement(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_CONTINUE)) {
        return parse_breakstatement(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_RETURN)) {
        return parse_returnstatement(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_TRY)) {
        return parse_trystatement(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_LEFTCURLYBRACKET)) {
        return parse_blockstatement(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_AT)) {
        return parse_breakpointstatement(p, out);
    } else {
        return parse_expressionstatement(p, out);
    }
    return false;
}

/** @brief Parse declarations
 *  @details Declarations define something (e.g. a variable or a function) OR
 *           a regular statement. They are *not* allowed in control flow statements. */
bool parse_declaration(parser *p, void *out) {
    bool success=false;
    if (parse_checktokenadvance(p, TOKEN_FUNCTION)) {
        success=parse_functiondeclaration(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_VAR)) {
        success=parse_vardeclaration(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_CLASS)) {
        success=parse_classdeclaration(p, out);
    } else if (parse_checktokenadvance(p, TOKEN_IMPORT)) {
        success=parse_importdeclaration(p, out);
    } else if (parse_checktoken(p, TOKEN_SYMBOL)) { // Typed var declaration ?
        success=parse_typedvardeclaration(p, out);
    } else {
        success=parse_statement(p, out);
    }
    
    //if (!success) parse_synchronize(p);
    return success;
}

/** Parses multiple declarations, separated by ; separators
 *  @param p    the parser
 *  @param end  token type to terminate on [use TOKEN_EOF if no special terminator]
 *  @returns    the syntaxtreeindx of the parsed expression */
bool parse_declarationmulti(parser *p, int n, tokentype *end, void *out) {
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED, current=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    
    while (!parse_checktokenmulti(p, n, end) && !parse_checktoken(p, TOKEN_EOF)) {
        PARSE_CHECK(parse_declaration(p, &current));
        
        /* If we now have more than one node, insert a sequence node */
        if (last!=SYNTAXTREE_UNCONNECTED) {
            PARSE_CHECK(parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, last, current, &current));
        }
        
        last = current;
    }
    
    *((syntaxtreeindx *) out) = current;
    
    return true;
}

/** Parse a program as a sequence of declarations and statements */
bool parse_program(parser *p, void *out) {
    tokentype terminator[] = { TOKEN_EOF };
    
    parse_declarationmulti(p, 1, terminator, &((syntaxtree *) p->out)->entry);
    
    return ERROR_SUCCEEDED(*p->err);
}

/* **********************************************************************
 * Parser definition table for morpho grammar
 * ********************************************************************** */

parserule rules[] = {
    PARSERULE_UNUSED(TOKEN_NEWLINE),
    PARSERULE_INFIX(TOKEN_QUESTION, parse_ternary, PREC_TERNARY),
    
    PARSERULE_PREFIX(TOKEN_STRING, parse_string),
    PARSERULE_PREFIX(TOKEN_INTERPOLATION, parse_interpolation),
    PARSERULE_PREFIX(TOKEN_INTEGER, parse_integer),
    PARSERULE_PREFIX(TOKEN_NUMBER, parse_number),
    PARSERULE_PREFIX(TOKEN_SYMBOL, parse_symbol),

    PARSERULE_MIXFIX(TOKEN_LEFTPAREN, parse_grouping, parse_call, PREC_CALL),
    PARSERULE_UNUSED(TOKEN_RIGHTPAREN),
    PARSERULE_MIXFIX(TOKEN_LEFTSQBRACKET, parse_list, parse_index, PREC_CALL),
    PARSERULE_UNUSED(TOKEN_RIGHTSQBRACKET),
    PARSERULE_PREFIX(TOKEN_LEFTCURLYBRACKET, parse_dictionary),
    PARSERULE_UNUSED(TOKEN_RIGHTCURLYBRACKET),
    PARSERULE_UNUSED(TOKEN_COLON),
    PARSERULE_UNUSED(TOKEN_SEMICOLON),
    PARSERULE_UNUSED(TOKEN_COMMA),
    
    PARSERULE_INFIX(TOKEN_PLUS, parse_binary, PREC_TERM),
    PARSERULE_MIXFIX(TOKEN_MINUS, parse_unary, parse_binary, PREC_TERM),
    PARSERULE_INFIX(TOKEN_STAR, parse_binary, PREC_FACTOR),
    PARSERULE_INFIX(TOKEN_SLASH, parse_binary, PREC_FACTOR),
    PARSERULE_INFIX(TOKEN_CIRCUMFLEX, parse_binary, PREC_POW),
    
    PARSERULE_UNUSED(TOKEN_PLUSPLUS),
    PARSERULE_UNUSED(TOKEN_MINUSMINUS),
    PARSERULE_INFIX(TOKEN_PLUSEQ, parse_assignby, PREC_ASSIGN),
    PARSERULE_INFIX(TOKEN_MINUSEQ, parse_assignby, PREC_ASSIGN),
    PARSERULE_INFIX(TOKEN_STAREQ, parse_assignby, PREC_ASSIGN),
    PARSERULE_INFIX(TOKEN_SLASHEQ, parse_assignby, PREC_ASSIGN),
    PARSERULE_UNUSED(TOKEN_HASH),
    PARSERULE_PREFIX(TOKEN_AT, parse_unary),
    
    PARSERULE_INFIX(TOKEN_DOT, parse_binary, PREC_CALL),
    PARSERULE_INFIX(TOKEN_DOTDOT, parse_range, PREC_RANGE),
    PARSERULE_INFIX(TOKEN_DOTDOTDOT, parse_range, PREC_RANGE),
    PARSERULE_PREFIX(TOKEN_EXCLAMATION, parse_unary),
    PARSERULE_UNUSED(TOKEN_AMP),
    PARSERULE_UNUSED(TOKEN_VBAR),
    PARSERULE_INFIX(TOKEN_DBLAMP, parse_binary, PREC_AND),
    PARSERULE_INFIX(TOKEN_DBLVBAR, parse_binary, PREC_OR),
    PARSERULE_INFIX(TOKEN_EQUAL, parse_binary, PREC_ASSIGN),
    PARSERULE_INFIX(TOKEN_EQ, parse_binary, PREC_EQUALITY),
    PARSERULE_INFIX(TOKEN_NEQ, parse_binary, PREC_EQUALITY),
    PARSERULE_INFIX(TOKEN_LT, parse_binary, PREC_COMPARISON),
    PARSERULE_INFIX(TOKEN_GT, parse_binary, PREC_COMPARISON),
    PARSERULE_INFIX(TOKEN_LTEQ, parse_binary, PREC_COMPARISON),
    PARSERULE_INFIX(TOKEN_GTEQ, parse_binary, PREC_COMPARISON),
    
    PARSERULE_PREFIX(TOKEN_TRUE, parse_bool),
    PARSERULE_PREFIX(TOKEN_FALSE, parse_bool),
    PARSERULE_PREFIX(TOKEN_NIL, parse_nil),
    PARSERULE_PREFIX(TOKEN_SELF, parse_self),
    PARSERULE_PREFIX(TOKEN_SUPER, parse_super),
    PARSERULE_PREFIX(TOKEN_IMAG, parse_complex),
    PARSERULE_UNUSED(TOKEN_PRINT),
    PARSERULE_UNUSED(TOKEN_VAR),
    PARSERULE_UNUSED(TOKEN_IF),
    PARSERULE_UNUSED(TOKEN_ELSE),
    PARSERULE_UNUSED(TOKEN_IN),
    PARSERULE_UNUSED(TOKEN_WHILE),
    PARSERULE_UNUSED(TOKEN_FOR),
    PARSERULE_UNUSED(TOKEN_DO),
    PARSERULE_UNUSED(TOKEN_BREAK),
    PARSERULE_UNUSED(TOKEN_CONTINUE),
    PARSERULE_UNUSED(TOKEN_FUNCTION),
    PARSERULE_UNUSED(TOKEN_RETURN),
    PARSERULE_UNUSED(TOKEN_CLASS),
    PARSERULE_UNUSED(TOKEN_IMPORT),
    PARSERULE_UNUSED(TOKEN_AS),
    PARSERULE_UNUSED(TOKEN_IS),
    PARSERULE_UNUSED(TOKEN_WITH),
    PARSERULE_UNUSED(TOKEN_TRY),
    PARSERULE_UNUSED(TOKEN_CATCH),
    
    PARSERULE_UNUSED(TOKEN_INCOMPLETE),
    PARSERULE_UNUSED(TOKEN_EOF),
    PARSERULE_UNUSED(TOKEN_NONE)
};

/* **********************************************************************
 * Obtain parse rules
 * ********************************************************************** */

/** Compares two parse rules */
int _parse_parserulecmp(const void *l, const void *r) {
    parserule *a = (parserule *) l;
    parserule *b = (parserule *) r;
    return ((int) a->type) - ((int) b->type);
}

/** Get the rule to parse an element of type tokentype. */
parserule *parse_getrule(parser *p, tokentype type) {
    if (p->parsetable.count==0) return &rules[type];
    
    parserule key = { .type = type };
    
    return bsearch(&key, p->parsetable.data, p->parsetable.count, sizeof(parserule), _parse_parserulecmp);
}

/* **********************************************************************
 * Customize parsers
 * ********************************************************************** */

/** Defines the parse table. */
bool parse_setparsetable(parser *p, parserule *rules) {
    varray_parseruleclear(&p->parsetable);
    
    for (int i=0; rules[i].type!=TOKEN_NONE; i++) {
        if (rules[i].prefix!=NULL || rules[i].infix!=NULL) {
            if (!varray_parseruleadd(&p->parsetable, &rules[i], 1)) return false;
        }
    }
    
    qsort(p->parsetable.data, p->parsetable.count, sizeof(parserule), _parse_parserulecmp);
    return true;
}

/** Sets the parse function to be called to start parsing */
void parse_setbaseparsefn(parser *p, parsefunction fn) {
    p->baseparsefn = fn;
}

/** Sets whether to skip new line tokens, and define the token type if so. */
void parse_setskipnewline(parser *p, bool skip, tokentype toknewline) {
    p->skipnewline = skip;
    p->toknewline = toknewline;
}

/** Set maximum recursion depth
 @warning: It is not guaranteed that values above PARSE_MAXRECURSIONDEPTH are achievable */
void parse_setmaxrecursiondepth(parser *p, int maxdepth) {
    p->maxrecursiondepth = maxdepth;
}


/* **********************************************************************
 * Interface
 * ********************************************************************** */

/** @brief Initialize a parser
 *  @param p       the parser to initialize
 *  @param lex   lexer to use
 *  @param err   an error structure to fill out if necessary
 *  @param tree Pointer to the output */
void parse_init(parser *p, lexer *lex, error *err, void *out) {
    p->current = TOKEN_BLANK;
    p->previous = TOKEN_BLANK;
    p->left = SYNTAXTREE_UNCONNECTED;
    p->lex=lex;
    p->err=err;
    p->out=out;
    p->nl=false;
    p->maxrecursiondepth=PARSE_MAXRECURSIONDEPTH;
    p->recursiondepth=0;
    varray_parseruleinit(&p->parsetable);
    varray_valueinit(&p->objects);
    
    // Configure parser to parse morpho by default
    parse_setbaseparsefn(p, parse_program);
    parse_setskipnewline(p, true, TOKEN_NEWLINE);
    parse_setparsetable(p, rules);
}

/** @brief Clear a parser */
void parse_clear(parser *p) {
    p->current = TOKEN_BLANK;
    p->previous = TOKEN_BLANK;
    p->left = SYNTAXTREE_UNCONNECTED;
    varray_parseruleclear(&p->parsetable);
    
    parse_clearobjects(p);
}

/** Generic entry point into the parser; call this from your own parser wrapper */
bool parse(parser *p) {
    PARSE_CHECK(parse_advance(p));
    bool success=(p->baseparsefn) (p, p->out);
    if (!success) parse_freeobjects(p);
    parse_clearobjects(p);
    return success;
}

/** Parse morpho source */
bool morpho_parse(parser *p) {
    lex_skipshebang(p->lex);
    bool success=parse(p);
    
    if (!success) syntaxtree_wipe((syntaxtree *) p->out);
    
    return success;
}

/* **********************************************************************
 * Other useful parsers
 * ********************************************************************** */

/** Convenience function to parse a string into an array of values
 * @param[in] string - string to parse
 * @param[in] nmax      - maximum number of values to read
 * @param[in] v            - value array, filled out on return
 * @param[out] n          - number of values read
 * @param[out] err      - error structure filled out if an error occurs
 * @returns true if successful, false otherwise. */
bool parse_stringtovaluearray(char *string, unsigned int nmax, value *v, unsigned int *n, error *err) {
    lexer l;
    token tok;
    unsigned int k=0;
    bool minus=false;
    lex_init(&l, string, 0);
    
    do {
        if (!lex(&l, &tok, err)) return false;
        switch(tok.type) {
            case TOKEN_INTEGER: {
                long f = strtol(tok.start, NULL, 10);
                v[k]=MORPHO_INTEGER((minus ? -f : f)); k++; minus=false;
            }
                break;
            case TOKEN_NUMBER: {
                double f = strtod(tok.start, NULL);
                v[k]=MORPHO_FLOAT((minus ? -f : f)); k++; minus=false;
            }
                break;
            case TOKEN_MINUS:
                minus=true;
                break;
            case TOKEN_COMMA:
            case TOKEN_EOF:
                break;
            default:
                morpho_writeerrorwithid(err, PARSE_UNRECGNZEDTOK, NULL, ERROR_POSNUNIDENTIFIABLE, ERROR_POSNUNIDENTIFIABLE);
                return false; 
                break;
        }
    } while (tok.type!=TOKEN_EOF && k<nmax);
    
    if (n) *n=k;
    
    lex_clear(&l);
    
    return true;
}

/* Parses a literal string, returning a value */
bool parse_value(const char *in, value *out) {
    lexer l;
    parser p;
    syntaxtree tree;
    error err;
    bool success=false;
    error_init(&err);
    syntaxtree_init(&tree);
    lex_init(&l, in, 1);
    parse_init(&p, &l, &err, &tree);
    if (parse(&p) && tree.tree.count>0) {
        syntaxtreenode node = tree.tree.data[tree.entry];
        
        if (SYNTAXTREE_ISLEAF(node.type)) {
            if (MORPHO_ISSTRING(node.content)) {
                *out = object_clonestring(node.content);
            } else *out = node.content;
            
            success=true;
        }
    }
    
    syntaxtree_clear(&tree);
    return success;
}


void parse_initialize(void) {
    /* Parse errors */
    morpho_defineerror(PARSE_INCOMPLETEEXPRESSION, ERROR_PARSE, PARSE_INCOMPLETEEXPRESSION_MSG);
    morpho_defineerror(PARSE_MISSINGPARENTHESIS, ERROR_PARSE, PARSE_MISSINGPARENTHESIS_MSG);
    morpho_defineerror(PARSE_EXPECTEXPRESSION, ERROR_PARSE, PARSE_EXPECTEXPRESSION_MSG);
    morpho_defineerror(PARSE_MISSINGSEMICOLON, ERROR_PARSE, PARSE_MISSINGSEMICOLON_MSG);
    morpho_defineerror(PARSE_MISSINGSEMICOLONEXP, ERROR_PARSE, PARSE_MISSINGSEMICOLONEXP_MSG);
    morpho_defineerror(PARSE_MISSINGSEMICOLONVAR, ERROR_PARSE, PARSE_MISSINGSEMICOLONVAR_MSG);
    morpho_defineerror(PARSE_VAREXPECTED, ERROR_PARSE, PARSE_VAREXPECTED_MSG);
    morpho_defineerror(PARSE_SYMBLEXPECTED, ERROR_PARSE, PARSE_SYMBLEXPECTED_MSG);
    morpho_defineerror(PARSE_BLOCKTERMINATOREXP, ERROR_PARSE, PARSE_BLOCKTERMINATOREXP_MSG);
    morpho_defineerror(PARSE_MSSNGSQBRC, ERROR_PARSE, PARSE_MSSNGSQBRC_MSG);
    morpho_defineerror(PARSE_MSSNGCOMMA, ERROR_PARSE, PARSE_MSSNGCOMMA_MSG);
    morpho_defineerror(PARSE_TRNRYMSSNGCOLON, ERROR_PARSE, PARSE_TRNRYMSSNGCOLON_MSG);
    morpho_defineerror(PARSE_IFLFTPARENMISSING, ERROR_PARSE, PARSE_IFLFTPARENMISSING_MSG);
    morpho_defineerror(PARSE_IFRGHTPARENMISSING, ERROR_PARSE, PARSE_IFRGHTPARENMISSING_MSG);
    morpho_defineerror(PARSE_WHILELFTPARENMISSING, ERROR_PARSE, PARSE_WHILELFTPARENMISSING_MSG);
    morpho_defineerror(PARSE_FORLFTPARENMISSING, ERROR_PARSE, PARSE_FORLFTPARENMISSING_MSG);
    morpho_defineerror(PARSE_FORSEMICOLONMISSING, ERROR_PARSE, PARSE_FORSEMICOLONMISSING_MSG);
    morpho_defineerror(PARSE_FORRGHTPARENMISSING, ERROR_PARSE, PARSE_FORRGHTPARENMISSING_MSG);
    morpho_defineerror(PARSE_FNNAMEMISSING, ERROR_PARSE, PARSE_FNNAMEMISSING_MSG);
    morpho_defineerror(PARSE_FNLEFTPARENMISSING, ERROR_PARSE, PARSE_FNLEFTPARENMISSING_MSG);
    morpho_defineerror(PARSE_FNRGHTPARENMISSING, ERROR_PARSE, PARSE_FNRGHTPARENMISSING_MSG);
    morpho_defineerror(PARSE_FNLEFTCURLYMISSING, ERROR_PARSE, PARSE_FNLEFTCURLYMISSING_MSG);
    morpho_defineerror(PARSE_CALLRGHTPARENMISSING, ERROR_PARSE, PARSE_CALLRGHTPARENMISSING_MSG);
    morpho_defineerror(PARSE_EXPECTCLASSNAME, ERROR_PARSE, PARSE_EXPECTCLASSNAME_MSG);
    morpho_defineerror(PARSE_CLASSLEFTCURLYMISSING, ERROR_PARSE, PARSE_CLASSLEFTCURLYMISSING_MSG);
    morpho_defineerror(PARSE_CLASSRGHTCURLYMISSING, ERROR_PARSE, PARSE_CLASSRGHTCURLYMISSING_MSG);
    morpho_defineerror(PARSE_EXPECTDOTAFTERSUPER, ERROR_PARSE, PARSE_EXPECTDOTAFTERSUPER_MSG);
    morpho_defineerror(PARSE_INCOMPLETESTRINGINT, ERROR_PARSE, PARSE_INCOMPLETESTRINGINT_MSG);
    morpho_defineerror(PARSE_VARBLANKINDEX, ERROR_COMPILE, PARSE_VARBLANKINDEX_MSG);
    morpho_defineerror(PARSE_IMPORTMISSINGNAME, ERROR_PARSE, PARSE_IMPORTMISSINGNAME_MSG);
    morpho_defineerror(PARSE_IMPORTMLTPLAS, ERROR_PARSE, PARSE_IMPORTMLTPLAS_MSG);
    morpho_defineerror(PARSE_IMPORTUNEXPCTDTOK, ERROR_PARSE, PARSE_IMPORTUNEXPCTDTOK_MSG);
    morpho_defineerror(PARSE_IMPORTASSYMBL, ERROR_PARSE, PARSE_IMPORTASSYMBL_MSG);
    morpho_defineerror(PARSE_IMPORTFORSYMBL, ERROR_PARSE, PARSE_IMPORTFORSYMBL_MSG);
    morpho_defineerror(PARSE_EXPECTSUPER, ERROR_COMPILE, PARSE_EXPECTSUPER_MSG);
    morpho_defineerror(PARSE_EXPECTMIXIN, ERROR_COMPILE, PARSE_EXPECTMIXIN_MSG);
    
    morpho_defineerror(PARSE_UNRECGNZEDTOK, ERROR_PARSE, PARSE_UNRECGNZEDTOK_MSG);
    morpho_defineerror(PARSE_DCTSPRTR, ERROR_PARSE, PARSE_DCTSPRTR_MSG);
    morpho_defineerror(PARSE_DCTTRMNTR, ERROR_PARSE, PARSE_DCTTRMNTR_MSG);
    morpho_defineerror(PARSE_SWTCHSPRTR, ERROR_PARSE, PARSE_SWTCHSPRTR_MSG);
    morpho_defineerror(PARSE_DCTENTRYSPRTR, ERROR_PARSE, PARSE_DCTENTRYSPRTR_MSG);
    morpho_defineerror(PARSE_EXPCTWHL, ERROR_PARSE, PARSE_EXPCTWHL_MSG);
    morpho_defineerror(PARSE_EXPCTCTCH, ERROR_PARSE, PARSE_EXPCTCTCH_MSG);
    morpho_defineerror(PARSE_ONEVARPR, ERROR_PARSE, PARSE_ONEVARPR_MSG);
    morpho_defineerror(PARSE_CATCHLEFTCURLYMISSING, ERROR_PARSE, PARSE_CATCHLEFTCURLYMISSING_MSG);
    morpho_defineerror(PARSE_VALRANGE, ERROR_PARSE, PARSE_VALRANGE_MSG);
    morpho_defineerror(PARSE_STRESC, ERROR_PARSE, PARSE_STRESC_MSG);
    morpho_defineerror(PARSE_RCRSNLMT, ERROR_PARSE, PARSE_RCRSNLMT_MSG);
    morpho_defineerror(PARSE_UNESCPDCTRL, ERROR_PARSE, PARSE_UNESCPDCTRL_MSG);
    morpho_defineerror(PARSE_INVLDUNCD, ERROR_PARSE, PARSE_INVLDUNCD_MSG);
}
