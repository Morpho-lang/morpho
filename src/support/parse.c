/** @file parse.c
 *  @author T J Atherton 
 *
 *  @brief Parser
*/

#include <string.h>
#include <ctype.h>
#include "parse.h"
#include "object.h"
#include "common.h"
#include "cmplx.h"
#include "syntaxtree.h"

/** Varrays of parse rules */
DEFINE_VARRAY(parserule, parserule)

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
    morpho_writeerrorwithid(p->err, id, tok->line, tok->posn, args);
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

/** @brief Continues parsing while tokens have a lower or equal precendece than a specified value.
 *  @param   p    the parser in use
 *  @param   precendence precedence value to keep below or equal to
 *  @returns syntaxtreeindx for the expression parsed */
bool parse_precedence(parser *p, precedence prec, void *out) {
    parsefunction prefixrule=NULL, infixrule=NULL;
    
    if (!parse_advance(p)) return false;
    
    parserule *rule = parse_getrule(p, p->previous.type);
    if (rule) prefixrule = rule->prefix;
    
    if (!rule || !prefixrule) {
        parse_error(p, true, PARSE_EXPECTEXPRESSION);
        return false;
    }
    
    if (!prefixrule(p, out)) return false;
    
    /* Now keep parsing while the tokens have lower precedence */
    rule=parse_getrule(p, p->current.type);
    while (rule!=NULL && prec <= rule->precedence) {
#ifdef MORPHO_NEWLINETERMINATORS
        /* Break if a newline is encountered before a function call */
        if (p->current.type==TOKEN_LEFTPAREN && p->nl) break;
#endif
        
        parse_advance(p);
        
        infixrule = parse_getrule(p, p->previous.type)->infix;
        if (infixrule) {
            if (!infixrule(p, out)) return false;
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
    if (!parse_checktoken(p, type)) return false;
    parse_advance(p);
    return true;
}

/** @brief Checks if the next token has the required type and advance if it does, otherwise generates an error.
 *  @param   p    the parser in use
 *  @param   type type to check
 *  @param   id   error id to generate if the token doesn't match
 *  @returns true on success */
bool parse_checkrequiredtoken(parser *p, tokentype type, errorid id) {
    if (parse_checktoken(p, type)) {
        parse_advance(p);
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
    
    for (unsigned int i=start; i<length; i++) {

        if (input[i]=='\n') {
            varray_charwrite(&str, input[i]);
        } else if (iscntrl(input[i])) {
            parse_error(p, true, PARSE_UNESCPDCTRL);
            goto parse_stringfromtokencleanup;
        } else if (input[i]!='\\') {
            varray_charwrite(&str, input[i]);
        } else {
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
        }
    }
    
    success=true;
    if (out) {
        *out = object_stringfromvarraychar(&str);
        if (!(*out)) parse_error(p, true, ERROR_ALLOCATIONFAILED);
    }
    
parse_stringfromtokencleanup:
    varray_charclear(&str);
    
    return success;
}

/** Parses a symbol token into a value with no processing. */
value parse_tokenasstring(parser *p) {
    value s = object_stringfromcstring(p->previous.start, p->previous.length);
    if (MORPHO_ISNIL(s)) parse_error(p, true, ERROR_ALLOCATIONFAILED, OBJECT_SYMBOLLABEL);

    return s;
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

/* ------------------------------------------
 * Parser implementation functions (parselets)
 * ------------------------------------------- */

bool parse_arglist(parser *p, tokentype rightdelimiter, unsigned int *nargs, void *out);
bool parse_variable(parser *p, errorid id, void *out);
bool parse_statementterminator(parser *p);
bool parse_checkstatementterminator(parser *p);
void parse_synchronize(parser *p);

/* ------------------------------------------
 * Utility functions for this parser
 * ------------------------------------------- */

/** @brief Parses an argument list
 * @param[in]  p     the parser
 * @param[in]  rightdelimiter  token type that denotes the end of the arguments list
 * @param[out] nargs the number of arguments
 * @returns indx of the arguments list
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
                    
                } else if (varg) parse_error(p, true, PARSE_ONEVARPR);
                varg = true; vargthis = true;
            }
            
            if (!parse_pseudoexpression(p, &current)) return false;

            if (vargthis) if (!parse_addnode(p, NODE_RANGE, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, current, &current)) return false;
            
            n++;
            
            if (!parse_addnode(p, NODE_ARGLIST, MORPHO_NIL, &start, prev, current, &current)) return false;
            
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
    parse_checkrequiredtoken(p, TOKEN_SYMBOL, id);
    return parse_symbol(p, out);
}

/** Parse a statement terminator  */
bool parse_statementterminator(parser *p) {
    if (parse_checktoken(p, TOKEN_SEMICOLON)) {
        parse_advance(p);
#ifdef MORPHO_NEWLINETERMINATORS
    } else if (p->nl || parse_checktoken(p, TOKEN_EOF) || parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET)) {
#endif
    } else if (parse_checktoken(p, TOKEN_IN) || parse_checktoken(p, TOKEN_ELSE)) {
    } else {
        parse_error(p, true, PARSE_MISSINGSEMICOLONEXP);
    }
    return true;
}

/** Checks whether a possible statement terminator is next */
bool parse_checkstatementterminator(parser *p) {
    return (parse_checktoken(p, TOKEN_SEMICOLON)
#ifdef MORPHO_NEWLINETERMINATORS
            || (p->nl)
            || parse_checktoken(p, TOKEN_EOF)
            || parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET)
#endif
            || parse_checktoken(p, TOKEN_IN)
            ) ;
}

/** @brief Keep parsing til the end of a statement boundary. */
void parse_synchronize(parser *p) {
    while (p->current.type!=TOKEN_EOF) {
        /** Align */
        if (p->previous.type == TOKEN_SEMICOLON) return;
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
                return;
            default:
                ;
        }
        
        parse_advance(p);
    }
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
    long f = strtol(p->previous.start, NULL, 10);
    if (!parse_validatestrtol(p, f)) return false;
    
    return parse_addnode(p, NODE_INTEGER, MORPHO_INTEGER(f), &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parses a number */
bool parse_number(parser *p, void *out) {
    double f = strtod(p->previous.start, NULL);
    if (!parse_validatestrtod(p, f)) return false;
    
    return parse_addnode(p, NODE_FLOAT, MORPHO_FLOAT(f), &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parse a complex number */
bool parse_complex(parser *p, void *out) {
    double f;
    if (p->previous.length==2) { // just a bare im symbol
        f = 1;
    } else {
        f = strtod(p->previous.start, NULL);
        if (!parse_validatestrtod(p, f)) return false;
    }
    value c = MORPHO_OBJECT(object_newcomplex(0,f));
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
    }

    return parse_addnode(p, NODE_SUPER, MORPHO_NIL, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parses a symbol */
bool parse_symbol(parser *p, void *out) {
    value s = object_stringfromcstring(p->previous.start, p->previous.length);
    if (MORPHO_ISNIL(s)) parse_error(p, true, ERROR_ALLOCATIONFAILED, OBJECT_SYMBOLLABEL);

    return parse_addnode(p, NODE_SYMBOL, s, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parses a string */
bool parse_string(parser *p, void *out) {
    value s;
    if (!parse_stringfromtoken(p, 1, p->previous.length-1, &s)) return false;
    
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
        if (!parse_expression(p, &key)) return false;
        if (!parse_checkrequiredtoken(p, TOKEN_COLON, PARSE_DCTSPRTR)) break;
        if (!parse_expression(p, &val)) return false;
        
        /* Create an entry node */
        if (!parse_addnode(p, NODE_DICTENTRY, MORPHO_NIL, &tok, key, val, &pair)) return false;
        
        /* These are linked into a chain of dictionary nodes */
        if (!parse_addnode(p, NODE_DICTIONARY, MORPHO_NIL, &tok, last, pair, &last)) return false;
        
        if (!parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET)) {
            if (!parse_checkrequiredtoken(p, TOKEN_COMMA, PARSE_MSSNGCOMMA)) return false;
        }
    };
    
    if (!parse_checkrequiredtoken(p, TOKEN_RIGHTCURLYBRACKET, PARSE_DCTTRMNTR)) return false;
    
    *((syntaxtreeindx *) out) = last;
    
    return true;
}

/** Parses a string interpolation. */
bool parse_interpolation(parser *p, void *out) {
    token tok = p->previous;
    
    /* First copy the string */
    value s;
    if (!parse_stringfromtoken(p, 1, tok.length-2, &s)) return false;
    
    syntaxtreeindx left=SYNTAXTREE_UNCONNECTED, right=SYNTAXTREE_UNCONNECTED;
    
    if (!parse_expression(p, &left)) goto parse_interpolation_cleanup;
    if (parse_checktokenadvance(p, TOKEN_STRING)) {
        if (!parse_string(p, &right)) goto parse_interpolation_cleanup;
    } else if (parse_checktokenadvance(p, TOKEN_INTERPOLATION)) {
        if (!parse_interpolation(p, &right)) goto parse_interpolation_cleanup;
    } else {
        parse_error(p, false, PARSE_INCOMPLETESTRINGINT);
        goto parse_interpolation_cleanup;
    }
    
    return parse_addnode(p, NODE_INTERPOLATION, s, &tok, left, right, (syntaxtreeindx *) out);
    
parse_interpolation_cleanup:
    morpho_freeobject(s);
    return false;
}

/** Parses an expression in parentheses */
bool parse_grouping(parser *p, void *out) {
    syntaxtreeindx new;
    if (!parse_expression(p, &new)) return false;
    if (!parse_addnode(p, NODE_GROUPING, MORPHO_NIL, &p->previous, new, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out)) return false;
    parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_MISSINGPARENTHESIS);
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
    if (!parse_precedence(p, PREC_UNARY, &right)) return false;
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
    } else {
        if (nodetype==NODE_ASSIGN &&
            parse_checktokenadvance(p, TOKEN_FUNCTION)) {
            if (!parse_anonymousfunction(p, &right)) return false;
        } else {
            if (!parse_precedence(p, rule->precedence + (assoc == LEFT ? 1 : 0), &right)) return false;
        }
    }
    
    /* Now add this node */
    return parse_addnode(p, nodetype, MORPHO_NIL, &start, left, right, (syntaxtreeindx *) out);
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
    } else {
        if (!parse_precedence(p, rule->precedence, &right)) return false;
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
    parse_expression(p, &right);
    syntaxtreeindx one=SYNTAXTREE_UNCONNECTED;
    if (!inclusive) {
        if (!parse_addnode(p, NODE_INTEGER, MORPHO_INTEGER(1), &start, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, &one)) return false;
        if (!parse_addnode(p, NODE_SUBTRACT, MORPHO_NIL, &start, right, one, &right)) return false;
    }
    syntaxtreeindx new;
    if (!parse_addnode(p, NODE_RANGE, MORPHO_NIL, &start, left, right, &new)) return false;
    
    if (parse_checktokenadvance(p, TOKEN_COLON)) {
        syntaxtreeindx step;
        if (!parse_expression(p, &step)) return false;
        
        if (!inclusive) parse_lookupnode(p, right)->right = step;
        if (!parse_addnode(p, NODE_RANGE, MORPHO_NIL, &start, new, step, &new)) return false;
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
    
    if (!parse_arglist(p, TOKEN_RIGHTPAREN, &nargs, &right)) return false;
    
    parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_CALLRGHTPARENMISSING);
    
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
    
    if (!parse_arglist(p, TOKEN_RIGHTSQBRACKET, &nindx, &right)) return false;
    
    parse_checkrequiredtoken(p, TOKEN_RIGHTSQBRACKET, PARSE_CALLRGHTPARENMISSING);
    
    return parse_addnode(p, NODE_INDEX, MORPHO_NIL, &start, left, right, (syntaxtreeindx *) out);
}

/** Parse list  */
bool parse_list(parser *p, void *out) {
    unsigned int nindx;
    token start = p->previous;
    
    syntaxtreeindx right;
    if (!parse_arglist(p, TOKEN_RIGHTSQBRACKET, &nindx, &right)) return false;
    parse_checkrequiredtoken(p, TOKEN_RIGHTSQBRACKET, PARSE_MSSNGSQBRC);

    return parse_addnode(p, NODE_LIST, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, right, out);
}

/** Parses an anonymous function */
bool parse_anonymousfunction(parser *p, void *out) {
    token start = p->previous;
    syntaxtreeindx args=SYNTAXTREE_UNCONNECTED,
                   body=SYNTAXTREE_UNCONNECTED;
    
    /* Parameter list */
    parse_checkrequiredtoken(p, TOKEN_LEFTPAREN, PARSE_FNLEFTPARENMISSING);
    if (!parse_arglist(p, TOKEN_RIGHTPAREN, NULL, &args)) return false;
    parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_FNRGHTPARENMISSING);
    
    /* Function body */
    if (parse_checktokenadvance(p, TOKEN_LEFTCURLYBRACKET)) { // fn (x) { ... }
        if (!parse_blockstatement(p, &body)) return false;
    } else {
        if (!parse_expression(p, &body)) return false; // Short form: fn (x) x
        parse_addnode(p, NODE_RETURN, MORPHO_NIL, &start, body, SYNTAXTREE_UNCONNECTED, &body);
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
        if (!parse_expression(p, &key)) return false;
        if (!parse_checkrequiredtoken(p, TOKEN_COLON, PARSE_SWTCHSPRTR)) break;
        tokentype terminators[] = { TOKEN_STRING, TOKEN_INTEGER, TOKEN_NUMBER, TOKEN_TRUE, TOKEN_FALSE, TOKEN_NIL, TOKEN_RIGHTCURLYBRACKET };
        if (!parse_declarationmulti(p, 7, terminators, &statements)) return false;
        
        /* Create an entry node */
        if (!parse_addnode(p, NODE_DICTENTRY, MORPHO_NIL, &tok, key, statements, &pair)) return false;
        
        /* These are linked into a chain of sequence nodes */
        if (!parse_addnode(p, NODE_DICTIONARY, MORPHO_NIL, &tok, last, pair, &last)) return false;
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
        
        if (!parse_variable(p, PARSE_VAREXPECTED, &symbol)) return false;
        
        if (parse_checktokenadvance(p, TOKEN_LEFTSQBRACKET)) {
            if (!parse_index(p, &symbol)) return false;
        }
        
        if (parse_checktokenadvance(p, TOKEN_EQUAL)) {
            if (!parse_pseudoexpression(p, &initializer)) return false;
        } else initializer=SYNTAXTREE_UNCONNECTED;
        
        if (!parse_addnode(p, NODE_DECLARATION, MORPHO_NIL, &start, symbol, initializer, &new)) return false;
        
        if (last!=SYNTAXTREE_UNCONNECTED) {
            if (!parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, last, new, &new)) return false;
        }
        
        last=new;
    } while (parse_checktokenadvance(p, TOKEN_COMMA));
    
    if (!parse_statementterminator(p)) return false;
    
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
    if (parse_checktokenadvance(p, TOKEN_SYMBOL)) {
        name=parse_tokenasstring(p);
    } else parse_error(p, false, PARSE_FNNAMEMISSING);
    
    /* Parameter list */
    parse_checkrequiredtoken(p, TOKEN_LEFTPAREN, PARSE_FNLEFTPARENMISSING);
    if (!parse_arglist(p, TOKEN_RIGHTPAREN, NULL, &args)) return false;
    parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_FNRGHTPARENMISSING);
    
    /* Function body */
    parse_checkrequiredtoken(p, TOKEN_LEFTCURLYBRACKET, PARSE_FNLEFTCURLYMISSING);
    if (!parse_blockstatement(p, &body)) return false;
    
    return parse_addnode(p, NODE_FUNCTION, name, &start, args, body, (syntaxtreeindx *) out);
}

/* Parses a class declaration */
bool parse_classdeclaration(parser *p, void *out) {
    value name=MORPHO_NIL;
    value sname=MORPHO_NIL;
    syntaxtreeindx sclass=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    /* Class name */
    if (parse_checktokenadvance(p, TOKEN_SYMBOL)) {
        name=parse_tokenasstring(p);
    } else parse_error(p, false, PARSE_EXPECTCLASSNAME);
    
    /* Extract a superclass name */
    if (parse_checktokenadvance(p, TOKEN_LT) || parse_checktokenadvance(p, TOKEN_IS)) {
        parse_checkrequiredtoken(p, TOKEN_SYMBOL, PARSE_EXPECTSUPER);
        sname=parse_tokenasstring(p);
        if (!parse_addnode(p, NODE_SYMBOL, sname, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, &sclass)) return false;
    }
    
    if (parse_checktokenadvance(p, TOKEN_WITH)) {
        do {
            parse_checkrequiredtoken(p, TOKEN_SYMBOL, PARSE_EXPECTSUPER);
            value mixin=parse_tokenasstring(p);
            
            syntaxtreeindx smixin;
            if (!parse_addnode(p, NODE_SYMBOL, mixin, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, &smixin)) return false;
                
            if (!parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &p->previous, smixin, sclass, &sclass)) return false; // Mixins end up being recorded in reverse order
            
        } while (parse_checktokenadvance(p, TOKEN_COMMA));
    }
    
    parse_checkrequiredtoken(p, TOKEN_LEFTCURLYBRACKET, PARSE_CLASSLEFTCURLYMISSING);
    /* Method declarations */
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED, current=SYNTAXTREE_UNCONNECTED;
    
    while (!parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET) && !parse_checktoken(p, TOKEN_EOF)) {
        if (!parse_functiondeclaration(p, &current)) return false;
        
        /* If we now have more than one node, insert a sequence node */
        if (last!=SYNTAXTREE_UNCONNECTED) {
            if (!parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, last, current, &current)) return false;
        }
        
        last = current;
    }
    
    parse_checkrequiredtoken(p, TOKEN_RIGHTCURLYBRACKET, PARSE_CLASSRGHTCURLYMISSING);
    
    return parse_addnode(p, NODE_CLASS, name, &start, sclass, current, (syntaxtreeindx *) out);
}

/** Parse an import declaration.
 *          IMPORT
 *         /              \
 *     module           FOR   or as
 *                    \
 *                   ( items )
 */
bool parse_importdeclaration(parser *p, void *out) {
    syntaxtreeindx modulename=SYNTAXTREE_UNCONNECTED, right=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    if (parse_checktokenadvance(p, TOKEN_STRING)) {
        if (!parse_string(p, &modulename)) return false;
    } else if (parse_checktokenadvance(p, TOKEN_SYMBOL)){
        if (!parse_symbol(p, &modulename)) return false;
    } else {
        parse_error(p, true, PARSE_IMPORTMISSINGNAME);
        return false;
    }
    
    if (!parse_checkstatementterminator(p)) {
        if (parse_checktokenadvance(p, TOKEN_AS)) {
            if (parse_checktokenadvance(p, TOKEN_SYMBOL)) {
                if (!parse_symbol(p, &right)) return false;
            } else parse_error(p, true, PARSE_IMPORTASSYMBL);
        } else if (parse_checktokenadvance(p, TOKEN_FOR)) {
            do {
                if (parse_checktokenadvance(p, TOKEN_SYMBOL)) {
                    syntaxtreeindx symbl;
                    if (!parse_symbol(p, &symbl)) return false;
                    if (!parse_addnode(p, NODE_FOR, MORPHO_NIL, &p->previous, right, symbl, &right)) return false;
                } else parse_error(p, true, PARSE_IMPORTFORSYMBL);
            } while (parse_checktokenadvance(p, TOKEN_COMMA));
        } else {
            parse_error(p, true, PARSE_IMPORTUNEXPCTDTOK);
        }
    }
    
    parse_statementterminator(p);
    
    return parse_addnode(p, NODE_IMPORT, MORPHO_NIL, &start, modulename, right, (syntaxtreeindx *) out);
}

/* -------------------------------
 * Statements
 * ------------------------------- */

/** Parse a print statement */
bool parse_printstatement(parser *p, void *out) {
    token start = p->previous;
    syntaxtreeindx left;
    if (!parse_pseudoexpression(p, &left)) return false;
    if (!parse_statementterminator(p)) return false;
    return parse_addnode(p, NODE_PRINT, MORPHO_NIL, &start, left, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parse an expression statement */
bool parse_expressionstatement(parser *p, void *out) {
    syntaxtreeindx new;
    if (!parse_expression(p, &new)) return false;
    if (!parse_statementterminator(p)) return false;
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
    syntaxtreeindx body = SYNTAXTREE_UNCONNECTED,
                   scope = SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    tokentype terminator[] = { TOKEN_RIGHTCURLYBRACKET };
    
    if (!parse_declarationmulti(p, 1, terminator, &body)) return false;
    if (parse_checktoken(p, TOKEN_EOF)) {
        parse_error(p, false, PARSE_INCOMPLETEEXPRESSION);
    } else {
        parse_checkrequiredtoken(p, TOKEN_RIGHTCURLYBRACKET, PARSE_MISSINGSEMICOLONEXP);
    }
    
    return parse_addnode(p, NODE_SCOPE, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, body, out);
}

/** Parse an if statement */
bool parse_ifstatement(parser *p, void *out) {
    syntaxtreeindx  cond=SYNTAXTREE_UNCONNECTED,
                    then=SYNTAXTREE_UNCONNECTED,
                    els=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    parse_checkrequiredtoken(p, TOKEN_LEFTPAREN, PARSE_IFLFTPARENMISSING);
    if (!parse_expression(p, &cond)) return false;
    parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_IFRGHTPARENMISSING);
    
    token thentok = p->current;
    if (!parse_statement(p, &then)) return false;
    
    if (parse_checktoken(p, TOKEN_ELSE)) {
        parse_advance(p);
        if (!parse_statement(p, &els)) return false;
        
        /* Create an additional node that contains both statements */
        if (!parse_addnode(p, NODE_THEN, MORPHO_NIL, &thentok, then, els, &then)) return false;
    }
    
    return parse_addnode(p, NODE_IF, MORPHO_NIL, &start, cond, then, out);
}

/** Parse a while statement */
bool parse_whilestatement(parser *p, void *out) {
    syntaxtreeindx  cond=SYNTAXTREE_UNCONNECTED,
                    body=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    parse_checkrequiredtoken(p, TOKEN_LEFTPAREN, PARSE_WHILELFTPARENMISSING);
    if (!parse_expression(p, &cond)) return false;
    parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_IFRGHTPARENMISSING);
    if (!parse_statement(p, &body)) return false;
    
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
 
    parse_checkrequiredtoken(p, TOKEN_LEFTPAREN, PARSE_FORLFTPARENMISSING);
    if (parse_checktokenadvance(p, TOKEN_SEMICOLON)) {
        
    } else if (parse_checktokenadvance(p, TOKEN_VAR)) {
        if (!parse_vardeclaration(p, &init)) return false;
    } else {
        if (!parse_expression(p, &init)) return false;
        while (parse_checktokenadvance(p, TOKEN_COMMA)) {
            syntaxtreeindx new;
            if (!parse_expressionstatement(p, &new)) return false;
            parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &p->current, init, new, &init);
        }
        parse_checktokenadvance(p, TOKEN_SEMICOLON);
    }
    
    if (parse_checktokenadvance(p, TOKEN_IN)) {
        /* If its an for..in loop, parse the collection */
        if (!parse_expression(p, &cond)) return false;
        forin=true;
    } else {
        /* Otherwise, parse the condition and final clause in a traditional for loop. */
        if (!parse_checktokenadvance(p, TOKEN_SEMICOLON)) {
            if (!parse_expressionstatement(p, &cond)) return false;
        }
        
        if (!parse_checktoken(p, TOKEN_RIGHTPAREN)) {
            if (!parse_expression(p, &final)) return false;
        }
    }
    parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_FORRGHTPARENMISSING);
    
    if (!parse_checkstatementterminator(p)) {
        if (!parse_statement(p, &body)) return false;
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
        if (!parse_addnode(p, NODE_IN, MORPHO_NIL, &start, init, cond, &innode)) return false;
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
        
        if (!parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, body, final, &loop)) return false;
        if (!parse_addnode(p, NODE_WHILE, MORPHO_NIL, &start, cond, loop, &whil)) return false;
        if (!parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, init, whil, &seq)) return false;
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
    
    parse_checkrequiredtoken(p, TOKEN_WHILE, PARSE_EXPCTWHL);
    parse_checkrequiredtoken(p, TOKEN_LEFTPAREN, PARSE_WHILELFTPARENMISSING);
    if (!parse_expression(p, &cond)) return false;
    parse_checkrequiredtoken(p, TOKEN_RIGHTPAREN, PARSE_IFRGHTPARENMISSING);
    
    /* Optional statement terminator */
    if (parse_checkstatementterminator(p)) {
        if (!parse_statementterminator(p)) return false;
    }
    
    return parse_addnode(p, NODE_DO, MORPHO_NIL, &start, body, cond, (syntaxtreeindx *) out);
}

/** Parses a break or continue statement */
bool parse_breakstatement(parser *p, void *out) {
    token start = p->previous;
    
    if (!parse_statementterminator(p)) return false;
    
    return parse_addnode(p, (start.type==TOKEN_BREAK ? NODE_BREAK: NODE_CONTINUE), MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED, (syntaxtreeindx *) out);
}

/** Parse a return statement */
bool parse_returnstatement(parser *p, void *out) {
    token start = p->previous;
    syntaxtreeindx left = SYNTAXTREE_UNCONNECTED;
    
    if (!parse_checkstatementterminator(p)) {
        if (!parse_pseudoexpression(p, &left)) return false;
    }
    
    if (!parse_statementterminator(p)) return false;
    
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
    
    if (!parse_statement(p, &try)) return false;
    
    parse_checkrequiredtoken(p, TOKEN_CATCH, PARSE_EXPCTCTCH);
    parse_checkrequiredtoken(p, TOKEN_LEFTCURLYBRACKET, PARSE_CATCHLEFTCURLYMISSING);
    
    if (!parse_switch(p, &catch)) return false;
    
    /* Optional statement terminator */
    if (parse_checkstatementterminator(p)) {
        if (!parse_statementterminator(p)) return false;
    }
    
    return parse_addnode(p, NODE_TRY, MORPHO_NIL, &start, try, catch, (syntaxtreeindx *) out);
}

/** Parse a breakpoint statement */
bool parse_breakpointstatement(parser *p, void *out) {
    token start = p->previous;
    
    if (parse_checkstatementterminator(p)) {
        if (!parse_statementterminator(p)) return false;
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
        if (!parse_declaration(p, &current)) return false;
        
        /* If we now have more than one node, insert a sequence node */
        if (last!=SYNTAXTREE_UNCONNECTED) {
            parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, last, current, &current);
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
    PARSERULE_UNUSED(TOKEN_QUESTION),
    
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
}

/** Entry point into the parser */
bool parse(parser *p) {
    lex_skipshebang(p->lex);
    if (!parse_advance(p)) return false;
    return (p->baseparsefn) (p, p->out);
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
                morpho_writeerrorwithid(err, PARSE_UNRECGNZEDTOK, ERROR_POSNUNIDENTIFIABLE, ERROR_POSNUNIDENTIFIABLE);
                return false; 
                break;
        }
    } while (tok.type!=TOKEN_EOF && k<nmax);
    
    if (n) *n=k;
    
    lex_clear(&l);
    
    return true;
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
    morpho_defineerror(PARSE_BLOCKTERMINATOREXP, ERROR_PARSE, PARSE_BLOCKTERMINATOREXP_MSG);
    morpho_defineerror(PARSE_MSSNGSQBRC, ERROR_PARSE, PARSE_MSSNGSQBRC_MSG);
    morpho_defineerror(PARSE_MSSNGCOMMA, ERROR_PARSE, PARSE_MSSNGCOMMA_MSG);
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
    morpho_defineerror(PARSE_IMPORTUNEXPCTDTOK, ERROR_PARSE, PARSE_IMPORTUNEXPCTDTOK_MSG);
    morpho_defineerror(PARSE_IMPORTASSYMBL, ERROR_PARSE, PARSE_IMPORTASSYMBL_MSG);
    morpho_defineerror(PARSE_IMPORTFORSYMBL, ERROR_PARSE, PARSE_IMPORTFORSYMBL_MSG);
    morpho_defineerror(PARSE_EXPECTSUPER, ERROR_COMPILE, PARSE_EXPECTSUPER_MSG);

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

void parse_finalize(void) {
    
}
