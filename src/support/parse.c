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
    if (p->err->id!=ERROR_NONE) return;
    
    va_start(args, id);
    morpho_writeerrorwithid(p->err, id, tok->line, tok->posn-tok->length, args);
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
        lex(l, &p->current, p->err);
        
        /* Skip any newlines encountered */
        if (p->current.type==TOKEN_NEWLINE) {
            p->nl=true;
            continue;
        }
        
        if (p->current.type != TOKEN_ERROR) break;
        UNREACHABLE("Unhandled error in parser.\n");
    }
    
    return (p->err->cat==ERROR_NONE);
}

/** @brief Checks if the next token has the required type, otherwise generates an error.
 *  @param   p    the parser in use
 *  @param   type type to check
 *  @param   id   error id to generate if the token doesn't match
 *  @returns true on success */
bool parse_consume(parser *p, tokentype type, errorid id) {
    if (p->current.type==type) {
        parse_advance(p);
        return true;
    }
    
    /* Raise an error */
    if (id!=ERROR_NONE) parse_error(p, true, id);
    return false;
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

/** Adds a node to the syntax tree. */
syntaxtreeindx parse_addnode(parser *p, syntaxtreenodetype type, value content, token *tok, syntaxtreeindx left, syntaxtreeindx right) {
    syntaxtreeindx new = syntaxtree_addnode(p->tree, type, content, tok->line, tok->posn, left, right);
    p->left=new; /* Record this for a future infix operator to catch */
    return new;
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
bool parse_matchtoken(parser *p, tokentype type) {
    if (!parse_checktoken(p, type)) return false;
    parse_advance(p);
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

/** Turn a token into a string */
value parse_stringfromtoken(parser *p, unsigned int start, unsigned int length) {
    char str[p->previous.length];
    const char *in=p->previous.start;
    unsigned int k=0;
    
    for (unsigned int i=start; i<length; i++) {
        if (in[i]!='\\') { // Escape characters
            str[k]=in[i]; k++;
        } else {
            if (i<length-1) switch (in[i+1]) {
                case 'n':
                    str[k]='\n'; break;
                case 't':
                    str[k]='\t'; break;
                case 'r':
                    str[k]='\r'; break;
                default:
                    str[k]=in[i+1]; break;
            }
            i++; k++;
        }
    }
    
    return object_stringfromcstring(str, k);
}

/** Parses a symbol token into a value */
value parse_symbolasvalue(parser *p) {
    value s = object_stringfromcstring(p->previous.start, p->previous.length);
    if (MORPHO_ISNIL(s)) parse_error(p, true, ERROR_ALLOCATIONFAILED, OBJECT_SYMBOLLABEL);

    return s;
}

parserule *parse_getrule(parser *p, tokentype type);

/* ------------------------------------------
 * Parser implementation functions (parselets)
 * ------------------------------------------- */

// Utility functions
syntaxtreeindx parse_arglist(parser *p, tokentype rightdelimiter, unsigned int *nargs);
syntaxtreeindx parse_variable(parser *p, errorid id);

// Base parsers for different elements of the grammar
syntaxtreeindx parse_precedence(parser *p, precedence precendence);
syntaxtreeindx parse_expression(parser *p);
syntaxtreeindx parse_pseudoexpression(parser *p);
syntaxtreeindx parse_statement(parser *p);
syntaxtreeindx parse_declaration(parser *p);
syntaxtreeindx parse_declarationmulti(parser *p, int n, tokentype *end);

// Simple literals
syntaxtreeindx parse_nil(parser *p);
syntaxtreeindx parse_integer(parser *p);
syntaxtreeindx parse_number(parser *p);
syntaxtreeindx parse_complex(parser *p);
syntaxtreeindx parse_bool(parser *p);
syntaxtreeindx parse_self(parser *p);
syntaxtreeindx parse_super(parser *p);
syntaxtreeindx parse_symbol(parser *p);

// Compound objects
syntaxtreeindx parse_string(parser *p);
syntaxtreeindx parse_dictionary(parser *p);
syntaxtreeindx parse_interpolation(parser *p);
syntaxtreeindx parse_grouping(parser *p);
syntaxtreeindx parse_unary(parser *p);
syntaxtreeindx parse_binary(parser *p);
syntaxtreeindx parse_assignby(parser *p);
syntaxtreeindx parse_call(parser *p);
syntaxtreeindx parse_index(parser *p);
syntaxtreeindx parse_list(parser *p);
syntaxtreeindx parse_anonymousfunction(parser *p);
syntaxtreeindx parse_switch(parser *p);

// Declarations
syntaxtreeindx parse_vardeclaration(parser *p);
syntaxtreeindx parse_functiondeclaration(parser *p);
syntaxtreeindx parse_classdeclaration(parser *p);
syntaxtreeindx parse_importdeclaration(parser *p);

// Statements
syntaxtreeindx parse_printstatement(parser *p);
syntaxtreeindx parse_expressionstatement(parser *p);
syntaxtreeindx parse_blockstatement(parser *p);
syntaxtreeindx parse_ifstatement(parser *p);
syntaxtreeindx parse_whilestatement(parser *p);
syntaxtreeindx parse_forstatement(parser *p);
syntaxtreeindx parse_dostatement(parser *p);
syntaxtreeindx parse_breakstatement(parser *p);
syntaxtreeindx parse_returnstatement(parser *p);
syntaxtreeindx parse_trystatement(parser *p);
syntaxtreeindx parse_breakpointstatement(parser *p);

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
syntaxtreeindx parse_arglist(parser *p, tokentype rightdelimiter, unsigned int *nargs) {
    syntaxtreeindx prev=SYNTAXTREE_UNCONNECTED, current=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    unsigned int n=0;
    bool varg=false;
    
    if (!parse_checktoken(p, rightdelimiter)) {
        do {
            bool vargthis = false;
            if (parse_matchtoken(p, TOKEN_DOTDOTDOT)) {
                // If we are trying to index something
                // then ... represents an open range
                if (rightdelimiter == TOKEN_RIGHTSQBRACKET) {
                    
                } else if (varg) parse_error(p, true, PARSE_ONEVARPR);
                varg = true; vargthis = true;
            }
            
            current=parse_pseudoexpression(p);

            if (vargthis) current=parse_addnode(p, NODE_RANGE, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, current);
            
            n++;
            
            current=parse_addnode(p, NODE_ARGLIST, MORPHO_NIL, &start, prev, current);
            
            prev = current;
        } while (parse_matchtoken(p, TOKEN_COMMA));
    }
    
    /* Output the number of args */
    if (nargs) *nargs=n;
    
    return current;
}

/** Parses a variable name, or raises and error if a symbol isn't found */
syntaxtreeindx parse_variable(parser *p, errorid id) {
    parse_consume(p, TOKEN_SYMBOL, id);
    return parse_symbol(p);
}

/** Parse a statement terminator  */
syntaxtreeindx parse_statementterminator(parser *p) {
    if (parse_checktoken(p, TOKEN_SEMICOLON)) {
        parse_advance(p);
#ifdef MORPHO_NEWLINETERMINATORS
    } else if (p->nl || parse_checktoken(p, TOKEN_EOF) || parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET)) {
#endif
    } else if (parse_checktoken(p, TOKEN_IN) || parse_checktoken(p, TOKEN_ELSE)) {
    } else {
        parse_error(p, true, PARSE_MISSINGSEMICOLONEXP);
    }
    return SYNTAXTREE_UNCONNECTED;
}

/* ------------------------------------------
 * Basic literals
 * ------------------------------------------- */

/** Parses nil */
syntaxtreeindx parse_nil(parser *p) {
    return parse_addnode(p, NODE_NIL,
        MORPHO_NIL, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses an integer */
syntaxtreeindx parse_integer(parser *p) {
    long f = strtol(p->previous.start, NULL, 10);
    return parse_addnode(p, NODE_INTEGER, MORPHO_INTEGER(f), &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses a number */
syntaxtreeindx parse_number(parser *p) {
    double f = strtod(p->previous.start, NULL);
    return parse_addnode(p, NODE_FLOAT, MORPHO_FLOAT(f), &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parse a complex number */
syntaxtreeindx parse_complex(parser *p) {
    double f;
    if (p->previous.length==2) { // just a bare im symbol
        f = 1;
    } else {
        f = strtod(p->previous.start,NULL);
    }
    value c = MORPHO_OBJECT(object_newcomplex(0,f));
    return parse_addnode(p, NODE_IMAG, c, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses a bool */
syntaxtreeindx parse_bool(parser *p) {
    return parse_addnode(p, NODE_BOOL,
        MORPHO_BOOL((p->previous.type==TOKEN_TRUE ? true : false)), &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses a self token */
syntaxtreeindx parse_self(parser *p) {
    return parse_addnode(p, NODE_SELF, MORPHO_NIL, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses a super token */
syntaxtreeindx parse_super(parser *p) {
    if (!parse_checktoken(p, TOKEN_DOT)) {
        parse_error(p, false, PARSE_EXPECTDOTAFTERSUPER);
    }

    return parse_addnode(p, NODE_SUPER, MORPHO_NIL, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses a symbol */
syntaxtreeindx parse_symbol(parser *p) {
    value s = object_stringfromcstring(p->previous.start, p->previous.length);
    if (MORPHO_ISNIL(s)) parse_error(p, true, ERROR_ALLOCATIONFAILED, OBJECT_SYMBOLLABEL);

    return parse_addnode(p, NODE_SYMBOL, s, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parses a string */
syntaxtreeindx parse_string(parser *p) {
    value s = parse_stringfromtoken(p, 1, p->previous.length-1);
    if (MORPHO_ISNIL(s)) parse_error(p, true, ERROR_ALLOCATIONFAILED, OBJECT_STRINGLABEL);
    return parse_addnode(p, NODE_STRING, s, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** @brief: Parses a dictionary.
 * @details Dictionaries are a list of key/value pairs,  { key : value, key: value } */
syntaxtreeindx parse_dictionary(parser *p) {
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED;
    last=parse_addnode(p, NODE_DICTIONARY, MORPHO_NIL, &p->current, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
    
    do {
        syntaxtreeindx key, val, pair;
        token tok=p->current; // Keep track of the token that corresponds to each key/value pair
        
        /* Parse the key/value pair */
        key=parse_expression(p);
        if (!parse_consume(p, TOKEN_COLON, PARSE_DCTSPRTR)) break;
        val=parse_expression(p);
        
        /* Create an entry node */
        pair=parse_addnode(p, NODE_DICTENTRY, MORPHO_NIL, &tok, key, val);
        
        /* These are linked into a chain of dictionary nodes */
        last=parse_addnode(p, NODE_DICTIONARY, MORPHO_NIL, &tok, last, pair);
        
        if (!parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET)) {
            if (!parse_consume(p, TOKEN_COMMA, PARSE_DCTENTRYSPRTR)) break;
        }
    } while(!parse_matchtoken(p, TOKEN_RIGHTCURLYBRACKET) && !parse_checktoken(p, TOKEN_EOF));
    
    return last;
}

/** Parses a string interpolation. */
syntaxtreeindx parse_interpolation(parser *p) {
    token tok = p->previous;
    
    /* First copy the string */
    //value s = object_stringfromcstring(tok.start+1, tok.length-3);
    value s = parse_stringfromtoken(p, 1, tok.length-2);
    if (MORPHO_ISNIL(s)) parse_error(p, true, ERROR_ALLOCATIONFAILED, OBJECT_STRINGLABEL);
    
    syntaxtreeindx left=SYNTAXTREE_UNCONNECTED, right=SYNTAXTREE_UNCONNECTED;
    
    left = parse_expression(p);
    if (parse_matchtoken(p, TOKEN_STRING)) {
        right = parse_string(p);
    } else if (parse_matchtoken(p, TOKEN_INTERPOLATION)) {
        right = parse_interpolation(p);
    } else {
        parse_error(p, false, PARSE_INCOMPLETESTRINGINT);
    }
    
    return parse_addnode(p, NODE_INTERPOLATION, s, &tok, left, right);
}

/** Parses an expression in parentheses */
syntaxtreeindx parse_grouping(parser *p) {
    syntaxtreeindx indx;
    indx = parse_addnode(p, NODE_GROUPING, MORPHO_NIL, &p->previous, parse_expression(p), SYNTAXTREE_UNCONNECTED);
    parse_consume(p, TOKEN_RIGHTPAREN, PARSE_MISSINGPARENTHESIS);
    return indx;
}

/** Parse a unary operator */
syntaxtreeindx parse_unary(parser *p) {
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
    return parse_addnode(p, nodetype, MORPHO_NIL, &start, parse_precedence(p, PREC_UNARY), SYNTAXTREE_UNCONNECTED);
}

/** Parse a binary operator */
syntaxtreeindx parse_binary(parser *p) {
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
            parse_matchtoken(p, TOKEN_FUNCTION)) {
            right=parse_anonymousfunction(p);
        } else {
            right = parse_precedence(p, rule->precedence + (assoc == LEFT ? 1 : 0));
        }
    }
    
    /* Now add this node */
    return parse_addnode(p, nodetype, MORPHO_NIL, &start, left, right);
}

/** Parse operators like +=, -=, *= etc. */
syntaxtreeindx parse_assignby(parser *p) {
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
        right = parse_precedence(p, rule->precedence);
    }
    
    right=parse_addnode(p, nodetype, MORPHO_NIL, &start, left, right);
    
    /* Now add this node */
    return parse_addnode(p, NODE_ASSIGN, MORPHO_NIL, &start, left, right);
}

/** Parses a range */
syntaxtreeindx parse_range(parser *p) {
    token start = p->previous;
    bool inclusive = (start.type==TOKEN_DOTDOT);
    
    syntaxtreeindx left=p->left;
    syntaxtreeindx right=parse_expression(p);
    syntaxtreeindx one=SYNTAXTREE_UNCONNECTED;
    if (!inclusive) {
        one=parse_addnode(p, NODE_INTEGER, MORPHO_INTEGER(1), &start, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
        right=parse_addnode(p, NODE_SUBTRACT, MORPHO_NIL, &start, right, one);
    }
    syntaxtreeindx out=parse_addnode(p, NODE_RANGE, MORPHO_NIL, &start, left, right);
    
    if (parse_matchtoken(p, TOKEN_COLON)) {
        syntaxtreeindx step=parse_expression(p);
        if (!inclusive) p->tree->tree.data[right].right=step;
        out=parse_addnode(p, NODE_RANGE, MORPHO_NIL, &start, out, step);
    }
    
    return out;
}

/** Parse a function call */
syntaxtreeindx parse_call(parser *p) {
    token start = p->previous;
    syntaxtreeindx left=p->left;
    syntaxtreeindx right;
    unsigned int nargs;
    
    /* Abandon parsing the function call if a newline is detected between the symbol and opening parenthesis */
//    if (parse_checkprevnl(p)) {
//        return p->left;
//    }
    
    right=parse_arglist(p, TOKEN_RIGHTPAREN, &nargs);
    
    parse_consume(p, TOKEN_RIGHTPAREN, PARSE_CALLRGHTPARENMISSING);
    
    return parse_addnode(p, NODE_CALL, MORPHO_NIL, &start, left, right);
}

/** Parse index
        index
       /          \
  symbol         indices
 */
syntaxtreeindx parse_index(parser *p) {
    token start = p->previous;
    syntaxtreeindx left=p->left;
    syntaxtreeindx right;
    unsigned int nindx;
    
    right=parse_arglist(p, TOKEN_RIGHTSQBRACKET, &nindx);
    
    parse_consume(p, TOKEN_RIGHTSQBRACKET, PARSE_CALLRGHTPARENMISSING);
    
    return parse_addnode(p, NODE_INDEX, MORPHO_NIL, &start, left, right);
}

/** Parse list  */
syntaxtreeindx parse_list(parser *p) {
    unsigned int nindx;
    token start = p->previous;
    
    syntaxtreeindx right=parse_arglist(p, TOKEN_RIGHTSQBRACKET, &nindx);
    parse_consume(p, TOKEN_RIGHTSQBRACKET, PARSE_CALLRGHTPARENMISSING);

    return parse_addnode(p, NODE_LIST, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, right);
}

/** Parses an anonymous function */
syntaxtreeindx parse_anonymousfunction(parser *p) {
    token start = p->previous;
    syntaxtreeindx args=SYNTAXTREE_UNCONNECTED,
                   body=SYNTAXTREE_UNCONNECTED;
    
    /* Parameter list */
    parse_consume(p, TOKEN_LEFTPAREN, PARSE_FNLEFTPARENMISSING);
    args=parse_arglist(p, TOKEN_RIGHTPAREN, NULL);
    parse_consume(p, TOKEN_RIGHTPAREN, PARSE_FNRGHTPARENMISSING);
    
    /* Function body */
    if (parse_matchtoken(p, TOKEN_LEFTCURLYBRACKET)) { // fn (x) { ... }
        body=parse_blockstatement(p);
    } else {
        body=parse_expression(p); // Short form: fn (x) x
        body=parse_addnode(p, NODE_RETURN, MORPHO_NIL, &start, body, SYNTAXTREE_UNCONNECTED);
    }
    
    return parse_addnode(p, NODE_FUNCTION, MORPHO_NIL, &start, args, body);
}

/** @brief: Parses a switch block
 * @details Switch blocks are key/statement pairs. Each pair is stored in a NODE_DICTIONARY list */
syntaxtreeindx parse_switch(parser *p) {
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED;
    
    while(!parse_matchtoken(p, TOKEN_RIGHTCURLYBRACKET) && !parse_checktoken(p, TOKEN_EOF)) {
        syntaxtreeindx key, statements, pair;
        token tok=p->current; // Keep track of the token that corresponds to each key/value pair
        
        /* Parse the key/value pair */
        key=parse_expression(p);
        if (!parse_consume(p, TOKEN_COLON, PARSE_SWTCHSPRTR)) break;
        tokentype terminators[] = { TOKEN_STRING, TOKEN_INTEGER, TOKEN_NUMBER, TOKEN_TRUE, TOKEN_FALSE, TOKEN_NIL, TOKEN_RIGHTCURLYBRACKET };
        statements=parse_declarationmulti(p, 7, terminators);
        
        /* Create an entry node */
        pair=parse_addnode(p, NODE_DICTENTRY, MORPHO_NIL, &tok, key, statements);
        
        /* These are linked into a chain of sequence nodes */
        last=parse_addnode(p, NODE_DICTIONARY, MORPHO_NIL, &tok, last, pair);
        
    };
    
    return last;
}

/* -------------------------------
 * Declarations
 * ------------------------------- */

/** Parses a variable declaration */
syntaxtreeindx parse_vardeclaration(parser *p) {
    syntaxtreeindx symbol, initializer, out=SYNTAXTREE_UNCONNECTED, last=SYNTAXTREE_UNCONNECTED;
    
    do {
        token start = p->previous;
        
        symbol=parse_variable(p, PARSE_VAREXPECTED);
        
        if (parse_matchtoken(p, TOKEN_LEFTSQBRACKET)) {
            symbol=parse_index(p);
        }
        
        if (parse_matchtoken(p, TOKEN_EQUAL)) {
            initializer=parse_pseudoexpression(p);
        } else initializer=SYNTAXTREE_UNCONNECTED;
        
        out=parse_addnode(p, NODE_DECLARATION, MORPHO_NIL, &start, symbol, initializer);
        
        if (last!=SYNTAXTREE_UNCONNECTED) {
            out=parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, last, out);
        }
        
        last=out;
    } while (parse_matchtoken(p, TOKEN_COMMA));
    
    parse_statementterminator(p);
    
    return out;
}

/** Parses a function declaration */
syntaxtreeindx parse_functiondeclaration(parser *p) {
    value name=MORPHO_NIL;
    token start = p->previous;
    syntaxtreeindx args=SYNTAXTREE_UNCONNECTED,
                   body=SYNTAXTREE_UNCONNECTED;
    
    /* Function name */
    if (parse_matchtoken(p, TOKEN_SYMBOL)) {
        name=parse_symbolasvalue(p);
    } else parse_error(p, false, PARSE_FNNAMEMISSING);
    
    /* Parameter list */
    parse_consume(p, TOKEN_LEFTPAREN, PARSE_FNLEFTPARENMISSING);
    args=parse_arglist(p, TOKEN_RIGHTPAREN, NULL);
    parse_consume(p, TOKEN_RIGHTPAREN, PARSE_FNRGHTPARENMISSING);
    
    /* Function body */
    parse_consume(p, TOKEN_LEFTCURLYBRACKET, PARSE_FNLEFTCURLYMISSING);
    body=parse_blockstatement(p);
    
    return parse_addnode(p, NODE_FUNCTION, name, &start, args, body);
}

/* Parses a class declaration */
syntaxtreeindx parse_classdeclaration(parser *p) {
    value name=MORPHO_NIL;
    value sname=MORPHO_NIL;
    syntaxtreeindx sclass=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    /* Class name */
    if (parse_matchtoken(p, TOKEN_SYMBOL)) {
        name=parse_symbolasvalue(p);
    } else parse_error(p, false, PARSE_EXPECTCLASSNAME);
    
    /* Extract a superclass name */
    if (parse_matchtoken(p, TOKEN_LT) || parse_matchtoken(p, TOKEN_IS)) {
        parse_consume(p, TOKEN_SYMBOL, PARSE_EXPECTSUPER);
        sname=parse_symbolasvalue(p);
        sclass=parse_addnode(p, NODE_SYMBOL, sname, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
    }
    
    if (parse_matchtoken(p, TOKEN_WITH)) {
        do {
            parse_consume(p, TOKEN_SYMBOL, PARSE_EXPECTSUPER);
            value mixin=parse_symbolasvalue(p);
            
            syntaxtreeindx smixin=parse_addnode(p, NODE_SYMBOL, mixin, &p->previous, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
                
            sclass = parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &p->previous, smixin, sclass); // Mixins end up being recorded in reverse order
            
        } while (parse_matchtoken(p, TOKEN_COMMA));
    }
    
    parse_consume(p, TOKEN_LEFTCURLYBRACKET, PARSE_CLASSLEFTCURLYMISSING);
    /* Method declarations */
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED, current=SYNTAXTREE_UNCONNECTED;
    
    while (!parse_checktoken(p, TOKEN_RIGHTCURLYBRACKET) && !parse_checktoken(p, TOKEN_EOF)) {
        current=parse_functiondeclaration(p);
        
        /* If we now have more than one node, insert a sequence node */
        if (last!=SYNTAXTREE_UNCONNECTED) {
            current = parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, last, current);
        }
        
        last = current;
    }
    
    parse_consume(p, TOKEN_RIGHTCURLYBRACKET, PARSE_CLASSRGHTCURLYMISSING);
    
    return parse_addnode(p, NODE_CLASS, name, &start, sclass, current);
}

/** Parse an import declaration.
 *          IMPORT
 *         /              \
 *     module           FOR   or as
 *                    \
 *                   ( items )
 */
syntaxtreeindx parse_importdeclaration(parser *p) {
    syntaxtreeindx modulename=SYNTAXTREE_UNCONNECTED, right=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    if (parse_matchtoken(p, TOKEN_STRING)) {
        modulename=parse_string(p);
    } else if (parse_matchtoken(p, TOKEN_SYMBOL)){
        modulename=parse_symbol(p);
    } else {
        parse_error(p, true, PARSE_IMPORTMISSINGNAME);
        return SYNTAXTREE_UNCONNECTED;
    }
    
    if (!parse_checkstatementterminator(p)) {
        if (parse_matchtoken(p, TOKEN_AS)) {
            if (parse_matchtoken(p, TOKEN_SYMBOL)) {
                right=parse_symbol(p);
            } else parse_error(p, true, PARSE_IMPORTASSYMBL);
        } else if (parse_matchtoken(p, TOKEN_FOR)) {
            do {
                if (parse_matchtoken(p, TOKEN_SYMBOL)) {
                    syntaxtreeindx symbl=parse_symbol(p);
                    right=parse_addnode(p, NODE_FOR, MORPHO_NIL, &p->previous, right, symbl);
                } else parse_error(p, true, PARSE_IMPORTFORSYMBL);
            } while (parse_matchtoken(p, TOKEN_COMMA));
        } else {
            parse_error(p, true, PARSE_IMPORTUNEXPCTDTOK);
        }
    }
    
    parse_statementterminator(p);
    
    return parse_addnode(p, NODE_IMPORT, MORPHO_NIL, &start, modulename, right);
}

/* -------------------------------
 * Statements
 * ------------------------------- */

/** Parse a print statement */
syntaxtreeindx parse_printstatement(parser *p) {
    token start = p->previous;
    syntaxtreeindx left = parse_pseudoexpression(p);
    parse_statementterminator(p);
    return parse_addnode(p, NODE_PRINT, MORPHO_NIL, &start, left, SYNTAXTREE_UNCONNECTED);
}

/** Parse an expression statement */
syntaxtreeindx parse_expressionstatement(parser *p) {
    syntaxtreeindx out = parse_expression(p);
    parse_statementterminator(p);
    return out;
}

/** @brief Parse a block statement.
 *  @details This wraps up a sequence of statements in a SCOPE node:
 *                     SCOPE
 *                    /     \
 *                   -       body
 **/
syntaxtreeindx parse_blockstatement(parser *p) {
    syntaxtreeindx body = SYNTAXTREE_UNCONNECTED,
                   scope = SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    tokentype terminator[] = { TOKEN_RIGHTCURLYBRACKET };
    
    body = parse_declarationmulti(p, 1, terminator);
    if (parse_checktoken(p, TOKEN_EOF)) {
        parse_error(p, false, PARSE_INCOMPLETEEXPRESSION);
    } else {
        parse_consume(p, TOKEN_RIGHTCURLYBRACKET, PARSE_MISSINGSEMICOLONEXP);
    }
    
    scope=parse_addnode(p, NODE_SCOPE, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, body);
    
    return scope;
}

/** Parse an if statement */
syntaxtreeindx parse_ifstatement(parser *p) {
    syntaxtreeindx  cond=SYNTAXTREE_UNCONNECTED,
                    then=SYNTAXTREE_UNCONNECTED,
                    els=SYNTAXTREE_UNCONNECTED,
                    out=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    parse_consume(p, TOKEN_LEFTPAREN, PARSE_IFLFTPARENMISSING);
    cond=parse_expression(p);
    parse_consume(p, TOKEN_RIGHTPAREN, PARSE_IFRGHTPARENMISSING);
    
    token thentok = p->current;
    then=parse_statement(p);
    
    if (parse_checktoken(p, TOKEN_ELSE)) {
        parse_advance(p);
        els=parse_statement(p);
        
        /* Create an additional node that contains both statements */
        then = parse_addnode(p, NODE_THEN, MORPHO_NIL, &thentok, then, els);
    }
    
    out=parse_addnode(p, NODE_IF, MORPHO_NIL, &start, cond, then);
    
    return out;
}

/** Parse a while statement */
syntaxtreeindx parse_whilestatement(parser *p) {
    syntaxtreeindx  cond=SYNTAXTREE_UNCONNECTED,
                    body=SYNTAXTREE_UNCONNECTED,
                    out=SYNTAXTREE_UNCONNECTED;
    token start = p->previous;
    
    parse_consume(p, TOKEN_LEFTPAREN, PARSE_WHILELFTPARENMISSING);
    cond=parse_expression(p);
    parse_consume(p, TOKEN_RIGHTPAREN, PARSE_IFRGHTPARENMISSING);
    body=parse_statement(p);
    
    out=parse_addnode(p, NODE_WHILE, MORPHO_NIL, &start, cond, body);
    
    return out;
}

/** Parse a for statement. */
syntaxtreeindx parse_forstatement(parser *p) {
    syntaxtreeindx init=SYNTAXTREE_UNCONNECTED, // Initializer
                   cond=SYNTAXTREE_UNCONNECTED, // Condition
                   body=SYNTAXTREE_UNCONNECTED, // Loop body
                   final=SYNTAXTREE_UNCONNECTED; // Final statement
    syntaxtreeindx out=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    bool forin=false;
 
    parse_consume(p, TOKEN_LEFTPAREN, PARSE_FORLFTPARENMISSING);
    if (parse_matchtoken(p, TOKEN_SEMICOLON)) {
        
    } else if (parse_matchtoken(p, TOKEN_VAR)) {
        init=parse_vardeclaration(p);
    } else {
        init=parse_expression(p);
        while (parse_matchtoken(p, TOKEN_COMMA)) {
            syntaxtreeindx new=parse_expressionstatement(p);
            init=parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &p->current, init, new);
        }
        parse_matchtoken(p, TOKEN_SEMICOLON);
    }
    
    if (parse_matchtoken(p, TOKEN_IN)) {
        /* If its an for..in loop, parse the collection */
        cond=parse_expression(p);
        forin=true;
    } else {
        /* Otherwise, parse the condition and final clause in a traditional for loop. */
        if (!parse_matchtoken(p, TOKEN_SEMICOLON)) {
            cond=parse_expressionstatement(p);
        }
        
        if (!parse_checktoken(p, TOKEN_RIGHTPAREN)) {
            final=parse_expression(p);
        }
    }
    parse_consume(p, TOKEN_RIGHTPAREN, PARSE_FORRGHTPARENMISSING);
    
    if (!parse_checkstatementterminator(p)) {
        body=parse_statement(p);
    }
    
    if (forin) {
        /* A for..in loop is parsed as follows:
         *
         *                 forin
         *                /     \
         *               in      body
         *              /  \
         *          init    collection
         */
         syntaxtreeindx innode=parse_addnode(p, NODE_IN, MORPHO_NIL, &start, init, cond);
         out=parse_addnode(p, NODE_FOR, MORPHO_NIL, &start, innode, body);
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
        syntaxtreeindx loop=parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, body, final);
        syntaxtreeindx whil=parse_addnode(p, NODE_WHILE, MORPHO_NIL, &start, cond, loop);
        syntaxtreeindx seq=parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, init, whil);
        out=parse_addnode(p,NODE_SCOPE, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, seq);
    }
    
    return out;
}

/** Parses a do...while loop */
syntaxtreeindx parse_dostatement(parser *p) {
    syntaxtreeindx body=SYNTAXTREE_UNCONNECTED, // Loop body
                   cond=SYNTAXTREE_UNCONNECTED; // Condition
    syntaxtreeindx out=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    
    body=parse_statement(p);
    
    parse_consume(p, TOKEN_WHILE, PARSE_EXPCTWHL);
    parse_consume(p, TOKEN_LEFTPAREN, PARSE_WHILELFTPARENMISSING);
    cond=parse_expression(p);
    parse_consume(p, TOKEN_RIGHTPAREN, PARSE_IFRGHTPARENMISSING);
    
    /* Optional statement terminator */
    if (parse_checkstatementterminator(p)) {
        parse_statementterminator(p);
    }
    
    out=parse_addnode(p, NODE_DO, MORPHO_NIL, &start, body, cond);
    
    return out;
}

/** Parses a break or continue statement */
syntaxtreeindx parse_breakstatement(parser *p) {
    token start = p->previous;
    
    parse_statementterminator(p);
    
    return parse_addnode(p, (start.type==TOKEN_BREAK ? NODE_BREAK: NODE_CONTINUE), MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/** Parse a return statement */
syntaxtreeindx parse_returnstatement(parser *p) {
    token start = p->previous;
    syntaxtreeindx left = SYNTAXTREE_UNCONNECTED;
    
    if (!parse_checkstatementterminator(p)) {
        left = parse_pseudoexpression(p);
    }
    
    parse_statementterminator(p);
    
    return parse_addnode(p, NODE_RETURN, MORPHO_NIL, &start, left, SYNTAXTREE_UNCONNECTED);
}

/** Parse a try/catch statement
        try
      /          \
    body        catch block */
syntaxtreeindx parse_trystatement(parser *p) {
    syntaxtreeindx try=SYNTAXTREE_UNCONNECTED, // Try block
                   catch=SYNTAXTREE_UNCONNECTED; // Catch dictionary
    syntaxtreeindx out=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    
    try=parse_statement(p);
    
    parse_consume(p, TOKEN_CATCH, PARSE_EXPCTCTCH);
    parse_consume(p, TOKEN_LEFTCURLYBRACKET, PARSE_CATCHLEFTCURLYMISSING);
    
    catch=parse_switch(p);
    
    /* Optional statement terminator */
    if (parse_checkstatementterminator(p)) {
        parse_statementterminator(p);
    }
    
    out=parse_addnode(p, NODE_TRY, MORPHO_NIL, &start, try, catch);
    
    return out;
}

/** Parse a breakpoint statement */
syntaxtreeindx parse_breakpointstatement(parser *p) {
    token start = p->previous;
    
    if (parse_checkstatementterminator(p)) {
        parse_statementterminator(p);
    }
    
    return parse_addnode(p, NODE_BREAKPOINT, MORPHO_NIL, &start, SYNTAXTREE_UNCONNECTED, SYNTAXTREE_UNCONNECTED);
}

/* -------------------------------
 * Categories of things to parse
 * ------------------------------- */

/** Parses an expression */
syntaxtreeindx parse_expression(parser *p) {
    return parse_precedence(p, PREC_ASSIGN);
}

/** Parses an expression that may include an anonymous function */
syntaxtreeindx parse_pseudoexpression(parser *p) {
    if (parse_matchtoken(p, TOKEN_FUNCTION)) {
        return parse_anonymousfunction(p);
    } else {
        return parse_expression(p);
    }
}

/** @brief Parse statements
 *  @details Statements are things that are allowed inside control flow statements */
syntaxtreeindx parse_statement(parser *p) {
    if (parse_matchtoken(p, TOKEN_PRINT)) {
        return parse_printstatement(p);
    } else if (parse_matchtoken(p, TOKEN_IF)) {
        return parse_ifstatement(p);
    } else if (parse_matchtoken(p, TOKEN_WHILE)) {
        return parse_whilestatement(p);
    } else if (parse_matchtoken(p, TOKEN_FOR)) {
        return parse_forstatement(p);
    } else if (parse_matchtoken(p, TOKEN_DO)) {
        return parse_dostatement(p);
    } else if (parse_matchtoken(p, TOKEN_BREAK)) {
        return parse_breakstatement(p);
    } else if (parse_matchtoken(p, TOKEN_CONTINUE)) {
        return parse_breakstatement(p);
    } else if (parse_matchtoken(p, TOKEN_RETURN)) {
        return parse_returnstatement(p);
    } else if (parse_matchtoken(p, TOKEN_TRY)) {
        return parse_trystatement(p);
    } else if (parse_matchtoken(p, TOKEN_LEFTCURLYBRACKET)) {
        return parse_blockstatement(p);
    } else if (parse_matchtoken(p, TOKEN_AT)) {
        return parse_breakpointstatement(p);
    } else {
        return parse_expressionstatement(p);
    }
    return SYNTAXTREE_UNCONNECTED;
}

/** @brief Parse declarations
 *  @details Declarations define something (e.g. a variable or a function) OR
 *           a regular statement. They are *not* allowed in control flow statements. */
syntaxtreeindx parse_declaration(parser *p) {
    syntaxtreeindx ret=SYNTAXTREE_UNCONNECTED;
    
    if (parse_matchtoken(p, TOKEN_FUNCTION)) {
        ret=parse_functiondeclaration(p);
    } else if (parse_matchtoken(p, TOKEN_VAR)) {
        ret=parse_vardeclaration(p);
    } else if (parse_matchtoken(p, TOKEN_CLASS)) {
        ret=parse_classdeclaration(p);
    } else if (parse_matchtoken(p, TOKEN_IMPORT)) {
        ret=parse_importdeclaration(p);
    } else {
        ret=parse_statement(p);
    }
    
    if (!ERROR_SUCCEEDED(*(p->err))) {
        parse_synchronize(p);
    }
    return ret;
}

/** Parses multiple declarations, separated by ; separators
 *  @param p    the parser
 *  @param end  token type to terminate on [use TOKEN_EOF if no special terminator]
 *  @returns    the syntaxtreeindx of the parsed expression */
syntaxtreeindx parse_declarationmulti(parser *p, int n, tokentype *end) {
    syntaxtreeindx last=SYNTAXTREE_UNCONNECTED, current=SYNTAXTREE_UNCONNECTED;
    token start = p->current;
    
    while (!parse_checktokenmulti(p, n, end) && !parse_checktoken(p, TOKEN_EOF)) {
        current=parse_declaration(p);
        
        /* If we now have more than one node, insert a sequence node */
        if (last!=SYNTAXTREE_UNCONNECTED) {
            current = parse_addnode(p, NODE_SEQUENCE, MORPHO_NIL, &start, last, current);
        }
        
        last = current;
    }
    
    return current;
}

/* -------------------------------
 * The parser definition table
 * ------------------------------- */

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
    PARSERULE_UNUSED(TOKEN_ERROR),
    PARSERULE_UNUSED(TOKEN_EOF)
};

/** Get the rule to parse an element of type tokentype. */
parserule *parse_getrule(parser *p, tokentype type) {
    return &rules[type];
}

/* -------------------------------
 * Parser implementation functions
 * ------------------------------- */

/** @brief Continues parsing while tokens have a lower or equal precendece than a specified value.
 *  @param   p    the parser in use
 *  @param   precendence precedence value to keep below or equal to
 *  @returns syntaxtreeindx for the expression parsed */
syntaxtreeindx parse_precedence(parser *p, precedence precendence) {
    parsefunction prefixrule=NULL, infixrule=NULL;
    syntaxtreeindx result;
    
    parse_advance(p);
    
    prefixrule = parse_getrule(p, p->previous.type)->prefix;
    
    if (!prefixrule) {
        parse_error(p, true, PARSE_EXPECTEXPRESSION);
        return SYNTAXTREE_UNCONNECTED;
    }
    
    result=prefixrule(p);
    
    /* Now keep parsing while the tokens have lower precedence */
    while (precendence <= parse_getrule(p, p->current.type)->precedence) {
#ifdef MORPHO_NEWLINETERMINATORS
        /* Break if a newline is encountered before a function call */
        if (p->current.type==TOKEN_LEFTPAREN && p->nl) break;
#endif
        
        parse_advance(p);
        
        infixrule = parse_getrule(p, p->previous.type)->infix;
        if (infixrule) result=infixrule(p);
        else parse_error(p, true, 0);
    }
    
    return result;
}

/* **********************************************************************
 * Interface
 * ********************************************************************** */

/** @brief Initialize a parser
 *  @param p       the parser to initialize
 *  @param lex   lexer to use
 *  @param err   an error structure to fill out if necessary
 *  @param tree Pointer to the output */
void parse_init(parser *p, lexer *lex, error *err, syntaxtree *tree) {
    p->current = TOKEN_BLANK;
    p->previous = TOKEN_BLANK;
    p->left = SYNTAXTREE_UNCONNECTED;
    p->lex=lex;
    p->err=err;
    p->tree=tree;
    p->nl=false;
}

/** @brief Clear a parser */
void parse_clear(parser *p) {
    p->current = TOKEN_BLANK;
    p->previous = TOKEN_BLANK;
    p->left = SYNTAXTREE_UNCONNECTED;
}

/** Entry point into the parser */
bool parse(parser *p) {
    parse_advance(p);
    tokentype terminator[] = { TOKEN_EOF };
    
    p->tree->entry = parse_declarationmulti(p, 1, terminator);
    
    return (p->err->cat==ERROR_NONE);
}

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
    morpho_defineerror(PARSE_SWTCHSPRTR, ERROR_PARSE, PARSE_SWTCHSPRTR_MSG);
    morpho_defineerror(PARSE_DCTENTRYSPRTR, ERROR_PARSE, PARSE_DCTENTRYSPRTR_MSG);
    morpho_defineerror(PARSE_EXPCTWHL, ERROR_PARSE, PARSE_EXPCTWHL_MSG);
    morpho_defineerror(PARSE_EXPCTCTCH, ERROR_PARSE, PARSE_EXPCTCTCH_MSG);
    morpho_defineerror(PARSE_ONEVARPR, ERROR_PARSE, PARSE_ONEVARPR_MSG);
    morpho_defineerror(PARSE_CATCHLEFTCURLYMISSING, ERROR_PARSE, PARSE_CATCHLEFTCURLYMISSING_MSG);
}

void parse_finalize(void) {
    
}
