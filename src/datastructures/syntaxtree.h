/** @file syntaxtree.h
 *  @author T J Atherton
 *
 *  @brief Syntax tree data structure for morpho
*/

#ifndef syntaxtree_h
#define syntaxtree_h

#include <stddef.h>
#include "value.h"
#include "varray.h"

typedef ptrdiff_t syntaxtreeindx;

/** @brief Type of node */
typedef enum {
    NODE_BASE,
    
    NODE_NIL,
    NODE_BOOL,
    NODE_FLOAT,
    NODE_INTEGER,
    NODE_STRING,
    NODE_SYMBOL,
    NODE_SELF,
    NODE_SUPER, 
    NODE_IMAG,
    
    NODE_LEAF, /* ^ All leafs should be above this enum value */
    
    NODE_NEGATE,
    NODE_NOT,
    
    NODE_UNARY, /* ^ All unary operators should be above this enum value */
    
    NODE_ADD,
    NODE_SUBTRACT,
    NODE_MULTIPLY,
    NODE_DIVIDE,
    NODE_POW,
    
    NODE_ASSIGN,
    
    NODE_EQ,
    NODE_NEQ,
    NODE_LT,
    NODE_GT,
    NODE_LTEQ,
    NODE_GTEQ,
    
    NODE_AND,
    NODE_OR,

    NODE_DOT,
    
    NODE_RANGE,
    
    NODE_OPERATOR, /* ^ All operators should be above this enum value */
    
    NODE_PRINT,
    NODE_DECLARATION,
    NODE_FUNCTION,
    NODE_METHOD,
    NODE_CLASS,
    NODE_RETURN,
    NODE_IF,
    NODE_THEN,
    NODE_WHILE,
    NODE_FOR,
    NODE_DO,
    NODE_IN,
    NODE_BREAK,
    NODE_CONTINUE,
    NODE_TRY,
    
    NODE_STATEMENT, /* ^ All statements should be above this enum value */
    
    NODE_GROUPING,
    NODE_SEQUENCE,
    NODE_DICTIONARY,
    NODE_DICTENTRY,
    NODE_INTERPOLATION,
    NODE_ARGLIST,
    NODE_SCOPE,
    NODE_CALL,
    NODE_INDEX,
    NODE_LIST,
    NODE_IMPORT,
    NODE_BREAKPOINT

} syntaxtreenodetype;

static inline bool syntaxtree_istype(syntaxtreenodetype type, syntaxtreenodetype lower, syntaxtreenodetype upper) {
    return ((type > lower) && (type < upper));
}

#define SYNTAXTREE_ISLEAF(x) syntaxtree_istype(x, NODE_BASE, NODE_LEAF)
#define SYNTAXTREE_ISUNARY(x) syntaxtree_istype(x, NODE_LEAF, NODE_UNARY)
#define SYNTAXTREE_ISOPERATOR(x) syntaxtree_istype(x, NODE_UNARY, NODE_OPERATOR)
#define SYNTAXTREE_ISSTATEMENT(x) syntaxtree_istype(x, NODE_OPERATOR, NODE_STATEMENT)

/** @brief A node on the syntax tree is defined by a value and indices of the left and right elements */
typedef struct _syntaxtreenode {
    syntaxtreenodetype type;
    value content;
    int line;
    int posn;
    syntaxtreeindx left;
    syntaxtreeindx right;
} syntaxtreenode;

#define SYNTAXTREE_UNCONNECTED -1

DECLARE_VARRAY(syntaxtreenode, syntaxtreenode);

DECLARE_VARRAY(syntaxtreeindx, syntaxtreeindx);

/* @brief A syntax tree data structure */
typedef struct {
    varray_syntaxtreenode tree;
    syntaxtreeindx entry;
} syntaxtree;

void syntaxtree_init(syntaxtree *tree);
void syntaxtree_clear(syntaxtree *tree);
void syntaxtree_print(syntaxtree *tree);

bool syntaxtree_addnode(syntaxtree *tree, syntaxtreenodetype type, value content, int line, int posn, syntaxtreeindx left, syntaxtreeindx right, syntaxtreeindx *out);

syntaxtreenode *syntaxtree_nodefromindx(syntaxtree *tree, syntaxtreeindx indx);

void syntaxtree_flatten(syntaxtree *tree, syntaxtreeindx indx, unsigned int ntypes, syntaxtreenodetype *types, varray_syntaxtreeindx *list);

#endif /* syntaxtree_h */
