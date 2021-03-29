/** @file syntaxtree.c
 *  @author T J Atherton
 *
 *  @brief Syntax tree data structure for morpho
*/

#include <stdio.h>
#include "syntaxtree.h"
#include "common.h"

DEFINE_VARRAY(syntaxtreenode, syntaxtreenode);
DEFINE_VARRAY(syntaxtreeindx, syntaxtreeindx);

/** @brief Initialize a syntax tree */
void syntaxtree_init(syntaxtree *tree) {
    varray_syntaxtreenodeinit(&tree->tree);
    tree->entry=0;
}

/** @brief Finalize a syntax tree */
void syntaxtree_clear(syntaxtree *tree) {
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("--Freeing syntax tree %p.\n",(void *) tree);
#endif
    /** Free attached objects */
    for (unsigned int i=0; i<tree->tree.count; i++) {
        syntaxtreenode *node = &tree->tree.data[i];
        if (MORPHO_ISOBJECT(node->content)) {
            object_free(MORPHO_GETOBJECT(node->content));
        }
    }
    varray_syntaxtreenodeclear(&tree->tree);
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    printf("------\n");
#endif
}

#ifdef MORPHO_DEBUG
static char * nodedisplay[] = {
    "",        // NODE_BASE
    
    "",        // NODE_NIL
    "",        // NODE_BOOL
    "",        // NODE_FLOAT,
    "",        // NODE_INTEGER,
    "",        // NODE_STRING,
    "",        // NODE_SYMBOL,
    "self",    // NODE_SELF,
    "super",   // NODE_SUPER,
    
    "",        // NODE_LEAF, /* ^ All leafs should be above this enum value */
    
    "-",       // NODE_NEGATE,
    "!",       // NODE_NOT,
    
    "",        // NODE_UNARY, /* ^ All unary operators should be above this enum value */
    
    "+",       // NODE_ADD,
    "-",       // NODE_SUBTRACT,
    "*",       // NODE_MULTIPLY,
    "/",       // NODE_DIVIDE,
    "^",       // NODE_POW,
    
    "=",       // NODE_ASSIGN,
    
    "==",      // NODE_EQ,
    "!=",      // NODE_NEQ,
    "<",       // NODE_LT,
    ">",       // NODE_GT,
    "<=",      // NODE_LTEQ,
    ">=",      // NODE_GTEQ,
    
    "and",     // NODE_AND
    "or",      // NODE_OR
    
    ".",       // NODE_DOT
    
    "..",      // NODE_RANGE
    
    "",        // NODE_OPERATOR, /* ^ All operators should be above this enum value */
    
    "print",   // NODE_PRINT
    ":=",      // NODE_DECLARATION
    "fn",      // NODE_FUNCTION
    "",        // NODE_METHOD
    "class",   // NODE_CLASS
    "return",  // NODE_RETURN
    "if",      // NODE_IF
    "else",    // NODE_THEN
    "while",   // NODE_WHILE
    "for",     // NODE_FOR
    "do",      // NODE_DO
    "in",      // NODE_IN
    "break",   // NODE_BREAK
    "continue",// NODE_CONTINUE
    
    "",        // NODE_STATEMENT
    
    "()",      // NODE_GROUPING
    ";",       // NODE_SEQUENCE
    "dict",    // NODE_DICTIONARY
    "keyval",  // NODE_DICTENTRY
    "\"",      // NODE_INTERPOLATION
    "arglist", // NODE_ARGLIST
    "{}",      // NODE_SCOPE
    "call",    // NODE_CALL
    "index",   // NODE_INDEX
    "list",    // NODE_LIST
    "import",  // NODE_IMPORT
    "@",       // NODE_BREAKPOINT
    
    "",        // NODE_STRUCTURAL
};

/** @brief Prints a single node of a syntax tree with indentation
 * @param base     pointer to head node
 * @param i        index of this node
 * @param indent   indentation level */
void syntaxtree_printnode(syntaxtreenode *base, syntaxtreeindx i, unsigned int indent) {
    syntaxtreenode *node = base + i;
    for (unsigned int i=0; i<indent; i++) printf("  ");
    if (!SYNTAXTREE_ISLEAF(node->type)) {
        char *display = nodedisplay[node->type];
        
        if (display) printf("%s",display);
        else UNREACHABLE("print syntax tree node type [Operator print rule not implemented]");
    } 
    
    if (SYNTAXTREE_ISLEAF(node->type)) {
        if (node->type==NODE_SELF || node->type==NODE_SUPER) {
            char *display = nodedisplay[node->type];
            if (display) printf("%s\n",display);
        } else {
            morpho_printvalue(node->content);
            printf("\n");
        }
    } else {
        if (node->type==NODE_FUNCTION || node->type==NODE_CLASS) {
            printf(" ");
            morpho_printvalue(node->content);
        } else if (node->type==NODE_INTERPOLATION) {
            printf("\" '");
            morpho_printvalue(node->content);
            printf("'");
        }
        printf("\n");
        if (node->left!=SYNTAXTREE_UNCONNECTED) syntaxtree_printnode(base, node->left,indent+1);
        if (!SYNTAXTREE_ISUNARY(node->type)) {
            if (node->right!=SYNTAXTREE_UNCONNECTED) {
                syntaxtree_printnode(base, node->right,indent+1);
            } else {
                for (unsigned int i=0; i<indent+1; i++) printf("  ");
                printf("-\n");
            }
        }
    }
}

/** @brief Prints a syntax tree */
void syntaxtree_print(syntaxtree *tree) {
    if (tree->tree.count && tree->entry!=SYNTAXTREE_UNCONNECTED) syntaxtree_printnode(tree->tree.data, tree->entry, 0);
}
#endif

/** @brief Adds a node to the syntax tree
 *  @param tree    tree to add to.
 *  @param type    type of node to add
 *  @param content a value to add
 *  @param left    } left ...
 *  @param right   } ...and right branches of the node.
 */
syntaxtreeindx syntaxtree_addnode(syntaxtree *tree, syntaxtreenodetype type, value content, int line, int posn, syntaxtreeindx left, syntaxtreeindx right) {
    syntaxtreenode new = {.content=content, .left = left, .right = right, .type = type, .line=line, .posn=posn};
    
    varray_syntaxtreenodeadd(&tree->tree, &new, 1);
    
    return (syntaxtreeindx) tree->tree.count-1;
}

/** Gets a syntaxtree node from its index */
syntaxtreenode *syntaxtree_nodefromindx(syntaxtree *tree, syntaxtreeindx indx) {
    return tree->tree.data+indx;
}

/* @brief Flattens a tree into a list of node pointers
 * @param[in] tree - the syntaxtree to traverse
 * @param[in] node - the starting node
 * @param[in] ntypes - number of node types to match (these will be flattened)
 * @param[in] types - list of node types to flatten
 * @param[in/out] list - list of nodes flattened from this node */
void syntaxtree_flatten(syntaxtree *tree, syntaxtreeindx indx, unsigned int ntypes, syntaxtreenodetype *types, varray_syntaxtreeindx *list) {
    if (indx==SYNTAXTREE_UNCONNECTED) return; 
    syntaxtreenode *node = syntaxtree_nodefromindx(tree, indx);
    if (!node) return;
    
    bool isdesiredtype=false;
    for (unsigned int i=0; i<ntypes; i++) {
        if (node->type==types[i]) isdesiredtype=true;
    }
    
    if (isdesiredtype) {
        if (node->left!=SYNTAXTREE_UNCONNECTED) syntaxtree_flatten(tree, node->left, ntypes, types, list);
        if (node->right!=SYNTAXTREE_UNCONNECTED) syntaxtree_flatten(tree, node->right, ntypes, types, list);
    } else {
        varray_syntaxtreeindxadd(list, &indx, 1);
    }
}
