/** @file parse.h
 *  @author T J Atherton and others (see below)
 *
 *  @brief Lexer and parser
*/

#ifndef parse_h
#define parse_h

#include <stdio.h>
#include "error.h"
#include "lex.h"
#include "syntaxtree.h"

/* **********************************************************************
* Parser error messages
* ********************************************************************** */

/* Parser */

#define COMPILE_INCOMPLETEEXPRESSION      "IncExp"
#define COMPILE_INCOMPLETEEXPRESSION_MSG  "Incomplete expression."

#define COMPILE_MISSINGPARENTHESIS        "MssngParen"
#define COMPILE_MISSINGPARENTHESIS_MSG    "Expect ')' after expression."

#define COMPILE_EXPECTEXPRESSION          "ExpExpr"
#define COMPILE_EXPECTEXPRESSION_MSG      "Expected expression."

#define COMPILE_MISSINGSEMICOLON          "MssngSemiVal"
#define COMPILE_MISSINGSEMICOLON_MSG      "Expect ; after value."

#define COMPILE_MISSINGSEMICOLONEXP       "MssngExpTerm"
#define COMPILE_MISSINGSEMICOLONEXP_MSG   "Expect expression terminator (; or newline) after expression."

#define COMPILE_MISSINGSEMICOLONVAR       "MssngSemiVar"
#define COMPILE_MISSINGSEMICOLONVAR_MSG   "Expect ; after variable declaration."

#define COMPILE_VAREXPECTED               "VarExpct"
#define COMPILE_VAREXPECTED_MSG           "Variable name expected after var."

#define COMPILE_BLOCKTERMINATOREXP        "MssngBrc"
#define COMPILE_BLOCKTERMINATOREXP_MSG    "Expected '}' to finish block."

#define COMPILE_IFLFTPARENMISSING         "IfMssngLftPrn"
#define COMPILE_IFLFTPARENMISSING_MSG     "Expected '(' after if."

#define COMPILE_IFRGHTPARENMISSING        "IfMssngRgtPrn"
#define COMPILE_IFRGHTPARENMISSING_MSG    "Expected ')' after condition."

#define COMPILE_WHILELFTPARENMISSING      "WhlMssngLftPrn"
#define COMPILE_WHILELFTPARENMISSING_MSG  "Expected '(' after while."

#define COMPILE_FORLFTPARENMISSING        "ForMssngLftPrn"
#define COMPILE_FORLFTPARENMISSING_MSG    "Expected '(' after for."

#define COMPILE_FORSEMICOLONMISSING       "ForMssngSemi"
#define COMPILE_FORSEMICOLONMISSING_MSG   "Expected ';'."

#define COMPILE_FORRGHTPARENMISSING       "ForMssngRgtPrn"
#define COMPILE_FORRGHTPARENMISSING_MSG   "Expected ')' after for clauses."

#define COMPILE_FNNAMEMISSING             "FnNoName"
#define COMPILE_FNNAMEMISSING_MSG         "Expected function or method name."

#define COMPILE_FNLEFTPARENMISSING        "FnMssngLftPrn"
#define COMPILE_FNLEFTPARENMISSING_MSG    "Expect '(' after name."

#define COMPILE_FNRGHTPARENMISSING        "FnMssngRgtPrn"
#define COMPILE_FNRGHTPARENMISSING_MSG    "Expect ')' after parameters."

#define COMPILE_FNLEFTCURLYMISSING        "FnMssngLftBrc"
#define COMPILE_FNLEFTCURLYMISSING_MSG    "Expect '{' before body."

#define COMPILE_CALLRGHTPARENMISSING      "CllMssngRgtPrn"
#define COMPILE_CALLRGHTPARENMISSING_MSG  "Expect ')' after arguments."

#define COMPILE_EXPECTCLASSNAME           "ClsNmMssng"
#define COMPILE_EXPECTCLASSNAME_MSG       "Expect class name."

#define COMPILE_CLASSLEFTCURLYMISSING     "ClsMssngLftBrc"
#define COMPILE_CLASSLEFTCURLYMISSING_MSG "Expect '{' before class body."

#define COMPILE_CLASSRGHTCURLYMISSING     "ClsMssngRgtBrc"
#define COMPILE_CLASSRGHTCURLYMISSING_MSG "Expect '}' after class body."

#define COMPILE_EXPECTDOTAFTERSUPER       "ExpctDtSpr"
#define COMPILE_EXPECTDOTAFTERSUPER_MSG   "Expect '.' after 'super'"

#define COMPILE_INCOMPLETESTRINGINT       "IntrpIncmp"
#define COMPILE_INCOMPLETESTRINGINT_MSG   "Incomplete string after interpolation."

#define COMPILE_VARBLANKINDEX             "EmptyIndx"
#define COMPILE_VARBLANKINDEX_MSG         "Empty capacity in variable declaration."

#define COMPILE_IMPORTMISSINGNAME         "ImprtMssngNm"
#define COMPILE_IMPORTMISSINGNAME_MSG     "Import expects a module or file name."

#define COMPILE_IMPORTUNEXPCTDTOK         "ImprtExpctFrAs"
#define COMPILE_IMPORTUNEXPCTDTOK_MSG     "Import expects a module or file name followed by for or as."

#define COMPILE_IMPORTASSYMBL             "ExpctSymblAftrAs"
#define COMPILE_IMPORTASSYMBL_MSG         "Expect symbol after as in import."

#define COMPILE_IMPORTFORSYMBL            "ExpctSymblAftrFr"
#define COMPILE_IMPORTFORSYMBL_MSG        "Expect symbol(s) after for in import."

#define COMPILE_EXPECTSUPER               "SprNmMssng"
#define COMPILE_EXPECTSUPER_MSG           "Expect superclass name."

#define PARSE_UNRECGNZEDTOK               "UnrcgnzdTok"
#define PARSE_UNRECGNZEDTOK_MSG           "Encountered an unrecognized token."

#define PARSE_DCTSPRTR                    "DctSprtr"
#define PARSE_DCTSPRTR_MSG                "Expected a colon separating a key/value pair in dictionary."

#define PARSE_SWTCHSPRTR                  "SwtchSprtr"
#define PARSE_SWTCHSPRTR_MSG              "Expected a colon after label."

#define PARSE_DCTENTRYSPRTR               "DctEntrySprtr"
#define PARSE_DCTENTRYSPRTR_MSG           "Expected a comma or '}'."

#define PARSE_EXPCTWHL                    "ExpctWhl"
#define PARSE_EXPCTWHL_MSG                "Expected while after loop body."

#define PARSE_EXPCTCTCH                   "ExpctCtch"
#define PARSE_EXPCTCTCH_MSG               "Expected catch after try statement."

#define PARSE_CATCHLEFTCURLYMISSING       "ExpctHndlr"
#define PARSE_CATCHLEFTCURLYMISSING_MSG   "Expected block of error handlers after catch."

#define PARSE_ONEVARPR                    "OneVarPr"
#define PARSE_ONEVARPR_MSG                "Functions can have only one variadic parameter."

/* **********************************************************************
* Parser
* ********************************************************************** */

/* -------------------------------------------------------
 * Define a Parser
 * ------------------------------------------------------- */

/** @brief A structure that defines the state of a parser */
typedef struct {
    token current; /** The current token */
    token previous; /** The previous token */
    syntaxtreeindx left;
    lexer *lex; /** Lexer to use */
    syntaxtree *tree; /** Syntax tree receiving output */
    error *err; /** Error structure to output errors to */
    bool nl; /** Was a newline encountered before the current token? */
} parser;

/* -------------------------------------------------------
 * Tokens are parsed into the AST by parserules
 * ------------------------------------------------------- */

/** @brief an enumerated type that defines precedence order in Morpho. */
typedef enum {
    PREC_NONE,
    PREC_LOWEST,
    PREC_ASSIGN,
    PREC_OR,
    PREC_AND,
    PREC_EQUALITY,
    PREC_COMPARISON,
    PREC_RANGE,
    PREC_TERM,
    PREC_FACTOR,
    PREC_UNARY,
    PREC_POW,
    PREC_CALL,
    PREC_HIGHEST
} precedence;

/** @brief Definition of a parse function. */
typedef syntaxtreeindx (*parsefunction) (parser *c);

/** @brief A parse rule will be defined for each token,
 * providing functions to parse the token if it is encountered in the
 * prefix or infix positions. The parse rule also defines the precedence. */
typedef struct {
    parsefunction prefix;
    parsefunction infix;
    precedence precedence;
} parserule;

/* -------------------------------------------------------
 * Prototypes for using a parser
 * ------------------------------------------------------- */

void parse_init(parser *p, lexer *lex, error *err, syntaxtree *tree);
bool parse(parser *p);

bool parse_stringtovaluearray(char *string, unsigned int nmax, value *v, unsigned int *n, error *err);

#endif /* parse_h */
