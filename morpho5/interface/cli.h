/** @file cli.h
 *  @author T J Atherton
 *
 *  @brief Command line interface
*/

#ifndef cli_h
#define cli_h

#include "morpho.h"
#include "varray.h"
#include "error.h"
#include "linedit.h"
#include "help.h"
#include "common.h"

#define CLI_PROMPT ">"
#define CLI_CONTINUATIONPROMPT "~"
#define CLI_QUIT "quit"
#define CLI_HELP "help"
#define CLI_SHORT_HELP "?"

#define CLI_NORMALCODE   "\033[0m"
#define CLI_REDCODE     "\033[0;31m"
#define CLI_BLUECODE     "\033[0;34m"

#ifdef MORPHO_COLORTERMINAL
#define CLI_ERRORCOLOR CLI_REDCODE
#define CLI_NORMALTEXT CLI_NORMALCODE
#else
#define CLI_ERRORCOLOR ""
#define CLI_NORMALTEXT ""
#endif

#define CLI_RUN                 0x1
#define CLI_DISASSEMBLE         0x2
#define CLI_DISASSEMBLESHOWSRC  0x4
#define CLI_DEBUG               0x8
#define CLI_OPTIMIZE            0x10

typedef unsigned int clioptions;

void cli_run(const char *in, clioptions opt);
void cli(clioptions opt);

char *cli_loadsource(const char *in);
void cli_disassemblewithsrc(program *p, char *src);
void cli_list(const char *in, int start, int end);

#endif /* cli_h */
