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
#define CLI_REDCODE      "\033[0;31m"
#define CLI_GREENCODE    "\033[0;32m"
#define CLI_YELLOWCODE   "\033[0;33m"
#define CLI_BLUECODE     "\033[0;34m"
#define CLI_PURPLECODE   "\033[0;35m"
#define CLI_CYANCODE     "\033[0;36m"
#define CLI_WHITECODE    "\033[0;37m"

#ifdef MORPHO_COLORTERMINAL
#define CLI_ERRORCOLOR CLI_REDCODE
#define CLI_NORMALTEXT CLI_NORMALCODE
#else
#define CLI_ERRORCOLOR ""
#define CLI_NORMALTEXT ""
#endif

#define CLI_RUN                 (1<<0)
#define CLI_DISASSEMBLE         (1<<1)
#define CLI_DISASSEMBLESHOWSRC  (1<<2)
#define CLI_DEBUG               (1<<3)
#define CLI_OPTIMIZE            (1<<4)
#define CLI_PROFILE             (1<<5)

typedef unsigned int clioptions;

void cli_run(const char *in, clioptions opt);
void cli(clioptions opt);

char *cli_loadsource(const char *in);
void cli_disassemblewithsrc(program *p, char *src);
void cli_list(const char *in, int start, int end);

#endif /* cli_h */
