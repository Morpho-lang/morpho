/** @file main.c
 *  @author T J Atherton
 *
 *  @brief Main entry point 
 */

#include <stdio.h>
#include <stdarg.h>
#include "cli.h"

#include "value.h"
#include "object.h"
#include "random.h"
#include "sparse.h"

int main(int argc, const char * argv[]) {
    morpho_initialize();
    
    clioptions opt = CLI_RUN | CLI_OPTIMIZE;
    const char *file = NULL;
    
    /* Process command line arguments */
    for (unsigned int i=1; i<argc; i++) {
        const char *option = argv[i];
        if (argv[i] && option[0]=='-') {
            switch (option[1]) {
                case 'D': /* Disassemble only */
                    opt^=CLI_RUN;
                    /* v note fallthrough */
                case 'd':
                    if (strncmp(option+1, "debug", strlen("debug"))==0) {
                        opt^=CLI_DEBUG;
                    } else { /* Disassemble */
                        opt |= CLI_DISASSEMBLE;
                        if (option[2]=='l' || option[2]=='L') {
                            /* Show lines of source alongside disassembly */
                            opt |= CLI_DISASSEMBLESHOWSRC;
                        }
                    }
                    break;
                case 'n':
                    if (strncmp(option+1, "nooptimize", strlen("nooptimize"))==0) {
                        opt^=CLI_OPTIMIZE;
                    }
                    break;
            }
        } else {
            file = option;
        }
    }
    
    if (file) cli_run(file, opt);
    else cli(opt);
    
    morpho_finalize();
    return 0;
}
