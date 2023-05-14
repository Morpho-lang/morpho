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
    clioptions opt = CLI_RUN;
    const char *file = NULL;
    int i=0;

    /* Process command line arguments */
    for (i=1; i<argc; i++) {
        const char *option = argv[i];
        if (argv[i] && option[0]=='-') {
            switch (option[1]) {
                case 'D': /* Disassemble only */
                    opt^=CLI_RUN;
                    /* v note fallthrough */
                case 'd':
                    if (strncmp(option+1, "debug", strlen("debug"))==0) {
                        opt|=CLI_DEBUG;
                    } else { /* Disassemble */
                        opt |= CLI_DISASSEMBLE;
                        if (option[2]=='l' || option[2]=='L') {
                            /* Show lines of source alongside disassembly */
                            opt |= CLI_DISASSEMBLESHOWSRC;
                        }
                    }
                    break;
                case 'O': /* Optimize */
                    opt|=CLI_OPTIMIZE;
                    break;
                case 'p':
#ifdef MORPHO_PROFILER
                    if (strncmp(option+1, "profile", strlen("profile"))==0) {
                        opt |= CLI_PROFILE;
                    }
#endif
                    break;
                case 'w': /* Workers */
                    {
                        const char *c=option+2;
                        int nw=0;
                        while (!isdigit(*c) && *c!='\0') c++;
                        if (isdigit(*c)) nw=atoi(c);
                        if (nw<0) nw=0;
                        morpho_setthreadnumber(nw);
                    }

                    break;
            }
        } else {
            file = option;
            break;
        }
    }

    morpho_initialize();
    
    if (i<argc) morpho_setargs(argc-i-1, argv+i+1); // Pass unprocessed args to the morpho runtime.

    if (file) cli_run(file, opt);
    else cli(opt);

    morpho_finalize();
    return 0;
}
