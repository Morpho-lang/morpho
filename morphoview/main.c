/** @file main.c
 *  @author T J Atherton
 *
 *  @brief Main entry point for morpho viewer application
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "command.h"
#include "display.h"

int main(int argc, const char * argv[]) {
    scene_initialize();
    display_initialize();
    bool temp = false;
    
    // Process arguments
    const char *file=NULL;
    for (unsigned int i=1; i<argc; i++) {
        const char *option = argv[i];
        if (argv[i] && option[0]=='-') {
            switch (option[1]) {
                case 't': /* Temporary file; delete after */
                    temp=true;
                    break;
            }
        } else {
            file = option;
        }
    }
    
    // Parse a command file if provided
    if (file) {
        char *buffer = NULL;
        //printf("Loading %s\n", file);
        
        if (command_loadinput(file, &buffer)) {
            command_parse(buffer);
        }
        
        free(buffer);
    }
    
    display_loop();
    
    display_finalize();
    scene_finalize();
    
    if (temp && file) command_removefile(file);
}
