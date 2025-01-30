/** @file system.h
 *  @author T J Atherton
 *
 *  @brief Defines System class to provide access to the runtime and system
 */

#ifndef system_h
#define system_h

#include <stdio.h>
#include "morpho.h"

/* -------------------------------------------------------
 * System class
 * ------------------------------------------------------- */

#define SYSTEM_CLASSNAME              "System"

#define SYSTEM_PLATFORM_METHOD        "platform"
#define SYSTEM_VERSION_METHOD         "version"
#define SYSTEM_CLOCK_METHOD           "clock"
#define SYSTEM_READLINE_METHOD        "readline"
#define SYSTEM_SLEEP_METHOD           "sleep"
#define SYSTEM_ARGUMENTS_METHOD       "arguments"
#define SYSTEM_EXIT_METHOD            "exit"

#define SYSTEM_HOMEFOLDER_METHOD      "homefolder"
#define SYSTEM_WORKINGFOLDER_METHOD   "workingfolder"
#define SYSTEM_SETWORKINGFOLDER_METHOD "setworkingfolder"

/* -------------------------------------------------------
 * System error messages
 * ------------------------------------------------------- */

#define SLEEP_ARGS                    "SystmSlpArgs"
#define SLEEP_ARGS_MSG                "Sleep method expects a time in seconds."

#define STWRKDR_ARGS                  "SystmStWrkDrArgs"
#define STWRKDR_ARGS_MSG              "Setworkingdirectory method expects a path name."

#define SYS_STWRKDR                   "SystmStWrkDr"
#define SYS_STWRKDR_MSG               "Couldn't set working directory."

void system_initialize(void);
void system_finalize(void);

#endif /* system_h */
