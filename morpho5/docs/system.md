[comment]: # (System help)
[version]: # (0.5)

# System
[tagsystem]: # (system)

The `System` class provides information and access to some features of the runtime environment. 

[showsubtopics]: # (subtopics)

## Platform
[tagplatform]: # (platform)

Detect which platform morpho was compiled for: 

    print System.platform() 

which returns `"macos"`, `"linux"`, `"unix"` or `"windows"`. 

## Version
[tagversion]: # (version)

Find the current version of morpho: 

    print System.version() 

## Clock
[tagclock]: # (clock)

Returns the system time in seconds, with at least millisecond granularity. Primarily intended for timing:

    var start=System.clock() 
    // Do something 
    print System.clock()-start 

Note that `System.clock` measures the actual physical time elapsed, not the time spent in a process. 

## Sleep
[tagsleep]: # (sleep)

Pauses the program for a specified number of seconds:

    System.sleep(0.5) // Sleep for half a second

## Readline
[tagreadline]: # (readline)

Reads a line of input from the console:

    var in = System.readline() 

## Arguments
[tagargunents]: # (arguments)

Returns a `List` of arguments passed to the current morpho on the command line. 

    var args = System.arguments() 
    for (e in args) print e 

Run a morpho program with arguments: 

    morpho5 program.morpho hello world

Note that, in line with UNIX conventions, command line arguments before the program file name are passed to the `morpho5` runtime; those after are passed to the morpho program via `System.arguments`. 

## Exit
[tagexit]: # (exit)

Stop execution of a program:

    System.exit() 
