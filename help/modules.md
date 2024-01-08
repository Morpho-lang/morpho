[comment]: # (Morpho modules help file)
[version]: # (0.5)

[toplevel]: #

# Modules
[tagmodules]: # (modules)

Morpho is extensible and provides a convenient module system that works like standard libraries in other languages. Modules may define useful variables, functions and classes, and can be made available using the `import` keyword. For example,

    import color

loads the `color` module that provides functionality related to color.

You can create your own modules; they're just regular morpho files that are stored in a standard place. On UNIX platforms, this is `/usr/local/share/morpho/modules`.

## Import
[tagimport]: # (import)
[tagas]: # (as)

Import provides access to the module system and including code from multiple source files.

To import code from another file, use import with the filename:

    import "file.morpho"

which immediately includes all the contents of `"file.morpho"`. Any classes, functions or variables defined in that file can now be used, which allows you to divide your program into multiple source files.

Morpho provides a number of built in modules--and you can write your own--which can be loaded like this:

    import color

which imports the `color` module.

You can selectively import symbols from a modules by using the `for` keyword:

    import color for HueMap, Red

which imports only the `HueMap` class and the `Red` variable.
