[comment]: # (Morpho modules help file)
[version]: # (0.5)

[toplevel]: #

# Modules
[tagmodules]: # (modules)

Morpho is extensible and provides a convenient module system that works like standard libraries in other languages. Modules may define useful variables, functions and classes, and can be made available using the `import` keyword. For example,

    import color

loads the `color` module that provides functionality related to color.

You can create your own modules; they're just regular morpho files that are stored in a standard place. On UNIX platforms, this is `/usr/local/share/morpho/modules`.

[showsubtopics]: # (subtopics)

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

You can also import a module using the 'as' keyword to place the symbols in a specified namespace:

    import color as col 

You can then use refer to specific symbols like this: 

    print col.Red 

(See the help topic 'namespaces' for more information.)

## Namespaces
[tagnamespace]: # (namespace)
[tagnamespaces]: # (namespaces)

A namespace is a collection of symbols that is imported from a module. 
You identify a namespace using the 'as' keyword when importing the module like this: 

    import color as col // 'col' is the namespace

Everything defined by the module with a unique symbol, including classes, functions and global variables, can be identified using the namespace, e.g. 

    print col.Red 

Since the symbols are only are defined in the namespace you imported them into, you can't refer to them directly: 

    print Red 

Using namespaces is recommended, becuase it helps prevent conflicts between modules.
