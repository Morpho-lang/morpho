# Release notes for 0.6.0

We're pleased to announce Morpho 0.6.0, which represents a great deal of behind-the-scenes work to ready the Morpho codebase for future developments. See our Roadmap document for more details. 

## Morpho now built as a shared library

Rather than the previous monolithic strucutre, the Morpho codebase has been divided into a shared library ("libmorpho") and a terminal application ("morpho-cli"). This means that Morpho can easily be embedded in other applications, and improves maintainability as these components can be updated separately. Morphoview has been migrated to a separate repository.

## Internal improvements

* Major code reorganization to improve the logical structure and maintainability of the morpho codebase. 
* Transitioned to Cmake build system to improve cross-platform compilation.
* Rewritten parser to improve error reporting and enable reuse across Morpho. 

## Improved quadrature

Functionals like `LineIntegral`, `AreaIntegral` and `VolumeIntegral` can now make use of a greatly improved quadrature routine. This will become the default in future versions of Morpho. Particularly in 3D, the new routine offers significantly improved performance, and can be extended in future. To use the new quadrature routine simply set the `method` optional argument: 

    var a = AreaIntegral(integrandfn, method = {})

The method Dictionary can specifically request particular quadrature rules or orders; more information will be in the dev guide. 

## Namespaces

You can now use the `import` keyword with a new keyword, `as`, to import the contents of a module into a given namespace: 

    import color as col

    print col.Red 

This helps Morpho programs avoid library conflicts and improves modularization.

## Tuple data type

Morpho now supports Tuples, an ordered immutable collection. The syntax is similar to Python: 

    var t = (0,1,2,3)

Tuples act much like Lists, but can be used as keys in a Dictionary. 

## Minor new features

* Formatted output for numbers is now available using the `format` method on the `Int` and `Float` classes. 
* JSON parser to enable data interchange with other applications.
* Errors can now be raised as "warnings", which are alerts to the user that do not interrupt execution.

## Improved documentation

Many previously un- or under-documented features have now been added to the interactive help. If you notice something that isn't well documented, please alert us via the issue tracker in Github. 

## Minor fixes

* Many improvements to the debugger, including better support for printing object properties. 
* Improved calculation of derivatives. 
* Bugfixes to closures, string interpolation, parallel force and energy calculations and many others.