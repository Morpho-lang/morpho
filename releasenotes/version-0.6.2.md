# Release notes for 0.6.2

We're pleased to announce Morpho 0.6.2, which is primarily a maintenance release and incorporates a number of bugfixes and improvements.

## Morphopm package manager

Alongside this release, we are pleased to announce a new package manager for morpho called `morphopm`, which makes installation of morpho packages significantly easier for users. Morphopm is available [on github](https://github.com/Morpho-lang/morpho-morphopm) and can also be installed via homebrew:

    brew tap morpho-lang/morpho
    brew install morpho-morphopm

## Benchmarks

The benchmarks folder, which used to contain a number of basic benchmarks for morpho, has been moved to a [new repository](https://github.com/Morpho-lang/morpho-benchmark) with several new benchmarks added. We will be using these to continue to improve morpho's performance.

## Ternary operator

Morpho now provides the ternary operator similar to other C-family languages:

    var a = (b < c ? b : c)

## Minor fixes

* Keywords can now be used as method and property labels.
* The povray module now produces silent output on linux if the quiet option is set.
* apply() now works properly with metafunctions.
* Bugfixes in the Sparse class.
* Improvements to resource locator.
