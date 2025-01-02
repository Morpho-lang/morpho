# Release notes for 0.6.2

We're pleased to announce Morpho 0.6.2, which is primarily a maintenance release and incorporates a number of minor bugfixes.

## Ternary operator

Morpho now provides the ternary operator, like many other C-family languages:

    var a = (b < c ? b : c)

## Minor fixes

* Keywords can now be used as method and property labels.
* The povray module now produces silent output on linux if the quiet option is set.
* apply() now works properly with metafunctions.
* Bugfixes in the Sparse class.