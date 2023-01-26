# Release notes for 0.5.5

We're pleased to announce morpho 0.5.5, which contains a number of important improvements, and notably some significant performance improvements.

## Documentation

We have added two new chapters to the manual, one on working with Meshes and the other describing the examples in detail. Additional chapters to follow. We've also improved the formatting of the manual, and a number of previously undocumented features are now documented in the manual and in the interactive help.

## Developer tools

Morpho now provides a profiler and a rewritten debugger. To use these, run with -profile or -debug respectively. Upon completion, the profiler will produce a report of the fraction of program execution time spent in each morpho function, allowing the programmer to identify optimization targets. We've used this tool to significantly improve a number of morpho components.

## Mixins

You can now create a class from multiple other classes (called a mixin) using the new `with` keyword: 

    class Foo is Boo with Hoo, Moo { ... }

Boo is the superclass of Foo, but methods defined in Hoo and Moo are copied into Foo before Foo's methods are defined. This enables greater modularity and facilitates code reuse.

## New linear algebra features

Its now possible to convert Sparse matrices to dense matrices and vice-versa by passing them to the relevant constructor function, e.g.

    var a = Sparse([[0,0,1],[1,1,1]])
    var b = Matrix(a)

You can assemble matrices in block form using other matrices:

    var c = Matrix([[a,0],[0,a]])

You can compute the eigenvalues and eigenvectors of a matrix with the new eigenvalues() and eigensystem() methods.

Preliminary work for numerical hessians is in place.

## Other improvements

* Interactive mode now supports UTF8 characters.
* Object now provides respondsto() and has() to determine the available methods and properties respectively. 
* Optimizing compiler [off by default; run with -O flag] lays the groundwork for significant future performance improvements.
* MeshGen and Delauney modules run significantly faster.
* Hydrogel functional is faster and dimensionally independent.
* Numerous minor bugfixes.
