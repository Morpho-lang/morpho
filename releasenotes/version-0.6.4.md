# Release notes for 0.6.4

We're pleased to announce Morpho 0.6.4, which contains many significant improvements and is the first release available natively on Windows. 

## New linear algebra implementation

The linear algebra package in morpho has been rewritten to support complex matrices as well as a more extensible implementation that should support other matrix types in future. These still use the BLAS and LAPACK numerical libraries, but more underlying capability is exposed. Both the `Matrix` and `ComplexMatrix` now support additional matrix decompositions, such as QR decomposition and SVD. The new implementation is highly compatible with existing code; in a few cases alternative error messages are generated and results are returned in Tuples rather than Lists. 

## Additional finite elements

In addition to the existing CG1 and CG2 implementations, morpho now supports a third order Lagrangian element CG3; this is valuable for problems involving hessians, for example. Support in the codebase for higher order elements has been improved, e.g. `MeshRefiner` is now able to refine a `Field` using these elements correctly. The `FiniteElementSpace` class now exposes additional information about a particular finite element to the user.

Interior penalty schemes are now available through a `Jump` functional, which integrates over the join between two connected elements. See the `jump` documentation for more details. (This feature is experimental). 

## Additional functionality available within integrand functions

Functions passed to `LineIntegral`, `AreaIntegral` and `VolumeIntegral` may call some special functions to access additional information: 

    * `elementid` returns the id of the current element
    * `hessian` returns the local hessian of a field 
    * `jacobian` returns the jacobian of the map from the reference element to the current element. 
    * `invjacobian` returns the inverse jacobian. 

## Improved type checking within the compiler 

The morpho compiler now allows variables to be declared with a distinct type, e.g. 

    String s = "Hello" 

and enforces these through a combination of static and runtime checks. Assignment of an incorrect type to a typed variable yields a type error. The compiler is able to infer many types automatically across arithmetic operations, function calls etc. A few new classes have been added, e.g. `Callable`, to assist in writing functions using multiple dispatch.

## Improvements to help system

The core help system has been moved from the terminal app to the morpho library, facilitating access by other interfaces. An improved markdown parser enables richer help entries. Enhanced search capabilities: the help system now provides a hint if there are multiple matches, and possible matches if there are none (e.g. if the user misspells something). Help text can now be retrieved programmatically using `System.help`.

## Folder improvements

New class methods `Folder.create` and `Folder.createRecursive` allow creation of new folders. The class method `Folder.normalizePath` converts a file path with arbitrary folder delimiters to the correct ones for the current platform. 

## Minor fixes

* Bug in `System.setworkingfolder` fixed.
* Rewritten multiple dispatch implementation fixes edge-cases in method resolution.
* Improved test suite coverage. 
* graphics package updated to support Windows fonts.
* Improvements to `Tuple` class to better match `List`'s capabilities.
* Fixed a problem with building morpho on ARM64 linux. 