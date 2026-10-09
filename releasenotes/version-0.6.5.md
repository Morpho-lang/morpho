# Release notes for 0.6.5

We're pleased to announce Morpho 0.6.5, which contains many significant improvements and is provided with a complete Windows installer.

## Rewritten graphics system

The `graphics` module has been extensively rewritten. New primitives available include `PointCloud`, `LineSet`. You can add primitives to a `Graphics` object with an optional scale/rotate/translate transformation, enabling reuse of primitives. The `display` method now returns a stable `id` for later use. 

    var g = Graphics()
    var id = g.display(Sphere([0, 0, 0], 0.2, color=Red))

A new `Scene` class is analogous to `Graphics`, but the contents can be changed after they are displayed, supporting animation. Use the id returned by `display` to move, recolor, remove, or replace an object. 

The `morphoview` application has been extensively rewritten, supporting interactive viewing, transparency and animations. `Morphoview` is now provided as a `morphoview` package that is installable with `morphopm`. 

## New plot interface

The `plot` module makes extensive use of the improved graphics system to provide axes, visually improved plots etc. A new `Plot` class is intended to simplify plotting, the old interface `plotmesh`, `plotselection`, and `plotfield`, remains available to support existing programs. To make a plot:

    import morphoview
    Show(Plot(mesh))

Morpho automatically detects what is to be plotted and works appropriately.

## Geometry performance improvements

Morpho's geometry stack has been carefully tuned to provide substantially improved performance through better algorithms, improvements to the quadrature routine and improvements to the multithreading model.

The `integrand` method now returns a `Field`, so the value on each element is available directly. Integrals accept a `tol` option and can choose how the error is measured through `errornorm`.

## Improved Fields

Fields can now store complex numbers and complex matrices. Create one with `ComplexField` or `ComplexMatrixField`, or pass a complex value to `Field`. Arithmetic works as it does for real fields:

    var f = ComplexField(mesh, 1+2im)

A piecewise constant `Field` can be constructed by specifying a grade: 

    var g = Field(mesh, 1, grade=1)

New methods `norm` and `sum` are provided. A `Selection` can be built by mapping a function over a `Field`.

## Module changes

The `VTK` and `povray` modules are now separate packages installable with `morphopm`. These packages have been updated to work with the new graphics system and additional functionality; povray now works properly on Windows, for example.

    morphopm install vtk
    morphopm install povray

Old copies are still shipped with Morpho, but print a warning to encourage migration. The `shapeopt` and `histogram` modules are deprecated and will be removed in a future release.

## Minor fixes

* `String.join` builds a `String` from a `List` of components.
* `System.subprocess` runs a command and returns a `Dictionary` with the output and exit status.
* `Selection.mesh()` returns the mesh a selection refers to.
* Metafunctions and closures now work across modules.
* Installed packages are searched before the built-in modules.
* Fixed a degenerate case in Delaunay triangulation.
* Help entries updated for the new graphics and plot interface.
* Additional constants provided in the `constants` module. 
* `KDTree.ismember` now returns the matching `KDTreeNode` or `false`. 
