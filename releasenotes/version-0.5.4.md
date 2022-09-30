# Release notes for 0.5.4

We're pleased to announce morpho 0.5.4, which contains many improvements. This is a large update with many new features as below. Going forward, new incremental work will be collected in the dev branch (we started doing this for this version) and we will move to a bimonthly release schedule. 

## Meshtools

The `meshtools` module has been extensively revised with many new features: 

* New `MeshPruner` class added that enables coarsening of meshes, analogous to `MeshRefiner`. 
* Refinement of 3D elements.
* Refinement of selections is improved, adding.
* Bugfixes for `MeshRefiner` and `MeshMerge` to prevent duplicate elements being generated in some circumstances. 

## Text

The `morphoview` application now supports text rendering. A number of modules have been updated to take advantage of this: 

* `plot` now provides `ScaleBar` objects that are useful for `plotfield`, as well as `plotmeshlabels` to label a mesh with element ids. 
* `graphics` now provides a `Text` class for textual elements, and has some performance improvements. 
* `color` now provides a number of new `ColorMap`s: `ViridisMap`, `InfernoMap`, `MagmaMap` and `PlasmaMap`, all of which are more friendly for people with color vision deficiency. 

## Variadic functions

You can now create functions that accept a variable number of parameters. Arguments passed to a function can be accessed as a `List`. 

    fn func(x, ...v) {
        for (a in v) print a
    }

Other improvements:

* New `VolumeIntegral` module to complement `AreaIntegral` and `LineIntegral`. 
* Internal improvements to the morpho virtual machine. 
* A `System` class to enable you to get platform information. 
* Numerous bugfixes. 
* Numerous improvements to the documentation.
* Improvements to the `povray` module. 
* New examples for `plot` module. 
* You can now translate the view in `morphoview` by right click and dragging or using alt-arrow keys. 