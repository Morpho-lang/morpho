[comment]: # (Meshgen module help)
[version]: # (0.5)

# Meshgen
[tagmeshgen]: # (meshgen)

The `meshgen` module is used to create `Mesh` objects corresponding to a specified domain. It provides the `MeshGen` class to perform the meshing, which are created with the following arguments:

    MeshGen(domain, boundingbox)

Domains are specified by a scalar function that is positive in the region to be meshed and locally smooth. For example, to mesh the unit disk:

    var dom = fn (x) -(x[0]^2+x[1]^2-1)

A `MeshGen` object is then created and then used to build the `Mesh` like this:

    var mg = MeshGen(dom, [-1..1:0.2, -1..1:0.2])
    var m = mg.build()

A bounding box for the mesh must be specified as a `List` of `Range` objects, one for each dimension. The increment on each `Range` gives an approximate scale for the size of elements generated.

To facilitate convenient creation of domains, a `Domain` class is provided that provides set operations `union`, `intersection` and `difference`.

`MeshGen` accepts a number of optional arguments:

* `weight` A scalar weight function that controls mesh density.
* `quiet` Set to `true` to suppress  `MeshGen` output.
* `method` a list of options that controls the method used.

Some method choices that are available include:

* `"FixedStepSize"` Use a fixed step size in optimization.
* `"StartGrid"` Start from a regular grid of points (the default).
* `"StartRandom"` Start from a randomly generated collection of points.

There are also a number of properties of a `MeshGen` object that can be set prior to calling `build` to control the operation of the mesh generation:

* `stepsize`, `steplimit` Stepsize used internally by the `Optimizer`
* `fscale` an internal "pressure"
* `ttol` how far the vertices are allowed to move before retriangulation
* `etol` energy tolerance for optimization problem
* `maxiterations` Maximum number of iterations of minimization +
  retriangulation (default is 100)

`MeshGen` picks default values that cover a reasonable range of uses.

[showsubtopics]: # (subtopics)

## Domain
[tagdomain]: # (domain)

The `Domain` class is used to conveniently build a domain by composing simpler elements. 

Create a `Domain` from a scalar function that is positive in the region of interest:

    var dom = Domain(fn (x) -(x[0]^2+x[1]^2-1))

You can pass it to `MeshGen` to specify the region to mesh: 

    var mg = MeshGen(dom, [-1..1:0.2, -1..1:0.2])

You can combine `Domain` objects using set operations `union`, `intersection` and `difference`: 

    var a = CircularDomain(Matrix([-0.5,0]), 1)
    var b = CircularDomain(Matrix([0.5,0]), 1)
    var c = CircularDomain(Matrix([0,0]), 0.3)
    var dom = a.union(b).difference(c)

## CircularDomain
[tagcirculardomain]: # (circulardomain)

Conveniently constructs a `Domain` object correspondiong to a disk. Requires the position of the center and a radius as arguments. 

Create a domain corresponding to the unit disk: 

    var c = CircularDomain([0,0], 1)

## RectangularDomain
[tagrectangulardomain]: # (rectangulardomain)

Conveniently constructs a `Domain` object corresponding to a rectangle. Requires a list of ranges as arguments. Works in arbitrary dimensions

Create a square `Domain`:

    var c = RectangularDomain([-1..1, -1..1])

## HalfSpaceDomain
[halfspacedomain]: # (halfspacedomain)

Conveniently constructs a `Domain` object correspondiong to a half space defined by a plane at `x0` and a normal `n`:

    var hs = HalfSpaceDomain(x0, n)

Note `n` is an "outward" normal, so points into the *excluded* region.

Half space corresponding to the allowed region `x<0`:

    var hs = HalfSpaceDomain(Matrix([0,0,0]), Matrix([1,0,0]))

Note that `HalfSpaceDomain`s cannot be meshed directly as they correspond to an infinite region. They are useful, however, for combining with other domains.

Create half a disk by cutting a `HalfSpaceDomain` from a `CircularDomain`:

    var c = CircularDomain([0,0], 1)
    var hs = HalfSpaceDomain(Matrix([0,0]), Matrix([-1,0]))
    var dom = c.difference(hs) 
    var mg = MeshGen(dom, [-1..1:0.2, -1..1:0.2], quiet=false)
    var m = mg.build()

## MshGnDim
[mshgndim]: # (mshgndim)

The `MeshGen` module currently supports 2 and 3 dimensional meshes. Higher dimensional meshing will be available in a future release; please contact the developer if you are interested in this functionality.