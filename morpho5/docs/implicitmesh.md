[comment]: # (Implicitmesh module help)
[version]: # (0.5)

# ImplicitMesh
[tagimplicitmesh]: # (implicitmesh)

The `implicitmesh` module allows you to build meshes from implicit functions. For example, the unit sphere could be specified using the function x^2+y^2+z^2-1 == 0.

To use the module, first import it:

    import implicitmesh

To create a sphere, first create an ImplicitMeshBuilder object with the implict function you'd like to use:

    var impl = ImplicitMeshBuilder(fn (x,y,z) x^2+y^2+z^2-1)

You can use an existing function (or method) as well as an anonymous function as above.

Then build the mesh,

    var mesh = impl.build(stepsize=0.25)

The `build` method takes a number of optional arguments:

* `start` - the starting point. If not provided, the value Matrix([1,1,1]) is used.
* `stepsize` - approximate lengthscale to use.
* `maxiterations` - maximum number of iterations to use. If this limit is exceeded, a partially built mesh will be returned.
