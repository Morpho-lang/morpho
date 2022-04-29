[comment]: # (Meshslice module help)
[version]: # (0.5)

# Meshslice
[tagmeshslice]: # (meshslice)

The `meshslice` module is used to slice a `Mesh` object along a given plane, yielding a new `Mesh` object of lower dimensionality. You can also use `meshslice` to project `Field` objects onto the new mesh.

To use the module, begin by importing it:

    import meshslice 

Then construct a `MeshSlicer` object, passing the mesh you want to slice in the constructor:

    var slice = MeshSlicer(mesh)

You then perform a slice by calling the `slice` method, passing the plane you want to slice through. This method returns a new `Mesh` object comprising the slice. A plane is defined by a point that lies on the plane `pt` and a direction normal to the plan `dirn`:

    var slc = slice.slice(pt, dirn)

Having performed a slice, you can then project any associated `Field` objects onto the sliced mesh by calling the `slicefield` method:

    var phi = Field(mesh, fn (x,y,z) x+y+z)
    var sphi = slice.slicefield(phi)

The new field returned by `slicefield` lives on the sliced mesh. You can slice any number of fields.

You can perform multiple slices with the same `MeshSlicer` simply by calling `slice` again with a different plane.

## SlcEmpty
[tagslcempty]: # (slcempty)

This error occurs if you try to use `slicefield` on a `MeshSlicer` without having performed a slice. For example:

    var slice = MeshSlicer(mesh)
    slice.slicefield(phi) // Throws SlcEmpty
    slice.slice([0,0,0],[1,0,0]) 

To fix, call `slice` before `slicefield`:

    var slice = MeshSlicer(mesh)
    slice.slice([0,0,0],[1,0,0]) 
    slice.slicefield(phi) // Now slices correctly 
