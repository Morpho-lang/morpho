[comment]: # (Morpho meshtools help file)
[version]: # (0.5)

# Meshtools
[tagmeshtools]: # (meshtools)

The Meshtools package contains a number of functions and classes to assist with creating and manipulating meshes.

[showsubtopics]: # (subtopics)

## AreaMesh
[tagareamesh]: # (areamesh)

This function creates a mesh composed of triangles from a parametric function. To use it:

    var m = AreaMesh(function, range1, range2, closed=boolean)

where

* `function` is a parametric function that has one parameter. It should return a list of coordinates or a column matrix corresponding to this parameter.
* `range1` is the Range to use for the first parameter of the parametric function.
* `range2` is the Range to use for the second parameter of the parametric function.
* `closed` is an optional parameter indicating whether to create a closed loop or not. You can supply a list where each element indicates whether the relevant parameter is closed or not.

To use `AreaMesh`, import the `meshtools` module:

    import meshtools

Create a square:

    var m = AreaMesh(fn (u,v) [u, v, 0], 0..1:0.1, 0..1:0.1)

Create a tube:

    var m = AreaMesh(fn (u, v) [v, cos(u), sin(u)], -Pi...Pi:Pi/4,
                     -1..1:0.1, closed=[true, false])

Create a torus:

    var c=0.5, a=0.2
    var m = AreaMesh(fn (u, v) [(c + a*cos(v))*cos(u),
                                (c + a*cos(v))*sin(u),
                                a*sin(v)], 0...2*Pi:Pi/16, 0...2*Pi:Pi/8, closed=true)

## LineMesh
[taglinemesh]: # (linemesh)

This function creates a mesh composed of line elements from a parametric function. To use it:

    var m = LineMesh(function, range, closed=boolean)

where

* `function` is a parametric function that has one parameter. It should return a list of coordinates or a column matrix corresponding to this parameter.
* `range` is the Range to use for the parametric function.
* `closed` is an optional parameter indicating whether to create a closed loop or not.

To use `LineMesh`, import the `meshtools` module:

    import meshtools

Create a circle:

    import constants
    var m = LineMesh(fn (t) [sin(t), cos(t), 0], 0...2*Pi:2*Pi/50, closed=true)

## PolyhedronMesh
[tagpolyhedronmesh]: # (polyhedron)

This function creates a mesh corresponding to a polyhedron. 

    var m = PolyhedronMesh(vertices, faces)

where `vertices` is a list of vertices and `faces` is a list of faces specified as a list of vertex indices.

To use `PolyhedronMesh`, import the `meshtools` module:

    import meshtools

Create a cube:

    var m = PolyhedronMesh([ [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
                             [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
                             [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
                             [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5]],
                           [ [0,1,3,2], [4,5,7,6], [0,1,5,4], 
                             [3,2,6,7], [0,2,6,4], [1,3,7,5] ])

*Note* that the vertices in each face list must be specified strictly in cyclic order.

## DelaunayMesh
[tagdelaunaymesh]: # (delaunaymesh)

The `DelaunayMesh` constructor function creates a `Mesh` object directly from a point cloud using the Delaunay triangulator.

    var pts = []
    for (i in 0...100) pts.append(Matrix([random(), random()]))
    var m=DelaunayMesh(pts)
    Show(plotmesh(m))

You can control the output dimension of the mesh (e.g. to create a 2D mesh embedded in 3D space) using the optional `outputdim` property. 

    var m = DelaunayMesh(pts, outputdim=3)

## Equiangulate
[tagequiangulate]: # (equiangulate)

Attempts to equiangulate a mesh, exchanging elements to improve their regularity.

    equiangulate(mesh)

*Note* this function modifies the mesh in place; it does not create a new mesh.

## ChangeMeshDimension
[tagchangemeshdimension]: # (changemeshdimension)

Changes the dimension in which a mesh is embedded. For example, you may have created a mesh in 2D that you now wish to use in 3D.

To use:

    var new = ChangeMeshDimension(mesh, dim)

where `mesh` is the mesh you wish to change, and `dim` is the new embedding dimension.

## MeshBuilder
[tagmeshbuiler]: # (meshbuilder)

The `MeshBuilder` class simplifies user creation of meshes. To use this class, begin by creating a `MeshBuilder` object:

    var build = MeshBuilder()

You can then add vertices, edges, etc. one by one using `addvertex`, `addedge`, `addface` and `addelement`. Each of these returns an element id:

    var id1=build.addvertex(Matrix([0,0,0]))
    var id2=build.addvertex(Matrix([1,1,1]))
    build.addedge([id1, id2])

Once the mesh is ready, call the `build` method to construct the `Mesh`:

    var m = build.build()

You can specify the dimension of the `Mesh` explicitly when initializing the `MeshBuilder`: 

    var mb = MeshBuilder(dimension=2)

or implicitly when adding the first vertex:

    var mb = MeshBuilder() 
    mb.addvertex([0,1]) // A 2D mesh

## MshBldDimIncnstnt
[tagmshblddimincnstnt]: # (mshblddimincnstnt)

This error is produced if you try to add a vertex that is inconsistent with the mesh dimension, e.g.

    var mb = MeshBuilder(dimension=2) 
    mb.addvertex([1,0,0]) // Throws an error! 

To fix this ensure all vertices have the correct dimension.

## MshBldDimUnknwn
[tagmshblddimunknwn]: # (mshblddimunknwn)

This error is produced if you try to add an element to a `MeshBuilder` object but haven't yet specified the dimension (at initialization) or by adding a vertex.

    var mb = MeshBuilder() 
    mb.addedge([0,1]) // No vertices have been added 

To fix this add the vertices first.

## MeshRefiner
[tagmeshrefiner]: # (meshrefiner)

The `MeshRefiner` class is used to refine meshes, and to correct associated data structures that depend on the mesh.

To prepare for refining, first create a `MeshRefiner` object either with a `Mesh`,

    var mr = MeshRefiner(mesh)

or with a list of objects that can include a `Mesh` as well as `Field`s and `Selection`s.

    var mr = MeshRefiner([mesh, field, selection ... ])

To perform the refinement, call the `refine` method. You can refine all elements,

    var dict = mr.refine()

or refine selected elements using a `Selection`,

    var dict = mr.refine(selection=select)

The `refine` method returns a `Dictionary` that maps old objects to new, refined objects. Use this to update your data structures.

    var newmesh = dict[oldmesh]

## MeshPruner
[tagmeshpruner]: # (meshpruner)

The `MeshPruner` class is used to prune excessive detail from meshes (a process that's sometimes referred to as coarsening), and to correct associated data structures that depend on the mesh.

First create a `MeshPruner` object either with a `Mesh`,

    var mp = MeshRefiner(mesh)

or with a list of objects that can include a `Mesh` as well as `Field`s and `Selection`s.

    var mp = MeshRefiner([mesh, field, selection ... ])

To perform the refinement, call the `prune` method with a `Selection`,

    var dict = mp.refine(select)

The `refine` method returns a `Dictionary` that maps old objects to new, refined objects. Use this to update your data structures.

    var newmesh = dict[oldmesh]

## MeshMerge
[tagmeshmerge]: # (meshmerge)
[tagmerge]: # (meshmerge)

The `MeshMerge` class is used to combine meshes into a single mesh, removing any duplicate elements.

To use, create a `MeshMerge` object with a list of meshes to merge,

    var mrg = MeshMerge([m1, m2, m3, ... ])

and then call the `merge` method to return a combined mesh:

    var newmesh = mrg.merge()
