[comment]: # (Mesh class help)
[version]: # (0.5)

# Mesh
[tagmesh]: # (Mesh)

The `Mesh` class provides support for meshes. Meshes may consist of different kinds of element, including vertices, line elements, facets or area elements, tetrahedra or volume elements.

To create a mesh, you can import it from a file:

    var m = Mesh("sphere.mesh")

or use one of the functions available in `meshtools` or `implicitmesh` packages.

Each type of element is referred to as belonging to a different `grade`. Point-like elements (vertices) are *grade 0*; line-like elements (edges) are *grade 1*; area-like elements (facets; triangles) are *grade 2* etc.

The `plot` package includes functions to visualize meshes.

[showsubtopics]: # (showsubtopics)

## Save
[tagsave]: # (Save)

Saves a mesh as a .mesh file.

    m.save("new.mesh")

## Vertexmatrix
[tagvertexmatrix]: # (vertexmatrix)

Returns the matrix of vertex positions used by the mesh. Each column corresponds to a vertex:

    var verts = m.vertexmatrix()
    print verts

## Setvertexmatrix
[tagsetvertexmatrix]: # (setvertexmatrix)

Replaces the matrix of vertex positions used by the mesh. The new matrix must have the same dimensions as the existing vertex matrix:

    m.setvertexmatrix(newverts)

## Vertexposition
[tagvertexposition]: # (vertexposition)

Retrieves the position of a vertex given an id:

    print m.vertexposition(id)

## Setvertexposition
[tagsetvertexposition]: # (setvertexposition)

Sets the position of a vertex given an id and a position vector:

    print m.setvertexposition(1, Matrix([0,0,0]))

## Resetconnectivity
[tagresetconnectivity]: # (resetconnectivity)

Clears any cached connectivity matrices associated with the mesh:

    m.resetconnectivity()

## Connectivitymatrix
[tagconnectivitymatrix]: # (connectivitymatrix)

Returns the connectivity matrix that maps elements of one grade to another. For example, to retrieve the vertex-to-edge connectivity:

    var c = m.connectivitymatrix(0, 1)
    print c

Here `0` refers to vertices and `1` to edges. Similarly, `m.connectivitymatrix(0, 2)` retrieves the vertex-to-facet connectivity.

## Addgrade
[tagaddgrade]: # (addgrade)

Adds a new grade to a mesh. This is commonly used when, for example, a mesh file includes facets but not edges. To add the missing edges:

    m.addgrade(1)

You can also provide an explicit sparse connectivity matrix for the new grade:

    m.addgrade(1, connectivity)

## Removegrade
[tagremovegrade]: # (removegrade)

Removes a grade and its associated connectivity from a mesh:

    m.removegrade(1)

## Addsymmetry
[tagaddsymmetry]: # (addsymmetry)

Adds a symmetry to a mesh. Experimental in version 0.5.

## Barycentric
[tagbarycentric]: # (barycentric)

Computes barycentric coordinates for a point inside a mesh element. You must supply the grade, the element id and the position matrix:

    var lambda = m.barycentric(2, element, Matrix([x, y, z]))
    print lambda

For a grade `g` element, the returned matrix contains `g+1` barycentric coordinates.

For example, for a triangular facet (`grade 2`), the returned matrix contains three barycentric coordinates whose sum is `1`.

## Maxgrade
[tagmaxgrade]: # (maxgrade)

Returns the highest grade element present:

    print m.maxgrade()

## Count
[tagcount]: # (count)

Counts the number of elements. If no argument is provided, returns the number of vertices. Otherwise, returns the number of elements present of a given grade:

    print m.count(2) // Returns the number of area-like elements. 

## Clone
[tagclone]: # (clone)

Creates a copy of a mesh:

    var copy = m.clone()
