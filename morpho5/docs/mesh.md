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

[showsuptopics]: # (showsuptopics)

## Save
[tagsave]: # (Save)

Saves a mesh as a .mesh file.

    m.save("new.mesh")

## Vertexposition
[tagvertexposition]: # (vertexposition)

Retrieves the position of a vertex given an id:

    print m.vertexposition(id)

## Setvertexposition
[tagsetvertexposition]: # (setvertexposition)

Sets the position of a vertex given an id and a position vector:

    print m.setvertexposition(1, Matrix([0,0,0]))

## Addgrade
[tagaddgrade]: # (addgrade)

Adds a new grade to a mesh. This is commonly used when, for example, a mesh file includes facets but not edges. To add the missing edges:

    m.addgrade(1)

## Addsymmetry
[tagaddsymmetry]: # (addsymmetry)

Adds a symmetry to a mesh. Experimental in version 0.5.

## Maxgrade
[tagmaxgrade]: # (maxgrade)

Returns the highest grade element present:

    print m.maxgrade()

## Count
[tagcount]: # (count)

Counts the number of elements. If no argument is provided, returns the number of vertices. Otherwise, returns the number of elements present of a given grade:

    print m.count(2) // Returns the number of area-like elements. 
