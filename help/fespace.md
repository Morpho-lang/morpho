[comment]: # (Finite element space help)
[version]: # (0.5)

# FiniteElementSpace
[tagfiniteelementspace]: # (FiniteElementSpace)
[tagfespace]: # (fespace)

The `FiniteElementSpace` class describes how a `Field` is discretized on a particular grade of a mesh.

You can create a finite element space directly from its label and grade:

    var fs = FiniteElementSpace("CG2", grade=2)

Available labels are `CG0` (piecewise constant: one value per element), `CG1`, `CG2` and `CG3` (continuous Lagrange elements). Each label exists on grades 1 (lines), 2 (triangles) and 3 (tetrahedra).

You can also obtain a finite element space from an existing field:

    var fs = f.finiteElementSpace()

[showsubtopics]: # (subtopics)

## Count
[tagcount]: # (count)

Returns the number of nodes in the finite element space:

    print fs.count()

## Grade
[taggrade]: # (grade)

Returns the mesh grade on which the finite element space is defined:

    print fs.grade()

## Layout
[taglayout]: # (layout)

Returns a sparse matrix describing how the degrees of freedom of a given field are laid out for this finite element space:

    var layout = fs.layout(f)
    print layout

This is useful when you need to understand how local node values are mapped into the underlying storage of a `Field`.

For example:

    var fs = f.finiteElementSpace()
    var layout = fs.layout(f)
    print layout

## NodeElementIndex
[tagnodeelementindex]: # (nodeelementindex)

Returns a tuple describing where a given node stores its degree of freedom in a field. The tuple has the form `(grade, element id, index)`:

    print fs.nodeElementIndex(0)

This can be used together with `Field` indexing to locate the value associated with a given node.

For example:

    var loc = fs.nodeElementIndex(0)
    print loc

## NodeCoords
[tagnodecoords]: # (nodecoords)

Returns barycentric coordinates for the nodes of the finite element space. With no argument, this returns a matrix containing the coordinates of every node:

    print fs.nodeCoords()

With an integer argument, it returns the barycentric coordinates for a single node:

    print fs.nodeCoords(0)

For a grade `g` space, each node coordinate is represented by `g+1` barycentric coordinates. For `CG0` the single node is at the element centroid.

For example, to inspect all node coordinates:

    var coords = fs.nodeCoords()
    print coords
