[comment]: # (Finite element space help)
[version]: # (0.5)

# FiniteElementSpace
[tagfiniteelementspace]: # (FiniteElementSpace)
[tagfespace]: # (fespace)

The `FiniteElementSpace` class describes how a `Field` is discretized on a particular grade of a mesh.

You will usually obtain a finite element space from an existing field:

    var fs = field.finiteElementSpace()

[showsubtopics]: # (showsubtopics)

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

    var layout = fs.layout(field)
    print layout

## NodeElementIndex
[tagnodeelementindex]: # (nodeelementindex)

Returns a tuple describing where a given node stores its degree of freedom in a field. The tuple has the form `(grade, element id, index)`:

    print fs.nodeElementIndex(0)

## NodeCoords
[tagnodecoords]: # (nodecoords)

Returns barycentric coordinates for the nodes of the finite element space. With no argument, this returns a matrix containing the coordinates of every node:

    print fs.nodeCoords()

With an integer argument, it returns the barycentric coordinates for a single node:

    print fs.nodeCoords(0)
