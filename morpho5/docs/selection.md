[comment]: # (Morpho selection class help file)
[version]: # (0.5)

# Selection
[tagselection]: # (selection)

The Selection class enables you to select components of a mesh for later use. You can supply a function that is applied to the coordinates of every vertex in the mesh, or select components like boundaries.

Create an empty selection:

    var s = Selection(mesh)

Select vertices above the z=0 plane using an anonymous function:

    var s = Selection(mesh, fn (x,y,z) z>0)

Select the boundary of a mesh:

    var s = Selection(mesh, boundary=true)

Selection objects can be composed using set operations:

    var s = s1.union(s2)

or
    var s = s1.intersection(s2)

To add additional grades, use the addgrade method. For example, to add areas:
    s.addgrade(2)

[showsubtopics]: # subtopics

## addgrade
[tagaddgrade]: # (addgrade)
Adds elements of the specified grade to a Selection. For example, to add edges to an existing selection, use

    s.addgrade(1)

By default, this only adds an element if *all* vertices in the element are currently selected. Sometimes, it's useful to be able to add elements for which only some vertices are selected. The optional argument `partials` allows you to do this:

    s.addgrade(1, partials=true)

Note that this method modifies the existing selection, and does not generate a new Selection object.

## removegrade
[tagremovegrade]: # (removegrade)
Removes elements of the specified grade from a Selection. For example, to remove edges from an existing selection, use

    s.removegrade(1)

Note that this method modifies the existing selection, and does not generate a new Selection object.

## idlistforgrade
[tagidlistforgrade]: # (idlistforgrade)
Returns a list of element ids included in the selection.

To find out which edges are selected:

    var edges = s.idlistforgrade(1)

## isselected
[tagisselected]: # (isselected)
Checks if an element id is selected, returning `true` or `false` accordingly.

To check if edge number 5 is selected:

    var f = s.isselected(1, 5))
