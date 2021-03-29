[comment]: # Graphics package help
[version]: # 0.5

#Graphics
[tag]: # graphics

The `graphics` package provides a number of classes to provide simple visualization capabilities. To use it, you first need to import the package:

    import graphics

The `Graphics` class acts as an abstract container for graphical information; to actually launch the display see the `Show` class. You can create an empty scene like this,

    var g = Graphics([])

or initialize the graphics with an element such as a mesh:

    var g = Graphics(mesh)

Additional elements can be added using the `display` method.

    g.display(element)

Morpho provides the following fundamental Graphical element classes:

    TriangleComplex

You can also use functions like `arrow`, `tube` and `cylinder` to create these elements conveniently.

#Show
[tag]: # Show

`Show` is used to launch an interactive graphical display using the external `morphoview` application. `Show` takes a `Graphics` object as an argument:

    var g = Graphics()
    Show(g)

#TriangleComplex
[tag]: # TriangleComplex

A `TriangleComplex` is a graphical element that can be used as part of a graphical display. It consists of a list of vertices and a connectivity matrix that selects which vertices are used in each triangle.

To create one, call the constructor with the following arguments:

    TriangleComplex(position, normals, colors, connectivity)

* `position` is a `Matrix` containing vertex positions as *columns*.
* `normals` is a `Matrix` with a normal for each vertex.
* `colors` is the color of the object.
* `connectivity` is a `Sparse` matrix where each column represents a triangle and rows correspond to vertices.

Add to a `Graphics` object using the `display` method. 

#Arrow
[tag]: # arrow

The `arrow` function creates an arrow. It takes four arguments:

    arrow(start, end, aspectratio, n)

* `start` and `end` are the two vertices. The arrow points `start` -> `end`.
* `aspectratio` controls the width of the arrow relative to its length
* `n` is an integer that controls the quality of the display. Higher `n` leads to a rounder arrow.

Display an arrow:

    var g = Graphics([])
    g.display(arrow([-1/2,-1/2,-1/2], [1/2,1/2,1/2], 0.1, 10))
    Show(g)

#Cylinder
[tag]: # cylinder

The `cylinder` function creates an cylinder. It takes four arguments:

    cylinder(start, end, aspectratio, n)

* `start` and `end` are the two vertices.
* `aspectratio` controls the width of the cylinder relative to its length
* `n` is an integer that controls the quality of the display. Higher `n` leads to a rounder cylinder.

Display an cylinder:

    var g = Graphics([])
    g.display(cylinder([-1/2,-1/2,-1/2], [1/2,1/2,1/2], 0.1, 10))
    Show(g)

#Tube
[tag]: # tube

The `tube` function connects a sequence of points to form a tube.

    tube(points, radius, n, closed)

* `points` is a list of points; this can be a list of lists or a `Matrix` with the positions as columns.
* `radius` is the radius of the tube
* `n` is an integer that controls the quality of the display. Higher `n` leads to a rounder tube.
* `closed` is a `bool` that indicates whether the tube should be closed to form a loop.

Draw a square:

    var a = tube([[-1/2,-1/2,0],[1/2,-1/2,0],[1/2,1/2,0],[-1/2,1/2,0]], 0.1, 20, true)
    var g = Graphics([])
    g.display(a)
