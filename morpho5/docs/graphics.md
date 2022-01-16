[comment]: # (Graphics module help)
[version]: # (0.5)

# Graphics
[taggraphics]: # (graphics)

The `graphics` module provides a number of classes to provide simple visualization capabilities. To use it, you first need to import the module:

    import graphics

The `Graphics` class acts as an abstract container for graphical information; to actually launch the display see the `Show` class. You can create an empty scene like this,

    var g = Graphics()

Additional elements can be added using the `display` method.

    g.display(element)

Morpho provides the following fundamental Graphical element classes:

    TriangleComplex

You can also use functions like `Arrow`, `Tube` and `Cylinder` to create these elements conveniently.

To combine graphics objects, use the add operator:

    var g1 = Graphics(), g2 = Graphics()
    // ...
    Show(g1+g2)

[show]: # (subtopics)

## Show
[tagshow]: # (Show)

`Show` is used to launch an interactive graphical display using the external `morphoview` application. `Show` takes a `Graphics` object as an argument:

    var g = Graphics()
    Show(g)

## TriangleComplex
[tagTriangleComplex]: # (TriangleComplex)

A `TriangleComplex` is a graphical element that can be used as part of a graphical display. It consists of a list of vertices and a connectivity matrix that selects which vertices are used in each triangle.

To create one, call the constructor with the following arguments:

    TriangleComplex(position, normals, colors, connectivity)

* `position` is a `Matrix` containing vertex positions as *columns*.
* `normals` is a `Matrix` with a normal for each vertex.
* `colors` is the color of the object.
* `connectivity` is a `Sparse` matrix where each column represents a triangle and rows correspond to vertices.

You can also provide optional arguments:

* `transmit` sets the transparency of the object. This parameter is only
used by the povray module as of now. Default is 0.
* `filter` sets the transparency of the object using a filter effect.
This parameter is only used by the povray module as of now. Default is 0. For the difference between `transmit` and `filter`, checkout the 
[POVRay documentation](http://xahlee.info/3d/povray-glassy.html).


Add to a `Graphics` object using the `display` method.

## Arrow
[tagArrow]: # (Arrow)

The `Arrow` function creates an arrow. It takes two arguments:

    arrow(start, end)

* `start` and `end` are the two vertices. The arrow points `start` -> `end`.

You can also provide optional arguments:

* `aspectratio` controls the width of the arrow relative to its length
* `n` is an integer that controls the quality of the display. Higher `n` leads to a rounder arrow.
* `color` is the color of the arrow. This can be a list of RGB values or a `Color` object
* `transmit` sets the transparency of the arrow. This parameter is only
used by the povray module as of now. Default is 0.
* `filter` sets the transparency of the arrow using a filter effect.
This parameter is only used by the povray module as of now. Default is 0. For the difference between `transmit` and `filter`, checkout the 
[POVRay documentation](http://xahlee.info/3d/povray-glassy.html).

Display an arrow:

    var g = Graphics([])
    g.display(Arrow([-1/2,-1/2,-1/2], [1/2,1/2,1/2], aspectratio=0.05, n=10))
    Show(g)

## Cylinder
[tagCylinder]: # (Cylinder)

The `Cylinder` function creates a cylinder. It takes two required arguments:

    cylinder(start, end)

* `start` and `end` are the two vertices.

You can also provide optional arguments:

* `aspectratio` controls the width of the cylinder relative to its length.
* `n` is an integer that controls the quality of the display. Higher `n` leads to a rounder cylinder.
* `color` is the color of the cylinder. This can be a list of RGB values or a `Color` object.
* `transmit` sets the transparency of the cylinder. This parameter is only
used by the povray module as of now. Default is 0.
* `filter` sets the transparency of the cylinder using a filter effect.
This parameter is only used by the povray module as of now. Default is 0. For the difference between `transmit` and `filter`, checkout the 
[POVRay documentation](http://xahlee.info/3d/povray-glassy.html).

Display an cylinder:

    var g = Graphics()
    g.display(Cylinder([-1/2,-1/2,-1/2], [1/2,1/2,1/2], aspectratio=0.1, n=10))
    Show(g)

## Tube
[tagTube]: # (Tube)

The `Tube` function connects a sequence of points to form a tube.

    Tube(points, radius)

* `points` is a list of points; this can be a list of lists or a `Matrix` with the positions as columns.
* `radius` is the radius of the tube.

You can also provide optional arguments:

* `n` is an integer that controls the quality of the display. Higher `n` leads to a rounder tube.
* `color` is the color of the tube. This can be a list of RGB values or a `Color` object.
* `closed` is a `bool` that indicates whether the tube should be closed to form a loop.
* `transmit` sets the transparency of the tube. This parameter is only
used by the povray module as of now. Default is 0.
* `filter` sets the transparency of the tube using a filter effect.
This parameter is only used by the povray module as of now. Default is 0. For the difference between `transmit` and `filter`, checkout the 
[POVRay documentation](http://xahlee.info/3d/povray-glassy.html).

Draw a square:

    var a = Tube([[-1/2,-1/2,0],[1/2,-1/2,0],[1/2,1/2,0],[-1/2,1/2,0]], 0.1, closed=true)
    var g = Graphics()
    g.display(a)

## Sphere
[tagSphere]: # (Sphere)

The `Sphere` function creates a sphere.

    Sphere(center, radius)

* `center` is the position of the center of the sphere; this can be a list or column `Matrix`.
* `radius` is the radius of the sphere

You can also provide an optional argument:

* `color` is the color of the sphere. This can be a list of RGB values or a `Color` object.
* `transmit` sets the transparency of the sphere. This parameter is only
used by the povray module as of now. Default is 0.
* `filter` sets the transparency of the sphere using a filter effect.
This parameter is only used by the povray module as of now. Default is 0. For the difference between `transmit` and `filter`, checkout the 
[POVRay documentation](http://xahlee.info/3d/povray-glassy.html).

Draw some randomly sized spheres:

    var g = Graphics()
    for (i in 0...10) {
      g.display(Sphere([random()-1/2, random()-1/2, random()-1/2], 0.1*(1+random()),       color=Gray(random())))
    }
    Show(g)
