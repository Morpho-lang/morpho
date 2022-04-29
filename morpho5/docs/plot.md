[comment]: # (Plot module help)
[version]: # (0.5)

# Plot
[tagplot]: # (plot)

The `plot` module provides visualization capabilities for Meshes, Selections and Fields. These functions produce Graphics objects that can be displayed with `Show`.

To use the module, first import it:

    import plot

[showsubtopics]: # (subtopics)

## Plotmesh
[tagplotmesh]: # (plotmesh)

Visualizes a `Mesh` object:

    var g = plotmesh(mesh)

Plotmesh accepts a number of optional arguments to control what is displayed:

* `selection` - Only elements in a provided `Selection` are drawn.
* `grade` - Only draw the specified grade. This can also be a list of multiple grades to draw.
* `color` - Draw the mesh in a provided `Color`.
* `filter` and `transmit` - Used by the `povray` module to indicate transparency.

## Plotselection
[tagplotselection]: # (plotselection)

Visualizes a `Selection` object:

    var g = plotselection(mesh, sel)

Plotselection accepts a number of optional arguments to control what is displayed:

* `grade` - Only draw the specified grade. This can also be a list of multiple grades to draw.
* `filter` and `transmit` - Used by the `povray` module to indicate transparency.

## Plotfield
[tagplotfield]: # (plotfield)

Visualizes a scalar `Field` object:

    var g = plotfield(field)

Plotfield accepts a number of optional arguments to control what is displayed:

* `grade` - Draw the specified grade.
* `colormap` - A `Colormap` object to use. The field is automatically scaled.
* `style` - Plot style. See below. 
* `filter` and `transmit` - Used by the `povray` module to indicate transparency.

Supported plot styles: 

* `default` - Color `Mesh` elements by the corresponding value of the `Field`.
* `interpolate` - Interpolate `Field` quantities onto higher elements.
