[comment]: # (Plot module help)
[version]: # (0.5.4)

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

## Plotmeshlabels
[tagplotmeshlabels]: # (plotmeshlabels)

Draws the ids for elements in a `Mesh`: 

    var g = plotmeshlabels(mesh) 

Plotmeshlabels accepts a number of optional arguments to control the output: 

* `grade` - Only draw the specified grade. This can also be a list of multiple grades to draw.
* `selection` - Only labels in a provided `Selection` are drawn.
* `offset` - Local offset vector for labels. Can be a `List`, a `Matrix` or a function. 
* `dirn` - Text direction for labels. Can be a `List`, a `Matrix` or a function. 
* `vertical` - Text vertical direction. Can be a `List`, a `Matrix` or a function. 
* `color` - Label color. Can be a `Color` object or a `Dictionary` of colors for each grade. 
* `fontsize` - Font size to use. 

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
* `scalebar` - A `Scalebar` object to use. 
* `style` - Plot style. See below. 
* `filter` and `transmit` - Used by the `povray` module to indicate transparency.
* `cmin` and `cmax` - Can be used to define the data range covered.
  Values beyond these limits will be colored by the lower/upper bound of
  the colormap accordingly.

Supported plot styles: 

* `default` - Color `Mesh` elements by the corresponding value of the `Field`.
* `interpolate` - Interpolate `Field` quantities onto higher elements.

## ScaleBar
[tagscalebar]: # (scalebar)

Represents a scalebar for a plot: 

    Show(plotfield(field, scalebar=ScaleBar(posn=[1.2,0,0])))

`ScaleBar`s can be created with many adjustable parameters: 

* `nticks` - Maximum number of ticks to show.  
* `posn` - Position to draw the `ScaleBar`. 
* `length` - Length of `ScaleBar` to draw. 
* `dirn` - Direction to draw the `ScaleBar` in. 
* `tickdirn` - Direction to draw the ticks in. 
* `colormap` - `ColorMap` to use.
* `textdirn` - Direction to draw labels in. 
* `textvertical` - Label vertical direction. 
* `fontsize` - Fontsize for labels
* `textcolor` - Color for labels 

You can draw the `ScaleBar` directly by calling the `draw` method: 

    sb.draw(min, max)

where `min` and `max` are the minimum and maximum values to display on the scalebar. 