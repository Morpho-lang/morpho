[comment]: # (Plot module help)
[version]: # (0.5.4)

# Plot
[tagplot]: # (plot)

The `plot` module draws a mesh, a selection, or a field. Import it with:

    import plot

A `Plot` is a `Graphics` you can open with `Show`:

    Show(Plot(mesh, grade=[0, 1], style="thick"))
    Show(Plot(selection, grade=[0, 1, 2], style=["shaded", "thick"]))
    Show(Plot(field, style="interpolate", scalebar=true))

`grade` chooses what to draw: `0` points, `1` lines, `2` faces. A list draws more than one. If you omit `grade`, only the highest grade is drawn.

`style` changes how those are drawn:

* `"thick"` draws spheres and tubes.
* `"shaded"` lights the faces.
* `"interpolate"` blends a field's colors across each face.
* `"spheres"` and `"tubes"` draw only the points or only the lines in the thick style.

Combine them with a list, as in `style=["shaded", "thick"]`. With no style, `Plot` draws points, plain lines, and flat faces.

`selection` on a mesh plot draws only the elements in that `Selection`. `color` sets the color. `axes=true` adds a box and axis labels. `scalebar=true` adds a color legend, or pass a `ScaleBar` to place it yourself. `title` and `background` set the window title and background color.

Legacy functions `plotmesh`, `plotselection`, and `plotfield` are provided for compatibility with older code. New code should simply use `Plot` as shown above. With no `style` specified, those functions draw the older thick style.

[showsubtopics]: # (subtopics)

## Refresh
[tagrefresh]: # (refresh)

A field plot can be changed in place:

    p.colormap(MagmaMap())
    p.range(0, 1)
    p.center(0)
    p.axes(true)
    p.scalebar(true)
    p.refresh()

`colormap` sets the color map and redraws. `range(cmin, cmax)` sets the values at the ends of the color map; `range()` with no arguments uses the field's own bounds again. `center` sets the value drawn with the middle color, which matters for a diverging map; `center()` clears it. `axes` and `scalebar` show or hide those parts. `refresh` redraws the plot from the current field.

## Plotmesh
[tagplotmesh]: # (plotmesh)

`plotmesh` is the older way to draw a `Mesh`. Prefer `Plot`:

    Plot(mesh, selection=sel, grade=[0, 1], color=Red, style="thick")

The older function draws the thick style if you omit `style`:

    var g = plotmesh(mesh)

* `selection` — draw only elements in a `Selection`.
* `grade` — one grade, or a list of grades.
* `color` — color for the mesh. Use `color.opacity(a)` for transparency.
* `style` — see above. If you omit it, `plotmesh` draws the thick style. `Plot` draws points, plain lines, and flat faces unless you set `style`.

## Plotmeshlabels
[tagplotmeshlabels]: # (plotmeshlabels)

Draws the ids for elements in a `Mesh`:

    var g = plotmeshlabels(mesh)

* `grade` — one grade, or a list of grades.
* `selection` — label only elements in a `Selection`.
* `offset` — where to place each label relative to the element. A list, a matrix, or a function.
* `dirn` — direction the text runs. A list, a matrix, or a function.
* `vertical` — upright direction of the text. A list, a matrix, or a function.
* `color` — a color, or a dictionary of colors for each grade.
* `fontsize` — size of the labels.

## Plotselection
[tagplotselection]: # (plotselection)

`plotselection` is the older way to draw a `Selection`. Selected elements are red and the rest of the mesh is gray. Prefer `Plot`:

    Plot(sel, grade=[0, 1, 2], style="thick")

The older function takes the mesh first, and draws the thick style if you omit `style`:

    var g = plotselection(mesh, sel)

* `grade` — one grade, or a list of grades.
* `style` — see above. If you omit it, `plotselection` draws the thick style.

`Plot(sel, color=Red)` draws the whole mesh in one color instead of red and gray.

## Plotfield
[tagplotfield]: # (plotfield)

`plotfield` is the older way to draw a scalar `Field`. Prefer `Plot`:

    Plot(field, colormap=ViridisMap(), style="interpolate", scalebar=true, cmin=0, cmax=1)

The older function draws the thick style if you omit `style`:

    var g = plotfield(field)

* `grade` — which grade to draw.
* `colormap` — a color map. The field is stretched to fit it.
* `scale` — set to `false` to use the field values as color-map positions directly.
* `scalebar` — `true`, or a `ScaleBar` to place the legend yourself.
* `selection` — draw only elements in a `Selection`.
* `style` — see above. If you omit it, `plotfield` draws the thick style. `"interpolate"` blends values across each face.
* `cmin` and `cmax` — values at the ends of the color map. Values outside this range keep the end colors.
* `center` — field value drawn with the middle color.

## Plotaxes
[tagplotaxes]: # (plotaxes)

Draws red, green, and blue arrows for the x, y, and z axes, starting at a point:

    plotaxes([0, 0, 0], size=1)

`size` is the length of each arrow. Add the result to a `Graphics` with `+`.

## ScaleBar
[tagscalebar]: # (scalebar)
[tagscalebarstrip]: # (scalebarstrip)

A color legend for a plot:

    Show(Plot(field, style="interpolate", scalebar=ScaleBar(posn=[1.2, 0, 0])))

* `nticks` — maximum number of ticks.
* `posn` — where to draw the bar.
* `length` — length of the bar.
* `dirn` — direction the bar runs.
* `tickdirn` — direction the ticks run.
* `colormap` — color map to show.
* `textdirn` — direction the labels run.
* `textvertical` — upright direction of the labels.
* `fontsize` — size of the labels.
* `textcolor` — color of the labels.

`ScaleBarStrip` draws a flat strip instead of a round bar.

Draw a bar on its own with `draw`. `min` and `max` are the values at the two ends:

    sb.draw(min, max)
