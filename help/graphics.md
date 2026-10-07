[comment]: # (Graphics module help)
[version]: # (0.5)

# Graphics
[taggraphics]: # (graphics)

The `graphics` module builds a `Graphics` from simple objects. Import it with:

    import graphics

Create an empty `Graphics` container:

    var g = Graphics()

Add an object with `display`. This returns an id you can keep if you want to change the object later:

    var id = g.display(Sphere([0, 0, 0], 0.2, color=Red))

Open the `Graphics` with `Show` (import `morphoview` first):

    import morphoview
    Show(g)

You can set the window title, the background color, and the lighting when you create the `Graphics`:

    var g = Graphics(title="Charges", background=White, light="threepoint")

Combine two `Graphics` with `+`. The result retains the title, background, and lights of the left hand `Graphics`:

    Show(g1 + g2)

The `extend` method copies the contents of another `Graphics` into the receiver and returns the new ids created: 

    var ids = g.extend(g2)

Basic objects you can display: `Sphere`, `Cylinder`, `Arrow`, `Tube`, `Polygon`, `Text`, `PointCloud`, `LineSet`, and `TriangleComplex`.

Give an object a color with `color=`. Use a named color such as `Red`, a list such as `[1, 0, 0]`, `Color(r, g, b)`, or a see-through color such as `Red.opacity(0.3)`. To give each point its own color, pass a `ColorTable`.

[showsubtopics]: # (subtopics)

## Show
[tagshow]: # (Show)

`Show` opens an interactive view of a `Graphics` using the `morphoview` module:

    import morphoview
    Show(g)

## Placing an object
[tagdisplay]: # (display)

The `display` method adds an object to a `Graphics`. You can control the placement of the object with additional arguments: a second argument translates it; `scale` changes its size, as one number or as `[x, y, z]`; `rotate` rotates it about an axis `[angle, x, y, z]` where `angle` is in radians:

    g.display(Sphere([0, 0, 0], 0.2), [1, 0, 0], scale=2)
    g.display(Arrow([0, 0, 0], [1, 0, 0]), rotate=[Pi/2, 0, 0, 1])

You can also provide a `color` for an object, overriding any color the object has; disable shading with `flat=true`, which is useful for legends and color bars.

## Sphere
[tagsphere]: # (Sphere)

Represents a sphere placed at `center` with a given `radius` and optional color: 

    Sphere(center, radius, color=Red)

The value `center` can be given as a `List` or a column matrix:

    g.display(Sphere([0, 0, 0], 0.25, color=Blue))

## Cylinder
[tagcylinder]: # (Cylinder)

Represents a cylinder drawn from `start` to `end` points, with an optional color:

    Cylinder(start, end, aspectratio=0.1, radius=nil, n=10, color=Red)

The parameters `start` and `end` can be given as a `List` or a column `Matrix`. Optional parameter `aspectratio` sets the thickness as a fraction of the length; `radius` sets the thickness directly and is used instead of `aspectratio`. The value `n` controls how smooth the cylinder is; a larger value produces a rounder cylinder.

    g.display(Cylinder([0, 0, 0], [0, 0, 1], radius=0.05, color=Green))

## Arrow
[tagarrow]: # (Arrow)

Represents an arrow drawn from `start` to `end` points, with an optional color:

    Arrow(start, end, aspectratio=0.1, radius=nil, n=10, color=Red)

The parameters `start` and `end` can be given as a `List` or a column `Matrix`. Optional parameter `aspectratio` sets the head size as a fraction of the length; if `radius` is omitted, it also sets the shaft thickness. The parameter `radius` sets the shaft thickness directly, leaving `aspectratio` to control only the head. The value `n` controls how smooth the arrow is; a larger value produces a rounder arrow.

    g.display(Arrow([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], aspectratio=0.15, color=Yellow))

## Tube
[tagtube]: # (Tube)

Represents a tube of radius `radius` drawn through a sequence of `points`, with an optional color:

    Tube(points, radius, n=10, closed=false, color=Red)

The parameter `points` can be given as a `List` of points, or a `Matrix` with one point per column. Optional parameter `closed`, if set to `true`, joins the last point back to the first. The value `n` controls how smooth the tube is; a larger value produces a rounder tube.

    g.display(Tube([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]],
                   0.05, closed=true, color=Orange))

## Polygon
[tagpolygon]: # (Polygon)

Represents a flat polygon with three or more corners, with an optional color:

    Polygon(position, color=Blue)

The parameter `position` can be given as a `List` of points, or a `Matrix` with one point per column. The corners must lie in one plane, and the shape must be convex. The order of the corners decides which side faces outward.

    g.display(Polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], color=Blue))

## PointCloud
[tagpointcloud]: # (PointCloud)

Represents a set of points, with an optional color:

    PointCloud(position, color=White)

The parameter `position` can be given as a `List` of points, or a `Matrix` with one point per column. Optional parameter `color` may be one color, or a `ColorTable` with one color per point.

    g.display(PointCloud([[0, 0, 0], [1, 0, 0], [0, 1, 0]], color=White))

## LineSet
[taglineset]: # (LineSet)

Represents line segments drawn through a sequence of points, with an optional color:

    LineSet(position, color=White, closed=false)

The parameter `position` can be given as a `List` of points, or a `Matrix` with one point per column. The segments connect the points in order. Optional parameter `closed`, when `true`, joins the last point back to the first. Pass pairs of point indices as `connectivity` to choose the segments yourself:

    LineSet(position, connectivity, color=Cyan)

    g.display(LineSet([[0, 0, 0], [1, 0, 0], [1, 1, 0]], color=White))
    g.display(LineSet([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1], [0, 2]], color=Cyan))

## TriangleComplex
[tagtrianglecomplex]: # (TriangleComplex)

Represents a surface made of triangles, with an optional color. This is a very general primitive that can be used to make many kinds of shape from a set of positions and connectivity:

    TriangleComplex(position, normals, color, connectivity)

The parameters `position` and `normals` are a `Matrix` with one column per point. Each column of `connectivity` is one triangle. Optional parameter `color` may be one color, or a `ColorTable` with one color per point.

## Text
[tagtext]: # (Text)

Represents text drawn from `position`, with an optional color:

    Text(text, position, dirn=[1, 0, 0], vertical=nil, size=10, font=nil, color=White)

The parameters `position`, `dirn`, and `vertical` can be given as a `List` or a column `Matrix`. Optional parameter `dirn` sets the direction the text runs; `vertical` sets the upright direction and, if omitted, the text stands upright in the x-z plane. Optional parameter `font` is a font name, such as `"Helvetica"`. The value `size` is the font size in points.

    g.display(Text("Hello", [0, 0, 0], size=72, dirn=[1, 0, 0], color=White))

## Light
[taglight]: # (Light)

By default a `Graphics` uses neutral lighting. Other choices are `"threepoint"` and `"off"`:

    var g = Graphics(light="threepoint")

You can also place your own lamps. A lamp is a position, with an optional color and brightness. A `Graphics` can have up to four.

    g.addLight([2, 2, 2], color=White, intensity=0.8)
    g.addLight(Light([-1, 0, 1], color=Blue, intensity=0.3))

`setLights` replaces the current lighting. `resetLights` returns to neutral lighting.

    g.setLights("off")
    g.resetLights()

## Scene
[tagscene]: # (Scene)

A `Scene` works similarly to `Graphics`, except that the contents can be modified supporting interactivity and animation. Each object added to the scene with `display` returns an id, as for `Graphics`. Use that id to move an object:

    var s = Scene()
    var id = s.display(Sphere([0, 0, 0], 0.2, color=Red))
    s.move(id, [1, 0, 0])
    s.move(id, scale=2)

You can also remove the object;

    s.remove(id)

or replace it with a different object, retaining the id;

    s.replace(id, Cylinder([0, 0, 0], [0, 0, 1], radius=0.05))

or even recolor it:

    s.recolor(id, Blue)

A special `morph` method updates a `TriangleComplex` whose points have moved, leaving the triangle connectivity unchanged. Use `replace` when the object itself should change.
