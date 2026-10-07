[comment]: # (Color module help)
[version]: # (0.5)

# Color
[tagcolor]: # (color)

The `color` module provides support for working with color. Colors are represented in morpho by `Color` objects. The module predefines some colors including `Red`, `Green`, `Blue`, `Black`, `White`, and `Clear`.

To use the module, use import as usual:

    import color

Create a spot `Color` from red, green, and blue component values, each from 0 to 1:

    var col = Color(0.5, 0.5, 0.5) // A 50% gray

You can provide the components in a single list or a column matrix instead. See `opacity` to make a color transparent.

Pure colors can be combined using arithmetic operations or using `Blend`:

    var purple = Red + Blue
    var pale = 0.5*Red + 0.5*White
    var mid = Blend(Red, Blue, 0.5)  // == (1-0.5)*Red + 0.5*Blue

The `color` module also provides `ColorMap`s, which provide a sequence of colors as a function of a continuous parameter, and `ColorTable`s, which define a set of colors given by an integer index. 

[showsubtopics]: # (subtopics)

## Opacity
[tagopacity]: # (opacity)
[tagtransparency]: # (transparency)

Colors can be constructed with a fourth parameter, sometimes referred to as alpha, that indicates their opacity, from 0 (transparent) to 1 (opaque):

    var faint = Color(1, 0, 0, 0.3)
    var faint = Red.opacity(0.3) // makes a version of Red 30% opaque

The constant `Clear` is fully transparent. Color constructors accept an additional opacity parameter: 

    Gray(0.2) // solid 20% gray
    Gray(0.2, 0.5) // 20% gray that is 50% transparent.

## RGB
[tagrgb]: # (rgb)

Gets the rgb components of a `Color` or `ColorMap` object as a list. Takes a single argument in the range 0 to 1, although the result will only depend on this argument if the object is a `ColorMap`.

    var col = Color(0.1, 0.5, 0.7)
    print col.rgb(0)

## Red
[tagred]: # (red)
Built in `Color` object for use with the `graphics` and `plot` modules.

## Green
[taggreen]: # (green)
Built in `Color` object for use with the `graphics` and `plot` modules.

## Blue
[tagblue]: # (blue)
Built in `Color` object for use with the `graphics` and `plot` modules.

## White
[tagwhite]: # (white)
Built in `Color` object for use with the `graphics` and `plot` modules.

## Black
[tagblack]: # (black)
Built in `Color` object for use with the `graphics` and `plot` modules.

## Cyan
[tagcyan]: # (cyan)
Built in `Color` object for use with the `graphics` and `plot` modules.

## Magenta
[tagmagenta]: # (magenta)
Built in `Color` object for use with the `graphics` and `plot` modules.

## Yellow
[tagyellow]: # (yellow)
Built in `Color` object for use with the `graphics` and `plot` modules.

## Brown
[tagbrown]: # (brown)
Built in `Color` object for use with the `graphics` and `plot` modules.

## Orange
[tagorange]: # (orange)
Built in `Color` object for use with the `graphics` and `plot` modules.

## Pink
[tagpink]: # (pink)
Built in `Color` object for use with the `graphics` and `plot` modules.

## Purple
[tagpurple]: # (purple)
Built in `Color` object for use with the `graphics` and `plot` modules.

## Gray
[taggray]: # (gray)

`Gray(x)` constructs a gray Color. The parameter `x` runs from 0 (black) to 1 (white). A second argument, if provided, sets the opacity:

    var c = Gray(0.2)    // 20% gray
    var c = Gray(0.2, 0.5) // 20% gray that is 50% opaque

## HSV
[taghsv]: # (hsv)

`HSV` builds a Color from hue, saturation, and value parameters. Hue is in degrees. Saturation and value run from 0 to 1. An optional fourth number sets the opacity.

    var c = HSV(200, 0.8, 1)

## ColorTable
[tagcolortable]: # (colortable)

A `ColorTable` assigns a different color to each point. You can build a `ColorTable` using a `Matrix`: each column represents one color. Use 3 rows for red, green, and blue, or 4 rows to include transparency:

    var cols = Matrix( ( (1, 0),
                         (0, 1),
                         (0, 0) ) )
    var colors = ColorTable(cols) // first point red, second point green
    var a = colors[0]             // the Color for point 0

## Colormap
[tagcolormap]: # (colormap)

A `ColorMap` converts a parameter, usually between 0 and 1, into a `Color`. `Color`s and `ColorMap`s provide many similar methods such as `rgb`, `opacity`, `lighten`, and `darken`.

Get one component, or all three:

    var col = HueMap()
    print col.red(0.5)
    print col.rgb(0)

Available color maps: `GradientMap`, `TundraMap`, `DivergingMap`, `CyclicMap`, `GrayMap`, `HueMap`, `ViridisMap`, `MagmaMap`, `InfernoMap`, and `PlasmaMap`.

## GradientMap
[taggradientmap]: # (gradientmap)

`GradientMap` blends between colors. Pass two colors, a list of colors, or a list of `(position, color)` pairs. Positions run from 0 to 1.

    var g = GradientMap(Blue, Red)
    var g = GradientMap([Blue, White, Red])
    var g = GradientMap([(0, Blue), (0.2, White), (1, Red)])

## TundraMap
[tagtundramap]: # (tundramap)

`TundraMap` fades from blue-gray through green to white.

## DivergingMap
[tagdivergingmap]: # (divergingmap)

`DivergingMap` is a three-color blend for data with a meaningful middle, such as positive and negative values.

    var d = DivergingMap(Blue, White, Red)

The middle color sits at 0.5. Pass a fourth number to move it, between 0 and 1. `BlueWhiteRedMap` is this blue-white-red map.

## CyclicMap
[tagcyclicmap]: # (cyclicmap)

`CyclicMap` repeats a list of colors. The last color blends back into the first.

    var c = CyclicMap([Red, Green, Blue])

## GrayMap
[taggraymap]: # (graymap)

`GrayMap` fades from black to white.

## HueMap
[taghuemap]: # (huemap)

`HueMap` cycles through vivid colors. It repeats on the interval 0 to 1.

## ViridisMap
[tagviridismap]: # (viridismap)

`ViridisMap` is a `Colormap` that displays a purple-green-yellow sequence.
It is perceptually uniform and intended to improve the accessibility of visualizations for viewers with color vision deficiency.

## MagmaMap
[tagmagmamap]: # (magmamap)

`MagmaMap` fades from black through purple to yellow.

## InfernoMap
[taginfernomap]: # (infernomap)

`InfernoMap` is a `Colormap` that displays a black-red-yellow sequence.
It is perceptually uniform and intended to improve the accessibility of visualizations for viewers with color vision deficiency.

## PlasmaMap
[tagplasmamap]: # (plasmamap)

`PlasmaMap` fades from blue through purple to yellow.
