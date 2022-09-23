[comment]: # (Color module help)
[version]: # (0.5)

# Color
[tagcolor]: # (color)

The `color` module provides support for working with color. Colors are represented in morpho by `Color` objects. The module predefines some colors including `Red`, `Green`, `Blue`, `Black`, `White`.

To use the module, use import as usual:

    import color

Create a Color object from an RGB pair:

    var col = Color(0.5,0.5,0.5) // A 50% gray

The `color` module also provides `ColorMap`s, which are give a sequence of colors as a function of a parameter; these are useful for plotting the values of a `Field` for example.

[showsubtopics]: # (subtopics)

## RGB
[tagrgb]: # (rgb)

Gets the rgb components of a `Color` or `ColorMap` object as a list. Takes a single argument in the range 0..1, although the result will only depend on this argument if the object is a `ColorMap`.

    var col = Color(0.1,0.5,0.7)
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

## Colormap
[tagcolormap]: # (colormap)
The `color` module provides `ColorMap`s which are subclasses of `Color` that map a single parameter in the range 0..1 onto a continuum of colors. `Color`s and `Colormap`s have the same interface.

Get the red, green or blue components of a color or colormap:

    var col = HueMap()
    print col.red(0.5) // argument can be in range 0..1

Get all three components as a list:

    col.rgb(0)

Create a grayscale:

    var c = Gray(0.2) // 20% gray

Available ColorMaps: `GradientMap`,  `GrayMap`, `HueMap`, `ViridisMap`, `MagmaMap`, `InfernoMap` and `PlasmaMap`.

## GradientMap
[taggradientmap]: # (gradientmap)

`GradientMap` is a `Colormap` that displays a white-green-purple sequence.

## GrayMap
[taggraymap]: # (graymap)

`GrayMap` is a `Colormap` that displays grayscales.

## HueMap
[taghuemap]: # (huemap)

`HueMap` is a `Colormap` that displays vivid colors. It is periodic on the interval 0..1.

## ViridisMap
[tagviridismap]: # (viridismap)

`ViridisMap` is a `Colormap` that displays a purple-green-yellow sequence.
It is perceptually uniform and intended to be improve the accessibility of visualizations for viewers with color vision deficiency.

## MagmaMap
[tagmagmamap]: # (magmamap)

`MagmaMap` is a `Colormap` that displays a black-red-yellow sequence.
It is perceptually uniform and intended to be improve the accessibility of visualizations for viewers with color vision deficiency.

## InfernoMap
[taginfernomap]: # (infernomap)

`InfernoMap` is a `Colormap` that displays a black-red-yellow sequence.
It is perceptually uniform and intended to be improve the accessibility of visualizations for viewers with color vision deficiency.

## PlasmaMap
[tagplasmamap]: # (plasmamap)

`InfernoMap` is a `Colormap` that displays a blue-red-yellow sequence. It is perceptually uniform and intended to be improve the accessibility of visualizations for viewers with color vision deficiency.
