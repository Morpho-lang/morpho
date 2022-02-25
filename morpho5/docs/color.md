[comment]: # (Color module help)
[version]: # (0.5)

# Color
[tagcolor]: # (color)

The `color` module provides support for working with color. Colors are represented in morpho by `Color` objects. The module predefines some colors including `Red`, `Green`, `Blue`, `Black`, `White`.

To use the module, use import as usual:

    import color

Create a Color object from an RGB pair:

    var col = Color(0.5,0.5,0.5) // A 50% gray

[showsubtopics]: # (subtopics)

## Colormap
[tagcolormap]: # (colormap)
The `color` module provides `ColorMap`s which are subclasses of `Color` that map a single parameter in the range 0..1 onto a continuum of colors. These include `GradientMap`,  `GrayMap` and `HueMap`. `Color`s and `Colormap`s have the same interface.

Get the red, green or blue components of a color or colormap:

    var col = HueMap()
    print col.red(0.5) // argument can be in range 0..1

Get all three components as a list:

    col.rgb(0)

Create a grayscale:

    var c = Gray(0.2) // 20% gray

## RGB
[tagrgb]: # (rgb)

Gets the rgb components of a `Color` or `ColorMap` object as a list. Takes a single argument in the range 0..1, although the result will only depend on this argument if the object is a `ColorMap`. 

    var col = Color(0.1,0.5,0.7)
    print col.rgb(0)
