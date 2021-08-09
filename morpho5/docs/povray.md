[comment]: # (Povray module help)
[version]: # (0.5)

# POVRay
[tagpovray]: # (povray)

The `povray` module provides integration with POVRay, a popular open source ray-tracing package for high quality graphical rendering. To use the module, first import it:

    import povray

To raytrace a graphic, begin by creating a `POVRaytracer` object:

    var pov = POVRaytracer(graphic)

Create a .pov file that can be run with POVRay:

    pov.write("out.pov")

Create, render and display a scene using POVRay:

    pov.render("out.pov")

The `POVRaytracer` constructor supports a number of optional arguments:

* `antialias` - whether to antialias the output or not
* `width` - image width
* `height` - image height
* `viewangle` - camera angle (higher means wider view)
* `viewpoint` - position of camera
