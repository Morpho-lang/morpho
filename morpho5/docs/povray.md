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

This also creates the .png file for the scene.

The `POVRaytracer` constructor supports an optional `camera` argument:

* `camera` - a `Camera` object (see below / help) containing the
  settings for the povray camera.

The `Camera` object can be initialized as follows:

    var camera = Camera()

This object contains the default settings of the camera, which can be
changed using the following optional arguments, or by just setting the
attributes after instantiation:

* `antialias` - whether to antialias the output or not (`true` by default)
* `width` - image width (`2048` by default)
* `height` - image height (`1536` by default)
* `viewangle` - camera angle (higher means wider view) (`24` by default)
* `viewpoint` - position of camera (`Matrix([0,0,-5])` by default)
* `look_at` - coordinate to look at (`Matrix([0,0,0])` by defualt)
* `sky` - orientation pointing to the sky (`Matrix([0,1,0])` by default)

The default settings generate a reasonable centered view of the x-y
plane.

These attributes can also be set directly for the `POVRaytracer` object:

    pov.look_at = Matrix([0,0,1])

The `render` method supports two optional boolean arguments:

* `quiet` - whether to suppress the parser and render statistics from `povray` or not (`false` by default)
* `display` - whether to turn on the graphic display while rendering or not (`true` by default) 

# Camera
[tagpovray]: # (camera)

The `Camera` object can be initialized as follows:

    var camera = Camera()

This object contains the default settings of the camera, which can be
changed using the following optional arguments, or by just setting the
attributes after instantiation:

* `antialias` - whether to antialias the output or not (`true` by default)
* `width` - image width (`2048` by default)
* `height` - image height (`1536` by default)
* `viewangle` - camera angle (higher means wider view) (`24` by default)
* `viewpoint` - position of camera (`Matrix([0,0,-5])` by default)
* `look_at` - coordinate to look at (`Matrix([0,0,0])` by defualt)
* `sky` - orientation pointing to the sky (`Matrix([0,1,0])` by default)

    camera.sky = Matrix([0,0,1])

The default settings generate a reasonable centered view of the x-y
plane.
