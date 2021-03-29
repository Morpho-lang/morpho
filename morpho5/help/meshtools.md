[comment]: # Morpho meshtools help file
[version]: # 0.5

#linemesh
[tag]: # linemesh

This function creates a mesh composed of line elements from a parametric function. To use it:

    var m = linemesh(function, range, closed=boolean)

where

  * function is a parameteric function that has one parameter. It should return a list of coordinates or a column matrix corresponding to this parameter.
  * range is the Range to use for the parametric function.
  * closed is an optional parameter indicating whether to create a closed loop or not.

To use `linemesh`, import the `meshtools` module:

    import meshtools.

Create a circle:

    import constants 
    var m = linemesh(fn (t) [sin(t), cos(t), 0], 0...2*Pi:2*Pi/50, closed=true)

#polyhedronmesh
[tag]: # polyhedronmesh


#plotmesh
[tag]: # plotmesh
