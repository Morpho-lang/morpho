[comment]: # (Functionals help)
[version]: # (0.5)

# Functionals
[tagfunctionals]: # (functionals)

A number of `functionals` are available in Morpho. Each of these represents an integral over some `Mesh` and `Field` objects (on a particular `Selection`) and are used to define energies and constraints in an `OptimizationProblem` provided by the `optimize` module.

Many functionals are built in. Additional functionals are available by importing the `functionals` module:

    import `functionals`

Functionals provide a number of standard methods:

* `total`(mesh) - returns the value of the integral with a provided mesh, selection and fields
* `integrand`(mesh) - returns the contribution to the integral from each element
* `gradient`(mesh) - returns the gradient of the functional with respect to vertex motions.
* `fieldgradient`(mesh, field) - returns the gradient of the functional with respect to components of the field

Each of these may be called with a mesh, a field and a selection.

[showsubtopics]: # (subtopics)

## Length
[taglength]: # (length)

A `Length` functional calculates the length of a line elements in a mesh.

Evaluate the length of a circular loop:

    import constants
    import meshtools
    var m = LineMesh(fn (t) [cos(t), sin(t), 0], 0...2*Pi:Pi/20, closed=true)
    var le = Length()
    print le.total(m)

## AreaEnclosed
[tagareaenclosed]: # (areaenclosed)

An `AreaEnclosed` functional calculates the area enclosed by a loop of line elements.

    var la = AreaEnclosed()

## Area
[tagarea]: # (area)

An `Area` functional calculates the area of the area elements in a mesh:

    var la = Area()
    print la.total(mesh)

## VolumeEnclosed
[tagvolumeenclosed]: # (volumeenclosed)

An `VolumeEnclosed` functional calculates the volume enclosed by a surface of line elements. Note that this estimate may be inaccurate for highly deformed surfaces.

    var lv = VolumeEnclosed()

## Volume
[tagvolume]: # (volume)

An `Volume` functional calculates the volume of volume elements.

    var lv = Volume()

## ScalarPotential
[tagscalarpotential]: # (scalarpotential)

The `ScalarPotential` functional is applied to point elements.

    var ls = ScalarPotential(potential, gradient)

You must supply two functions (which may be anonymous) that return the potential and gradient respectively.

This functional is often used to implement level set constraints. For example, to confine a set of points to a sphere:

    import optimize
    fn sphere(x,y,z) { return x^2+y^2+z^2-1 }
    fn grad(x,y,z) { return Matrix([2*x, 2*y, 2*z]) }
    var lsph = ScalarPotential(sphere, grad)
    problem.addlocalconstraint(lsph)

See the thomson example.

## LinearElasticity
[taglinearelasticity]: # (linearelasticity)

The `LinearElasticity` functional measures the linear elastic energy away from a reference state.

You must initialize with a reference mesh:

    var le = LinearElasticity(mref)

Manually set the poisson's ratio and grade to operate on:

    le.poissonratio = 0.2
    le.grade = 2

## EquiElement
[tagequielement]: # (equielement)

The `EquiElement` functional measures the discrepency between the size of elements adjacent to each vertex. It can be used to equalize elements for regularization purposes.

## LineCurvatureSq
[taglinecurvaturesq]: # (linecurvaturesq)

The `LineCurvatureSq` functional measures the integrated curvature squared of a sequence of line elements.

## LineTorsionSq
[taglinetorsionsq]: # (linetorsionsq)

The `LineTorsionSq` functional measures the integrated torsion squared of a sequence of line elements.

## MeanCurvatureSq
[tagmeancurvsq]: # (meancurvaturesq)

The `MeanCurvatureSq` functional computes the integrated mean curvature over a surface.

## GaussCurvature
[taggausscurv]: # (gausscurvature)

The `GaussCurvature` computes the integrated gaussian curvature over a surface.

## GradSq
[taggradsq]: # (gradsq)

The `GradSq` functional measures the integral of the gradient squared of a field. The field can be a scalar, vector or matrix function.

Initialize with the required field:

    var le=GradSq(phi)

## Nematic
[tagnematic]: # (nematic)

The `Nematic` functional measures the elastic energy of a nematic liquid crystal.

    var lf=Nematic(nn)

There are a number of optional parameters that can be used to set the splay, twist and bend constants:

    var lf=Nematic(nn, ksplay=1, ktwist=0.5, kbend=1.5, pitch=0.1)

These are stored as properties of the object and can be retrieved as follows:

    print lf.ksplay

## NematicElectric
[tagnematic]: # (nematic)

The `NematicElectric` functional measures the integral of a nematic and electric coupling term integral((n.E)^2) where the electric field E may be computed from a scalar potential or supplied as a vector.

Initialize with a director field `nn` and a scalar potential `phi`:
    var lne = NematicElectric(nn, phi)

## NormSq
[tagnormsq]: # (normsq)

The `NormSq` functional measures the elementwise L2 norm squared of a field.

## LineIntegral
[taglineintegral]: # (lineintegral)

The `LineIntegral` functional computes the line integral of a function. You supply an integrand function that takes a position matrix as an argument.

To compute `integral(x^2+y^2)` over a line element:

    var la=LineIntegral(fn (x) x[0]^2+x[1]^2)

The function `tangent()` returns a unit vector tangent to the current element:

    var la=LineIntegral(fn (x) x.inner(tangent()))

You can also integrate functions that involve fields:

    var la=LineIntegral(fn (x, n) n.inner(tangent()), n)

where `n` is a vector field. The local interpolated value of this field is passed to your integrand function. More than one field can be used; they are passed as arguments to the integrand function in the order you supply them to `LineIntegrand`.

## AreaIntegral
[tagareaintegral]: # (areaintegral)

The `AreaIntegral` functional computes the area integral of a function. You supply an integrand function that takes a position matrix as an argument.

To compute integral(x*y) over an area element:

    var la=AreaIntegral(fn (x) x[0]*x[1])

You can also integrate functions that involve fields:

    var la=AreaIntegral(fn (x, phi) phi^2, phi)

More than one field can be used; they are passed as arguments to the integrand function in the order you supply them to `AreaIntegrand`.

## FloryHuggins
[tagfloryhuggins]: # (floryhuggins)

The `FloryHuggins` functional computes the Flory-Huggins mixing energy over an element:

    a*phi*log(phi) + b*(1-phi)+log(1-phi) + c*phi*(1-phi)

where a, b and c are parameters you can supply. The value of phi is calculated from a reference mesh that you provide on initializing the Functional: 

    var lfh = FloryHuggins(mref)

Manually set the coefficients and grade to operate on:

    lfh.a = 1; lfh.b = 1; lfh.c = 1;
    lfh.grade = 2
