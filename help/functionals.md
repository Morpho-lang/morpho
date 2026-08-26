[comment]: # (Functionals help)
[version]: # (0.5)

# Functionals
[tagfunctionals]: # (functionals)

A number of `functionals` are available in Morpho. Each of these represents an integral over some `Mesh` and `Field` objects (on a particular `Selection`) and are used to define energies and constraints in an `OptimizationProblem` provided by the `optimize` module.

Many functionals are built in. Additional functionals are available by importing the `functionals` module:

    import functionals

Functionals provide a number of standard methods:

* `total`(mesh) — value of the integral; also `total`(mesh, selection)
* `integrand`(mesh) — contribution from each element as a `Matrix`; also `integrand`(mesh, selection)
* `integrand`(mesh, id) — contribution from a single element as a `Float`; `integrand`(mesh, grade, id) sets the grade explicitly
* `gradient`(mesh) — derivative with respect to vertex motion; also `gradient`(mesh, selection)
* `hessian`(mesh) — sparse second derivative with respect to vertex motion, where provided; also `hessian`(mesh, selection)
* `fieldgradient`(field) — derivative with respect to field values, where provided

[showsubtopics]: # (subtopics)

## Total
[tagtotal]: # (total)

The `total` method returns the value of a functional:

    print fnl.total(mesh)
    print fnl.total(mesh, selection)

## Integrand
[tagintegrand]: # (integrand)

The `integrand` method returns the contribution from each element:

    print fnl.integrand(mesh)
    print fnl.integrand(mesh, selection)

Evaluate a single element with the functional's default grade, or choose an explicit grade:

    print fnl.integrand(mesh, id)
    print fnl.integrand(mesh, grade, id)

## Gradient
[taggradient]: # (gradient)

The `gradient` method returns the derivative of a functional with respect to vertex motion:

    print fnl.gradient(mesh)
    print fnl.gradient(mesh, selection)

## Fieldgradient
[tagfieldgradient]: # (fieldgradient)

Functionals that depend on a field may provide a `fieldgradient` method that returns the derivative with respect to field values:

    print fnl.fieldgradient(f)
    print fnl.fieldgradient(f, mesh)

If the functional stores more than one field (for example `NematicElectric` or a `LineIntegral` with several fields), pass the field you want the derivative with respect to.

## Hessian
[taghessian]: # (hessian)

Some functionals provide a `hessian` method:

    print fnl.hessian(mesh)
    print fnl.hessian(mesh, selection)

For example, a typical workflow for a field-dependent functional is

    var value = fnl.total(mesh)
    var gradx = fnl.gradient(mesh)
    var gradf = fnl.fieldgradient(f)

where `value` is the functional value, `gradx` is the derivative with respect to vertex positions and `gradf` is the derivative with respect to field values.

For example, for a field-dependent functional:

    var fnl = GradSq(phi)
    print fnl.total(mesh)
    print fnl.gradient(mesh)
    print fnl.fieldgradient(phi)

## Length
[taglength]: # (length)

A `Length` functional calculates the length of a line element in a mesh.

Evaluate the length of a circular loop:

    import constants
    import meshtools
    var m = LineMesh(fn (t) [cos(t), sin(t), 0], 0...2*Pi:Pi/20, closed=true)
    var le = Length()
    print le.total(m)

See the `Functionals` entry for general information about functionals.

## AreaEnclosed
[tagareaenclosed]: # (areaenclosed)

An `AreaEnclosed` functional calculates the area enclosed by a loop of line elements.

    var la = AreaEnclosed()

Evaluate the area enclosed of a circular loop:

    import constants
    import meshtools
    var m = LineMesh(fn (t) [cos(t), sin(t), 0], 0...2*Pi:Pi/20, closed=true)
    var larea = AreaEnclosed()
    print larea.total(m)

`AreaEnclosed` sums unsigned triangle areas from the origin to each edge. Oriented meshes are not yet supported, so a non-convex loop may give an incorrect result.

See the `Functionals` entry for general information about functionals.

## Area
[tagarea]: # (area)

An `Area` functional calculates the area of the area elements in a mesh:

    var la = Area()
    print la.total(mesh)

See the `Functionals` entry for general information about functionals.

## VolumeEnclosed
[tagvolumeenclosed]: # (volumeenclosed)

A `VolumeEnclosed` functional is used to calculate the volume enclosed by a surface. Note that this estimate may become inaccurate for highly deformed surfaces.

    var lv = VolumeEnclosed()
    print lv.total(mesh)

See the `Functionals` entry for general information about functionals.

## Volume
[tagvolume]: # (volume)

A `Volume` functional calculates the volume of volume elements.

    var lv = Volume()

See the `Functionals` entry for general information about functionals.

## ScalarPotential
[tagscalarpotential]: # (scalarpotential)

The `ScalarPotential` functional is applied to point elements.

    var ls = ScalarPotential(potential)

You must supply a function (which may be anonymous) that returns the potential. You may optionally provide a function that returns the gradient as well at initialization:

    var ls = ScalarPotential(potential, gradient)

This functional is often used to constrain the mesh to the level set of a function. For example, to confine a set of points to a sphere:

    import optimize
    fn sphere(x,y,z) { return x^2+y^2+z^2-1 }
    fn grad(x,y,z) { return Matrix([2*x, 2*y, 2*z]) }
    var lsph = ScalarPotential(sphere, grad)
    problem.addlocalconstraint(lsph)

See the thomson example for use of this technique.

See the `Functionals` entry for general information about functionals.

## LinearElasticity
[taglinearelasticity]: # (linearelasticity)

The `LinearElasticity` functional measures the linear elastic energy away from a reference state. 

You must initialize with a reference mesh:

    var le = LinearElasticity(mref)

Manually set the poisson's ratio and grade to operate on:

    le.poissonratio = 0.2
    le.grade = 2

The energy for each element in the Mesh is computed as follows: First the Gram matrix `S` is computed for the element as well as the Gram matrix `F` for the corresponding element in the reference Mesh. These quantities are used to compute the Cauchy-Green strain tensor:

    C = (F S^-1 - I)/2

The energy density is then: 

    mu*Tr(C^2) + 1/2*lambda*Tr(C)^2

where mu and lambda are the Lamé parameters. The total energy is found by multiplying the energy density by the volume or area of the element as appropriate. 

See the `Functionals` entry for general information about functionals.

## EquiElement
[tagequielement]: # (equielement)

The `EquiElement` functional measures the discrepency between the size of elements adjacent to each vertex. It can be used to equalize elements for regularization purposes.

See the `Functionals` entry for general information about functionals.

## LineCurvatureSq
[taglinecurvaturesq]: # (linecurvaturesq)

The `LineCurvatureSq` functional measures the integrated curvature squared of a sequence of line elements.

Compute the total squared curvature of a loop:

    import constants
    import meshtools
    var m = LineMesh(fn (t) [cos(t), sin(t), 0], 0...2*Pi:Pi/20, closed=true)
    var larea = LineCurvatureSq()
    print larea.total(m)

See the `Functionals` entry for general information about functionals.

## LineTorsionSq
[taglinetorsionsq]: # (linetorsionsq)

The `LineTorsionSq` functional measures the integrated torsion squared of a sequence of line elements.

Compute the total squared torsion of a helix:

    import constants
    import meshtools
    var m = LineMesh(fn (t) [cos(t), sin(t), t], 0...2*Pi:Pi/20, closed=true)
    var larea = LineTorsionSq()
    print larea.total(m)

See the `Functionals` entry for general information about functionals.

## MeanCurvatureSq
[tagmeancurvsq]: # (meancurvaturesq)

The `MeanCurvatureSq` functional computes the integrated mean curvature over a surface.

Compute the integrated mean squared curvature of the unit sphere:

    import implicitmesh
    var impl = ImplicitMeshBuilder(fn (x,y,z) x^2+y^2+z^2-1)
    var mesh = impl.build(stepsize=0.25)
    var lmsq = MeanCurvatureSq() 
    print lmsq.total(mesh) 

See the `Functionals` entry for general information about functionals.

## GaussCurvature
[taggausscurv]: # (gausscurvature)

The `GaussCurvature` computes the integrated gaussian curvature over a surface.

Note that for surfaces with a boundary, the integrand is correct only for the interior points. To compute the geodesic curvature of the boundary in that case, you can set the optional flag `geodesic` to `true` and compute the total on the boundary selection.
Here is an example for a 2D disk mesh.

    var mesh = Mesh("disk.mesh")
    mesh.addgrade(1)

    var whole = Selection(mesh, fn(x,y,z) true)
    var bnd = Selection(mesh, boundary=true)
    var interior = whole.difference(bnd)

    var gauss = GaussCurvature()
    print gauss.total(mesh, interior) // expect: 0
    gauss.geodesic = true
    print gauss.total(mesh, bnd) // expect: 2*Pi

See the `Functionals` entry for general information about functionals.

## GradSq
[taggradsq]: # (gradsq)

The `GradSq` functional measures the integral of the gradient squared of a field. The field can be a scalar, vector or matrix function.

Initialize with the required field:

    var le=GradSq(phi)

Compute the integral of GradSq(phi):

    print le.total(mesh)
    print le.fieldgradient(phi)

See the `Functionals` entry for general information about functionals.

## Nematic
[tagnematic]: # (nematic)

The `Nematic` functional measures the elastic energy of a nematic liquid crystal.

    var lf=Nematic(nn)

There are a number of optional parameters that can be used to set the splay, twist and bend constants:

    var lf=Nematic(nn, ksplay=1, ktwist=0.5, kbend=1.5, pitch=0.1)

These are stored as properties of the object and can be retrieved as follows:

    print lf.ksplay

See the `Functionals` entry for general information about functionals.

## NematicElectric
[tagnematic]: # (nematic)

The `NematicElectric` functional measures the integral of a nematic and electric coupling term integral((n.E)^2) where the electric field E may be computed from a scalar potential or supplied as a vector.

Initialize with a director field `nn` and a scalar potential `phi`:

    var lne = NematicElectric(nn, phi)

Differentiate with respect to either stored field:

    print lne.fieldgradient(nn)
    print lne.fieldgradient(phi)

See the `Functionals` entry for general information about functionals.

## NormSq
[tagnormsq]: # (normsq)

The `NormSq` functional measures the elementwise L2 norm squared of a field.

See the `Functionals` entry for general information about functionals.

## Quadrature
[tagquadrature]: # (quadrature)

`LineIntegral`, `AreaIntegral` and `VolumeIntegral` accept an optional `method` dictionary that selects the quadrature rule and the adaptive stopping test:

    AreaIntegral(fn (x) x[0]*x[1], method={ })
    AreaIntegral(fn (x) x[0]*x[1], method={ "rule": "cubtri7" })
    AreaIntegral(fn (x) x[0]*x[1], method={ "rule": "tri4", "adapt": false })
    AreaIntegral(fn (x) x[0]*x[1], method={ "errornorm": "sum" })
    AreaIntegral(fn (x) x[0]*x[1], method={ "tol": 1e-8 })

The recognized keys are:

* `rule` — a `String` naming a quadrature rule, or `"hybrid2d"` for the default two-dimensional strategy. Unknown names raise `IntgrtrRlNtFnd`.
* `degree` — an `Int` requesting a rule of at least that degree when `rule` is omitted.
* `adapt` — a `Bool`. The default is `true`. With `adapt=false`, a named rule is evaluated once and no p- or h-refinement is done.
* `errornorm` — `"max"` (the default) or `"sum"`. Any other value, or a non-string, raises `IntgrtrMthdTyp`.
* `tol` — a `Float` (the default is `1e-6`). Integers are accepted. A non-numeric value raises `IntgrtrMthdTyp`.

With `errornorm: "max"`, h-refinement stops when the largest element error is below `tol` times the absolute value of the last root estimate. `"sum"` uses the older, more conservative test on the summed element errors. On a vertex singularity the true global error under `"max"` can sit a little above `tol`; use `"sum"` if you need the tighter bound.

Useful named rules include:

1D: `gauss1`/`kronrod3`, `gauss2`/`kronrod5`, `gauss5`/`kronrod11`, `gauss7`/`kronrod15` with `midpoint`/`simpson` for educational purposes.
2D: `tri4`, `tri10`, `tri20`, `cubtri7`, `cubtri19` and `cools7`/`cools16`.
3D: `keast4`, `keast5`, `tet5`, `tet6` and `grundmann3d0`–`grundmann3d5`.

## LineIntegral
[taglineintegral]: # (lineintegral)

The `LineIntegral` functional computes the line integral of a function. You supply an integrand function that takes a position matrix as an argument.

To compute `integral(x^2+y^2)` over a line element:

    var la=LineIntegral(fn (x) x[0]^2+x[1]^2)

The function `tangent()` returns a unit vector tangent to the current element:

    var la=LineIntegral(fn (x) x.inner(tangent()))

You can also integrate functions that involve fields:

    var la=LineIntegral(fn (x, n) n.inner(tangent()), n)

where `n` is a vector field. The local interpolated value of this field is passed to your integrand function. More than one field can be used; they are passed as arguments to the integrand function in the order you supply them to `LineIntegral`.

The gradient of a field is available within an integrand function using the `gradient()` function.

The field derivative of the integral is `fieldgradient(f)` for a field `f` supplied to the constructor.

An optional `method` dictionary selects the quadrature; see the `quadrature` help entry for further details.

See the `Functionals` entry for general information about functionals.

## AreaIntegral
[tagareaintegral]: # (areaintegral)

The `AreaIntegral` functional computes the area integral of a function. You supply an integrand function that takes a position matrix as an argument.

To compute `integral(x*y)` over an area element:

    var la=AreaIntegral(fn (x) x[0]*x[1])

You can also integrate functions that involve fields:

    var la=AreaIntegral(fn (x, phi) phi^2, phi)

The local facet normal can be accessed in an integrand using the `normal()` function:

    var la=AreaIntegral(fn (x) x.inner(normal())^2)

More than one field can be used; they are passed as arguments to the integrand function in the order you supply them to `AreaIntegral`.

The gradient of a field is available within an integrand function using the `gradient()` function.

An optional `method` dictionary selects the quadrature; see the `quadrature` help entry for further details.

See the `Functionals` entry for general information about functionals.

## VolumeIntegral
[tagvolumeintegral]: # (volumeintegral)

The `VolumeIntegral` functional computes the volume integral of a function. You supply an integrand function that takes a position matrix as an argument.

To compute integral(x*y*z) over an volume element:

    var la=VolumeIntegral(fn (x) x[0]*x[1]*x[2])

You can also integrate functions that involve fields:

    var la=VolumeIntegral(fn (x, phi) phi^2, phi)

More than one field can be used; they are passed as arguments to the integrand function in the order you supply them to `VolumeIntegral`.

The gradient of a field is available within an integrand function using the `gradient()` function.

An optional `method` dictionary selects the quadrature; see the `quadrature` help entry for further details.

See the `Functionals` entry for general information about functionals.

## Hydrogel
[taghydrogel]: # (hydrogel)

The `Hydrogel` functional computes the Flory-Rehner energy over an element:

    (a*phi*log(phi) + b*(1-phi)+log(1-phi) + c*phi*(1-phi))*V + 
    d*(log(phiref/phi)/3 - (phiref/phi)^(2/3) + 1)*V0

The first three terms come from the Flory-Huggins mixing energy, whereas
the fourth term proportional to d comes from the Flory-Rehner elastic
energy.

The value of phi is calculated from a reference mesh
that you provide on initializing the Functional: 

    var lfh = Hydrogel(mref)

Here, a, b, c, d and phiref are parameters you can supply (they are `nil`
by default), V is the current volume and V0 is the reference volume of a
given element. You also need to supply the initial value of phi, labeled
as phi0, which is assumed to be the same for all the elements. 
Manually set the coefficients and grade to operate on:

    lfh.a = 1; lfh.b = 1; lfh.c = 1; lfh.d = 1;
    lfh.grade = 2, lfh.phi0 = 0.5, lfh.phiref = 0.1

See the `Functionals` entry for general information about functionals.

## Jump
[tagjump]: # (jump)

The `Jump` functional computes an interface contribution over interior codimension-1 mesh elements. It evaluates your integrand on interfaces shared by exactly two parent elements and ignores boundary interfaces.

Initialize a jump functional with an integrand and any fields it depends on:

    var j = Jump(fn (x, phi) jumpdn(phi)^2, phi)

The integrand receives the interface position `x` followed by the interpolated field values in the order supplied to `Jump`.

Within a `Jump` integrand, the special function `jumpdn(field)` returns the jump in the normal derivative of a supplied field across the current interface.

The field derivative is `fieldgradient(phi)` for a field supplied to the constructor.

`Jump` also accepts a `method` dictionary. A `strategy` of `"centroid"` (the default) or `"quadrature"` selects how the interface is sampled. In `"quadrature"` mode the other keys are those documented under `quadrature`:

    var j = Jump(fn (x, phi) jumpdn(phi)^2, phi, method={ "strategy": "quadrature" })

See the `Functionals` entry for general information about functionals.
