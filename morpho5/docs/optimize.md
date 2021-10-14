[comment]: # (Morpho optimize help file)
[version]: # (0.5)

# Optimize
[tagoptimize]: # (optimize)

The `optimize` package contains a number of functions and classes to perform shape optimization.

[showsubtopics]: # (subtopics)

## OptimizationProblem
[tagoptimizationproblem]: # (optimizationproblem)

An `OptimizationProblem` object defines an optimization problem, which may include functionals to optimize as well as global and local constraints.

Create an `OptimizationProblem` with a mesh:

    var problem = OptimizationProblem(mesh)

Add an energy:

    var la = Area()
    problem.addenergy(la)

Add an energy that operates on a selected region, and with an optional prefactor:

    problem.addenergy(la, selection=sel, prefactor=2)

Add a constraint:

    problem.addconstraint(la)

Add a local constraint (here a onesided level set constraint):

    var ls = ScalarPotential(fn (x,y,z) z, fn (x,y,z) Matrix([0,0,1]))
    problem.addlocalconstraint(ls, onesided=true)

## Optimizer
[tagoptimizer]: # (optimizer)

`Optimizer` objects are used to optimize `Mesh`es and `Field`s. You should use the appropriate subclass: `ShapeOptimizer` or `FieldOptimizer` respectively.

[showsubtopics]: # (subtopics)

## ShapeOptimizer
[tagshapeoptimizer]: # (shapeoptimizer)

A `ShapeOptimizer` object performs shape optimization: it moves the vertex positions to reduce an overall energy.

Create a `ShapeOptimizer` object with an `OptimizationProblem` and a `Mesh`:

    var sopt = ShapeOptimizer(problem, m)

Take a step down the gradient with fixed stepsize:

    sopt.relax(5) // Takes five steps

Linesearch down the gradient:

    sopt.linesearch(5) // Performs five linesearches

Control a number of properties of the optimizer:

    sopt.stepsize=0.1 // The stepsize to take
    sopt.steplimit=0.5 // Maximum stepsize for optimizing methods
    sopt.etol = 1e-8 // Energy convergence tolerance 
    sopt.ctol = 1e-9 // Tolerance to which constraints are satisfied
    sopt.maxconstraintsteps = 20 // Maximum number of constraint steps to use

## FieldOptimizer
[tagfieldoptimizer]: # (fieldoptimizer)

A `FieldOptimizer` object performs field optimization: it changes elements of a `Field` to reduce an overall energy.

Create a `FieldOptimizer` object with an `OptimizationProblem` and a `Field`:

    var sopt = FieldOptimizer(problem, fld)

Field optimizers provide the same options and methods as Shape optimizers: see the `ShapeOptimizer` documentation for details.
