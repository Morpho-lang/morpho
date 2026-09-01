[comment]: # (Field class help)
[version]: # (0.5)

# Field
[tagfield]: # (Field)

Fields are used to store information, including numbers or matrices, associated with the elements of a `Mesh` object. 

You can create a `Field` by applying a function,

    var f = Field(mesh, fn (x, y, z) x+y+z)

or by supplying a single constant value,

    var f = Field(mesh, Matrix([1,0,0]))

Fields can then be added and subtracted using the `+` and `-` operators, including elementwise addition or subtraction of a number.

To access elements of a `Field`, use index notation:

    print f[id] // Prints element id from lowest active grade
    print f[g, id] // Prints element id on grade `g`
    print f[g, id, index] // Prints quantity `index` on element `id` on grade `g`

Fields are associated with a `FiniteElementSpace` to allow calculus operations, including integration, local derivatives, etc. Unless another `FiniteElementSpace` is specified, the default is piecewise-linear (`CG1`) with values defined on vertices.

Create a `Field` with a specified `FiniteElementSpace`:

    var f = Field(mesh, fn (x, y, z) x+y+z, finiteelementspace=FiniteElementSpace("CG2"))

Create a piecewise constant (`CG0`) `Field` just by specifying a grade:

    var f = Field(mesh, 1, grade=1) // Field is defined on line elements

Create a `Field` with no `FiniteElementSpace` attached, i.e. a raw container:

    var f = Field(mesh, fn (x, y, z) x+y+z, finiteelementspace=nil)

[showsubtopics]: # (subtopics)

## Mesh
[tagmesh]: # (mesh)

Returns the Mesh associated with a Field object:

    print f.mesh() 

## Grade
[taggrade]: # (grade)

An integer `grade=N` with `N>=1` creates a piecewise-constant `CG0` field on that grade:

    var f = Field(mesh, 0, grade=2)

Each facet then stores one value, initialized to `0`. `grade=0` is the same as the default `CG1` vertex field.

A function passed with `grade=N` is sampled at the nodes of that space (the element centroid for `CG0`), not at the mesh vertices.

You can store more than one item per element by supplying a list to the `grade` option indicating how many items you want to store on each grade. For example,

    var f = Field(mesh, 1.0, grade=[0,2,1])

stores two numbers on the line (grade 1) elements and one number on the facets (grade 2) elements. Each number in the field is initialized to the value `1.0`. A list also opts out of a finite element space.

## Shape
[tagshape]: # (shape)

The `shape` method returns a tuple indicating the number of items stored on each element of a particular grade. This has the same format as the sequence you supply to the `grade` option of the `Field` constructor. For example,

    (1, 0, 2)

would indicate one item stored on each vertex and two items stored on each facet.

## FiniteElementSpace
[tagfiniteelementspace]: # (finiteelementspace)

Returns the `FiniteElementSpace` used to discretize the field, or `nil` if the field is a raw container:

    var fs = f.finiteElementSpace()
    print fs.grade()

See also `FiniteElementSpace`.

## Prototype
[tagprototype]: # (prototype)

Returns the prototype value used by the field:

    print f.prototype()

## EvalElement
[tagevalelement]: # (evalelement)

Evaluates a field inside a specific element using barycentric coordinates:

    print f.evalElement(el, [0.2, 0.3, 0.5])

You can supply the barycentric coordinates either as a `List` or as a column `Matrix`.

The number of barycentric coordinates should match the number of vertices of the reference element, i.e. `grade+1`.

For example:

    var val = f.evalElement(el, [0.2, 0.3, 0.5])
    print val

## ElementDofs
[tagelementdofs]: # (elementdofs)

Returns a list describing which field entries contribute to a given element. Each entry is a tuple of the form `(grade, element id, index)`:

    print f.elementDofs(el)

A typical return value might look like

    [ (0, 3, 0), (0, 7, 0), (0, 8, 0) ]

for a linear field on a triangular element.

For example:

    var dofs = f.elementDofs(el)
    print dofs

## Norm
[tagnorm]: # (norm)

Returns the Frobenius norm of the values stored in the field:

    print f.norm()

This is the same as `f.linearize().norm()`.

## Linearize
[taglinearize]: # (linearize)

Returns a matrix containing the data stored by the field:

    var mat = f.linearize()
    print mat

## __linearize
[tagxlinearize]: # (__linearize)

Returns the underlying storage matrix directly.

This method is intended for low-level use:

    var mat = f.__linearize()

## Op
[tagop]: # (op)

The `op` method applies a function to every item stored in a `Field`, returning the result as elements of a new `Field` object. For example,

    f.op(fn (x) x.norm())

calls the `norm` method on each element stored in `f`.

Additional `Field` objects may be supplied as extra arguments to `op`. These must have the same shape (the same number of items stored on each grade). The function supplied to `op` will now be called with the corresponding element from each field as arguments. For example,

    f.op(fn (x,y) x.inner(y), g)

calculates an elementwise inner product between the elements of Fields `f` and `g`.
