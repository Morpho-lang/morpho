[comment]: # (Field class help)
[version]: # (0.5)

# Field
[tagfield]: # (Field)

Fields are used to store information, including numbers or matrices, associated with the elements of a `Mesh` object.

You can create a `Field` by applying a function to each of the vertices,

    var f = Field(mesh, fn (x, y, z) x+y+z)

or by supplying a single constant value,

    var f = Field(mesh, Matrix([1,0,0]))

Fields can then be added and subtracted using the `+` and `-` operators.

To access elements of a `Field`, use index notation:

    print f[grade, element, index]

where
* `grade` is the grade to select
* `element` is the element id
* `index` is the element index

As a shorthand, it's possible to omit the grade and index; these are then both assumed to be `0`:

    print f[2]

[showsubtopics]: # (subtopics)

## Grade
[taggrade]: # (grade)

To create fields that include grades other than just vertices, use the `grade` option to `Field`. This can be just a grade index,

    var f = Field(mesh, 0, grade=2)

which creates an empty field with `0` for each of the facets of the mesh `mesh`.

You can store more than one item per element by supplying a list to the `grade` option indicating how many items you want to store on each grade. For example,

    var f = Field(mesh, 1.0, grade=[0,2,1])

stores two numbers on the line (grade 1) elements and one number on the facets (grade 2) elements. Each number in the field is initialized to the value `1.0`.

## Shape
[tagshape]: # (shape)

The `shape` method returns a list indicating the number of items stored on each element of a particular grade. This has the same format as the list you supply to the `grade` option of the `Field` constructor. For example,

    [1,0,2]

would indicate one item stored on each vertex and two items stored on each facet.

## Op
[tagop]: # (op)

The `op` method applies a function to every item stored in a `Field`, returning the result as elements of a new `Field` object. For example,

    f.op(fn (x) x.norm())

calls the `norm` method on each element stored in `f`.

Additional `Field` objects may be supplied as extra arguments to `op`. These must have the same shape (the same number of items stored on each grade). The function supplied to `op` will now be called with the corresponding element from each field as arguments. For example,

    f.op(fn (x,y) x.inner(y), g)

calculates an elementwise inner product between the elements of Fields `f` and `g`.
