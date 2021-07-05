[comment]: # (Morpho variables help file)
[version]: # (0.5)

[toplevel]: #

# Variables
[tagvar]: # (var)

Variables are defined using the `var` keyword followed by the variable name:

    var a

Optionally, an initial assignment may be given:

    var a = 1

Variables defined in a block of code are visible only within that block, so

    var greeting = "Hello"
    {
        var greeting = "Goodbye"
        print greeting
    }
    print greeting

will print

*Goodbye*
*Hello*

Multiple variables can be defined at once by separating them with commas

    var a, b=2, c[2]=[1,2]

where each can have its own initializer (or not).

## Indexing
[taglb]: # ([)
[tagrb]: # (])
[tagindex]: # (index)
[tagsub]: # (subscript)

Morpho provides a number of collection objects, such as `List`, `Range`, `Array`, `Dictionary`, `Matrix` and `Sparse`, that can contain more than one value. Index notation (sometimes called subscript notation) is used to access elements of these objects.

To retrieve an item from a collection, you use the `[` and `]` brackets like this:

    var a = List("Apple", "Bag", "Cat")
    print a[0]

which prints *Apple*. Note that the first element is accessed with `0` not `1`.

Similarly, to set an entry in a collection, use:

    a[0]="Adder"

which would replaces the first element in `a` with `"Adder"`.

Some collection objects need more than one index,

    var a = Matrix([[1,0],[0,1]])
    print a[0,0]

and others such as `Dictionary` use non-numerical indices,

    var b = Dictionary()
    b["Massachusetts"]="Boston"
    b["California"]="Sacramento"

as in this dictionary of state capitals.
