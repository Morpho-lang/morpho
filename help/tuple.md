[comment]: # (Tuple class help)
[version]: # (0.6.0)

# Tuple
[tagtuple]: # (Tuple)

Tuples are collection objects that contain a sequence of values each associated with an integer index. Unlike Lists, they can't be changed after creation, so they form Morpho's immutable sequence type.

Create a tuple like this:

    var tuple = (1, 2, 3)

Look up values using index notation:

    tuple[0]

Indexing can also be done with slices:

	tuple[0..2]

Loop over elements of a tuple:

    for (i in tuple) print i

[showsubtopics]: # (subtopics)

## ismember
[tagismember]: # (ismember)

Tests if a value is a member of a tuple:

    var tuple = (1,2,3)
    print tuple.ismember(1) // expect: true

## Join
[tagjoin]: # (join)

Join two tuples together:

    var t1 = (1,2,3), t2 = (4, 5, 6)
    print t1.join(t2) // expect: (1,2,3,4,5,6)

## Sort
[tagsort]: # (sort)

Sorts the contents of a tuple into ascending order, returning a new tuple:

    var tuple = (4,3,2,1)
    print tuple.sort() // prints (1, 2, 3, 4)

You can provide your own function to use to compare values in the tuple:

    tuple.sort(fn (a, b) a-b)

This function should return a negative value if `a<b`, a positive value if `a>b` and `0` if `a` and `b` are equal.

## Order
[tagorder]: # (order)

Returns a tuple of indices that would, if used in order, sort a tuple. For example

    var tuple = (2,3,1)
    print tuple.order() // prints (2,0,1)

would produce `(2,0,1)`.

## Reverse
[tagreverse]: # (reverse)

Returns a reversed copy of a tuple, leaving the original unchanged:

    var tuple = (1,2,3)
    print tuple.reverse() // prints (3,2,1)

## Roll
[tagroll]: # (roll)

Returns a copy of a tuple with its contents rolled by a specified number of positions, leaving the original unchanged:

    var tuple = (1,2,3)
    print tuple.roll(1)  // prints (3,1,2)
    print tuple.roll(-1) // prints (2,3,1)

## tostring
[tagtostring]: # (tostring)

Converts a tuple to a string:

    var tuple = (1,2,3)
    print tuple.tostring() // prints (1, 2, 3)
