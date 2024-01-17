[comment]: # (Tuple class help)
[version]: # (0.6.0)

# Tuple
[tagtuple]: # (Tuple)

Tuples are collection objects that contain a sequence of values each associated with an integer index. Unlike Lists, they can't be changed after creation.

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

Join two lists together:

    var t1 = (1,2,3), t2 = (4, 5, 6)
    print t1.join(t2) // expect: (1,2,3,4,5,6)
