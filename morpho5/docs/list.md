[comment]: # (List class help)
[version]: # (0.5)

# List
[taglist]: # (List)

Lists are collection objects that contain a sequence of values each associated with an integer index.

Create a list like this:

    var list = [1, 2, 3]

Lookup values using index notation:

    list[0]

You can change list entries like this:

    list[0] = "Hello"

Create an empty list:

    var list = []

Loop over elements of a list:

    for (i in list) print i

[showsubtopics]: # (subtopics)

## Append
[tagappend]: # (Append)

Adds an element to the end of a list:

    list = []
    list.append("Foo")

## Pop
[tagpop]: # (pop)

Remove the last element from a list, returning the element removed:

    print list.pop()

## Sort
[tagsort]: # (sort)

Sorts a list:

    list.sort()

You can provide your own function to use to compare values in the list

    list.sort(fn (a, b) a-b)

This function should return a negative value if `a<b`, a positive value if `a>b` and `0` if `a` and `b` are equal.

## Order
[tagorder]: # (order)

Returns a list of indices that would, if used in order, would sort a list. For example

    var list = [2,3,1]
    print list.order() // expect: [2,0,1]

would produce `[2,0,1]`

## Remove
[tagremove]: # (remove)

Remove any occurrences of a value from a list:

    var list = [1,2,3]
    list.remove(1)

## ismember
[tagismember]: # (ismember)

Tests if a value is a member of a list:

    var list = [1,2,3]
    print list.ismember(1) // expect: true

## Add
[tagadd]: # (add)

Join two lists together:

    var l1 = [1,2,3], l2 = [4, 5, 6]
    print l1+l2 // expect: [1,2,3,4,5,6]
