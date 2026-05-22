[comment]: # (Sparse class help)
[version]: # (0.5)

# Sparse
[tagsparse]: # (Sparse)

The Sparse class provides support for sparse matrices. An empty sparse matrix can be initialized with a given size,

    var a = Sparse(nrows,ncols)

Alternatively, a matrix can be created from an array of triplets,  

    var a = Sparse([[row, col, value] ...])

For example,

    var a = Sparse([[0,0,2], [1,1,-2]])

creates the matrix

    [ 2 0 ]
    [ 0 -2 ]

Once a sparse matrix is created, you can use all the regular arithmetic operators with matrix operands, e.g.

    a+b
    a*b

[showsubtopics]: # (showsubtopics)

## Rowindices
[tagrowindices]: # (rowindices)

Returns the row indices of the nonzero entries in a specified column:

    print a.rowindices(0)

## Setrowindices
[tagsetrowindices]: # (setrowindices)

Replaces the row indices of the nonzero entries in a specified column:

    a.setrowindices(0, [0,2,4])

## Colindices
[tagcolindices]: # (colindices)

Returns the column indices that contain nonzero entries:

    print a.colindices()
