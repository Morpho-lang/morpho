[comment]: # Sparse class help
[version]: # 0.5

#Sparse
[tag]: # Sparse

The Sparse class provides support for sparse matrices. An empty sparse matrix can be initialized with a given size,

    var a = Sparse(nrows,ncols)

Alternatively, a matrix can be created from an array of triplets,  

    var a = Matrix([[row, col, value] ...])
    
For example

    var a = Matrix([[0,0,2], [1,1,-2]])
    
creates the matrix

    [ 2 0 ]
    [ 0 -2 ]

Once a sparse matrix is created, you can use all the regular arithmetic operators with matrix operands, e.g.

    a+b
    a*b
