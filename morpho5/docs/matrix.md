[comment]: # (Matrix class help)
[version]: # (0.5)

# Matrix
[tagmatrix]: # (Matrix)

The Matrix class provides support for matrices. A matrix can be initialized with a given size,

    var a = Matrix(nrows,ncols)

where all elements are initially set to zero. Alternatively, a matrix can be created from an array,

    var a = Matrix([[1,2], [3,4]])

or a Sparse matrix,

    var a = Sparse([[0,0,1],[1,1,1],[2,2,1]])
    var b = Matrix(a)

You can create a column vector like this,

    var v = Matrix([1,2])

Finally, you can create a Matrix by assembling other matrices like this,

    var a = Matrix([[0,1],[1,0]])
    var b = Matrix([[a,0],[0,a]]) // produces a 4x4 matrix 

Once a matrix is created, you can use all the regular arithmetic operators with matrix operands, e.g.

    a+b
    a*b

The division operator is used to solve a linear system, e.g.

    var a = Matrix([[1,2],[3,4]])
    var b = Matrix([1,2])

    print b/a

yields the solution to the system a*x = b.

[showsubtopics]: # (subtopics)

## Assign
[tagassign]: # (Assign)

Copies the contents of matrix B into matrix A: 

    A.assign(B)

The two matrices must have the same dimensions.

## Dimensions
[tagdimensions]: # (Dimensions)

Returns the dimensions of a matrix:

    var A = Matrix([1,2,3]) // Create a column matrix 
    print A.dimensions()    // Expect: [ 3, 1 ]

## Eigenvalues
[tageigenvalues]: # (Eigenvalues)

Returns a list of eigenvalues of a Matrix:

    var A = Matrix([[0,1],[1,0]])
    print A.eigenvalues() // Expect: [1,-1]

## Eigensystem
[tageigensystem]: # (Eigensystem)

Returns the eigenvalues and eigenvectors of a Matrix:

    var A = Matrix([[0,1],[1,0]])
    print A.eigensystem() 

Eigensystem returns a two element list: The first element is a List of eigenvalues. The second element is a Matrix containing the corresponding eigenvectors as its columns:

    print A.eigensystem()[0]
    // [ 1, -1 ]
    print A.eigensystem()[1]
    // [ 0.707107 -0.707107 ]
    // [ 0.707107 0.707107 ]

## Inner
[taginner]: # (Inner)

Computes the Frobenius inner product between two matrices:

    var prod = A.inner(B)

## Inverse
[taginverse]: # (Inverse)

Returns the inverse of a matrix if it is invertible. Raises a
`MtrxSnglr` error if the matrix is singular. E.g.

    var m = Matrix([[1,2],[3,4]])
    var mi = m.inverse()

yields the inverse of the matrix `m`, such that mi*m is the identity
matrix.

## Norm
[tagnorm]: # (Norm)

Returns a matrix norm. By default the L2 norm is returned: 

    var a = Matrix([1,2,3,4])
    print a.norm() // Expect: sqrt(30) = 0.5477..

## Reshape
[tagreshape]: # (Reshape)

Changes the dimensions of a matrix such that the total number of elements remains constant:

    var A = Matrix([[1,3],[2,4]])
    A.reshape(1,4) // 1 row, 4 columns
    print A // Expect: [ 1, 2, 3, 4 ]

Note that elements are stored in column major-order.

## Sum
[tagsum]: # (Sum)

Returns the sum of all entries in a matrix:

    var sum = A.sum() 

## Transpose
[tagtranspose]: # (Transpose)

Returns the transpose of a matrix: 

    var At = A.transpose()

## Trace
[tagtrace]: # (Trace)

Computes the trace (the sum of the diagonal elements) of a square matrix:

    var tr = A.trace()
