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

The `ComplexMatrix` class provides the corresponding support for complex-valued matrices and supports the same core indexing, slicing, decomposition and arithmetic operations, together with methods such as `real`, `imag`, `conjugate` and `conjTranspose`.

Once a matrix is created, you can use all the regular arithmetic operators with matrix operands, e.g.

    a+b
    a*b

You can retrieved individual matrix entries with specified indices:

    print a[0,0]

or create a submatrix using slices:

	print a[0..1,0..1]

If a matrix is a row or column vector, it can also be sliced with a single argument:

    var v = Matrix([1,2,3])
    print v[0..1]

The division operator is used to solve a linear system, e.g.

    var a = Matrix([[1,2],[3,4]])
    var b = Matrix([1,2])

    print b/a

yields the solution to the system `a*x = b`.

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
    print A.dimensions()    // Expect: (3, 1)

## Column
[tagcolumn]: # (column)

Returns a specified column of a matrix as a column matrix:

    var v = A.column(0)

## Setcolumn
[tagsetcolumn]: # (setcolumn)

Replaces a specified column of a matrix:

    A.setcolumn(0, Matrix([1,2,3]))

## Eigenvalues
[tageigenvalues]: # (Eigenvalues)

Returns a tuple of eigenvalues of a Matrix:

    var A = Matrix([[0,1],[1,0]])
    print A.eigenvalues() // Expect: (1,-1)

## Eigensystem
[tageigensystem]: # (Eigensystem)

Returns the eigenvalues and eigenvectors of a Matrix:

    var A = Matrix([[0,1],[1,0]])
    print A.eigensystem() 

Eigensystem returns a two element tuple: The first element is a tuple of eigenvalues. The second element is a Matrix containing the corresponding eigenvectors as its columns:

    print A.eigensystem()[0]
    // (1, -1)
    print A.eigensystem()[1]
    // [ 0.707107 -0.707107 ]
    // [ 0.707107 0.707107 ]

## SVD
[tagsvd]: # (SVD)

The 'svd' method returns the singular value decomposition of a matrix as a three element tuple:

    var svd = A.svd()

The return value contains the left singular vectors, singular values, and right singular vectors in that order.

If `A` is a matrix, its singular value decomposition factors it as

    A = U S V^T

where `U` and `V` are orthogonal matrices and `S` contains the singular values of `A`.

The SVD is useful for understanding the numerical rank of a matrix, solving least-squares problems, and analyzing the dominant modes or directions present in the data represented by a matrix.

## QR
[tagqr]: # (QR)

The 'qr' method returns the QR decomposition of a matrix as a two element tuple:

    var qr = A.qr()

If `A` is a matrix, its QR decomposition factors it as

    A = Q R

where `Q` is orthogonal and `R` is upper triangular.

The QR decomposition is useful for solving linear least-squares problems and for constructing numerically stable orthogonal bases from the columns of a matrix.

## Inner
[taginner]: # (Inner)

Computes the Frobenius inner product between two matrices:

    var prod = A.inner(B)

## Outer
[tagouter]: # (Outer)

Computes the outer produce between two vectors: 

    var prod = A.outer(B)

Note that `outer` always treats both vectors as column vectors. 

## Inverse
[taginverse]: # (Inverse)

Returns the inverse of a matrix if it is invertible. Raises a
`MtrxSnglr` error if the matrix is singular. E.g.

    var m = Matrix([[1,2],[3,4]])
    var mi = m.inverse()

yields the inverse of the matrix `m`, such that `mi*m` is the identity
matrix.

## Norm
[tagnorm]: # (Norm)

Returns a matrix norm. By default the Frobenius norm is returned:

    var a = Matrix([1,2,3,4])
    print a.norm() // Expect: sqrt(30) = 5.47723...

You can select a different supported norm by supplying an argument:

    import constants
    print a.norm(1)   // Expect: 10 (L1 norm)
    print a.norm(Inf) // Expect: 4 (Infinity norm)

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

## Roll
[tagroll]: # (Roll)

Rotates values in a Matrix about a given axis by a given shift:

    var r = A.roll(shift, axis)

Elements that roll beyond the last position are re-introduced at the first.

## IdentityMatrix
[tagidentitymatrix]: # (IdentityMatrix)

Constructs an identity matrix of a specified size:

    var a = IdentityMatrix(size)

## ComplexMatrix
[tagcomplexmatrix]: # (ComplexMatrix)

The `ComplexMatrix` class provides support for complex-valued matrices. It can be initialized in the same ways as `Matrix`, for example

    var a = ComplexMatrix(2,2)
    var b = ComplexMatrix((1+1im, 2+2im, 3+3im))
    var c = ComplexMatrix([[1+1im, 2], [3, 4-1im]])

`ComplexMatrix` supports the same core indexing, slicing, arithmetic and decomposition methods as `Matrix`, including `inverse`, `norm`, `sum`, `trace`, `transpose`, `eigenvalues`, `eigensystem`, `svd`, `qr`, `reshape` and `roll`.

As with `Matrix`, single-argument slicing is supported for row and column vectors.

For `ComplexMatrix`, `eigenvalues()` and the first component of `eigensystem()` may contain complex values.

In addition, `ComplexMatrix` provides methods for accessing and manipulating the complex structure:

    var r = c.real()
    var i = c.imag()
    var z = c.conjugate()
    var h = c.conjTranspose()

`conjugate()` returns the elementwise complex conjugate of the matrix, while `conjTranspose()` returns the conjugate transpose (also called the Hermitian transpose).

Mixed arithmetic between `Matrix` and `ComplexMatrix` is supported, with the result promoted to complex where needed.

## Real
[tagreal]: # (real)

Returns a `Matrix` containing the real part of a `ComplexMatrix`:

    print c.real()

## Imag
[tagimag]: # (imag)

Returns a `Matrix` containing the imaginary part of a `ComplexMatrix`:

    print c.imag()

## ConjTranspose
[tagconjtranspose]: # (conjtranspose)

Returns the conjugate transpose of a `ComplexMatrix`:

    print c.conjTranspose()
