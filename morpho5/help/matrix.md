[comment]: # Matrix class help
[version]: # 0.5

#Matrix
[tag]: # Matrix

The Matrix class provides support for matrices. A matrix can be initialized with a given size,

    var a = Matrix(nrows,ncols)

where all elements are initially set to zero. Alternatively, a matrix can be created from an array,

    var a = Matrix([[1,2], [3,4]])

You can create a column vector like this,

    var v = Matrix([1,2])

Once a matrix is created, you can use all the regular arithmetic operators with matrix operands, e.g.

    a+b
    a*b

The division operator is used to solve a linear system, e.g.

    var a = Matrix([[1,2],[3,4]])
    var b = Matrix([1,2])

    print b/a

yields the solution to the system a*x = b.

##MtrxIncmptbl
[tag]: # MtrxIncmptbl

This error occurs when an arithmetic operation is performed on two 'incompatible' matrices. For example, two matrices must have the same dimensions, i.e. the same number of rows and columns, to be added or subtracted,

    var a = Matrix([[1,2],[3,4]])
    var b = Matrix([[1]])
    print a+b // generates a `MtrxIncmptbl` error.

Or to be multiplied together, the number of columns of the left hand matrix must equal the number of rows of the right hand matrix.

    var a = Matrix([[1,2],[3,4]])
    var b = Matrix([1,2])
    print a*b // ok
    print b*a // generates a `MtrxIncmptbl` error.
