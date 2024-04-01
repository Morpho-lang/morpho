[comment]: # (Errors help file)
[version]: # (0.5)

# Errors
[tagerror]: # (error)
[tagerrors]: # (errors)

When an error occurs in running a morpho program, an error message is displayed together with an explanation of where in the program that the error happened.

You can make your own custom errors using the `Error` class: 

    var myerr = Error("Tag", "A message")

Use the `throw` method to raise the error, interrupting execution unless the error is caught: 

    myerr.throw() 

or 

    myerr.throw("A custom message") 

You can also use the `warning` method to alert the user of a potential issue that doesn't need the program to be interrupted. 

    myerr.warning() 

[showsubtopics]: # (subtopics)

## Alloc
[tagalloc]: # (alloc)

This error may occur when creating new objects or resizing them. It typically indicates that the computer is under memory pressure.

## Intrnl
[tagintrnl]: # (intrnl)

This error indicates an internal problem with morpho. Please contact the developers for support.

## InvldOp
[taginvldop]: # (invldop)

This error occurs when an operator like `+` or `-` is given operands that it doesn't understand. For example,

    print "Hello" * "Goodbye" // Causes 'InvldOp'

causes this error because the multiplication operator doesn't know how to multiply strings.

If the operands are objects, this means that the objects don't provide a method for the requested operation, e.g. for

    print object1 / object2

`object1` would need to provide a `div()` method that can successfully handle `object2`.

## CnctFld
[tagcnctfld]: # (cnctfld)

This error occurs when concatenation of strings or other objects fails, typically because of low memory.

## Uncallable
[taguncallable]: # (uncallable)

This error occurs when you try to call something that isn't a method or a function. Here, we initialize a variable with a string and call it:

    var f = "Not a function"
    f() // Causes 'Uncallable'

## GlblRtrn
[tagglblrtrn]: # (glblrtrn)

This error occurs when morpho encounters a `return` keyword outside of a function or method definition.

## InstFail
[taginstfail]: # (instfail)

This error occurs when morpho tried to create a new object, but something went wrong.

## NotAnObj
[tagnotanobj]: # (notanobj)

This error occurs if you try to access a property of something that isn't an object:  

    var a = 1
    a.size = 5

## ObjLcksPrp
[tagobjlcksprp]: # (objlcksprp)

This error occurs if you try to access a property or method that hasn't been defined for an object:

    var a = Object()
    print a.pifflepaffle

or

    print a.foo()

## NoInit
[tagnoinit]: # (noinit)

This error can occur if you try to create a new object from a class that doesn't have an `init` method:

    class Foo { }
    var a = Foo(0.3)

Here, the argument to `Foo` causes the `NoInit` error because no `init` method is available to process it.

## NotAnInst
[tagnotaninst]: # (notaninst)

This error occurs if you try to invoke a method on something that isn't an object:

    var a = 4
    print a.foo()

## ClssLcksMthd
[tagclsslcksmthd]: # (clsslcksmthd)

This error occurs if you try to invoke a method on a class that doesn't exist:

    class Foo { }
    print Foo.foo()

## InvldArgs
[taginvldargs]: # (invldargs)

This error occurs if you call a function with the wrong number of arguments:

    fn f(x) { return x }
    f(1,2)

## NotIndxbl
[tagnotindxbl]: # (notindxbl)

This error occurs if you try to index something that isn't a collection:

    var a = 0.3
    print a[1]

## IndxBnds
[tagindxbnds]: # (indxbnds)

This error can occur when selecting an entry from a collection object (such as a list) if the index supplied is bigger than the number of entries:

    var a = [1,2,3]
    print a[10]

## NonNmIndx
[tagnonnmindx]: # (nonnmindx)

This error occurs if you try to index an array with a non-numerical index:

    var a[2,2]
    print a["foo","bar"]

## ArrayDim
[tagarraydim]: # arraydim

This error occurs if you try to index an array with the wrong number of indices:

    var a[2,2]
    print a[1]

## DbgQuit
[tagdbgquit]: # (dbgquit)

This notification is generated after selecting `Quit` within the debugger. Execution of the program is halted and control returns to the user.    

## SymblUndf
[tagsymblundf]: # (symblundf)

This error occurs if you refer to something that has not been previously declared, for example trying to use a variable of call a function that doesn't exist. It's possible that the symbol is spelt incorrectly, or that the capitalization doesn't match the definition (*morpho* symbols are case-sensitive).

A common problem is to try to assign to a variable that hasn't yet been declared:

    a = 5

To fix this, prefix with `var`:

    var a = 5


## MtrxIncmptbl
[tagmtrxincmptbl]: # (mtrxincmptbl)

This error occurs when an arithmetic operation is performed on two 'incompatible' matrices. For example, two matrices must have the same dimensions, i.e. the same number of rows and columns, to be added or subtracted,

    var a = Matrix([[1,2],[3,4]])
    var b = Matrix([[1]])
    print a+b // generates a `MtrxIncmptbl` error.

Or to be multiplied together, the number of columns of the left hand matrix must equal the number of rows of the right hand matrix.

    var a = Matrix([[1,2],[3,4]])
    var b = Matrix([1,2])
    print a*b // ok
    print b*a // generates a `MtrxIncmptbl` error.
