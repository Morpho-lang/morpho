[comment]: # Morpho language help file
[version]: # 0.5

#Errors
[tag]: #error

##Alloc
[tag]: # Alloc

This error may occur when creating new objects or resizing them. It typically indicates that the computer is under memory pressure.

##Intrnl
[tag]: # Intrnl

This error indicates an internal problem with morpho. Please contact the developers for support.

##InvldOp
[tag]: # InvldOp

This error occurs when an operator like `+` or `-` is given operands that it doesn't understand. For example,

    print "Hello" * "Goodbye" // Causes 'InvldOp'

causes this error because the multiplication operator doesn't know how to multiply strings.

If the operands are objects, this means that the objects don't provide a method for the requested operation, e.g. for

    print object1 / object2

`object1` would need to provide a `div()` method that can successfully handle `object2`.

##CnctFld
[tag]: # CnctFld

This error occurs when concatenation of strings or other objects fails, typically because of low memory.

##Uncallable
[tag]: # Uncallable

This error occurs when you try to call something that isn't a method or a function. Here, we initialize a variable with a string and call it:

    var f = "Not a function"
    f() // Causes 'Uncallable'

##GlblRtrn
[tag]: # GlblRtrn

This error occurs when morpho encounters a `return` keyword outside of a function or method definition.

##InstFail
[tag]: # InstFail

This error occurs when morpho tried to create a new object, but something went wrong.

##NotAnObj
[tag]: # NotAnObj

This error occurs if you try to access a property of something that isn't an object:  

    var a = 1
    a.size = 5

##ObjLcksPrp
[tag]: # ObjLcksPrp

This error occurs if you try to access a property or method that hasn't been defined for an object:

    var a = Object()
    print a.pifflepaffle

or

    print a.foo()

##NoInit
[tag]: # NoInit

This error can occur if you try to create a new object from a class that doesn't have an `init` method:

    class Foo { }
    var a = Foo(0.3)

Here, the argument to `Foo` causes the `NoInit` error because no `init` method is available to process it.

##NotAnInst
[tag]: # NotAnInst

This error occurs if you try to invoke a method on something that isn't an object:

    var a = 4
    print a.foo()

##ClssLcksMthd
[tag]: # ClssLcksMthd

This error occurs if you try to invoke a method on a class that doesn't exist:

    class Foo { }
    print Foo.foo()

##InvldArgs
[tag]: # InvldArgs

This error occurs if you call a function with the wrong number of arguments:

    fn f(x) { return x }
    f(1,2)

##NotIndxbl
[tag]: # NotIndxbl

This error occurs if you try to index something that isn't a collection:

    var a = 0.3
    print a[1]

##IndxBnds
[tag]: # IndxBnds

This error can occur when selecting an entry from a collection object (such as a list) if the index supplied is bigger than the number of entries:

    var a = [1,2,3]
    print a[10]

##NonNmIndx
[tag]: # NonNmIndx

This error occurs if you try to index an array with a non-numerical index:

    var a[2,2]
    print a["foo","bar"]

##ArrayDim
[tag]: # ArrayDim

This error occurs if you try to index an array with the wrong number of indices:

    var a[2,2]
    print a[1]

##DbgQuit
[tag]: # DbgQuit

This notification is generated after selecting `Quit` within the debugger. Execution of the program is halted and control returns to the user.    

##SymblUndf
[tag]: # SymblUndf

This error occurs if you refer to something that has not been previously declared, for example trying to use a variable of call a function that doesn't exist. It's possible that the symbol is spelt incorrectly, or that the capitalization doesn't match the definition (*morpho* symbols are case-sensitive).

A common problem is to try to assign to a variable that hasn't yet been declared:

    a = 5

To fix this, prefix with `var`:

    var a = 5
