[comment]: # (Errors help file)
[version]: # (0.5)

# Errors
[tagerror]: # (error)
[tagerrors]: # (errors)
[tagerrors]: # (throw)
[tagerrors]: # (warning)

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

## Throw
[tagthrow]: # (throw)

Raises an `Error`, interrupting execution unless it is caught:

    myerr.throw()

You can optionally supply a custom message:

    myerr.throw("A custom message")

## Warning
[tagwarning]: # (warning)

Displays an `Error` as a warning without interrupting execution:

    myerr.warning()

To see the full list of morpho errors, look at the `errorlist` help entry.

# Error list
[tagerrorrlist]: # (error list)
[tagerrorlist]: # (errorlist)

A list of morpho errors:

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

## DvZr
[tagdvzr]: # (dvzr)

This error occurs when attempting to divide by zero:

    var a = 5 / 0 // Causes 'DvZr'

## StckOvflw
[tagstckovflw]: # (stckovflw)

This error occurs when the call stack exceeds its maximum depth, typically due to excessive recursion or deeply nested function calls.

## ErrStckOvflw
[tagerrstckovflw]: # (errstckovflw)

This error occurs when the error handler stack overflows, typically due to errors occurring within error handlers.

## Exit
[tagexit]: # (exit)

This error is generated when the virtual machine is halted, typically when the program exits normally.

## MltplDsptchFld
[tagmltpldsptchfld]: # (mltpldsptchfld)

This error occurs when multiple dispatch cannot find a method implementation that matches the provided arguments:

    class A { }
    class B { }
    fn method(A a) { }
    fn method(B b) { }
    method(1) // Causes 'MltplDsptchFld' - no matching method for integer

## TypeChk
[tagtypechk]: # (typechk)

This error occurs when there is a type violation, such as attempting to assign a value of one type to a variable declared with a different type:

    String x = 5 // Causes 'TypeChk'

## NoOptArg
[tagnooptarg]: # (nooptarg)

This error occurs when you try to pass optional arguments to a function that doesn't accept them:

    fn f(x) { return x }
    f(1, y=2) // Causes 'NoOptArg'

## UnkwnOptArg
[tagunkwnoptarg]: # (unkwnoptarg)

This error occurs when you pass an unknown optional argument to a function:

    fn f(x, y=1) { return x + y }
    f(1, z=2) // Causes 'UnkwnOptArg'

## InvldArgsBltn
[taginvldargsbltn]: # (invldargsbltn)

This error occurs when a built-in function is called with arguments of the wrong type:

    print(1, 2, 3) // If print expects a string, causes 'InvldArgsBltn'

## ArrayArgs
[tagarrayargs]: # (arrayargs)

This error occurs when creating an Array with invalid arguments. Arrays must be created with integer dimensions:

    var a = Array("invalid") // Causes 'ArrayArgs'

## ArrayInit
[tagarrayinit]: # (arrayinit)

This error occurs when an Array initializer is not an array or list:

    var a = Array(2, 2, "invalid") // Causes 'ArrayInit'

## ArrayCmpt
[tagarraycmpt]: # (arraycmpt)

This error occurs when an Array initializer has dimensions that don't match the requested dimensions:

    var a = Array(2, 2, [[1,2,3]]) // Causes 'ArrayCmpt' if dimensions don't match

## ArrayIndx
[tagarrayindx]: # (arrayindx)

This error occurs when indexing an Array with non-integer indices:

    var a[2,2]
    a["x", "y"] // Causes 'ArrayIndx'

## BrkOtsdLp
[tagbrkotsdlp]: # (brkotsdlp)

This error occurs when a `break` statement is encountered outside of a loop:

    break // Causes 'BrkOtsdLp'

## CntOtsdLp
[tagcntotsdlp]: # (cntotsdlp)

This error occurs when a `continue` statement is encountered outside of a loop:

    continue // Causes 'CntOtsdLp'

## ClssCrcRf
[tagclsscrcrf]: # (clsscrcrf)

This error occurs when a class attempts to inherit from itself:

    class A < A { } // Causes 'ClssCrcRf'

## ClssDplctImpl
[tagclssdplctimpl]: # (clssdplctimpl)

This error occurs when a class has duplicate method implementations with the same signature:

    class A {
        fn method() { }
        fn method() { } // Causes 'ClssDplctImpl'
    }

## ClssLnrz
[tagclsslnrz]: # (clsslnrz)

This error occurs when morpho cannot linearize a class hierarchy due to conflicting inheritance order. Check parent and ancestor classes for inheritance issues.

## SlfOtsdClss
[tagslfotsdclss]: # (slfotsdclss)

This error occurs when `self` is used outside of a class method:

    print self // Causes 'SlfOtsdClss'

## SprOtsdClss
[tagsprotsdclss]: # (sprotsdclss)

This error occurs when `super` is used outside of a class method:

    print super // Causes 'SprOtsdClss'

## SprSelMthd
[tagsprselmthd]: # (sprselmthd)

This error occurs when `super` is used incorrectly. It can only be used to select a method:

    super // Causes 'SprSelMthd'
    super.method() // OK

## SprNtFnd
[tagsprntfnd]: # (sprntfnd)

This error occurs when a superclass cannot be found:

    class A < NonExistent { } // Causes 'SprNtFnd'

## TooMnyArg
[tagtoomnyarg]: # (toomnyarg)

This error occurs when too many arguments are passed to a function:

    fn f(x) { return x }
    f(1, 2, 3) // Causes 'TooMnyArg'

## TooMnyPrm
[tagtoomnyprm]: # (toomnyprm)

This error occurs when a function is defined with too many parameters (exceeding the maximum allowed).

## TooMnyCnst
[tagtoomnycnst]: # (toomnycnst)

This error occurs when a program has too many constants (exceeding the maximum allowed).

## VblDcl
[tagvbldcl]: # (vbldcl)

This error occurs when a variable is declared multiple times in the same scope:

    var x = 1
    var x = 2 // Causes 'VblDcl'

## FlNtFnd
[tagflntfnd]: # (flntfnd)

This error occurs when a file cannot be found:

    import "nonexistent.morpho" // Causes 'FlNtFnd'

## MdlNtFnd
[tagmdlntfnd]: # (mdlntfnd)

This error occurs when a module cannot be found:

    import nonexistent // Causes 'MdlNtFnd'

## ImprtFld
[tagimprtfld]: # (imprtfld)

This error occurs when an import statement fails:

    import "broken.morpho" // Causes 'ImprtFld' if the file has errors

## UnrslvdFrwdRf
[tagunrslvdfrwdrf]: # (unrslvdfrwdrf)

This error occurs when a function is called before it is defined in the same scope:

    f() // Causes 'UnrslvdFrwdRf'
    fn f() { }

## MltVarPrmtr
[tagmltvarprmtr]: # (mltvarprmtr)

This error occurs when a function has more than one variadic parameter:

    fn f(...args1, ...args2) { } // Causes 'MltVarPrmtr'

## VarPrLst
[tagvarprlst]: # (varprlst)

This error occurs when fixed parameters are placed after a variadic parameter:

    fn f(...args, x) { } // Causes 'VarPrLst'

## OptPrmDflt
[tagoptprmdflt]: # (optprmdflt)

This error occurs when an optional parameter's default value is not a constant:

    var x = 1
    fn f(y: x) { } // Causes 'OptPrmDflt'

## MssngLoopBdy
[tagmssngloopbdy]: # (mssngloopbdy)

This error occurs when a loop statement is missing its body:

    for (var i = 0; i < 10; i++) // Causes 'MssngLoopBdy'

## NstdClss
[tagnstdclss]: # (nstdclss)

This error occurs when attempting to define a class within another class:

    class A {
        class B { } // Causes 'NstdClss'
    }

## InvldAssgn
[taginvldassgn]: # (invldassgn)

This error occurs when attempting to assign to an invalid target:

    5 = 10 // Causes 'InvldAssgn'

## FnPrmSymb
[tagfnprmsymb]: # (fnprmsymb)

This error occurs when function parameters are not symbols:

    fn f(5) { } // Causes 'FnPrmSymb'

## PptyNmRqd
[tagpptynmrqd]: # (pptynmrqd)

This error occurs when a property name is required but not provided.

## InitRtn
[taginitrtn]: # (initrtn)

This error occurs when attempting to return a value from an initializer method:

    class A {
        init() {
            return 5 // Causes 'InitRtn'
        }
    }

## MssngIndx
[tagmssngindx]: # (mssngindx)

This error occurs when indexing syntax is incomplete, missing required indices.

## MssngIntlzr
[tagmssngintlzr]: # (mssngintlzr)

This error occurs when a typed variable is declared without an initializer:

    var x: String // Causes 'MssngIntlzr' if initialization is required

## TypeErr
[tagtypeerr]: # (typeerr)

This error occurs when there is a type violation during assignment:

    var x: String
    x = 5 // Causes 'TypeErr'

## UnknwnType
[tagunknwntype]: # (unknwntype)

This error occurs when an unknown type is referenced:

    var x: UnknownType // Causes 'UnknwnType'

## UnknwnNmSpc
[tagunknwnnmspc]: # (unknwnnmspc)

This error occurs when an unknown namespace is referenced:

    import unknown::module // Causes 'UnknwnNmSpc'

## UnknwnTypeNmSpc
[tagunknwntypenmspc]: # (unknwntypenmspc)

This error occurs when an unknown type is referenced in a namespace:

    var x: unknown::Type // Causes 'UnknwnTypeNmSpc'

## SymblUndfNmSpc
[tagsymblundfnmspc]: # (symblundfnmspc)

This error occurs when a symbol is not defined in the specified namespace:

    unknown::symbol // Causes 'SymblUndfNmSpc'

## IncExp
[tagincexp]: # (incexp)

This error occurs when an expression is incomplete:

    var x = 5 + // Causes 'IncExp'

## MssngParen
[tagmssngparen]: # (mssngparen)

This error occurs when a closing parenthesis is missing:

    fn f(x // Causes 'MssngParen'

## ExpExpr
[tagexpexpr]: # (expexpr)

This error occurs when an expression is expected but not found:

    var x = // Causes 'ExpExpr'

## MssngExpTerm
[tagmssngexpterm]: # (mssngexpterm)

This error occurs when an expression terminator (semicolon or newline) is missing after an expression.

## VarExpct
[tagvarexpct]: # (varexpct)

This error occurs when a variable name is expected after `var`:

    var // Causes 'VarExpct'

## SymblExpct
[tagsymblexpct]: # (symblexpct)

This error occurs when a symbol is expected but not found.

## MssngBrc
[tagmssngbrc]: # (mssngbrc)

This error occurs when a closing brace is missing:

    fn f() { // Causes 'MssngBrc'

## MssngSqBrc
[tagmssngsqbrc]: # (mssngsqbrc)

This error occurs when a closing square bracket is missing:

    var x = [1, 2 // Causes 'MssngSqBrc'

## MssngComma
[tagmssngcomma]: # (mssngcomma)

This error occurs when a comma is expected:

    var x = [1 2] // Causes 'MssngComma'

## TrnryMssngColon
[tagtrnymssngcolon]: # (trnymssngcolon)

This error occurs when a colon is missing in a ternary operator:

    var x = true ? 1 // Causes 'TrnryMssngColon'

## IfMssngLftPrn
[tagifmssnglftprn]: # (ifmssnglftprn)

This error occurs when a left parenthesis is missing after `if`:

    if x > 0 { } // Causes 'IfMssngLftPrn'

## IfMssngRgtPrn
[tagifmssngrgtprn]: # (ifmssngrgtprn)

This error occurs when a right parenthesis is missing after an if condition:

    if (x > 0 { } // Causes 'IfMssngRgtPrn'

## WhlMssngLftPrn
[tagwhlmssnglftprn]: # (whlmssnglftprn)

This error occurs when a left parenthesis is missing after `while`:

    while x > 0 { } // Causes 'WhlMssngLftPrn'

## ForMssngLftPrn
[tagformssnglftprn]: # (formssnglftprn)

This error occurs when a left parenthesis is missing after `for`:

    for var i = 0; i < 10; i++ { } // Causes 'ForMssngLftPrn'

## ForMssngRgtPrn
[tagformssngrgtprn]: # (formssngrgtprn)

This error occurs when a right parenthesis is missing after for clauses:

    for (var i = 0; i < 10; i++ { } // Causes 'ForMssngRgtPrn'

## FnNoName
[tagfnoname]: # (fnoname)

This error occurs when a function or method name is expected but not found:

    fn () { } // Causes 'FnNoName'

## FnMssngLftPrn
[tagfnmssnglftprn]: # (fnmssnglftprn)

This error occurs when a left parenthesis is missing after a function name:

    fn f { } // Causes 'FnMssngLftPrn'

## FnMssngRgtPrn
[tagfnmssngrgtprn]: # (fnmssngrgtprn)

This error occurs when a right parenthesis is missing after function parameters:

    fn f(x { } // Causes 'FnMssngRgtPrn'

## FnMssngLftBrc
[tagfnmssnglftbrc]: # (fnmssnglftbrc)

This error occurs when a left brace is missing before a function body:

    fn f() // Causes 'FnMssngLftBrc'

## CllMssngRgtPrn
[tagcllmssngrgtprn]: # (cllmssngrgtprn)

This error occurs when a right parenthesis is missing after function call arguments:

    f(x // Causes 'CllMssngRgtPrn'

## ClsNmMssng
[tagclsnmssng]: # (clsnmssng)

This error occurs when a class name is expected but not found:

    class { } // Causes 'ClsNmMssng'

## ClsMssngLftBrc
[tagclsmssnglftbrc]: # (clsmssnglftbrc)

This error occurs when a left brace is missing before a class body:

    class A // Causes 'ClsMssngLftBrc'

## ClsMssngRgtBrc
[tagclsmssngrgtbrc]: # (clsmssngrgtbrc)

This error occurs when a right brace is missing after a class body:

    class A { // Causes 'ClsMssngRgtBrc'

## ExpctDtSpr
[tagexpctdtspr]: # (expctdtspr)

This error occurs when a dot is expected after `super`:

    super method() // Causes 'ExpctDtSpr'
    super.method() // OK

## SprNmMssng
[tagsprnmssng]: # (sprnmssng)

This error occurs when a superclass name is expected but not found:

    class A < { } // Causes 'SprNmMssng'

## MxnNmMssng
[tagmxnnmssng]: # (mxnnmssng)

This error occurs when a mixin class name is expected but not found.

## IntrpIncmp
[tagintrpincmp]: # (intrpincmp)

This error occurs when a string interpolation is incomplete:

    var x = "Hello ${" // Causes 'IntrpIncmp'

## EmptyIndx
[tagemptyindx]: # (emptyindx)

This error occurs when a variable declaration has an empty capacity:

    var x[] // Causes 'EmptyIndx'

## ImprtMssngNm
[tagimprtmssngnm]: # (imprtmssngnm)

This error occurs when an import statement is missing a module or file name:

    import // Causes 'ImprtMssngNm'

## ImprtMltplAs
[tagimprtmltplas]: # (imprtmltplas)

This error occurs when an import statement has multiple `as` clauses:

    import module as A as B // Causes 'ImprtMltplAs'

## ImprtExpctFrAs
[tagimprtexpctfras]: # (imprtexpctfras)

This error occurs when an import statement doesn't have the expected format:

    import module invalid // Causes 'ImprtExpctFrAs'

## ExpctSymblAftrAs
[tagexpctsymblaftras]: # (expctsymblaftras)

This error occurs when a symbol is expected after `as` in an import:

    import module as // Causes 'ExpctSymblAftrAs'

## ExpctSymblAftrFr
[tagexpctsymblaftrfr]: # (expctsymblaftrfr)

This error occurs when a symbol is expected after `for` in an import:

    import module for // Causes 'ExpctSymblAftrFr'

## DctSprtr
[tagdctsprtr]: # (dctsprtr)

This error occurs when a colon is missing in a dictionary key-value pair:

    var d = {"key" "value"} // Causes 'DctSprtr'

## DctEntrySprtr
[tagdctentrysprtr]: # (dctentrysprtr)

This error occurs when a comma is missing between dictionary entries:

    var d = {"a": 1 "b": 2} // Causes 'DctEntrySprtr'

## DctTrmntr
[tagdcttrmntr]: # (dcttrmntr)

This error occurs when a closing brace is missing in a dictionary:

    var d = {"a": 1 // Causes 'DctTrmntr'

## SwtchSprtr
[tagswtchsprtr]: # (swtchsprtr)

This error occurs when a colon is missing after a switch label:

    switch x {
        case 1 // Causes 'SwtchSprtr'
    }

## ExpctWhl
[tagexpctwhl]: # (expctwhl)

This error occurs when `while` is expected after a do-while loop body:

    do {
        // body
    } // Causes 'ExpctWhl'

## ExpctCtch
[tagexpctctch]: # (expctctch)

This error occurs when `catch` is expected after a `try` statement:

    try {
        // code
    } // Causes 'ExpctCtch'

## ExpctHndlr
[tagexpcthndlr]: # (expcthndlr)

This error occurs when an error handler block is expected after `catch`:

    try {
        // code
    } catch // Causes 'ExpctHndlr'

## InvldLbl
[taginvldlbl]: # (invldlbl)

This error occurs when an invalid label is used in a catch statement.

## OneVarPr
[tagonevarpr]: # (onevarpr)

This error occurs when a function has more than one variadic parameter (same as `MltVarPrmtr`).

## ValRng
[tagvalrng]: # (valrng)

This error occurs when a value is out of the expected range.

## StrEsc
[tagstresc]: # (stresc)

This error occurs when an unrecognized escape sequence is used in a string:

    var s = "\q" // Causes 'StrEsc'

## RcrsnLmt
[tagrcrsnlmt]: # (rcrsnlmt)

This error occurs when the parser recursion depth is exceeded, typically due to deeply nested expressions.

## UnescpdCtrl
[tagunescpdctrl]: # (unescpdctrl)

This error occurs when an unescaped control character is found in a string literal.

## InvldUncd
[taginvlduncd]: # (invlduncd)

This error occurs when an invalid unicode escape sequence is used in a string:

    var s = "\uZZZZ" // Causes 'InvldUncd'

## UnrcgnzdTok
[tagunrcgnzdtok]: # (unrcgnzdtok)

This error occurs when the parser encounters an unrecognized token.

## UntrmComm
[taguntrmcomm]: # (untrmcomm)

This error occurs when a multiline comment is not terminated:

    /* This comment // Causes 'UntrmComm'

## UntrmStrng
[taguntrmstrng]: # (untrmstrng)

This error occurs when a string literal is not terminated:

    var s = "This string // Causes 'UntrmStrng'

## UnrgnzdTkn
[tagunrgnzdtkn]: # (unrgnzdtkn)

This error occurs when the lexer encounters an unrecognized token.

## MtrxBnds
[tagmtrxbnds]: # (mtrxbnds)

This error occurs when attempting to access a matrix element with an index that is out of bounds:

    var m = Matrix([[1,2],[3,4]])
    print m[10, 10] // Causes 'MtrxBnds'

## MtrxInvldIndx
[tagmtrxinvldindx]: # (mtrxinvldindx)

This error occurs when matrix indices are not integers:

    var m = Matrix([[1,2],[3,4]])
    print m["x", "y"] // Causes 'MtrxInvldIndx'

## MtrxInvldNumIndx
[tagmtrxinvldnumindx]: # (mtrxinvldnumindx)

This error occurs when a matrix is indexed with the wrong number of indices:

    var m = Matrix([[1,2],[3,4]])
    print m[1] // Causes 'MtrxInvldNumIndx' (needs two indices)

## MtrxCns
[tagmtrxcns]: # (mtrxcns)

This error occurs when the Matrix constructor is called with invalid arguments. It should be called with dimensions or an array/list/matrix initializer:

    var m = Matrix("invalid") // Causes 'MtrxCns'

## MtrxIdnttyCns
[tagmtrxidnttycns]: # (mtrxidnttycns)

This error occurs when IdentityMatrix is called with invalid arguments. It expects a single dimension:

    var m = IdentityMatrix() // Causes 'MtrxIdnttyCns'

## MtrxInvldInit
[tagmtrxinvldinit]: # (mtrxinvldinit)

This error occurs when an invalid initializer is passed to the Matrix constructor:

    var m = Matrix([["invalid"]]) // Causes 'MtrxInvldInit' if incompatible

## MtrxInvldArg
[tagmtrxinvldarg]: # (mtrxinvldarg)

This error occurs when matrix arithmetic methods receive invalid arguments:

    var m = Matrix([[1,2],[3,4]])
    m + "string" // Causes 'MtrxInvldArg'

## MtrxRShpArg
[tagmtrxrshparg]: # (mtrxrshparg)

This error occurs when the reshape method is called with invalid arguments. It requires two integer arguments:

    var m = Matrix([[1,2],[3,4]])
    m.reshape("invalid") // Causes 'MtrxRShpArg'

## MtrxIncmptbl
[tagmtrxincmptbl]: # (mtrxincmptbl)

This error occurs when matrices have incompatible shapes for an operation. See the main documentation above for examples.

## MtrxSnglr
[tagmtrxsnglr]: # (mtrxsnglr)

This error occurs when attempting to invert a singular (non-invertible) matrix:

    var m = Matrix([[1,2],[2,4]]) // Singular matrix
    m.inverse() // Causes 'MtrxSnglr'

## MtrxNtSq
[tagmtrxntsq]: # (mtrxntsq)

This error occurs when a matrix operation requires a square matrix but a non-square matrix is provided:

    var m = Matrix([[1,2,3],[4,5,6]]) // 2x3 matrix
    m.inverse() // Causes 'MtrxNtSq'

## MtrxOpFld
[tagmtrxopfld]: # (mtrxopfld)

This error occurs when a matrix operation fails for an unspecified reason.

## MtrxNrmArgs
[tagmtrxnrmargs]: # (mtrxnrmargs)

This error occurs when the norm method is called with invalid arguments. It expects an optional numerical argument:

    var m = Matrix([[1,2],[3,4]])
    m.norm("invalid") // Causes 'MtrxNrmArgs'

## MtrxStClArgs
[tagmtrxstclargs]: # (mtrxstclargs)

This error occurs when `setColumn` is called with invalid arguments. It expects an integer column index and a column matrix:

    var m = Matrix([[1,2],[3,4]])
    m.setColumn("invalid", Matrix([1,2])) // Causes 'MtrxStClArgs'

The older method name `setcolumn` is retained for compatibility but is deprecated.

## LnAlgMtrxIncmptbl
[taglnalgmtrxincmptbl]: # (lnalgmtrxincmptbl)

This error occurs when matrices have incompatible shapes in linear algebra operations.

## LnAlgMtrxIndxBnds
[taglnalgmtrxindxbnds]: # (lnalgmtrxindxbnds)

This error occurs when a matrix index is out of bounds in linear algebra operations.

## LnAlgMtrxSnglr
[taglnalgmtrxsnglr]: # (lnalgmtrxsnglr)

This error occurs when a matrix is singular in linear algebra operations.

## LnAlgMtrxNtSq
[taglnalgmtrxntsq]: # (lnalgmtrxntsq)

This error occurs when a matrix is not square in linear algebra operations.

## LnAlgLapackArgs
[taglnalglapackargs]: # (lnalglapackargs)

This error occurs when a LAPACK function is called with invalid arguments.

## LnAlgMtrxOpFld
[taglnalgmtrxopfld]: # (lnalgmtrxopfld)

This error occurs when a matrix operation fails in the linear algebra library.

## LnAlgMtrxNtSpprtd
[taglnalgmtrxntspprtd]: # (lnalgmtrxntspprtd)

This error occurs when an operation is not supported for a particular matrix type.

## LnAlgMtrxInvldArg
[taglnalgmtrxinvldarg]: # (lnalgmtrxinvldarg)

This error occurs when invalid arguments are passed to a matrix method in the linear algebra library.

## LnAlgMtrxNnNmrclArg
[taglnalgmtrxnnnmrclarg]: # (lnalgmtrxnnnmrclarg)

This error occurs when a matrix method requires numerical arguments but receives non-numerical ones.

## LnAlgMtrxNrmArgs
[taglnalgmtrxnrmargs]: # (lnalgmtrxnrmargs)

This error occurs when the norm method is called with an unsupported argument. It requires 1 or inf:

    var m = Matrix([[1,2],[3,4]])
    m.norm(2) // Causes 'LnAlgMtrxNrmArgs' if 2 is not supported

## LnAlgInvldArg
[taglnalginvldarg]: # (lnalginvldarg)

This error occurs when matrix arithmetic methods receive invalid arguments:

    var m = Matrix([[1,2],[3,4]])
    m + "string" // Causes 'LnAlgInvldArg'

## SprsCns
[tagsprscns]: # (sprscns)

This error occurs when the Sparse constructor is called with invalid arguments. It should be called with dimensions or an array initializer:

    var s = Sparse("invalid") // Causes 'SprsCns'

## SprsInvldInit
[tagsprsinvldinit]: # (sprsinvldinit)

This error occurs when an invalid initializer is passed to the Sparse constructor.

## SprsSt
[tagsprsst]: # (sprsst)

This error occurs when attempting to set a sparse matrix element fails.

## SprsCnvFld
[tagsprscnvfld]: # (sprscnvfld)

This error occurs when sparse format conversion fails.

## SprsOpFld
[tagsprsopfld]: # (sprsopfld)

This error occurs when a sparse matrix operation fails.

## CmplxCns
[tagcmplxcns]: # (cmplxcns)

This error occurs when the Complex constructor is called with invalid arguments. It should be called with two floats:

    var c = Complex(1) // Causes 'CmplxCns'

## CmplxInvldArg
[tagcmplxinvldarg]: # (cmplxinvldarg)

This error occurs when complex arithmetic methods receive invalid arguments:

    var c = Complex(1, 2)
    c + "string" // Causes 'CmplxInvldArg'

## CmpxArg
[tagcmpxarg]: # (cmpxarg)

This error occurs when a complex operation receives unexpected arguments.

## LstArgs
[taglstargs]: # (lstargs)

This error occurs when a List is created with invalid arguments. Lists must be called with integer dimensions:

    var l = List("invalid") // Causes 'LstArgs'

## LstNumArgs
[taglstnumargs]: # (lstnumargs)

This error occurs when a List is indexed with more than one argument:

    var l = [1, 2, 3]
    l[1, 2] // Causes 'LstNumArgs'

## LstAddArgs
[taglstaddargs]: # (lstaddargs)

This error occurs when the add method receives invalid arguments. It requires a list:

    var l = [1, 2, 3]
    l.add("invalid") // Causes 'LstAddArgs'

## LstSrtFn
[taglstsrtfn]: # (lstsrtfn)

This error occurs when a list sort function doesn't return an integer:

    var l = [3, 1, 2]
    l.sort(fn(a, b) { return "invalid" }) // Causes 'LstSrtFn'

## EntryNtFnd
[tagentryntfnd]: # (entryntfnd)

This error occurs when an entry is not found in a list:

    var l = [1, 2, 3]
    l.remove(10) // Causes 'EntryNtFnd'

## TplArgs
[tagtplargs]: # (tplargs)

This error occurs when a Tuple is created with invalid arguments. Tuples must be called with integer dimensions:

    var t = Tuple("invalid") // Causes 'TplArgs'

## TpmNumArgs
[tagtpmnumargs]: # (tpmnumargs)

This error occurs when a Tuple is indexed with more than one argument:

    var t = (1, 2, 3)
    t[1, 2] // Causes 'TpmNumArgs'

## DctKyNtFnd
[tagdctkyntfnd]: # (dctkyntfnd)

This error occurs when a key is not found in a dictionary:

    var d = {"a": 1}
    print d["b"] // Causes 'DctKyNtFnd'

## DctStArg
[tagdctstarg]: # (dctstarg)

This error occurs when dictionary set methods (union, intersection, difference) receive invalid arguments. They expect a dictionary:

    var d1 = {"a": 1}
    d1.union("invalid") // Causes 'DctStArg'

## FlOpnFld
[tagflopnfld]: # (flopnfld)

This error occurs when a file cannot be opened:

    var f = File("nonexistent.txt", "read") // Causes 'FlOpnFld' if file doesn't exist

## FlNmMssng
[tagflnmssng]: # (flnmssng)

This error occurs when a filename is missing in a File operation:

    var f = File() // Causes 'FlNmMssng'

## FlNmArgs
[tagflnmargs]: # (flnmargs)

This error occurs when the first argument to File is not a filename:

    var f = File(123, "read") // Causes 'FlNmArgs'

## FlMode
[tagflmode]: # (flmode)

This error occurs when the second argument to File is not a valid mode. It should be 'read', 'write', or 'append':

    var f = File("test.txt", "invalid") // Causes 'FlMode'

## FlWrtArgs
[tagflwrtargs]: # (flwrtargs)

This error occurs when File.write receives non-string arguments:

    var f = File("test.txt", "write")
    f.write(123) // Causes 'FlWrtArgs'

## FlWrtFld
[tagflwrtfld]: # (flwrtfld)

This error occurs when writing to a file fails.

## FldrExpctPth
[tagfldrexpctpth]: # (fldrexpctpth)

This error occurs when folder methods receive invalid arguments. They expect a path:

    Folder.exists(123) // Causes 'FldrExpctPth'

## NtFldr
[tagntfldr]: # (ntfldr)

This error occurs when a path is not a folder:

    Folder.exists("file.txt") // May cause 'NtFldr' if it's a file, not a folder

## FldrCrtFld
[tagfldrcrtfld]: # (fldrcrtfld)

This error occurs when folder creation fails:

    Folder.create("/invalid/path") // Causes 'FldrCrtFld'

## RngArgs
[tagrngargs]: # (rngargs)

This error occurs when Range receives invalid arguments. It expects numerical arguments: a start, an end, and an optional stepsize:

    Range("invalid") // Causes 'RngArgs'

## RngStpSz
[tagrngstpsz]: # (rngstpsz)

This error occurs when a Range stepsize is too small:

    Range(0, 10, 0.0000001) // May cause 'RngStpSz' if too small

## ExpctNmArgs
[tagexpctnmargs]: # (expctnmargs)

This error occurs when a function expects numerical arguments but receives non-numerical ones:

    sqrt("string") // Causes 'ExpctNmArgs'

## ExpctArgNm
[tagexpctargnm]: # (expctargnm)

This error occurs when a function expects a single numerical argument but receives something else:

    abs() // Causes 'ExpctArgNm'

## TypArgNm
[tagtypargnm]: # (typargnm)

This error occurs when a function expects one argument but receives a different number:

    type() // May cause 'TypArgNm' if no arguments provided

## MnMxArgs
[tagmnmxargs]: # (mnmxargs)

This error occurs when min or max functions receive invalid arguments. They expect at least one numerical argument, list, or matrix:

    min() // Causes 'MnMxArgs'

## ApplyArgs
[tagapplyargs]: # (applyargs)

This error occurs when the apply function receives fewer than two arguments:

    apply() // Causes 'ApplyArgs'

## ApplyNtCllble
[tagapplyntcllble]: # (applyntcllble)

This error occurs when apply receives a non-callable object as its first argument:

    apply("not a function", [1, 2, 3]) // Causes 'ApplyNtCllble'

## FrmtArg
[tagfrmtarg]: # (frmtarg)

This error occurs when the format method receives invalid arguments. It requires a format string:

    "test".format(123) // Causes 'FrmtArg' if format string expected

## InvldFrmt
[taginvldfrmt]: # (invldfrmt)

This error occurs when an invalid format string is provided:

    "test".format("%Z") // May cause 'InvldFrmt' if %Z is invalid

## ErrorArgs
[tagerrorargs]: # (errorargs)

This error occurs when the Error constructor is called with invalid arguments. It must be called with a tag and a default message:

    Error("Tag") // Causes 'ErrorArgs'

## Err
[tagerr]: # (err)

This is a generic error tag used for general error conditions.

## EnmrtArgs
[tagenmrtargs]: # (enmrtargs)

This error occurs when the enumerate method receives invalid arguments. It expects a single integer argument:

    var obj = Object()
    obj.enumerate("invalid") // Causes 'EnmrtArgs'

## IndxArgs
[tagindxargs]: # (indxargs)

This error occurs when the index method receives invalid arguments. It expects a String property name:

    var obj = Object()
    obj.index(123) // Causes 'IndxArgs'

## SetIndxArgs
[tagsetindxargs]: # (setindxargs)

This error occurs when the setindex method receives invalid arguments. It expects an index and a value:

    var obj = Object()
    obj.setindex(1) // Causes 'SetIndxArgs' (missing value)

## RspndsToArg
[tagrspndstoarg]: # (rspndstoarg)

This error occurs when the respondsto method receives invalid arguments. It expects a single string argument or no argument:

    var obj = Object()
    obj.respondsto(123) // Causes 'RspndsToArg'

## HasArg
[taghasarg]: # (hasarg)

This error occurs when the has method receives invalid arguments. It expects a single string argument or no argument:

    var obj = Object()
    obj.has(123) // Causes 'HasArg'

## IsMmbrArg
[tagismmbrarg]: # (ismmbrarg)

This error occurs when the ismember method receives invalid arguments. It expects a single argument:

    var obj = Object()
    obj.ismember() // Causes 'IsMmbrArg'

## ObjCantClone
[tagobjcantclone]: # (objcantclone)

This error occurs when attempting to clone an object that cannot be cloned:

    var obj = Object()
    obj.clone() // May cause 'ObjCantClone' if cloning not supported

## ObjImmutable
[tagobjimmutable]: # (objimmutable)

This error occurs when attempting to modify an immutable object:

    var obj = Object()
    // If obj is immutable:
    obj.property = "value" // Causes 'ObjImmutable'

## ObjNoPrp
[tagobjnoprp]: # (objnoprp)

This error occurs when an object does not provide properties:

    var obj = Object()
    obj.property // May cause 'ObjNoPrp' if object doesn't support properties

## InvocationArgs
[taginvocationargs]: # (invocationargs)

This error occurs when Invocation is called with invalid arguments. It must be called with an object and a method name:

    Invocation() // Causes 'InvocationArgs'

## SystmSlpArgs
[tagsystmslpargs]: # (systmslpargs)

This error occurs when the sleep method receives invalid arguments. It expects a time in seconds:

    sleep("invalid") // Causes 'SystmSlpArgs'

## SystmStWrkDr
[tagsystmstwrkdr]: # (systmstwrkdr)

This error occurs when setting the working directory fails:

    System.setworkingdirectory("/invalid/path") // Causes 'SystmStWrkDr'

## SystmStWrkDrArgs
[tagsystmstwrkdrargs]: # (systmstwrkdrargs)

This error occurs when setworkingdirectory receives invalid arguments. It expects a path name:

    System.setworkingdirectory(123) // Causes 'SystmStWrkDrArgs'

## JSONPrsArgs
[tagjsonprsargs]: # (jsonprsargs)

This error occurs when JSON.parse receives invalid arguments. It requires a string:

    JSON.parse(123) // Causes 'JSONPrsArgs'

## JSONObjctKey
[tagjsonobjctkey]: # (jsonobjctkey)

This error occurs when a JSON object key is not a string:

    JSON.parse('{123: "value"}') // Causes 'JSONObjctKey'

## JSONNmbrFrmt
[tagjsonnmbrfrmt]: # (jsonnmbrfrmt)

This error occurs when a number in JSON is improperly formatted:

    JSON.parse('{"num": 1.2.3}') // Causes 'JSONNmbrFrmt'

## JSONExtrnsTkn
[tagjsonextrnstkn]: # (jsonextrnstkn)

This error occurs when there is an extraneous token after a JSON element:

    JSON.parse('{"a": 1} extra') // Causes 'JSONExtrnsTkn'

## JSONBlnkElmnt
[tagjsonblnkelmnt]: # (jsonblnkelmnt)

This error occurs when a blank element is found in JSON:

    JSON.parse('[,]') // Causes 'JSONBlnkElmnt'

## MshFlNtFnd
[tagmshflntfnd]: # (mshflntfnd)

This error occurs when a mesh file cannot be found:

    var m = Mesh("nonexistent.mesh") // Causes 'MshFlNtFnd'

## MshVrtMtrxDim
[tagmshvrtmtrxdim]: # (mshvrtmtrxdim)

This error occurs when vertex matrix dimensions are inconsistent with the mesh.

## MshLdVrtDim
[tagmshldvrtdim]: # (mshldvrtdim)

This error occurs when a vertex has inconsistent dimensions when loading a mesh file.

## MshLdVrtCrd
[tagmshldvrtcrd]: # (mshldvrtcrd)

This error occurs when a vertex has non-numerical coordinates when loading a mesh file.

## MshLdPrsErr
[tagmshldprserr]: # (mshldprserr)

This error occurs when there is a parse error in a mesh file.

## MshLdVrtNm
[tagmshldvrtnm]: # (mshldvrtnm)

This error occurs when an element has an incorrect number of vertices when loading a mesh file.

## MshLdVrtId
[tagmshldvrtid]: # (mshldvrtid)

This error occurs when a vertex id is not an integer when loading a mesh file.

## MshLdVrtNtFnd
[tagmshldvrtntfnd]: # (mshldvrtntfnd)

This error occurs when a vertex is not found when loading a mesh file.

## MshInvldDim
[tagmshinvlddim]: # (mshinvlddim)

This error occurs when `Mesh` is constructed with a negative dimension:

    var m = Mesh(-1) // Causes 'MshInvldDim'

## MshInvldId
[tagmshinvldid]: # (mshinvldid)

This error occurs when an invalid element id is used:

    var m = Mesh()
    m.element(-1) // Causes 'MshInvldId'

## MshAddGrdOutOfBnds
[tagmshaddgrdoutofbnds]: # (mshaddgrdoutofbnds)

This error occurs when attempting to add elements of a grade that exceeds the mesh's maximum grade:

    var m = Mesh()
    m.addgrade(10) // Causes 'MshAddGrdOutOfBnds' if max grade is lower

## MshAddSymMsngTrnsfrm
[tagmshaddsymmsngtrnsfrm]: # (mshaddsymmsngtrnsfrm)

This error occurs when addsymmetry receives an object that doesn't provide a transform method:

    var m = Mesh()
    var obj = Object()
    m.addsymmetry(obj) // Causes 'MshAddSymMsngTrnsfrm'

## SlBnd
[tagslbnd]: # (slbnd)

This error occurs when a mesh has no boundary elements:

    var m = Mesh()
    m.boundary() // Causes 'SlBnd' if no boundary exists

## SlMsh
[tagslmsh]: # (slmsh)

This error occurs when a set operation is applied to Selections that refer to different Meshes:

    var s1 = Selection(mesh1)
    var s2 = Selection(mesh2)
    s1.union(s2) // Causes 'SlMsh'

## FldArgs
[tagfldargs]: # (fldargs)

This error occurs when Field receives invalid optional arguments. It allows `grade` and `finiteelementspace` as optional arguments:

    Field(mesh, foo=1) // Causes 'FldArgs'

## FldBnds
[tagfldbnds]: # (fldbnds)

This error occurs when a Field index is out of bounds:

    var f = Field(mesh)
    f[100, 100, 100] // Causes 'FldBnds' if out of bounds

## FldIncmptbl
[tagfldincmptbl]: # (fldincmptbl)

This error occurs when fields have incompatible shapes:

    var f1 = Field(mesh1)
    var f2 = Field(mesh2)
    f1 + f2 // Causes 'FldIncmptbl' if shapes incompatible

## FldIncmptblVal
[tagfldincmptblval]: # (fldincmptblval)

This error occurs when an assignment value has an incompatible shape with field elements:

    var f = Field(mesh)
    f[0, 0, 0] = Matrix([[1,2,3,4]]) // Causes 'FldIncmptblVal' if shape doesn't match

## FldOp
[tagfldop]: # (fldop)

This error occurs when Field.op receives extra arguments that are not Fields:

    var f = Field(mesh)
    f.op(fn (x) x, "not a field") // Causes 'FldOp'

A non-callable first argument raises `MltplDsptchFld` instead.

## FldOpFn
[tagfldopfn]: # (fldopfn)

This error occurs when Field.op cannot construct a Field from the return value of the function:

    var f = Field(mesh)
    f.op(fn(x) { return "invalid" }, f) // Causes 'FldOpFn'

## FnSpcArgs
[tagfnspcargs]: # (fnspcargs)

This error occurs when `FiniteElementSpace` is given an invalid `grade` option. The constructor takes a label, with an optional integer grade:

    FiniteElementSpace("CG1", grade="x") // Causes 'FnSpcArgs'

## FnSpcNtFnd
[tagfnspcntfnd]: # (fnspcntfnd)

This error occurs when a function space cannot be found for the requested label and grade:

    FiniteElementSpace("nonexistent", grade=1) // Causes 'FnSpcNtFnd'

## FnctlELNtFnd
[tagfnctleltfnd]: # (fnctleltfnd)

This error occurs when a mesh doesn't provide elements of the grade a functional maps over, or the functional cannot act on that grade:

    var func = Volume()
    func.integrand(mesh) // Causes 'FnctlELNtFnd' on a surface mesh

Jump raises the same error if parent connectivity for the interface grade is missing. GradSq raises it on line meshes, where the gradient is not implemented.

## FnctlFESpc
[tagfnctlfespc]: # (fnctlfespc)

This error occurs when a Field's finite element space cannot be used with this functional. For example, integrating a line Field over area elements, or using a piecewise-constant (`CG0`) Field with `Jump` or `NormSq`:

    AreaIntegral(fn (x, q) q, Field(m, grade=1)).total(m) // Causes 'FnctlFESpc' on a surface mesh

## FnctlNoFESpc
[tagfnctlnofespc]: # (fnctlnofespc)

Line, area and volume integrals, and Jump, need a Field with a finite element space. This error is raised if the Field was created without one (`finiteelementspace=nil`). Leave that option off to use the default `CG1` space.

    LineIntegral(fn (x, q) q, Field(m, finiteelementspace=nil)).total(m) // Causes 'FnctlNoFESpc'

## FnctlArgs
[tagfnctlargs]: # (fnctlargs)

This error occurs when a functional constructor or prepare step is given invalid arguments (a missing Field, reference mesh, or option). 

    var func = Length()
    func.integrand() // Causes 'MltplDsptchFld'

## VolEnclZero
[tagvolenclzero]: # (volenclzero)

This error occurs when VolumeEnclosed detects an element of zero size. Check that a mesh point is not coincident with the origin:

    var func = VolumeEnclosed()
    func.total(mesh) // Causes 'VolEnclZero' if an element is coincident with the origin

## HydrglFldGrd
[taghydrglfldgrd]: # (hydrglfldgrd)

This error occurs when Hydrogel is given `phi0` as a Field that lacks scalar elements in the grade Hydrogel maps over.

## HydrglZrRfVl
[taghydrglzrrfvl]: # (hydrglzrrfvl)

This warning occurs when a Hydrogel reference element has a tiny volume.

## HydrglBnds
[taghydrglbnds]: # (hydrglbnds)

This warning occurs when `phi` is outside `(0, 1)` in a Hydrogel calculation. The value is clamped and evaluation continues.

## SclrPtFnCllbl
[tagsclrptfncllbl]: # (sclrptfncllbl)

This error occurs when a ScalarPotential function is not callable:

    var a = ScalarPotential()
    a.function = 0.4
    a.integrand(mesh) // Causes 'SclrPtFnCllbl'

## IntgrlArgs
[tagintgrlargs]: # (intgrlargs)

This error occurs when an Integral or Jump is constructed with invalid arguments. It requires a callable, followed by zero or more Fields. `method`, if present, must be a Dictionary:

    LineIntegral(fn (x) x[0], method="Foo") // Causes 'IntgrlArgs'

## IntgrlFld
[tagintgrlfld]: # (intgrlfld)

This error occurs when `grad` or `hess` cannot tell which Field you mean. Pass the Field object, not the interpolated value, if more than one Field is in scope:

    AreaIntegral(fn (x, fl, gl) grad(fl).inner(grad(g)), f, g) // Causes 'IntgrlFld'

## IntgrlDffEvl
[tagintgrldffevl]: # (intgrldffevl)

This error occurs when `grad` or `hess` evaluation fails in an Integral, or the finite element space does not support that derivative.

## IntgrlSpclFn
[tagintgrlspclfn]: # (intgrlspclfn)

This error occurs when a special function such as `tangent`, `normal` or `grad` is used outside an Integral, or on the wrong grade of element:

    tangent() // Causes 'IntgrlSpclFn'

## IntgrlNested
[tagintgrlnested]: # (intgrlnested)

This error occurs when an Integral or Jump is evaluated from inside another Integral or Jump integrand. Nested evaluation is not supported:

    AreaIntegral(fn (x) LineIntegral(fn (y) 1).total(m)).total(m) // Causes 'IntgrlNested'

## JumpUnimpl
[tagjumpunimpl]: # (jumpunimpl)

This error occurs when a Jump integrand uses an evaluation that is not implemented yet, such as a normal-derivative jump that the finite element space cannot provide.

## IntgrtrSbdvns
[tagintgrtrsbdvns]: # (intgrtrsbdvns)

This error occurs when too many subdivisions are needed in evaluating an integral, possibly indicating a singularity:

    // Occurs during numerical integration when subdivision limit is exceeded

## IntgrtrRlNtFnd
[tagintgrtrrlntfnd]: # (intgrtrrlntfnd)

This error occurs when an integrator quadrature rule cannot be found:

    var method = {"rule": "nonexistent"}
    // Causes 'IntgrtrRlNtFnd' when rule doesn't exist

## IntgrtrRlUnavlb
[tagintgrtrrlunavlb]: # (intgrtrrlunavlb)

This error occurs when no quadrature rule is available that matches the provided integrator method dictionary, including `"hybrid2d"` used outside two dimensions:

    var method = {"rule": "invalid", "degree": 100}
    // Causes 'IntgrtrRlUnavlb' if no matching rule

## IntgrtrMthdTyp
[tagintgrtrmthdtyp]: # (intgrtrmthdtyp)

This error occurs when an integrator method dictionary option has the wrong type, or when `errornorm` is not `"max"` or `"sum"`:

    var method = {"rule": 123} // Causes 'IntgrtrMthdTyp' since rule must be a String
    LineIntegral(fn (x) x[0], method={ "errornorm": "l2" }) // Causes 'IntgrtrMthdTyp'
    LineIntegral(fn (x) x[0], method={ "tol": "tight" }) // Causes 'IntgrtrMthdTyp'

## DbgSymbl
[tagdbgsymbl]: # (dbgsymbl)

This error occurs in the debugger when a symbol cannot be found in the current context:

    // Occurs when debugging and accessing a symbol that doesn't exist

## DbgSymblPrpty
[tagdbgsymblprpty]: # (dbgsymblprpty)

This error occurs in the debugger when a symbol lacks a requested property:

    // Occurs when debugging and accessing a property that doesn't exist

## DbgInvldRg
[tagdbginvldrg]: # (dbginvldrg)

This error occurs in the debugger when an invalid register is accessed:

    // Occurs when debugging and accessing an invalid register

## DbgInvldGlbl
[tagdbginvldglbl]: # (dbginvldglbl)

This error occurs in the debugger when an invalid global is accessed:

    // Occurs when debugging and accessing an invalid global

## DbgInvldInstr
[tagdbginvldinstr]: # (dbginvldinstr)

This error occurs in the debugger when an invalid instruction is encountered:

    // Occurs when debugging and encountering an invalid instruction

## DbgRgObj
[tagdbgrgobj]: # (dbgrgobj)

This error occurs in the debugger when a register doesn't contain an object:

    // Occurs when debugging and expecting an object in a register

## DbgStPrp
[tagdbgstprp]: # (dbgstprp)

This error occurs in the debugger when attempting to set a property on an object that doesn't support it:

    // Occurs when debugging and trying to set a property
