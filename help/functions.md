[comment]: # (Morpho functions help file)
[version]: # (0.5)

[toplevel]: #

# Functions
[tagfn]: # (fn)
[tagfun]: # (fun)
[tagfunction]: # (function)

A function in morpho is defined with the `fn` keyword, followed by the function's name, a list of parameters enclosed in parentheses, and the body of the function in curly braces. This example computes the square of a number:

    fn sqr(x) {
      return x*x
    }

Once a function has been defined you can evaluate it like any other morpho function.

    print sqr(2)

Functions can accept optional parameters:

    fn fun(x, quiet=true ) { if (!quiet) print "Loud!" }

Morpho functions can also be defined to restrict the type of accepted parameters:

    fn f(List x) { print "A list!" }

Multiple implementations can be defined that accept different numbers of parameters and parameter types:

    fn f() { print "No parameters!" }
    fn f(String x) { print "A string!" }
    fn f(Tuple x) { print "A tuple!" }

The correct implementation is then selected at runtime: 

    f("Hello World!") // expect: A string! 

[show]: # (subtopics) 

## Variadic
[tagvariadic]: # (variadic)

As well as regular parameters, functions can also be defined with *variadic* parameters: 

    fn func(x, ...v) {
        for (a in v) print a
    }

This function can then be called with 1 or more arguments: 

    func(1)
    func(1, 2)
    func(1, 2, 3) // All valid! 

The variadic parameter `v` captures all the extra arguments supplied. Functions cannot be defined with more than one variadic parameter. 

You can mix regular, variadic and optional parameters. Variadic parameters come before optional parameters:

    fn func(x, ...v, optional=true) {
        // 
    }

## Optional
[tagoptional]: # (optional)

Functions can also be defined with *optional* parameters:

    fn func(a=1) {
        print a 
    }

Each optional parameter must be defined with a default value (here `1`). The function can then be called either with or without the optional parameter: 

    func()    // a == 1 due to default value
    func(a=2) // a == 2 supplied by the user

Note that optional parameters may not be typed. 

## Return
[tagreturn]: # (return)

The `return` keyword is used to exit from a function, optionally passing a given value back to the caller. `return` can be used anywhere within a function. The below example calculates the `n` th Fibonacci number,

    fn fib(n) {
      if (n<2) return n
      return fib(n-1) + fib(n-2)
    }

by returning early if `n<2`, otherwise returning the result by recursively calling itself.

## Signature
[tagsignature]: # (signature)

The *signature* of a function is a list of the types of arguments in its definition: 

    fn f(x) {}         // Accepts one parameter of any type
    fn f(x,y) {}       // Accepts two parameters of any type
    fn f(List x, y) {} // The first parameter must be a list
    fn f(List x, List y) {} // Both parameters must be lists

While you can define multiple implementations of a function that accept different parameter types, you can only define one implementation with a unique signature.

Note that optional and variadic parameters are not typed.

# Multiple dispatch
[tagmultiple]: # (multiple)
[tagdispatch]: # (dispatch)
[tagmultipledispatch]: # (multipledispatch)

Morpho supports *multiple dispatch*, whereby you can define several implementations of a function that accept different types:

    fn f(List x) { return "Accepts a list" }
    fn f(String x) { return "Accepts a string" }

Morpho chooses the appropriate implementation at runtime: 

    f([1,2,3]) // Selects the List implementation

Any classes you define can be used as types:

    class A {} 
    class B is A {} 
    class C is B {} 

    fn f(A x) { return "A" }
    fn f(B x) { return "B" }

Morpho selects the *closest match*, so that: 

    print f(A()) // expect: A
    print f(B()) // expect: B
    print f(C()) // expect: B

Class `C` inherits from both `B` and `A`, but because it directly inherits from `B`, that's the closer match. 

# Closures
[tagclosures]: # (closures)
[tagclosure]: # (closure)

Functions in morpho can form *closures*, i.e. they can enclose information from their local context. In this example, 

    fn foo(a) {
        fn g() { return a } 
        return g
    }

the function `foo` returns a function that captures the value of `a`. If we now try calling `foo` and then calling the returned functions,

    var p=foo(1), q=foo(2) 
    print p() // expect: 1 
    print q() // expect: 2
    
we can see that `p` and `q` seem to contain different copies of `g` that encapsulate the value that `foo` was called with. 

Morpho hints that a returned function is actually a closure by displaying it with double brackets: 

    print foo(1) // expect: <<fn g>> 
