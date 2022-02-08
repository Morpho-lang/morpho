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

## Return
[tagreturn]: # (return)

The `return` keyword is used to exit from a function, optionally passing a given value back to the caller. `return` can be used anywhere within a function. The below example calculates the `n` th Fibonacci number,

    fn fib(n) {
      if (n<2) return n
      return fib(n-1) + fib(n-2)
    }

by returning early if `n<2`, otherwise returning the result by recursively calling itself.

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
