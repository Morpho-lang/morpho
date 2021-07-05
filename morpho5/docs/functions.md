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
