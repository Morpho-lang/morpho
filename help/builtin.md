[comment]: # (Builtin function help)
[version]: # (0.5)

# Builtin functions
[tagbuiltin]: # (builtin)

Morpho provides a number of built-in functions.

[showsubtopics]: # (subtopics)

## Random
[tagrandom]: # (random)
[tagrand]: # (rand)

The `random` function generates a random number from a uniform distribution on the interval [0,1].

    print random() 

See also `randomnormal` and `randomint`.

## Randomnormal
[tagrandomnormal]: # (randomnormal)

The `randomnormal` function generates a random number from a normal (gaussian) distribution with unit variance and zero offset.

    print randomnormal() 

See also `random` and `randomint`.

## Randomint
[tagrandomint]: # (randomint)

The `randomint` function generates a random integer with a specified maximum value.

    print randomint(10) // Generates a random integer [0,10)

## isnil
[tagisnil]: # (isnil)

Returns `true` if a value is `nil` or `false` otherwise.

## isint
[tagisint]: # (isint)

Returns `true` if a value is an integer or `false` otherwise.

## isfloat
[tagisfloat]: # (isfloat)

Returns `true` if a value is a floating point number or `false` otherwise.

## isbool
[tagisbool]: # (isbool)

Returns `true` if a value is a boolean or `false` otherwise.

## isobject
[tagisobject]: # (isobject)

Returns `true` if a value is an object or `false` otherwise.

## isstring
[tagisstring]: # (isstring)

Returns `true` if a value is a string or `false` otherwise.

## isclass
[tagisclass]: # (isclass)

Returns `true` if a value is a class or `false` otherwise.

## isrange
[tagisrange]: # (isrange)

Returns `true` if a value is a range or `false` otherwise.

## isdictionary
[tagisdictionary]: # (isdictionary)

Returns `true` if a value is a dictionary or `false` otherwise.

## islist
[tagislist]: # (islist)

Returns `true` if a value is a list or `false` otherwise.

## isarray
[tagisarray]: # (isarray)

Returns `true` if a value is an array or `false` otherwise.

## ismatrix
[tagismatrix]: # (ismatrix)

Returns `true` if a value is a matrix or `false` otherwise.

## issparse
[tagissparse]: # (issparse)

Returns `true` if a value is a sparse matrix or `false` otherwise.

## isinf
[tagisinf]: # (isinf)

Returns `true` if a value is infinite or `false` otherwise.

## isnan
[tagisnan]: # (isnan)

Returns `true` if a value is a Not a Number or `false` otherwise.

## iscallable
[tagiscallable]: # (iscallable)

Returns `true` if a value is callable or `false` otherwise.

## isfinite
[tagisfinite]: # (isfinite)

Returns `true` if a value is finite or `false` otherwise.

    print isfinite(1) // expect: true 
    print isfinite(1/0) // expect: false 

## isnumber
[tagisnumber]: # (isnumber)

Returns `true` if a value is a real number, or `false` otherwise.

    print isnumber(1) // expect: true 
    print isnumber(Object()) // expect: false

## ismesh
[tagismesh]: # (ismesh)

Returns `true` if a value is a `Mesh`, or `false` otherwise.

## isselection
[tagisselection]: # (isselection)

Returns `true` if a value is a `Selection`, or `false` otherwise.

## isfield
[tagisfield]: # (isfield)

Returns `true` if a value is a `Field`, or `false` otherwise.

## Apply
[tagapply]: # (apply)

Apply calls a function with the arguments provided as a list:

    apply(f, [0.5, 0.5]) // calls f(0.5, 0.5) 
    
It's often useful where a function or method and/or the number of parameters isn't known ahead of time. The first parameter to apply can be any callable object, including a method invocation or a closure. 

You may also instead omit the list and use apply with multiple arguments: 

    apply(f, 0.5, 0.5) // calls f(0.5, 0.5)
    
There is one edge case that occurs when you want to call a function that accepts a single list as a parameter. In this case, enclose the list in another list: 

    apply(f, [[1,2]]) // equivalent to f([1,2])

## Abs
[tagabs]: # (abs)

Returns the absolute value of a number: 

    print abs(-10) // prints 10 

## Sign
[tagsign]: # (sign)

Gives the sign of a number: 

    print sign(4) // expect: 1
    print sign(-10.0) // expect: -1
    print sign(0) // expect: 0

## Arctan
[tagarctan]: # (arctan)

Returns the arctangent of an input value that lies from `-Inf` to `Inf`. You can use one argument:

    print arctan(0) // expect: 0

or use two arguments to return the angle in the correct quadrant:

    print arctan(x, y)

Note the order `x`, `y` differs from some other languages.

## Exp
[tagexp]: # (exp)

Exponential function `e^x`. Inverse of `log`.

    print exp(0) // expect: 1 
    print exp(Pi*im) // expect: -1 + 0im

## Log
[taglog]: # (log)

Natural logarithm function. Inverse of `exp`.

    print log(1) // expect: 0 

## Log10
[taglog10]: # (log10)

Base 10 logarithm function.

    print log10(10) // expect: 1

## Sin
[tagsin]: # (sin)

Sine trigonometric function.

    print sin(0) // expect: 0 

## Sinh
[tagsinh]: # (sinh)

Hyperbolic sine trigonometric function.

    print sinh(0) // expect: 0 

## Cos
[tagcos]: # (cos)

Cosine trigonometric function.

    print cos(0) // expect: 1

## Cosh
[tagcosh]: # (cosh)

Hyperbolic cosine trigonometric function.

    print cosh(0) // expect: 1

## Tan
[tagtan]: # (tan)

Tangent trigonometric function.

    print tan(0) // expect: 0 

## Tanh
[tagtanh]: # (tanh)

Hyperbolic tangent trigonometric function.

    print tanh(0) // expect: 0 

## Asin
[tagasin]: # (asin)

Inverse sine trigonometric function. Returns a value on the interval    `[-Pi/2,Pi/2]`.

    print asin(0) // expect: 0 

## Acos
[tagacos]: # (acos)

Inverse cosine trigonometric function. Returns a value on the interval  `[-Pi/2,Pi/2]`.

    print acos(1) // expect: 0 

## Sqrt
[tagsqrt]: # (sqrt)

Square root function.

    print sqrt(4) // expect: 2

## Min
[tagmin]: # (min)

Finds the minimum value of its arguments. If any of the arguments are Objects and are enumerable, (e.g. a `List`), `min` will search inside them for a minimum value. Accepts any number of arguments. 

    print min(3,2,1) // expect: 1 
    print min([3,2,1]) // expect: 1 
    print min([3,2,1],[0,-1,2]) // expect: -2 

## Max
[tagmax]: # (max)

Finds the maximum value of its arguments. If any of the arguments are Objects and are enumerable, (e.g. a `List`), `max` will search inside them for a maximum value. Accepts any number of arguments. 

    print min(3,2,1) // expect: 3 
    print min([3,2,1]) // expect: 3
    print min([3,2,1],[0,-1,2]) // expect: 3 

## Bounds
[tagbounds]: # (bounds)

Returns both the results of `min` and `max` as a list, Providing a set of bounds for its arguments and any enumerable objects within them.

    print bounds(1,2,3) // expect: [1,3]
    print bounds([3,2,1],[0,-1,2]) // expect: [-1,3]
