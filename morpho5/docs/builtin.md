[comment]: # (Builtin function help)
[version]: # (0.5)

# Builtin functions
[tagbuiltin]: # (builtin)

Morpho provides a number of built-in functions.

[showsubtopics]: # (subtopics)

## arctan
[tagarctan]: # (arctan)

Returns the arctangent of an input value that lies from `-Inf` to `Inf`. You can use one argument:

    print arctan(0) // expect: 0

or use two arguments to return the angle in the correct quadrant:

    print arctan(x, y)

Note the order `x`, `y` differs from some other languages.

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

## Apply
[tagapply]: # (apply)

Apply calls a function with the arguments provided as a list:

    apply(f, [0.5, 0.5]) // calls f(0.5, 0.5) 
    
It's often useful where a function or method and/or the number of parameters isn't known ahead of time. The first parameter to apply can be any callable object, including a method invocation or a closure. 

You may also instead omit the list and use apply with multiple arguments: 

    apply(f, 0.5, 0.5) // calls f(0.5, 0.5)
    
There is one edge case that occurs when you want to call a function that accepts a single list as a parameter. In this case, enclose the list in another list: 

    apply(f, [[1,2]]) // equivalent to f([1,2])
