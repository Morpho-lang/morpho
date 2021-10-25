[comment]: # (Morpho syntax help file)
[version]: # (0.5)

[toplevel]: #

# Syntax
[tagsyntax]: # (syntax)

Morpho provides a flexible object oriented language similar to other languages in the C family (like C++, Java and Javascript) with a simplified syntax.

Morpho programs are stored as plain text with the .morpho file extension. A program can be run from the command line by typing

    morpho5 program.morpho

## Comments
[tagcomment]: # (comment)
[tagcomments]: # (comments)
[tagcommentvar]: # (//)
[tagcommentvarr]: # (/*)
[tagcommentvarrr]: # (*/)
Two types of comment are available. The first type is called a 'line comment' whereby text after `//` on the same line is ignored by the interpreter.

    a.dosomething() // A comment

Longer 'block' comments can be created by placing text between `/*` and `*/`. Newlines are ignored

    /* This
       is
       a longer comment */

In contrast to C, these comments can be nested

    /* A nested /* comment */ */

enabling the programmer to quickly comment out a section of code.

## Symbols
[tagsymbols]: # (symbols)
[tagnames]: # (names)

Symbols are used to refer to named entities, including variables, classes, functions etc. Symbols must begin with a letter or underscore _ as the first character and may include letters or numbers as the remainder. Symbols are case sensitive.

    asymbol
    _alsoasymbol
    another_symbol
    EvenThis123
    YET_ANOTHER_SYMBOL

Classes are typically given names with an initial capital letter. Variable names are usually all lower case.

## Newlines
[tagnewlines]: # (newlines)
[tagnewline]: # (newline)

Strictly, morpho ends statements with semicolons like C, but in practice these are usually optional and you can just start a new line instead. For example, instead of

    var a = 1; // The ; is optional

you can simply use

    var a = 1

If you want to put several statements on the same line, you can separate them with semicolons:

    var a = 1; print a

There are a few edge cases to be aware of: The morpho parser works by accepting a newline anywhere it expects to find a semicolon. To split a statement over multiple lines, signal to morpho that you plan to continue by leaving the statement unfinished. Hence, do this:

    print a +
          1

rather than this:

    print a   // < Morpho thinks this is a complete statement
          + 1 // < and so this line will cause a syntax error


## Booleans
[tagtrue]: # (true)
[tagfalse]: # (false)
[tagbooleans]: # (true)

Comparison operations like `==`, `<` and `>=` return `true` or `false` depending on the result of the comparison. For example,

    print 1==2

prints `false`. The constants `true` or `false` are provided for you to use in your own code:

    return true

## Nil
[tagnil]: # (nil)

The keyword `nil` is used to represent the absence of an object or value.

Note that in `if` statements, a value of `nil` is treated like `false`.

    if (nil) {
        // Never executed.
    }

## Blocks
[tagblocks]: # (blocks)
[tagblock]: # (block)

Code is divided into *blocks*, which are delimited by curly brackets like this:

    {
      var a = "Hello"
      print a
    }

This syntax is used in function declarations, loops and conditional statements.

Any variables declared within a block become *local* to that block, and cannot be seen outside of it. For example,

    var a = "Foo"
    {
      var a = "Bar"
      print a
    }
    print a

would print "Bar" then "Foo"; the version of `a` inside the code block is said to *shadow* the outer version.

## Precedence
[tagprecedence]: # (precedence)

Precedence refers to the order in which morpho evaluates operations. For example,

    print 1+2*3

prints `7` because `2*3` is evaluated before the addition; the operator `*` is said to have higher precedence than `+`.

You can always modify the order of evaluation by using brackets:  

    print (1+2)*3 // prints 9

## Print
[tagprint]: # (print)

The `print` keyword is used to print information to the console. It can be followed by any value, e.g.

    print 1
    print true
    print a
    print "Hello"
