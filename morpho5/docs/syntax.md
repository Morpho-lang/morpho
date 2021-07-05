[comment]: # (Morpho syntax help file)
[version]: # (0.5)

[toplevel]: #

# Syntax
[tagsyntax]: # (syntax)

Morpho provides a flexible object oriented language similar to other languages in the C family (like C++, Java and Javascript) with a simplified syntax.

Morpho programs are stored as plain text with the .morpho file extension. A program can be run from the command line by typing

    morpho program.morpho

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

Morpho accepts newlines in place of a semicolon to end a statement.

    var a = 1; //

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

## Precedence

## Print
[tagprint]: # (print)

Print is used to print information to the console. It can be followed by any value, e.g.

    print 1
    print true
    print a
    print "Hello"
