[comment]: # (Parser module help)
[version]: # (0.5)

# Parser
[tagparser]: # (parser)

The `parser` module reads a stream of text one token at a time. Use it to scan a `File` or a string for words, numbers, and punctuation.

To use the module, first import it:

    import parser

Create a `Parser` from a `Tokenizer`, or directly from a stream. The stream form builds a `Tokenizer` with the default options. `next`, `peek`, and `eof` forward to that tokenizer.

Read a short expression:

    var tok = Tokenizer(StringStream("1 + 2.5e-1"), delimiters="+")
    var p = Parser(tok)
    print p.integer() // 1
    p.expect("+")
    print p.float()   // 0.25

[showsubtopics]: # (subtopics)

## StringStream
[tagstringstream]: # (stringstream)

A `StringStream` reads a string one character at a time, in the same way a `File` does. Create one from the text you want to scan:

    var s = StringStream("ab")
    print s.readChar() // a

`readChar` returns `nil` at the end of the string, and `eof` is then `true`. Pass a `StringStream` or a `File` to `Tokenizer`.

## Token
[tagtoken]: # (token)

A `Token` is one word, number, or punctuation mark from the input. It records where that text was found.

* `text` is the characters of the token.
* `line` and `column` are the location of its first character. Both start at 1.
* `quote` is `"` when the token was written in double quotes, and `nil` otherwise.

`prnt` prints the text together with its line and column:

    token.prnt()

## Tokenizer
[tagtokenizer]: # (tokenizer)

A `Tokenizer` splits a stream into `Token`s, skipping whitespace and comments. Create one from a `File` or a `StringStream`:

    var tok = Tokenizer(stream)
    var token = tok.next()

`next` returns the next `Token`, or `nil` at the end of the input. `peek(n)` looks ahead without consuming a token. `n` defaults to 0, so `peek()` is the next token, and `n` must be a nonnegative integer.

Optional arguments:

* `whitespace` characters to skip. The default is space, tab, carriage return, and newline.
* `delimiters` characters that are returned as their own tokens, such as `"+-*/"`.
* `lineComment` a nonempty string, such as `"//"`. The rest of the line is skipped, and the newline is left in place.
* `blockComment` a list of two nonempty strings, the start and end markers, such as `["/*", "*/"]`.

Text in double quotes is one token. `\"` and `\\` are escapes; any other backslash is kept. A `+` or `-` after `e` or `E` stays with the number, so `1e-3` is one token even when `-` is a delimiter.

## Expect
[tagexpect]: # (expect)

`expect` reads the next token and checks that its text matches:

    p.expect("+")

The matching `Token` is returned. If the input has ended, or the text is different, `expect` throws `ParseErr` and reports the line and column. Pass your own `Error` with `error=` to throw that instead:

    p.expect("+", error=myerr)

## Integer
[taginteger]: # (integer)

`integer` and `float` read the next token and convert it with `Int` and `Float`:

    print p.integer()
    print p.float()

A missing token, or a token that is not a number, throws `ParseErr`.

## TokenErr
[tagtokenerr]: # (tokenerr)

This error occurs when a `Tokenizer` is given a bad option, or when the input ends in the middle of a comment, quote, or escape. For example:

    var tok = Tokenizer(StringStream("/* oops"), blockComment=["/*", "*/"])
    tok.next() // Throws TokenErr

`lineComment` must be a nonempty string. `blockComment` must be a list of two nonempty strings. `peek` must be given a nonnegative integer.

To fix, close the comment or quote, or correct the optional argument.

## ParseErr
[tagparseerr]: # (parseerr)

This error occurs when `expect`, `integer`, or `float` does not find the text it expected. For example:

    var p = Parser(StringStream("1 + 2"))
    p.expect("*") // Throws ParseErr

The message includes the line and column of the token that was found, or reports that the input has ended.

To fix, expect the token that is actually next, or check it with `peek` before calling `expect`.
