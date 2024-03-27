[comment]: # (Morpho language help file)
[version]: # (0.5)

# Help
[tag]: # (help)

Morpho provides an online help system. To get help about a topic called `topicname`, type

    help topicname

A list of available topics is provided below and includes language keywords like `class`, `fn` and `for`, built in classes like `Matrix` and `File` or information about functions like `exp` and `random`.

Some topics have additional subtopics: to access these type

    help topic subtopic

For example, to get help on a method for a particular class, you could type

    help Classname.methodname

Note that `help` ignores all punctuation.

You can also use `?` as a shorthand synonym for `help`

    ? topic

A useful feature is that, if an error occurs, simply type `help` to get more information about the error.

[showtopics]: # (topics)

# Quit
[tagquit]: # (quit)

The `quit` CLI command quits `morpho` run in interactive mode and returns to the shell.
