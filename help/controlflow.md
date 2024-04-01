[comment]: # (Morpho control flow help file)
[version]: # (0.5)

[toplevel]: #

# Control Flow
[tagcontrol]: # (control)

Control flow statements are used to determine whether and how many times a selected piece of code is executed. These include:

* `if` - Selectively execute a piece of code if a condition is met.
* `else` - Execute a different block of code if the test in an `if` statement fails.
* `for` - Repeatedly execute a section of code with a counter
* `while` - Repeatedly execute a section of code while a condition is true.

## If
[tagif]: # (if)
[tagelse]: # (else)

`If` allows you to selectively execute a section of code depending on whether a condition is met. The simplest version looks like this:

    if (x<1) print x

where the body of the loop, `print x`, is only executed if x is less than 1. The body can be a code block to accommodate longer sections of code:

    if (x<1) {
        ... // do something
    }

If you want to choose between two alternatives, use `else`:

    if (a==b) {
        // do something
    } else {
        // this code is executed only if the condition is false
    }

You can even chain multiple tests together like this:

    if (a==b) {
        // option 1
    } else if (a==c) {
        // option 2
    } else {
        // something else
    }

## While
[tagwhile]: # (while)

While loops repeat a section of code while a condition is true. For example,

    var k=1
    while (k <= 4) { print k; k+=1 }
           ^cond   ^body

prints the numbers 1 to 4. The loop has two sections: `cond` is the condition to be executed and `body` is the section of code to be repeated.

Simple loops like the above example, especially those that involve counting out a sequence of numbers, are more conveniently written using a `for` loop,

    for (k in 1..4) print k

Where `while` loops can be very useful is where the state of an object is being changed in the loop, e.g.

    var a = List(1,2,3,4)
    while (a.count()>0) print a.pop()

which prints 4,3,2,1.

## Do
[tagdo]: # (do)

A `do`...`while` loop repeats code while a condition is true---similar to a `while` loop---but the test happens at the end:

    var k=1
    do {
      print k;
      k+=1
    } while (k<5)

which prints 1,2,3,4

Hence this type of loop executes at least one interation

## For
[tagfor]: # (for)
[tagin]: # (in)

For loops allow you to repeatedly execute a section of code. They come in two versions: the simpler version looks like this,

    for (var i in 1..5) print i

which prints the numbers 1 to 5 in turn. The variable `i` is the *loop variable*, which takes on a different value each iteration. `1..5` is a range, which denotes a sequence of numbers. The *body* of the loop,  `print i`, is the code to be repeatedly executed.

Morpho will implicitly insert a `var` before the loop variable if it's missing, so this works too:

    for (i in 1..5) print i

If you want your loop variable to count in increments other than 1, you can specify a stepsize in the range:

    for (i in 1..5:2) print i
                   ^step

Ranges need not be integer:

    for (i in 0.1..0.5:0.1) print i

You can also replace the range with other kinds of collection object to loop over their contents:

    var a = Matrix([1,2,3,4])
    for (x in a) print x

Morpho iterates over the collection object using an integer *counter variable* that's normally hidden. If you want to know the current value of the counter (e.g. to get the index of an element as well as its value), you can use the following:

    var a = [1, 2, 3]
    for (x, i in a) print "${i}: ${x}"

Morpho also provides a second form of `for` loop similar to that in C:

    for (var i=0; i<5; i+=1) { print i }
         ^start   ^test ^inc.  ^body

which is executed as follows:
  start: the variable `i` is declared and initially set to zero.
  test: before each iteration, the test is evaluated. If the test is `false`, the loop terminates.
  body: the body of the loop is executed.
  inc: the variable `i` is increased by 1.

You can include any code that you like in each of the sections.

## Break
[tagbreak]: # (break)

`Break` is used inside loops to finish the loop early. For example

    for (i in 1..5) {
        if (i>3) break // --.
        print i        //   | (Once i>3)
    }                  //   |
    ...                // <-'

would only print 1, 2 and 3. Once the condition `i>3` is true, the `break` statement causes execution to continue after the loop body.

Both `for` and `while` loops support break.

## Continue
[tagcontinue]: # (continue)

`Continue` is used inside loops to skip over the rest of an iteration. For example

    for (i in 1..5) {     // <-.
        print "Hello"          |
        if (i>3) continue // --'
        print i
    }                     

prints "Hello" five times but only prints 1, 2 and 3. Once the condition `i>3` is true, the `continue` statement causes execution to transfer to the start of the loop body.

Traditional `for` loops also support `continue`:

                    // v increment
    for (var i=0; i<5; i+=1) {
        if (i==2) continue
        print i
    }

Since `continue` causes control to be transferred *to the increment section* in this kind of loop, here the program prints 0..4 but the number 2 is skipped.

Use of `continue` with `while` loops is possible but isn't recommended as it can easily produce an infinite loop!

    var i=0
    while (i<5) {
        if (i==2) continue
        print i
        i+=1
    }

In this example, when the condition `i==2` is `true`, execution skips back to the start, but `i` *isn't* incremented. The loop gets stuck in the iteration `i==2`.

## Try
[tagtry]: # (try)
[tagcatch]: # (catch)

A `try` and `catch` statement allow you handle errors. For example

    try {
      // Do something
    } catch {
      "Tag" : // Handle the error
    }

Code within the block after the `try` keyword is executed. If an error is generated then Morpho looks to see if the tag associated with the error matches any of the labels in the `catch` block. If it does, the code after the matching label is executed. If no error occurs, the catch block is skipped entirely.
