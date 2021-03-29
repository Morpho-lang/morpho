[comment]: # Morpho language help file
[version]: # 0.5

#Functions
[tag]: # fn
[tag]: # fun
[tag]: # function

A function in morpho is defined with the `fn` keyword, followed by the function's name, a list of parameters enclosed in parentheses, and the body of the function in curly braces. This example computes the square of a number:

    fn sqr(x) {
      return x*x
    }

Once a function has been defined you can evaluate it like any other morpho function.

    print sqr(2)

#Return
[tag]: # return

The `return` keyword is used to exit from a function, optionally passing a given value back to the caller. `return` can be used anywhere within a function. The below example calculates the `n` th Fibonacci number,

    fn fib(n) {
      if (n<2) return n
      return fib(n-1) + fib(n-2)
    }

by returning early if `n<2`, otherwise returning the result by recursively calling itself.

#Variables
[tag]: # var

Variables are defined using the `var` keyword followed by the variable name:

    var a

Optionally, an initial assignment may be given:

    var a = 1

Variables defined in a block of code are visible only within that block, so

    var greeting = "Hello"
    {
        var greeting = "Goodbye"
        print greeting
    }
    print greeting

will print

*Goodbye*
*Hello*

Multiple variables can be defined at once by separating them with commas

    var a, b=2, c[2]=[1,2]

where each can have its own initializer (or not).

#Classes
[tag]: # class

Classes are defined using the `class` keyword followed by the name of the class.
The definition includes methods  that the class responds to. The special `init` method
is called whenever an object is created.

    class Cake {
        init(type) {
            self.type = type
        }

        eat() {
            print "A delicious "+type+" cake"
        }
    }

Objects are created by calling the class as if it was a function:

    var c = Cake("carrot")

Methods are called using the . operator:

    c.eat()

#Super
[tag]: # super

The keyword `super` allows you to access methods provided by an object's superclass rather than its own. This is particularly useful when the programmer wants a class to extend the functionality of a parent class, but needs to make sure the old behavior is still maintained.

For example, consider the following pair of classes:

    class Lunch {
        init(type) { self.type=type }
    }

    class Soup < Lunch {
        init(type) {
            print "Delicious soup!"
            super.init(type)
        }
    }

The subclass Soup uses `super` to call the original initializer.

#If
[tag]: # if
[tag]: # else

If allows you to selectively execute a section of code depending on whether a condition is met. The simplest version looks like this:

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

#While
[tag]: # while

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

#For
[tag]: # for
[tag]: # in

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

#Break
[tag]: # break

Break is used inside loops to finish the loop early. For example

    for (i in 1..5) {
        if (i>3) break // --.
        print i        //   | (Once i>3)
    }                  //   |
    ...                // <-'

would only print 1,2 and 3. Once the condition `i>3` is true, the `break` statement causes execution to continue after the loop body.

Both `for` and `while` loops support break.

#Continue
[tag]: # continue

Continue is used inside loops to skip over the rest of an iteration. For example

    for (i in 1..5) {     // <-.
        print "Hello"          |
        if (i>3) continue // --'
        print i
    }                     

prints "Hello" five times but only prints 1,2 and 3. Once the condition `i>3` is true, the `continue` statement causes execution to transfer to the start of the loop body.

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

#Indexing
[tag]: # [
[tag]: # ]
[tag]: # index
[tag]: # subscript

Morpho provides a number of collection objects, such as `List`, `Range`, `Array`, `Dictionary`, `Matrix` and `Sparse`, that can contain more than one value. Index notation is used to access elements of these objects.

To retrieve an item from a collection, you use the `[` and `]` brackets like this:

    var a = List("Apple", "Bag", "Cat")
    print a[0]

which prints *Apple*. Note that the first element is accessed with `0` not `1`.

Similarly, to set an entry in a collection, use:

    a[0]="Adder"

which would replaces the first element in `a` with `"Adder"`.

Some collection objects need more than one index,

    var a = Matrix([[1,0],[0,1]])
    print a[0,0]

and others such as `Dictionary` use non-numerical indices,

    var b = Dictionary()
    b["Massachusetts"]="Boston"
    b["California"]="Sacramento"

as in this dictionary of state capitals.
