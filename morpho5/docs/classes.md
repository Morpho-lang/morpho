[comment]: # (Morpho classes help file)
[version]: # (0.5)

[toplevel]: #

# Classes
[tagclass]: # (class)

Classes are defined using the `class` keyword followed by the name of the class.
The definition includes methods  that the class responds to. The special `init` method
is called whenever an object is created.

    class Cake {
        init(type) {
            self.type = type
        }

        eat() {
            print "A delicious "+self.type+" cake"
        }
    }

Objects are created by calling the class as if it was a function:

    var c = Cake("carrot")

Methods are called using the . operator:

    c.eat()

## Self
[tagself]: # (self)

The `self` keyword is used to access an object's properties and methods from within its definition.

    class Vehicle {
      init (type) { self.type = type }

      drive () { print "Driving my ${self.type}." }
    }

## Super
[tagsuper]: # (super)

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
