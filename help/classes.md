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

[showsubtopics]: # (subtopics)

## Is
[tagis]: # (is)

The `is` keyword is used to specify a class's superclass:

    class A is B {

    }

All methods defined by the superclass `B` are copied into the new class `A`, *before* any methods specified in the class definition. Hence, you can replace methods from the superclass simply by defining a method with the same name.

## With
[tagwith]: # (with)
[tagmixin]: # (mixin)

The `with` keyword is used together with `is` to insert additional methods into a class definition *without* making them the superclass. These are often called `mixins`. These methods are inserted after the superclass's methods. Multiple classes can be specified after `with`; they are added in the order specified.

    class A is B with C, D {

    }

Here `B` is the superclass of `A`, but methods defined by `C` and `D` are also available to `A`. If `B`, `C` and `D` define methods with the same name, those in `C` take precedence over any in `B` and those in `D` take precedence over `B` and `C`. 

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

    class Soup is Lunch {
        init(type) {
            print "Delicious soup!"
            super.init(type)
        }
    }

The subclass Soup uses `super` to call the original initializer.
