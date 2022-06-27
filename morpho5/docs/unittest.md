[comment]: # (Unittest module help)
[version]: # (0.5)

# Unit Test

[tagunittest]: # (unittest)

The `unittest` module is used to facilitate creating unit test to lock down behavior in code. Once you have unit tests written for your code they can be run anytime the code is changed to verify the expected behavior.

To create a unit test you first import the module

    import unittest

Then create a new test class that inherits from the base class `unittest` and end the definition by creating an instance of the test class

    exampleTest is UnitTest { 
        ...
    } exampleTest()

Methods of this class are then test methods that run when an instance of this class is created. It is recommended that these methods be named so that they let the reader know exactly what is being tested. These methods are not part of the source code and never called by name so brevity is not a consideration.

## Assert

In a test method `self.assert(value)` will test if the value is true and `self.assertEqual(value1, value2)` will test if the contents of the values are equal. If an assert fails then the test that calls it will fail with an appropriate message. If you want to check set equality of a list you can use `self.assertSetEqualityOfList(list1,list2)`.

## Test Fixtures

The special method `fixture()` will run before each of the test methods are executed. This can be used to set up common data for tests.

## Floating point tolerance

Equality for floating point numbers defaults to a tolerance of 1e-10 but can be changed by setting the `self.tol` property in a test or fixture.

## Assume Fail

Sometimes it is beneficial to write a test when you have identified a bug but have yet to fix it. In these cases setting

    self.assumeFail = "Reason"

will force a test to pass regardless of asserts or errors and display a message that the test was assumed to fail because of the `"Reason"`.

## Example

The following is an example test for the exponential operator with a fixture and an test that assumes failure

    import unittest
    class power_spec is UnitTest {
        fixture(){
            self.myNumber = 3
        }
        integer_raised_to_an_integer(){
            self.assertEqual(2^2,4)
        }
        float_raised_to_an_integer(){
            self.assertEqual(2.0^3,8)
        }
        integer_raised_to_a_float(){
            self.assertEqual(2^3.0, 8)
        }
        float_raised_to_a_float(){
            self.assertEqual(2.0^3.0, 8 )
        }
        exponent_binds_before_uminus(){
            self.assertEqual(-1^2, -1)
            var x = -1
            self.assertEqual(x^2, 1)
            self.assertEqual((-1)^2, 1)
        }
        higher_exponents_happen_first(){
            self.assertEqual(2^3^2, 512)
        }
        variable_to_exponent(){
            self.assertEqual(self.myNumber^self.myNumber,27)
        }
        assume_failure(){
            self.assumeFail = "Example Reason"
            self.assertEqual(2^3,7)
        }
    } power_spec()

This produces the following report:

    ===================================
    Running test suite: @power_spec
    ===================================
    Running test higher_exponents_happen_first:
    Passed
    Running test variable_to_exponent:
    Passed
    Running test exponent_binds_before_uminus:
    Passed
    Running test integer_raised_to_a_float:
    Passed
    Running test float_raised_to_a_float:
    Passed
    Running test assume_failure:
    Assumed Fail: Example Reason
    Running test integer_raised_to_an_integer:
    Passed
    Running test float_raised_to_an_integer:
    Passed
    ===================================
        Ran 8 tests  
        No Failures Detected        
    ===================================

A failure would look like

    Running test assume_failure:
    Assertion Failed 8 != 7, difference is 1 which is above the tolerance of 1e-10
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     assume_failure: FAILED
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
