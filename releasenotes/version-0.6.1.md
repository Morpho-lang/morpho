# Release notes for 0.6.1

We're pleased to announce Morpho 0.6.1, which incorporates very important new language features and sets morpho up for future improvements.

## Types

Morpho now supports types. Variables can be declared with a specified type like so

    String s = "Hello" 

and the type of function parameters can be specified like

    fn f(String s, List l) { }

## Multiple dispatch

Morpho now supports *multiple dispatch*, whereby you can define multiple implementations of a function that accept different parameter types. The correct implementation to use is selected at runtime: 

    fn f(String x) { }
    fn f(List x) { }

Methods defined on classes also support this mechanism. You can still specify parameters that without a type, in which case all types are accepted. Multiple dispatch is implemented efficiently (it incurs only a small overhead relative to a traditional function call) and is very useful to remove complex type checking. We are using this feature to improve how morpho works internally, as well as to implement new morpho packages. 

## Additional hessians 

LineCurvatureSq and LineTorsionSq now provide the hessian() method. 

## Preliminary support for finite element discretizations

We have begun to include support for additional discretizations beyond the linear elements supported by prior versions of Morpho in the codebase. This feature is a work in progress and not yet completely ready for use; we expect to complete it in forthcoming releases. 

## Minor fixes

* Bugfixes to parallelization.
* Error messages now refer to the module in which the error was found. 
* Can now call throw() and warning() directly on the Error class. 
