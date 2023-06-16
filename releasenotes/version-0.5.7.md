# Release notes for 0.5.7

We're pleased to announce morpho 0.5.7, which is the final release in the 0.5 series. This release contains a number of improvements and bugfixes. 

## Windows install instructions fixed

We have updated the installation instructions for Windows with this release to work with either WSL1 or WSL2. 

## Gradients in AreaIntegral and VolumeIntegral

You can now compute the local gradient of a field using the grad() function within the integrand supplied to AreaIntegral and VolumeIntegral. This significantly enhances the number of models morpho can handle. 

## Improved System class

* System.print(), System.readline() and System.sleep() methods added. 

* System.clock() now reports wall time (useful for testing the effect of parallelization)

* System.arguments() provides access to the command line options morpho was run with. 

## Minor improvements

* New Matrix.roll() and List.roll() methods shift the contents of a List or Matrix respectively. 

* Field.linearize() provides access to the underlying Matrix store. 

* IdentityMatrix() constructor function. 

* Debugger now supports printing of global variables and object properties. 

* Fix issues with compilation on some platforms.

* Experimental support for accessing integrand values for individual elements on some functionals. 

* Numerous minor bugfixes.
