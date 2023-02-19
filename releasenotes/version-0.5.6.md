# Release notes for 0.5.6

We're pleased to announce morpho 0.5.6, which contains a number of improvements, particularly focussed on performance and extensibility. 

## Parallelized Force and Energy calculations

Morpho now supports parallelized force and energy calculations, which can lead to significant speedups for some programs. To use these, run morpho with -w flag and supply the number of worker threads to use: 

    morpho5 -w4 program.morpho

Further features for parallel programming will appear in future releases. 

## Resources and Packages

The morpho runtime can now look for resource files---help files, morpho files, etc.---in multiple places. The default location is now configurable at installation, and also via a .morphopackages file stored in the user home directory. This enables morpho modules to live in their own git repository, together with resource files, and should make it easier for users to contribute to morpho. More details are in the dev guide. 

## Extensions

It's now possible to extend morpho through dynamic libraries written in C and linked at runtime. From the user's perspective, these work just like modules using the `import` keyword. 

## New manual chapter on visualization

We continue to improve the manual, and now include a chapter on visualization. The developer guide has also been updated. 

## Other improvements

* Improvements to morpho's object model. New Function, Closures and Invocation classes provided that respond to standard methods. 
* Fixes to some functionals to work correctly with 2D meshes. 
* You can now supply anonymous functions in the arguments to a function.
* You can set the minimum and maximum values for plotfield using the optional cmin and cmax and arguments. 
* Manual contains additional information on installation