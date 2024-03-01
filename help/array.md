[comment]: # (Array class help)
[version]: # (0.5)

# Array
[tagarray]: # (Array)

Arrays are collection objects that can have any number of indices. Their size is set when they are created:

    var a[5]
    var b[2,2]
    var c[nv,nv,nv]

Values can be retrieved with appropriate indices:

    print a[0,0]

Arrays can be indexed with slices:

	print a[[0,2,4],2]
	print a[1,0..2]

Any morpho value can be stored in an array element

    a[0,0] = [1,2,3]

[showsubtopics]: # (subtopics)

## Dimensions
[tagdimensions]: # (Dimensions)

Get the dimensions of an Array object:

    var a[2,2]
    print a.dimensions() // expect: [ 2, 2 ]
