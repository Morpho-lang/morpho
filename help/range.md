[comment]: # (Morpho range help file)
[version]: # (0.5)

# Range
[tagrange]: # (range)

Ranges represent a sequence of numerical values. There are two ways to create them depending on whether the upper value is included or not:

    var a = 1..5  // inclusive version, i.e. [1,2,3,4,5]
    var b = 1...5 // exclusive version, i.e. [1,2,3,4]

By default, the increment between values is 1, but you can use a different value like this:

    var a = 1..5:0.5 // 1 - 5 with an increment of 0.5.

You can also create Range objects using the appropriate constructor function:

    var a = Range(1,5,0.5)

Ranges are particularly useful in writing loops:

    for (i in 1..5) print i

They can easily be converted to a list of values:

    var c = List(1..5)

To find the number of elements in a Range, use the `count` method

    print (1..5).count()

[showmethodsrange]: #
