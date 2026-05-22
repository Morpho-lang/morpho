[comment]: # (Morpho json help file)
[version]: # (0.5)

[toplevel]: #

# JSON
[tagjson]: # (json)

The `JSON` class provides import and export functionality for the JSON (JavaScript Object Notation) interchange file format as defined by IETF RFC 7159. 

To parse a string that contains JSON, use the `parse` method: 

    var a = JSON.parse("[1,2,3,4]")
    print a // expect: [ 1, 2, 3, 4 ]

Elements in the JSON string are converted to equivalent morpho values.

To convert basic data types to JSON, use the `JSON.tostring` class method: 

    var b = JSON.tostring([1,2,3])

The exporter supports `nil`, boolean values `true` and `false`, numbers, `String`s as well as `List` and `Dictionary` objects that may contain any of the supported types. 

[showsubtopics]: # (subtopics)

## Parse
[tagparse]: # (parse)

Parses a JSON string and converts it into the corresponding Morpho value:

    var a = JSON.parse("[1,2,3,4]")
    print a

If the JSON contains arrays or objects, these are converted into Morpho `List` and `Dictionary` values.

## tostring
[tagtostring]: # (tostring)

Converts supported Morpho values into a JSON string:

    var s = JSON.tostring([1,2,3])
    print s

This method supports `nil`, booleans, numbers, `String`, `List` and `Dictionary` values containing supported data.
