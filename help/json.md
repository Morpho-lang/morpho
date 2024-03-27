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

To convert basic data types to JSON, use the `tostring` method: 

    var b = JSON.tostring([1,2,3])

The exporter supports `nil`, boolean values `true` and `false`, numbers, `String`s as well as `List` and `Dictionary` objects that may contain any of the supported types. 
