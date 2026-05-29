[comment]: # (Dictionary help)
[version]: # (0.5)

# Dictionary
[tag]: # (Dictionary)

Dictionaries are collection objects that associate a unique *key* with a particular *value*. Keys can be any kind of morpho value, including numbers, strings and objects.

An example dictionary mapping states to capitals:

    var dict = { "Massachusetts" : "Boston",
                 "New York" : "Albany",
                 "Vermont" : "Montpelier" }

Look up values by a given key with index notation:

    print dict["Vermont"]

You can change the value associated with a key, or add new elements to the dictionary like this:

    dict["Maine"]="Augusta"

Create an empty dictionary using the `Dictionary` constructor function:

    var d = Dictionary()

Loop over keys in a dictionary:

    for (k in dict) print k

[showsubtopics]: # (subtopics)

## Keys
[tagkeys]: # (keys)

The `keys` method returns a Morpho `List` of the keys.

    var keys = dict.keys() // will return ["Massachusetts", "New York", "Vermont"]

## Contains
[tagcontains]: # (contains)

The `contains` method returns a `Bool` value for whether the `Dictionary` contains a given key.

    print dict.contains("Vermont") // true
    print dict.contains("New Hampshire") // false

## Remove
[tagremove]: # (remove)

The `remove` method removes a given key from the `Dictionary`.

    dict.remove("Vermont")
    print dict // { New York : Albany, Massachusetts : Boston }

## Clear
[tagclear]: # (clear)

The `clear` method removes all the `(key, value)` pairs from the dictionary, resulting in an empty dictionary.

    dict.clear()
    print dict // {  }

## Union
[tagunion]: # (union)

The `union` method combines two dictionaries. If the same key is present in both, the value from the second dictionary is used:

    var a = { "x" : 1, "y" : 2 }
    var b = { "y" : 5, "z" : 3 }
    print a.union(b) // { x : 1, y : 5, z : 3 }

The `+` operator provides the same operation for dictionaries:

    print a+b

## Intersection
[tagintersection]: # (intersection)

The `intersection` method returns a dictionary containing only the keys present in both dictionaries:

    var a = { "x" : 1, "y" : 2 }
    var b = { "y" : 5, "z" : 3 }
    print a.intersection(b)

## Difference
[tagdifference]: # (difference)

The `difference` method returns a dictionary containing only the keys present in the first dictionary but not the second:

    var a = { "x" : 1, "y" : 2 }
    var b = { "y" : 5 }
    print a.difference(b)

The `-` operator provides the same operation for dictionaries:

    print a-b
