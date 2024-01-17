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

The `keys` method returns a Morpho List of the keys.

    var keys = dict.keys() // will return ["Massachusetts", "New York", "Vermont"]

The `contains` method returns a Bool value for whether the Dictionary
contains a given key.

    print dict.contains("Vermont") // true
    print dict.contains("New Hampshire") // false

The `remove` method removes a given key from the Dictionary.

    dict.remove("Vermont")
    print dict // { New York : Albany, Massachusetts : Boston }

The `clear` method removes all the (key, value) pairs fromt the
dictionary, resulting in an empty dictionary. 

    dict.clear()

    print dict // {  }
