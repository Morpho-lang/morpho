[comment]: # (Dictionary help)
[version]: # (0.5)

# Dictionary
[tag]: # (Dictionary)

Dictionaries are collection objects that associate a unique *key* with a particular *value*. Keys can be any kind of morpho value, including numbers, strings and objects.

An example dictionary mapping states to capitals:

    var dict = { "Massachusetts" : "Boston",
                 "New York" : "Albany",
                 "Vermont" : "Montpelier" }

Lookup values by a given key with index notation:

    print dict["Vermont"]

You can change the value associated with a key, or add new elements to the dictionary like this:

    dict["Maine"]="Augusta"

Create an empty dictionary using the `Dictionary` constructor function:

    var d = Dictionary()

Loop over keys in a dictionary:

    for (k in dict) print k
