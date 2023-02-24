[comment]: # (Values help)
[version]: # (0.5)

# Values
[tagvalues]: # (values)

Values are the basic unit of information in morpho: All functions in morpho accept values as arguments and return values. 

[showsubtopics]: # (subtopics)

## Int
[tagint]: # (int)

Morpho provides integers, which work as you would expect in other languages, although you rarely need to worry about the distinction between floats and integers. 

Convert a floating point number to an Integer: 

    print Int(1.3) // expect: 1

Convert a string to an integer:

    print Int("10")+1 // expect: 11

## Float
[tagfloat]: # (float)

Morpho provides double precision floating point numbers. 

Convert a string to a floating point number:

    print Float("1.2e2")+1 // expect: 121

## Ceil
[tagceil]: # (ceil)

Returns the smallest integer larger than or equal to its argument:

    print ceil(1.3) // expect: 2

## Floor
[tagfloor]: # (floor)

Returns the largest integer smaller than or equal to its argument:

    print floor(1.3) // expect: 1
