[comment]: # (Complex help)
[version]: # (0.5)

# Complex
[tagcomplex]: # (complex)
[tagim]: # (im)

Morpho provides complex numbers. The keyword `im` is used to denote the imaginary part of a complex number:

    var a=1+5im 
    print a*a

Print values on the unit circle in the complex plane:

    import constants 
    for (phi in 0..Pi:Pi/5) print exp(im*phi)

Get the real and imaginary parts of a complex number:

    print real(a) 
    print imag(a) 

or alternatively:

    print a.real()
    print a.imag() 

[showsuptopics]: # subtopics

## Angle
[tagangle]: # (angle)

Returns the angle `phi` associated with the polar representation of a complex number `r*exp(im*phi)`:

    print z.angle() 

## Conj
[tagconjugate]: # (conjugate)
[tagconj]: # (conj)

Returns the complex conjugate of a number:

    print z.conj() 
