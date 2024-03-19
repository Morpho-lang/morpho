#include "colors.inc"
background { rgb <1, 1, 1> }
camera {location <0, 0, 10>up <0,1,0> right <-1.33,0,0> angle 15look_at <0, 0, 0> sky <0, 1, 0> }
sphere { <0, 0, 0>, 0.8 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
light_source {<10, 10, 10> color White}
light_source {<0, 0, 10> color White}
light_source {<-10, -10, 10> color White}
