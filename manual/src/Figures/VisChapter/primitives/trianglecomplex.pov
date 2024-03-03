#include "colors.inc"
background { rgb <1, 1, 1> }
camera {location <0, 0, 10>up <0,1,0> right <-1.33,0,0> angle 15look_at <0, 0, 0> sky <0, 1, 0> }
mesh2 {
vertex_vectors { 3, 
<0, 0, 0>, 
<1, 0, 0>, 
<0, 1, 0>
}
normal_vectors { 3, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>
}
face_indices { 1, 
<0, 1, 2>, 
}
 texture {  pigment { rgb <1, 0, 0>  } }
}
light_source {<10, 10, 10> color White}
light_source {<0, 0, 10> color White}
light_source {<-10, -10, 10> color White}
