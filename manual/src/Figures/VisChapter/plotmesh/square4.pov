#include "colors.inc"
background { rgb <1, 1, 1> }
camera {location <0, 0, 10>up <0,1,0> right <-1.33,0,0> angle 18look_at <0, 0, 0> sky <0, 1, 0> }
sphere { <-0.5, -0.5, 0>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <0, -0.5, 0>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <0.5, -0.5, 0>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <-0.5, 0, 0>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <0, 0, 0>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <0.5, 0, 0>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <-0.5, 0.5, 0>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <0, 0.5, 0>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <0.5, 0.5, 0>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
mesh2 {
vertex_vectors { 24, 
<-0.5, -0.5, 0>, 
<0, -0.5, 0>, 
<-0.5, 0, 0>, 
<0, -0.5, 0>, 
<-0.5, 0, 0>, 
<0, 0, 0>, 
<0, -0.5, 0>, 
<0.5, -0.5, 0>, 
<0, 0, 0>, 
<0.5, -0.5, 0>, 
<0, 0, 0>, 
<0.5, 0, 0>, 
<-0.5, 0, 0>, 
<0, 0, 0>, 
<-0.5, 0.5, 0>, 
<0, 0, 0>, 
<-0.5, 0.5, 0>, 
<0, 0.5, 0>, 
<0, 0, 0>, 
<0.5, 0, 0>, 
<0, 0.5, 0>, 
<0.5, 0, 0>, 
<0, 0.5, 0>, 
<0.5, 0.5, 0>
}
normal_vectors { 24, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>, 
<0, 0, 1>
}
texture_list { 24, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }, 
texture{ pigment{ rgb <0.5, 0.5, 0.5>  } }
}
face_indices { 8, 
<0, 1, 2>,0,1,2, 
<3, 4, 5>,3,4,5, 
<6, 7, 8>,6,7,8, 
<9, 10, 11>,9,10,11, 
<12, 13, 14>,12,13,14, 
<15, 16, 17>,15,16,17, 
<18, 19, 20>,18,19,20, 
<21, 22, 23>,21,22,23, 
}
}
light_source {<10, 10, 10> color White}
light_source {<0, 0, 10> color White}
light_source {<-10, -10, 10> color White}
