#include "colors.inc"
background { rgb <1, 1, 1> }
camera {location <2.28086, 6.08229, 7.60286>up <0,1,0> right <-1.33,0,0> angle 15look_at <0, 0, 0> sky <0, 1, 0> }
sphere { <-0.5, -0.5, -0.5>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <0.5, -0.5, -0.5>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <-0.5, 0.5, -0.5>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <0.5, 0.5, -0.5>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <-0.5, -0.5, 0.5>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <0.5, -0.5, 0.5>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <-0.5, 0.5, 0.5>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
sphere { <0.5, 0.5, 0.5>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
cylinder { <-0.5, -0.5, 0.5>, <0.5, -0.5, 0.5>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
cylinder { <0.5, -0.5, 0.5>, <0.5, 0.5, 0.5>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
cylinder { <-0.5, 0.5, 0.5>, <0.5, 0.5, 0.5>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
cylinder { <-0.5, -0.5, 0.5>, <-0.5, 0.5, 0.5>, 0.025 texture {  pigment { rgb <0.5, 0.5, 0.5> } } }
text {  ttf "cyrvetic.ttf" "0" 0.1, 0 
  pigment { rgb <1, 0, 0>  }
  scale 0.133333 
  translate -0.578868*x + -0.578868*y + -0.578868*z 
  matrix < 1,  0,  0, 
 0,  1,  0, 
 0,  0,  -1, 
 0,  0,  0> 
 }

text {  ttf "cyrvetic.ttf" "1" 0.1, 0 
  pigment { rgb <1, 0, 0>  }
  scale 0.133333 
  translate 0.478868*x + -0.578868*y + -0.578868*z 
  matrix < 1,  0,  0, 
 0,  1,  0, 
 0,  0,  -1, 
 0,  0,  0> 
 }

text {  ttf "cyrvetic.ttf" "2" 0.1, 0 
  pigment { rgb <1, 0, 0>  }
  scale 0.133333 
  translate -0.578868*x + 0.478868*y + -0.578868*z 
  matrix < 1,  0,  0, 
 0,  1,  0, 
 0,  0,  -1, 
 0,  0,  0> 
 }

text {  ttf "cyrvetic.ttf" "3" 0.1, 0 
  pigment { rgb <1, 0, 0>  }
  scale 0.133333 
  translate 0.478868*x + 0.478868*y + -0.578868*z 
  matrix < 1,  0,  0, 
 0,  1,  0, 
 0,  0,  -1, 
 0,  0,  0> 
 }

text {  ttf "cyrvetic.ttf" "4" 0.1, 0 
  pigment { rgb <1, 0, 0>  }
  scale 0.133333 
  translate -0.578868*x + -0.578868*y + 0.478868*z 
  matrix < 1,  0,  0, 
 0,  1,  0, 
 0,  0,  -1, 
 0,  0,  0> 
 }

text {  ttf "cyrvetic.ttf" "5" 0.1, 0 
  pigment { rgb <1, 0, 0>  }
  scale 0.133333 
  translate 0.478868*x + -0.578868*y + 0.478868*z 
  matrix < 1,  0,  0, 
 0,  1,  0, 
 0,  0,  -1, 
 0,  0,  0> 
 }

text {  ttf "cyrvetic.ttf" "6" 0.1, 0 
  pigment { rgb <1, 0, 0>  }
  scale 0.133333 
  translate -0.578868*x + 0.478868*y + 0.478868*z 
  matrix < 1,  0,  0, 
 0,  1,  0, 
 0,  0,  -1, 
 0,  0,  0> 
 }

text {  ttf "cyrvetic.ttf" "7" 0.1, 0 
  pigment { rgb <1, 0, 0>  }
  scale 0.133333 
  translate 0.478868*x + 0.478868*y + 0.478868*z 
  matrix < 1,  0,  0, 
 0,  1,  0, 
 0,  0,  -1, 
 0,  0,  0> 
 }

light_source {<10, 10, 10> color White}
light_source {<-10, -10, 10> color White}
