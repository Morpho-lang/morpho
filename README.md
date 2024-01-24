# morpho-libmorpho

[![Build](https://github.com/Morpho-lang/morpho-libmorpho/actions/workflows/build.yml/badge.svg)](https://github.com/Morpho-lang/morpho-libmorpho/actions/workflows/build.yml)
[![Test suite](https://github.com/Morpho-lang/morpho-libmorpho/actions/workflows/buildandtest.yml/badge.svg)](https://github.com/Morpho-lang/morpho-libmorpho/actions/workflows/buildandtest.yml)
 
The morpho language built as a dynamic library for embedding in other applications. Part of development efforts for version 0.6.0 - presently not recommended for public use.

To install, clone this repository:

    git clone https://github.com/Morpho-lang/morpho-libmorpho.git

and then,

    cd morpho-libmorpho
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make install 

You may need to use sudo make install.

[![No NAN Boxing](https://github.com/Morpho-lang/morpho-libmorpho/actions/workflows/nonanboxing.yml/badge.svg)](https://github.com/Morpho-lang/morpho-libmorpho/actions/workflows/nonanboxing.yml)
