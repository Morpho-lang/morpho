# Morpho
The Morpho language. Morpho is a programmable environment for shape optimization. Morpho aims to be:

* **Familiar**. Morpho uses syntax similar to other C-family languages. The syntax fits on a postcard, so it's easy to learn.
* **Fast**. Morpho programs run as efficiently as other well-implemented dynamic languages like *wren* or *lua* (Morpho is significantly faster than Python, for example). Morpho leverages numerical libraries like *BLAS*, *LAPACK* and *SUITESPARSE* to provide high performance.
* **Class-based**. A morpho program involves creating and manipulating objects, which greatly simplifies operation.
* **Extendable**. Morpho is, in effect, an embeddable language oriented for scientific applications. Functionality is easy to add via packages.

Documentation is available on [readthedocs](https://morpho-lang.readthedocs.io/en/latest/) and via an extensive [user manual](https://github.com/Morpho-lang/morpho/blob/main/manual/manual.pdf).

*This material is based upon work supported by the National Science Foundation under grants DMR-1654283 and OAC-2003820.*

## Installation

Morpho can be installed as follows:

### macOS

1. Install the [Homebrew](https://brew.sh) package manager, following instructions on the homebrew site.

2. Install dependencies. Open the Terminal application and type:

    brew update

    brew install glfw suite-sparse

3. Obtain the source by cloning this repository:

    git clone https://github.com/Morpho-lang/morpho.git

4. Navigate to the `morpho5` folder within the downloaded repository and build the application:

    cd morpho/morpho5

    make install

(Some users may need to use `sudo make install`)

5. Navigate to the `morphoview` folder and build the viewer application:

    cd ../morphoview

    make install

(Some users may need to use `sudo make install`)

6. Check that the application works by typing

    morpho5

### Unix and Linux

2. Install morpho's dependencies using your distribution's package manager (or manually if you prefer). For example, on Ubuntu you would type

    sudo apt install libglfw3

    sudo apt install libsuitesparse-dev

    sudo apt install liblapacke

3. Obtain the source by cloning this repository:

    git clone https://github.com/Morpho-lang/morpho.git

4. Navigate to the `morpho5` folder within the downloaded repository and build the application:

    cd morpho/morpho5

    sudo make -f Makefile.linux install

5. Navigate to the `morphoview` folder and build the viewer application:

    cd ../morphoview

    sudo make -f Makefile.linux install

6. Check that the application works by typing

    morpho5

Note that the build script places morpho5 and morphoview in the `/usr/local` file structure; this can easily be changed if a different location is preferred.
