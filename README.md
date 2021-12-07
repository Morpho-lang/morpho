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

```
brew update

brew install glfw suite-sparse
```

3. Obtain the source by cloning this repository:

```
git clone https://github.com/Morpho-lang/morpho.git
```

4. Navigate to the `morpho5` folder within the downloaded repository and build the application:

```
cd morpho/morpho5

make install
```

(Some users may need to use `sudo make install`)

5. Navigate to the `morphoview` folder and build the viewer application:

```
cd ../morphoview

make install
```

(Some users may need to use `sudo make install`)

6. Check that the application works by typing

```
morpho5
```

### Unix and Linux

2. Install morpho's dependencies using your distribution's package manager (or manually if you prefer). For example, on Ubuntu you would type
```
sudo apt install libglfw3

sudo apt install libsuitesparse-dev

sudo apt install liblapacke-dev
```

3. Obtain the source by cloning this repository:

```
git clone https://github.com/Morpho-lang/morpho.git
```

4. Navigate to the `morpho5` folder within the downloaded repository and build the application:

```
cd morpho/morpho5

sudo make -f Makefile.linux install
```

5. Navigate to the `morphoview` folder and build the viewer application:

```
cd ../morphoview

sudo make -f Makefile.linux install
```

6. Check that the application works by typing

```
morpho5
```

Note that the build script places morpho5 and morphoview in the `/usr/local` file structure; this can easily be changed if a different location is preferred.

### Windows via Windows Subsystem for Linux (WSL)

#### Install WSL

The instructions to install the Ubuntu App are [here](https://ubuntu.com/tutorials/ubuntu-on-windows#1-overview).

Once the Ubuntu terminal is working in Windows, you can install Morpho through it in almost the same way as a Linux system, with the addition of an X windows manager to handle visualizations.

Unless mentioned otherwise, all the commands below are run in the Ubuntu terminal.

#### Install Morpho

1\. Install the dependencies

You can install the dependencies using the Advanced Package Tool or apt.

First update the apt package list and then update existing packages.

```
sudo apt update

sudo apt upgrade
```


The dependencies for morpho can be then installed as follows:
```
sudo apt install libglfw3-dev

sudo apt install libsuitesparse-dev

sudo apt install liblapacke-dev

sudo apt install povray
```

To build the code you will also need to install build-essentials:

```
sudo apt install build-essential
```

2\. Obtain the morpho source by cloning the Morpho repository:

```
git clone https://github.com/Morpho-lang/morpho.git
```

3\. Navigate to the morpho5 folder within the downloaded repository and build the application:

```
cd morpho/morpho5

sudo make -f Makefile.linux install
```

4\. Navigate to the morphoview folder and build the viewer application:

```
cd ../morphoview

sudo make -f Makefile.linux install
```

5\. Check that the application works by typing

```
morpho5
```

6\. Get the visualization working in WSL:

Now a window manager must be installed so that the WSL can create windows.

On Windows, install [VcXsrv](https://sourceforge.net/projects/vcxsrv/).

It shows up as XLaunch in the Windows start menu.

Open Xlaunch. Then,

* choose 'Multiple windows', set display number to 0, and hit 'Next'
* choose `start no client' and hit 'Next'
* <b>Unselect</b> 'native opengl' and hit 'Next'
* Hit 'Finish'

In Ubuntu download a package containing a full suite of desktop utilities that allows for the use of windows.

```
sudo apt install ubuntu-desktop mesa-utils
````

Tell ubuntu which display to use

```
export DISPLAY=localhost:0
```

To set the DISPLAY variable on launch add the line
```
DISPLAY=localhost:0
```
to ~/.bashrc


Test that the window system is working
```
glxgears
```


7\. Test the thomson program!

Navigate to the thomson example in the examples directory and run it.

If you are in the `morphoview` directory

```
cd ../examples/thomson

morpho5 thomson.morpho
```

This example starts with randomly distributed charges on a sphere and minimizing electric potential.

It should generate an interactive figure of points on a sphere.
