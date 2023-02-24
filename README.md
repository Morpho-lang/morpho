![Morpho](manual/src/Figures/morphologosmall.png#gh-light-mode-only)![Morpho](manual/src/Figures/morphologosmall-white.png#gh-dark-mode-only)

[![Build and Test](https://github.com/Morpho-lang/morpho/actions/workflows/CI.yml/badge.svg)](https://github.com/Morpho-lang/morpho/actions/workflows/CI.yml)
[![Examples](https://github.com/Morpho-lang/morpho/actions/workflows/Examples.yml/badge.svg)](https://github.com/Morpho-lang/morpho/actions/workflows/Examples.yml)

The Morpho language. Morpho is a programmable environment for shape optimization and scientific computing tasks more generally. Morpho aims to be:

* **Familiar**. Morpho uses syntax similar to other C-family languages. The syntax fits on a postcard, so it's easy to learn.
* **Fast**. Morpho programs run as efficiently as other well-implemented dynamic languages like *wren* or *lua* (Morpho is significantly faster than Python, for example). Morpho leverages numerical libraries like *BLAS*, *LAPACK* and *SUITESPARSE* to provide high performance.
* **Class-based**. A morpho program involves creating and manipulating objects, which greatly simplifies operation.
* **Extendable**. Functionality is easy to add via packages, both in Morpho and in C or other compiled languages.

*Morpho is based upon work supported by the National Science Foundation under grants DMR-1654283 and OAC-2003820.*

## Learn and use morpho

Documentation is available on [readthedocs](https://morpho-lang.readthedocs.io/en/latest/) and via an extensive [user manual](https://github.com/Morpho-lang/morpho/blob/main/manual/manual.pdf). A [Slack community](https://join.slack.com/t/morphoco/shared_invite/zt-1o6azavwl-XMtjjFwxW~P6C8rc~YbBlA) is also available for people interested in using morpho and seeking support. 

In academic publications, please cite morpho as: 

*Joshi, C. et al., "Morpho -- A programmable environment for shape optimization and shapeshifting problems", arXiv:2208.07859 (2022)*
 
We expect to update this once the paper is published. 

Participation in the morpho community, both as users and developers, is bound by our [Code of Conduct](CODE_OF_CONDUCT.md). 

## Contributing 

Morpho is under active development and we welcome contributions! Please see the [Contributor's guide](CONTRIBUTING.md) for more information about how you can get involved in the morpho project. For those interested in extending morpho or working with the source a [Developer guide](https://github.com/Morpho-lang/morpho/blob/main/devguide/devguide.pdf) is also provided. 

We provide a [Roadmap](https://github.com/Morpho-lang/morpho/wiki/Road-Map) for future development plans that might give you ideas for how you could contribute.

We also welcome bug reports and suggestions: Please feel free to use the *Issues* feature on our github repository to bring these to the developers' attention. 

## Installation

Morpho can be installed as follows:

- [macOS](#macos)
- [macOS M1](#macos-m1)
- [Unix and Linux](#unix-and-linux)
- [Windows](#windows-via-windows-subsystem-for-linux-wsl)

### macOS

The recommended approach to installing morpho on macOS is to use the [Homebrew](https://brew.sh) package manager.

1\. Install [Homebrew](https://brew.sh), following instructions on the homebrew site. 

2\. In the terminal type: 

```
brew update
brew tap morpho-lang/morpho
brew install morpho
```

You may be prompted by homebrew to install additional components. For some users, it may be necessary to install XCode from the App Store. We recommend you also obtain this git repository so that you can try the examples, read the manual etc. which are not installed by homebrew. To do so simply cd to any convenient folder in the Terminal and type:  

```
git clone https://github.com/Morpho-lang/morpho.git
```

### Unix and Linux

These instructions assume a distribution that uses the apt package manager. You may need to find equivalent packages for other distributions. 

1\. Make sure your version of apt is up to date. 

```
sudo apt update
sudo apt upgrade
```

2\. Ensure you have basic developer tools installed. Some distributions omit these to save space.

```
sudo apt install build-essential
```

3\. Install morpho's dependencies using your distribution's package manager. 

```
sudo apt install libglfw3-dev libsuitesparse-dev liblapacke-dev povray libfreetype6-dev
```

4\. Obtain the source by cloning this repository:

```
git clone https://github.com/Morpho-lang/morpho.git
```

5\. Navigate to the `morpho5` folder within the downloaded repository and build the application:

```
cd morpho/morpho5

sudo make -f Makefile.linux install
```

6\. Navigate to the `morphoview` folder and build the viewer application:

```
cd ../morphoview

sudo make -f Makefile.linux install
```

7\. Check that the application works by typing

```
morpho5
```

Note that the build script places morpho5 and morphoview in the `/usr/local` file structure; this can easily be changed if a different location is preferred. See the manual for details. 

### Windows via Windows Subsystem for Linux (WSL)

#### Install WSL

The instructions to install the Ubuntu App are [here](https://ubuntu.com/tutorials/ubuntu-on-windows#1-overview).

Once the Ubuntu terminal is working in Windows, you can install Morpho through it in almost the same way as a Linux system, with the addition of an X windows manager to handle visualizations.

Unless mentioned otherwise, all the commands below are run in the Ubuntu terminal.

#### Install Morpho

Follow the instructions for the linux install above. 

#### Get visualization working in WSL

Now a window manager must be installed so that the WSL can create windows.

1\. On Windows, install [VcXsrv](https://sourceforge.net/projects/vcxsrv/).

It shows up as XLaunch in the Windows start menu.

2\. Open Xlaunch. Then,

* choose 'Multiple windows', set display number to 0, and hit 'Next'
* choose `start no client' and hit 'Next'
* <b>Unselect</b> 'native opengl' and hit 'Next'
* Hit 'Finish'

3\. Within Ubuntu download a package containing a full suite of desktop utilities that allows for the use of windows.

```
sudo apt install ubuntu-desktop mesa-utils
````

4\. Tell ubuntu which display to use

```
export DISPLAY=localhost:0
```

5\. To set the DISPLAY variable on login type

```
echo export DISPLAY=localhost:0 >> ~/.bashrc
```

6\. Test that the window system is working
```
glxgears
```

#### Test that visualization works

Navigate to the thomson example in the examples directory and run it. If you are in the `morphoview` directory, you can do this by typing,

```
cd ../examples/thomson

morpho5 thomson.morpho
```

This example starts with randomly distributed charges on a sphere and minimizing electric potential. It should generate an interactively rotatable figure of points on a sphere. See the manual for details. 

---
### Other Tests
[![NoNaNBoxing Tests](https://github.com/Morpho-lang/morpho/actions/workflows/NoNanBoxing.yml/badge.svg)](https://github.com/Morpho-lang/morpho/actions/workflows/NoNanBoxing.yml)
[![Garbage Colletor Stress Test](https://github.com/Morpho-lang/morpho/actions/workflows/GarbageCollectorTest.yml/badge.svg)](https://github.com/Morpho-lang/morpho/actions/workflows/GarbageCollectorTest.yml)
