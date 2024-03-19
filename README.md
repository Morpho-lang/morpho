![Morpho](manual/src/Figures/morphologosmall.png#gh-light-mode-only)![Morpho](manual/src/Figures/morphologosmall-white.png#gh-dark-mode-only)

[![Build](https://github.com/Morpho-lang/morpho-libmorpho/actions/workflows/build.yml/badge.svg)](https://github.com/Morpho-lang/morpho-libmorpho/actions/workflows/build.yml)
[![Test suite](https://github.com/Morpho-lang/morpho-libmorpho/actions/workflows/buildandtest.yml/badge.svg)](https://github.com/Morpho-lang/morpho-libmorpho/actions/workflows/buildandtest.yml)

The Morpho language. Morpho is a programmable environment for shape optimization and scientific computing tasks more generally. Morpho aims to be:

* **Familiar**. Morpho uses syntax similar to other C-family languages. The syntax fits on a postcard, so it's easy to learn.
* **Fast**. Morpho programs run as efficiently as other well-implemented dynamic languages like *wren* or *lua* (Morpho is often significantly faster than Python, for example). Morpho leverages numerical libraries like *BLAS*, *LAPACK* and *SUITESPARSE* to provide high performance.
* **Class-based**. Morpho is highly object-oriented, which simplifies coding and enables reusability.
* **Extendable**. Functionality is easy to add via packages, both in Morpho and in C or other compiled languages.

*Morpho is based upon work supported by the National Science Foundation under grants DMR-1654283 and OAC-2003820.*

## Learn and use morpho

Documentation is available on [readthedocs](https://morpho-lang.readthedocs.io/en/latest/) and via an extensive [user manual](https://github.com/Morpho-lang/morpho/blob/main/manual/manual.pdf). A [Slack community](https://join.slack.com/t/morphoco/shared_invite/zt-1o6azavwl-XMtjjFwxW~P6C8rc~YbBlA) is also available for people interested in using morpho and seeking support. 

In academic publications, please cite morpho as: 

*Joshi, C. et al., "Morpho -- A programmable environment for shape optimization and shapeshifting problems", [arXiv:2208.07859](https://arxiv.org/abs/2208.07859) (2022)*
 
We expect to update this once the paper is published. 

Participation in the morpho community, both as users and developers, is bound by our [Code of Conduct](CODE_OF_CONDUCT.md). 

## Contributing 

Morpho is under active development and we welcome contributions! Please see the [Contributor's guide](CONTRIBUTING.md) for more information about how you can get involved in the morpho project. For those interested in extending morpho or working with the source a [Developer guide](https://github.com/Morpho-lang/morpho-devguide) is also provided in a separate repository. 

We provide a [Roadmap](https://github.com/Morpho-lang/morpho/wiki/Road-Map) for future development plans that might give you ideas for how you could contribute.

We also welcome bug reports and suggestions: Please feel free to use the *Issues* feature on our github repository to bring these to the developers' attention. 

## Installation

Code in this repository builds morpho as a shared library. 

Morpho can be installed as follows:

- [macOS](#macos)
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

These instructions assume a distribution that uses the apt package manager to obtain dependencies. You may need to find equivalent packages for other distributions. 

1\. Make sure your version of apt is up to date. 

```
sudo apt update
sudo apt upgrade
```

2\. Ensure you have basic developer tools and the Cmake build system installed. 

```
sudo apt install build-essential cmake 
```

3\. Install morpho's dependencies using your distribution's package manager. 

```
sudo apt install libglfw3-dev libsuitesparse-dev liblapacke-dev povray libfreetype6-dev
```

4\. Obtain the source by cloning this repository:

```
git clone https://github.com/Morpho-lang/morpho.git
```

5\. Build the morpho library:

```
cd morpho
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
sudo make install 
```

6\. You will also need to install the terminal application to use Morpho. This is hosted in a (separate git repository)[https://github.com/Morpho-lang/morpho-cli] but the process is very similar to libmorpho:

First navigate out of the morpho folder:

```
cd ../../
```

and then obtain and build morpho-cli:

```
git clone https://github.com/Morpho-lang/morpho-cli

cd morpho-cli
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
sudo make install 
```

7\. Check that the application works by typing

```
morpho6
```

8\. You should also install the morphoview application, which is hosted in a (separate git repository)[https://github.com/Morpho-lang/morpho-morphoview].

First navigate out of the morpho-cli folder:

```
cd ../../
```

and then obtain and build morpho-morphoview:

```
git clone https://github.com/Morpho-lang/morpho-morphoview

cd morpho-morphoview
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
sudo make install 
```

Note that the build script places libmorpho, morpho6 and morphoview in the `/usr/local` file structure; this can easily be changed if a different location is preferred. See the manual for details. 

### Windows via Windows Subsystem for Linux (WSL2)

#### Install WSL2

If you don't have WSL2 installed on your Windows computer, follow the instructions to install the Ubuntu App [here](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview). Follow all the steps in this link to ensure that graphics are working. 

#### Install Morpho

Once the Ubuntu terminal is working in Windows, you can install Morpho the same way as in Linux by running the commands in the [instructions](#unix-and-linux) in the Ubuntu terminal.

If you are using WSL2, then the installation is complete.

#### Graphics On WSL1

If you instead are working on WSL1, then you need to follow these instructions to get graphics running.

Unless mentioned otherwise, all the commands below are run in the Ubuntu terminal.

A window manager must be installed so that the WSL1 can create windows.

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
[![No NAN Boxing](https://github.com/Morpho-lang/morpho-libmorpho/actions/workflows/nonanboxing.yml/badge.svg)](https://github.com/Morpho-lang/morpho-libmorpho/actions/workflows/nonanboxing.yml)
