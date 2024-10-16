![Morpho](https://github.com/Morpho-lang/morpho-manual/blob/main/src/Figures/morphologosmall.png#gh-light-mode-only)![Morpho](https://github.com/Morpho-lang/morpho-manual/blob/main/src/Figures/morphologosmall-white.png#gh-dark-mode-only)

[![Build](https://github.com/Morpho-lang/morpho/actions/workflows/build.yml/badge.svg)](https://github.com/Morpho-lang/morpho/actions/workflows/build.yml)
[![Test suite](https://github.com/Morpho-lang/morpho/actions/workflows/buildandtest.yml/badge.svg)](https://github.com/Morpho-lang/morpho/actions/workflows/buildandtest.yml)

The Morpho language. Morpho is a programmable environment for shape optimization and scientific computing tasks more generally. Morpho aims to be:

* **Familiar**. Morpho uses syntax similar to other C-family languages. The syntax fits on a postcard, so it's easy to learn.
* **Fast**. Morpho programs run as efficiently as other well-implemented dynamic languages like *wren* or *lua* (Morpho is often significantly faster than Python, for example). Morpho leverages numerical libraries like *BLAS*, *LAPACK* and *SUITESPARSE* to provide high performance.
* **Class-based**. Morpho is highly object-oriented, which simplifies coding and enables reusability.
* **Extendable**. Functionality is easy to add via packages, both in Morpho and in C or other compiled languages.

*Morpho is based upon work supported by the National Science Foundation under grants DMR-1654283 and OAC-2003820.*

## Learn and use morpho

Documentation is available on [readthedocs](https://morpho-lang.readthedocs.io/en/latest/), an extensive [user manual](https://github.com/Morpho-lang/morpho-manual/blob/main/manual.pdf) and a [developer guide](https://github.com/Morpho-lang/morpho-devguide/blob/main/devguide.pdf). A [Slack community](https://join.slack.com/t/morphoco/shared_invite/zt-1o6azavwl-XMtjjFwxW~P6C8rc~YbBlA) is also available for people interested in using morpho and seeking support.

**New!** We now have a sequence of tutorial videos on our [Youtube channel](https://www.youtube.com/@Morpho-lang) to help you learn Morpho: 

* An [introduction to the Morpho language](https://youtu.be/eVPGWpNDeq4)
* Introduction to [shape optimization with Morpho](https://youtu.be/odCkR0PDKa0)

In academic publications, please cite morpho as:

*Joshi, C. et al., "Morpho -- A programmable environment for shape optimization and shapeshifting problems", [arXiv:2208.07859](https://arxiv.org/abs/2208.07859) (2022)*

We expect to update this once the paper is published.

Participation in the morpho community, both as users and developers, is bound by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Contributing

Morpho is under active development and we welcome contributions! Please see the [Contributor's guide](CONTRIBUTING.md) for more information about how you can get involved in the morpho project. For those interested in extending morpho or working with the source a [Developer guide](https://github.com/Morpho-lang/morpho-devguide) is also provided in a separate repository. 

We provide a [Roadmap](https://github.com/Morpho-lang/morpho/wiki/Road-Map) for future development plans that might give you ideas for how you could contribute.

We also welcome bug reports and suggestions: Please feel free to use the *Issues* feature on our github repository to bring these to the developers' attention.

## Installation

Code in this repository builds morpho as a shared library. Morpho also requires two subsidiary programs, a [terminal app](https://github.com/Morpho-lang/morpho-cli), and a [viewer application](https://github.com/Morpho-lang/morpho-morphoview).

For this release, morpho can be installed on all supported platforms using the homebrew package manager. Alternatively, the program can be installed from source as described below. We are continuously working on improving morpho installation, and hope to provide additional mechanisms for installation in upcoming releases.

### Installation with homebrew

The simplest way to install morpho is through the [homebrew package manager](https://brew.sh). To do so:

1. If not already installed, install homebrew on your machine as described on the [homebrew website](https://brew.sh)

2. Open a terminal and type:

```
brew update
brew tap morpho-lang/morpho
brew install morpho morpho-cli morpho-morphoview
```

If you need to uninstall morpho, simply open a terminal and type `brew uninstall morpho-cli morpho-morphoview morpho`. It's very important to uninstall the homebrew morpho in this way before attempting to install from source as below.

### Install from source

The second way to install morpho is by compiling the source code directly. Morpho now leverages the [Cmake](https://cmake.org) build system, which enables platform independent builds. Windows users must first install Windows Subsystem for Linux; some instructions to do so are found below.

#### Gather dependencies

You can use any appropriate package manager to install morpho's dependencies via the terminal. Using homebrew (preferred on macOS):

```
brew update
brew install cmake glfw suite-sparse freetype povray libgrapheme
```
Using apt (preferred on Ubuntu):

```
sudo apt update
sudo apt upgrade
sudo apt install build-essential cmake
sudo apt install libglfw3-dev libsuitesparse-dev liblapacke-dev povray libfreetype6-dev libunistring-dev
```

#### Build the morpho shared library

You can build the shared library, hosted in this repository.

1. Obtain the source by cloning the repository:

```
git clone https://github.com/Morpho-lang/morpho.git
```

2. Navigate to the morpho folder and build the library:

```
cd morpho
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
sudo make install
```

3. Navigate back out of the morpho folder:

```
cd ../../
```

#### Build the morpho terminal app

The [terminal app](https://github.com/Morpho-lang/morpho-cli) provides an interactive interface to morpho, and can also run morpho files. To build it:

1. Obtain the source by cloning the github public repository:

```
git clone https://github.com/Morpho-lang/morpho-cli.git
```

2. Navigate to the morpho-cli folder and build the library:

```
cd morpho-cli
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
sudo make install
```

3. Check it works by typing:

```
morpho6
```

4. Assuming that the morpho terminal app starts correctly, type `quit` to return to the shell and then

```
cd ../../
```

to navigate back out of the morph-cli folder.

#### Build the morphoview viewer application

[Morphoview](https://github.com/Morpho-lang/morpho-morphoview) is a simple viewer application to visualize morpho results.

1. Obtain the source by cloning the github public repository:

```
git clone https://github.com/Morpho-lang/morpho-morphoview.git
```

2. Navigate to the morpho-cli folder and build the library:

```
cd morpho-morphoview
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
sudo make install
```

3. Check it works by typing:

```
morphoview
```

which should simply run and quit normally. You can then type

```
cd ../../
```

to navigate back out of the morpho-morphoview folder.

### Windows via Windows Subsystem for Linux (WSL2)

Windows support is provided through Windows Subsystem for Linux (WSL), which is an environment that enables windows to run linux applications. We highly recommend using WSL2, which is the most recent version and provides better support for GUI applications; some instructions for WSL1 are provided [in the manual](https://github.com/Morpho-lang/morpho-manual/blob/main/manual.pdf). Detailed information on running GUI applications in WSL2 is found on the [Microsoft WSL support page](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps).

1. Begin by installing the [Ubuntu App](https://ubuntu.com/desktop/wsl) from the Microsoft store. 

2. Once the Ubuntu terminal is working in Windows, you can install morpho either through homebrew or by building from source. 

---
### Other Tests
[![No NAN Boxing](https://github.com/Morpho-lang/morpho/actions/workflows/nonanboxing.yml/badge.svg)](https://github.com/Morpho-lang/morpho/actions/workflows/nonanboxing.yml)
