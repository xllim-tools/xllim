Here you will find instructions on how to compile and run Kernelo.
You may skip to Running Kernelo in * if you don't intend to build the module from source.

[[_TOC_]]

# Building on Ubuntu 20.04

## Install dependecies
```
sudo apt install gcc cmake python3-dev libatlas-base-dev libarmadillo-dev libboost-dev
```

## Build the Python extension
```
$ python3 setup.py build_ext --inplace -vvv
```
Now you can run import kernelo in Python:
```
>>> import kernelo
```

# Running Kernelo in Docker
TODO

# Running Kernelo in Vagrant
TODO

# Building on Ubuntu 18.04 (obsolete)

## Install pip
Change /usb/bin/python link to python3
```
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
sudo apt install python3-pip
python -V
pip -V
pip install gcovr
```

## Install a more recent cmake
[Source](https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line)
Ubuntu 18.04 provides an old version of cmake.
```
sudo apt purge --auto-remove cmake
sudo apt install software-properties-common
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt update
sudo apt install cmake
```

## Install a more recent gcc
[Source](https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/)
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9
gcc --version
```
