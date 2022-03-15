# Building on Ubuntu 20.04

## Install dependecies
```
sudo apt install gcc cmake python3-dev libarmadillo-dev libboost-dev libgtest-dev
```
Apparently the cmake setup is not building the googletest code present in ``external/lib``.
The ``libgtest-dev`` of the system must be used istead.

## Build the project
```
$ mkdir build ; cd build
$ cmake ../
$ cmake --build .
```
## Genrating test coverage reports
Inria's SonarCube has only the community [C++ plugin](https://github.com/SonarOpenCommunity/sonar-cxx).
This means that it cannot process gcov coverage reports directly and they need to be converted to an XML format using ``gcovr``.

Installing gcovr:
```
$ pip install gcovr
```
Build the project for coverage test:
```
(build_dir)$ cmake -DCMAKE_BUILD_TYPE=Coverage ../
(build_dir)$ cmake --build .
```
Run whatever tests you have.
You may need to create links to input files.
```
()$ ln -s cpptest/functionalModel_tests/test_shkuratov.json .
()$ ln -s cpptest/functionalModel_tests/test_hapke.json .
(build_dir)$ ./TestLearning
(build_dir)$ ./TestGeneration
```
Run gcovr
```
(build_dir)$ gcovr -r ../ . --sonarqube coverage.xml
```
If you are mostly interested in par line count coverage, you may use ``--exclude-unreachable-branches --exclude-throw-branches`` options with ``gcovr`` to reduce the number of overly reported compilation branches.

To push the report to Sonarqube, make sure the  ``sonar.coverageReportPaths=build/coverage.xml`` property is set in ``sonar-project.properties`` file.

## Install Sonarqube scanner
[Source](https://techexpert.tips/sonarqube/sonarqube-scanner-installation-ubuntu-linux/)
```
sudo apt install unzip wget nodejs
wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.7.0.2747-linux.zip
unzip sonar-scanner-cli-4.7.0.2747-linux.zip
```
Add sonarscanner location to PATH, by adding it to ```/etc/profile.d/sonar-scanner.sh```.

# Building on Ubuntu 18.04

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
