# Generating test coverage reports

## Build the project and C++ tests
Apparently the cmake setup is not building the googletest code present in ``external/lib``.
The ``libgtest-dev`` of the system must be used istead.

```
$ mkdir build ; cd build
$ cmake ../
$ cmake --build .
```

# Install gcovr
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

# CI setup

* Get a VM with docker on it
