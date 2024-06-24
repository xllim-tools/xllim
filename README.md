# Summary

[[_TOC_]]


# Description

The Kernelo-GLLiM software is implemented in C++ with Python bindings written in Cython. 5.7k lines of C++
code implements the GLLiM solver, including 2k lines of tests and 800 lines of Cython.
It integrates features such as:
1. Forward model functionnals. These can be implemented as C++ or as pure Python functions. The forward
model can be used to generate data following a distribution and refine GLLiM results by sampling the PDF
using various strategies.
2. Multi-initialization options
3. Post-GLLiM refinement methods such as Importance Sampling (IS) and Iterative Mixture Importance Sampling
(IMIS)
4. Post-processing analysis, including confidence quantification on predictions, detection of multiple solutions, and
permutation of predictions in case of signal regularity.

Kernelo-GLLiM is distributed as a compiled shared library in a Docker container, and is integrated in the [Planet-GLLiM](https://gitlab.inria.fr/kernelo-mistis/planet-gllim-front-end) 
astrophysics application, that is distributed as a Docker image for local use and as a data processing service
on [Allgo-18](https://allgo18.inria.fr/).


# Documentation and API reference

The API reference of the Kernelo module is available [here](https://kernelo-mistis.gitlabpages.inria.fr/kernelo-gllim-is/). 
For more information you can find a complete scientific documentation in the [Planet-GLLiM documentation](https://kernelo-mistis.gitlabpages.inria.fr/planet-gllim-front-end/)


# How to run Kernelo-GLLiM ?

Kernelo-GLLiM is distributed as a compiled shared library in a Docker container thus Docker is required. However is you are using Ubuntu 20.04 you can compile or run the module without Docker.

## Using Docker

### First steps

1. Install Docker following [these instrucitons](https://docs.docker.com/engine/install/)
2. Connect to Inria's GitLab
```
docker login registry.gitlab.inria.fr
```
3. Pull the docker image
```
docker pull registry.gitlab.inria.fr/kernelo-mistis/planet-gllim-front-end/kernelo_python_runner:master
```
4. Create your container
```
docker run -it --name [myContainer] registry.gitlab.inria.fr/kernelo-mistis/planet-gllim-front-end/kernelo_python_runner:master
```
Once inside the container you can manage your workspace, install dependencies, run commands... Enter *exit* to exit the container.

### Use your container

5. Copy local files into your container.
```
docker cp [myFile] [myContainer]:/home/
```
6. Start your container
```
docker start [myContainer]
```
7. Enter into your container in interactive mode
```
docker exec -it [myContainer] bash
```
8. Stop your container
```
docker stop [myContainer]
```

Note that changes made to the docker container (installing packages etc.) are **persistent**. However be careful not to delete your container, otherwise all modifications made within it would be lost. You can also bind your container to a volume with -v option. More details at docker [documentation](https://docs.docker.com/reference/cli/docker/).


## Run on Ubuntu 20.04

Kernelo is build on Ubuntu 20.04, so if it your OS you can run it without Docker. It may also work with other Linux distribution but it is not tested.
1. Get the kernelo extension 
The extension .so file is then stored as artifact, and can be downloaded from [GitLab's CI page](https://gitlab.inria.fr/kernelo-mistis/kernelo-gllim-is/-/pipelines). Click on the menu on the right side of the latest succesful job and select ``build_job:archive``. This will download an ``archive.zip`` file containing the extension ``.so`` file.
2. Install dependecies
```
sudo apt install python3 python3-numpy libatlas3-base libarmadillo9
```
3. Copy the .so extension file into your working directory and start Python 3
```
python3
>>> import kernelo
```


## Build on Ubuntu 20.04

If you want to build the projet.

1. Clone the projet.
```
git clone https://gitlab.inria.fr/kernelo-mistis/kernelo-gllim-is.git
cd kernelo-gllim-is
```
2. Install dependecies
```
sudo apt update
sudo apt-get install -y --no-install-recommends gcc g++ \
	python3-dev cython3 python3-numpy python3-pip \
	libatlas-base-dev libarmadillo-dev libboost-dev
pip3 install -U pip wheel setuptools cyarma
```
3. Install python requirements
```
pip install -r requirements.txt
```
4. Build the Python extension
```
python3 setup.py bdist_wheel -vvv
```
5. Install the extension in your Python environnemnt.
Wheel file may have slightly different name.
```
pip3 install --no-cache-dir dist/kernelo-0.1-cp38-cp38-linux_x86_64.whl
```
6. Now you can import kernelo in Python 3:
```
python3
>>> import kernelo
```


## Using Vagrant *(not tested)*
Vagrant allows to run processes in a VM provided run by VirtualBox.
It simplifies the setup and configuration of the VM.
By default when you start a vagrant box (the VM) from a folder, this folder is made available in the VM at ``/vagrant/``.
You can run any scripts in this directory and write results in it, and they will be written in the directory from which you started the VM on your host.
1. Install Vagrant following instructions available [here](https://www.vagrantup.com/downloads)
2. Init a Ubuntu 20.04 Vagrant box in the project's directory
```
Kernelo_dir$ vagrant init ubuntu/focal64
```
3. Start the box, connect to it, install dependecies and go to the /vagrant direcotry on the box
```
Kernelo_dir$ vagrant up
Kernelo_dir$ vagrant ssh
vagrant-box:/$ sudo apt install python3 python3-numpy libatlas3-base libarmadillo9
vagrant-box:/$ cd /vagrant
vagrant-box:/vagrant/$
```
Now you can run your programs on the Vagrant box while using the /vagrant directory for input and output of data.
Any changes you make to the box (install packages etc.) are persistent, and you can safely logout from the box and shut it down with ``vagrant halt``.



# Licence
This software is licensed under the GNU GPL-compatible [CeCILL-C licence](LICENCE.txt).
While the software is free, we would appreciate it if you send us an e-mail at ``kernelo.gliim at inria.fr`` to let us know how you use it.
Also, please contact us if the licence does not meet your needs.
