🔥**NEW : xLLiM v2 is available**🔥
> The new version of xLLiM is available. It has a more efficient code structure, new optimized and parallelised methods and depenbdencies update (*armadillo10*, *pybind11*, *carma*) 

[[_TOC_]]
 
# Presentation

``xLLiM`` (formerly Kernelo) is a python module performing efficient inversion of high-dimensional models within a Bayesian framework using GLLiM (Gaussian Locally-Linear Mapping) model approximation. It integrates features such as:
* Forward model functionnals. These can be implemented as C++ or as pure Python functions. The forward model can be used to generate data following a distribution and refine GLLiM results by sampling the PDF using various strategies.
* Multi-initialization options
* Post-GLLiM refinement methods such as Importance Sampling (IS) and Iterative Mixture Importance Sampling (IMIS)
* Post-processing analysis, including confidence quantification on predictions, detection of multiple solutions, andpermutation of predictions in case of signal regularity.

Kernelo-GLLiM is distributed as a compiled shared library in a Docker container, and is integrated in the [Planet-GLLiM](https://gitlab.inria.fr/kernelo-mistis/planet-gllim-front-end) 
astrophysics application, that is distributed as a Docker image for local use and as a data processing service
on [Allgo-18](https://allgo18.inria.fr/).


# Documentation and API reference

The API reference of the xLLiM module is available [here](https://kernelo-mistis.gitlabpages.inria.fr/kernelo-gllim-is/). 
For more information you can find a complete scientific documentation in the [Planet-GLLiM documentation](https://kernelo-mistis.gitlabpages.inria.fr/planet-gllim-front-end/)


# Dependencies
The library only runs in a Docker container. All the dependencies are installed in the *xllim_notebook* and the minimal *xllim_runner* docker images. The running dependencies are
| Name                 | version           |
|----------------------|-------------------|
| Armadillo            | 10.8.2            |
| Python               | 3.10.12           |
| Numpy                | 1.21.5            |

The library is built with
| Name                 | version           |
|----------------------|-------------------|
| Ubuntu               | Jammy 22.04.3 LTS |
| g++                  | 11.4.0            |
| Armadillo            | 10.8.2            |
| OpenMP               | 4.5               |
| Python               | 3.10.12           |
| Pybind11             | 2.11.1            |
| Carma                | 0.6.7             |


# How to run xLLiM ?

xLLiM is distributed as a compiled shared library in a Docker container thus Docker is required. We highly recommand to run our library with Docker container technology. However is you are using Ubuntu 22.04 you can compile or run the module without Docker.

## Multi-platform installation with Docker

### Prerequisite
Docker Engine is available on a variety of Linux platforms, macOS and Windows 10. You can find the installation documentation on [Docker's documentation website](https://docs.docker.com/)
#### Linux & MacOs
* [Docker Engine](https://docs.docker.com/engine/) ([Installation instructions](https://docs.docker.com/engine/install/))
* [Docker Compose](https://docs.docker.com/compose/) ([Installation instructions](https://docs.docker.com/compose/install/))

#### Windows
* Docker Desktop that contains everything to run a Docker Compose ([Installation instructions](https://docs.docker.com/desktop/windows/install/)). You will be asked to install a linux kernel on the first Docker Desktop run. Choose Ubuntu-20.04. [Instructions to install the linux kernel](https://docs.microsoft.com/en-us/windows/wsl/install)

### The minimal xLLiM image

This image is built from Ubuntu 22.04 and contains sufficient dependencies for xLLiM to run. 

### First steps

1. Connect to Inria's GitLab
```
docker login registry.gitlab.inria.fr
```
2. Pull the docker image
```
docker pull registry.gitlab.inria.fr/kernelo-mistis/planet-gllim-front-end/xllim_runner:master
```
3. Create your container
```
docker run -it --name [myContainer] registry.gitlab.inria.fr/kernelo-mistis/planet-gllim-front-end/xllim_runner:master
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


### The Jupyter notebook image

The image is built from the offical [jupyter/scipy-notebook](https://jupyter-docker-stacks.readthedocs.io/en/latest/) image adapted to xLLiM dependencies. This image offers the familiar JupyterLab user interface within a Python-based datascience environment.

### First steps

1. Connect to Inria's GitLab
```
docker login registry.gitlab.inria.fr
```
2. Get the Dockerfile. You achieve this by using curl or wget.
```
curl --location -o jupyter.Dockerfile "https://gitlab.inria.fr/kernelo-mistis/kernelo-gllim-is/-/raw/v2/jupyter.Dockerfile?ref_type=heads&inline=false"
```
3. Build the docker image named *xllim_jupyter_notebook*
```
docker build -f jupyter.Dockerfile -t "xllim_jupyter_notebook" --no-cache-filter install .
```
4. Run the container *xllim_notebook* and bind the volume to your current working directory.
```
docker run -it --name xllim_notebook --detach -p 8888:8888 -v "${PWD}":/home/jovyan/work --user root -e CHOWN_EXTRA="/home/jovyan/work" xllim_jupyter_notebook
```
5. Get the JupyterLab web adress and have fun !
```
docker logs xllim_notebook | grep -oP 'http://127.0.0.1:8888/lab\?token=\w+' | head -n 1
```

### Use your container

Once your container is set up it is very easy to use your xLLiM environment. All changes made to the docker container (installing packages, etc.) are **persistent**. Be careful not to delete your container, otherwise all modifications made within it would be lost. You can start and stop the container with the two simple commands below:
```
docker stop xllim_notebook
docker start xllim_notebook
```


## Run on Ubuntu 22.04

``xLLiM`` is built on Ubuntu 22.04, so if it your OS you can run it without Docker. It may also work with other Linux distribution but it is not tested.
1. Get the xLLiM extension 
The extension .so file is then stored as artifact, and can be downloaded from [GitLab's CI page](https://gitlab.inria.fr/kernelo-mistis/kernelo-gllim-is/-/pipelines). Click on the menu on the right side of the latest succesful job and select ``build_job:archive``. This will download an ``archive.zip`` file containing the extension ``.so`` file.
2. Install dependecies
```
sudo apt install python3 python3-numpy libarmadillo10 libgomp1
```
3. Copy the .so extension file into your working directory and start Python 3
```
python3
>>> import xllim
```


## Build on Ubuntu 22.04

If you want to build the projet.

1. Clone the projet.
```
git clone https://gitlab.inria.fr/kernelo-mistis/kernelo-gllim-is.git
cd kernelo-gllim-is
```
2. Install xLLiM dependencies

```
sudo apt update

# Install compilation tools
sudo apt-get install -y --no-install-recommends g++ cmake make

# Install Armadillo-related dependencies
sudo apt-get install -y --no-install-recommends libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev libarmadillo-dev

# Install Python-related dependencies
sudo apt-get install -y --no-install-recommends python3-dev

# Boost required for boost/property_tree
sudo apt-get install -y --no-install-recommends libboost-dev
```
3. Build with CMake

You have to define target installation directory. It can indicate the path to your Python packages. The Debug build type can be either *Debug* (-O0 -g -fsanitize=address) or *Release* (-O2 -DNDEBUG).
```
cd /path/to/kernelo-gllim-is/build
cmake -S .. -B . -DPYTHON_LIBRARY_DIR="/home/user/.local/lib/python3.10/site-packages"  -DCMAKE_BUILD_TYPE=Debug
make install
```
4. Now you can import xLLiM in Python 3:
```
python3
>>> import xllim
```


# Licence
This software is licensed under the GNU GPL-compatible [CeCILL-C licence](LICENCE.txt).
While the software is free, we would appreciate it if you send us an e-mail at ``kernelo.gliim at inria.fr`` to let us know how you use it.
Also, please contact us if the licence does not meet your needs.
