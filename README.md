Here you will find instructions on how to compile and run Kernelo.
You may skip to Running Kernelo in * if you don't intend to build the module from source.

[[_TOC_]]

# Building on Ubuntu 20.04
1. Install dependecies
```
sudo apt install gcc cmake python3-dev libatlas-base-dev libarmadillo-dev libboost-dev libboost-all-dev libatlas-base-dev
```
2. Install python requirements
```
pip install -r requirements.txt
```
3. Build the Python extension
```
$ python3 setup.py build_ext --inplace -vvv
```
4. Now you can import kernelo in Python 3:
```
>>> import kernelo
```

# Running Kernelo without building it
Python extensions are build by the Gitlab CI every time a commit is made.
The extension .so file is then stored as artifact, and can be downloaded from
[GitLab's CI page](https://gitlab.inria.fr/kernelo-mistis/kernelo-gllim-is/-/pipelines).
Once you have the .so file you can import kernelo in Python 3:
```
>>> import kernelo
```

### Download the pre-built Python extension
1. Go to [GitLab's CI page](https://gitlab.inria.fr/kernelo-mistis/kernelo-gllim-is/-/pipelines).
2. Click on the menu on the right side of the latest succesful job and select ``build_job:archive``.
This will download an ``archive.zip`` file containing the extension ``.so`` file.

## Running Kernelo on Ubuntu 20.04
1. Install dependecies
```
sudo apt install python3 python3-numpy libatlas3-base libarmadillo9
```
2. Copy the .so extension file into your working directory and start Python 3
```
$ python3
>>> import kernelo
```

## Running Kernelo on Windows or Mac
Kernelo is build on Linux Ubuntu, but can be executed on other systems using virtualisation.
Here we provide instructions how to do it using Docker and Vagrant.

### Prerequisite
1. Clone the Kernelo repository
```
$ git clone git@gitlab.inria.fr:kernelo-mistis/kernelo-gllim-is.git Kernelo_dir
$ cd Kernelo_dir
```
2. Copy the .so extension file into ``Kernelo_dir``

### Using Docker
1. Install Docker following [these instrucitons]()
2. Build the runner image
```
$ docker build -f Builder.dockerfile --target runner -t kernelo_runner .
```
3. Start the runner with the ``Kernelo_dir`` mounted at ``/kernelo``, and go to that directory
```
$ docker run -i -t -v "$(pwd)":/kernelo kernelo_runner
root@kernelo_runner:/# cd kernelo/
```
4. Now you can import kernelo in Python 3:
```
root@kernelo_runner:/kernelo# python3
>>> import kernelo
```
Note that changes made to the docker image (installing packages etc.) are not persistent.

### Using Vagrant
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
Please refer to the [lincence page](LICENCE.txt)
If the licence does not meet your needs, please send a request at ``kernelo.gliim at inria.fr``.
