.. _installation-usage:

Installation and Usage
----------------------

xLLiM is distributed as a compiled shared library in a Docker container thus Docker is required. For this example we suggest you build the **Jupyter notebook image**.
The image is built from the offical [jupyter/scipy-notebook](https://jupyter-docker-stacks.readthedocs.io/en/latest/) image adapted to xLLiM dependencies. 
This image offers the familiar JupyterLab user interface within a Python-based datascience environment.

First steps
***********

1. Connect to Inria's GitLab:

.. code-block:: bash

   docker login registry.gitlab.inria.fr

2. Get the Dockerfile. You achieve this by using curl or wget:

.. code-block:: bash

   curl --location -o jupyter.Dockerfile "https://gitlab.inria.fr/kernelo-mistis/kernelo-gllim-is/-/raw/v2/jupyter.Dockerfile?ref_type=heads&inline=false"

3. Build the docker image named *xllim_jupyter_notebook*:

.. code-block:: bash

   docker build -f jupyter.Dockerfile -t "xllim_jupyter_notebook" .

4. Run the container *xllim_notebook* and bind the volume to your current working directory:

.. code-block:: bash

   docker run -it --name xllim_notebook --detach -p 8888:8888 -v "${PWD}":/home/jovyan/work xllim_jupyter_notebook

5. Get the JupyterLab web address and have fun!:

.. code-block:: bash

   docker logs xllim_notebook | grep -oP 'http://127.0.0.1:8888/lab\?token=\w+' | head -n 1

Use your container
******************

Once your container is set up it is very easy to use your xLLiM environment. 
All changes made to the docker container (installing packages, etc.) are **persistent**. 
Be careful not to delete your container, otherwise all modifications made within it would be lost. 
You can start and stop the container with the two simple commands below:

.. code-block:: bash

   docker stop xllim_notebook
   docker start xllim_notebook


Go on the JupyterLab web-based user interface. Here is your personal workspace tree structure:

.. code-block:: console

    /home/jovyan/
    +-- examples/
    |   +-- example_hapke.ipynb
    |   +-- example.ipynb
    |   +-- JSC1_BRDF.json
    +-- work/
        // your local directory bounded to this docker container


You can now open the *example.ipynb* notebook and try it !



:ref:`Main methods <main-methods>`