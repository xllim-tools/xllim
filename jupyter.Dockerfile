# syntax=docker/dockerfile:1

FROM quay.io/jupyter/scipy-notebook:x86_64-ubuntu-22.04 AS jupyter_notebook

# Adjust the Jupyter image to xLLiM dependencies
RUN conda install -y --no-pin python=3.10.12
RUN conda install -y --no-pin numpy=1.21.5
RUN conda install -y --no-pin conda-forge::armadillo=10.8.2

# Download and place xllim shared library in python local packages
RUN mkdir -p /home/jovyan/.local/lib/python3.10/site-packages
RUN curl --location -o artefact.zip "https://gitlab.inria.fr/api/v4/projects/kernelo-mistis%2Fkernelo-gllim-is/jobs/artifacts/v2/download?job=build_job"
RUN unzip artefact.zip
RUN rm artefact.zip
RUN mv *.so /home/jovyan/.local/lib/python3.10/site-packages/xllim.so

# Copy the examples directory notebook
RUN curl --location -o examples.zip  "https://gitlab.inria.fr/kernelo-mistis/kernelo-gllim-is/-/archive/v2/kernelo-gllim-is-v2.zip?path=examples"
RUN unzip examples.zip
RUN mv kernelo-gllim-is-v2-examples/examples .
RUN rm examples.zip
RUN rmdir kernelo-gllim-is-v2-examples