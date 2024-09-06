# syntax=docker/dockerfile:1

FROM quay.io/jupyter/scipy-notebook:ubuntu-22.04 AS jupyter_notebook

# Adjust the Jupyter image to xLLiM dependencies
RUN conda install -y --no-pin python=3.10.12
RUN conda install -y --no-pin numpy=1.21.5
RUN conda install -y --no-pin conda-forge::armadillo=10.8.2

# Download and place xllim shared library in python local packages
RUN mkdir -p /home/jovyan/.local/lib/python3.10/site-packages
RUN curl --location -o artefact.zip "https://gitlab.inria.fr/api/v4/projects/kernelo-mistis%2Fkernelo-gllim-is/jobs/artifacts/v2/download?job=build_job"
# RUN curl --location --output artefact.zip --header "JOB-TOKEN: $CI_JOB_TOKEN" "https://gitlab.inria.fr/api/v4/projects/kernelo-mistis%2Fkernelo-gllim-is/jobs/artifacts/v2/download?job=build_job"'
RUN ls
RUN ls /home/jovyan/.local/lib/python3.10/site-packages
RUN unzip artefact.zip
RUN rm artefact.zip
RUN mv *.so /home/jovyan/.local/lib/python3.10/site-packages/xllim.so
RUN ls
RUN ls /home/jovyan/.local/lib/python3.10/site-packages
# COPY xllim.so /home/jovyan/.local/lib/python3.10/site-packages

# Copy the example notebook
RUN curl -o /home/jovyan/example_notebook.ipynb https://gitlab.inria.fr/kernelo-mistis/kernelo-gllim-is/-/raw/v2/tests/pythonTests/example_notebook.ipynb?ref_type=heads&inline=false
RUN ls
RUN ls /home/jovyan
# COPY example_notebook.ipynb /home/jovyan