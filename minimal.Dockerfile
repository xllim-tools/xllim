ARG image
FROM $image

# Download and place xllim shared library in python global dist packages
RUN apt update
RUN apt install -y --no-install-recommends curl
RUN curl --location -o artefact.zip "https://gitlab.inria.fr/api/v4/projects/kernelo-mistis%2Fkernelo-gllim-is/jobs/artifacts/v2/download?job=build_job"
RUN unzip artefact.zip
RUN rm artefact.zip
RUN mv *.so /usr/lib/python3/dist-packages/xllim.so