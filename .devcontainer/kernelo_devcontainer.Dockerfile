FROM kernelo_coverage
# In development mode we need all the tools from kernelo_builder and kernelo_coverage

# python runner
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils software-properties-common python3-pip vim
