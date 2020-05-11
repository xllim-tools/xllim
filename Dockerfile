FROM jupyter/base-notebook:latest

MAINTAINER sami-djouadi

USER root

RUN apt-get update && \
	apt-get install -y --no-install-recommends build-essential cmake g++ python3.7 python3.7-dev libopenblas-dev liblapack-dev libatlas-base-dev gfortran

RUN mkdir /home/libraries && \
	cd /home/libraries && \
	echo "installing OpenBLAS" && \
	wget --quiet https://github.com/xianyi/OpenBLAS/archive/v0.3.9.tar.gz && \
	tar -xzf v0.3.9.tar.gz && \
	cd OpenBLAS-0.3.9/ && \
	make || true && \
	cd .. && \
	rm v0.3.9.tar.gz && \
	rm -r OpenBLAS-0.3.9

RUN cd /home/libraries && \
	echo "installing LAPACK" && \
	wget https://github.com/Reference-LAPACK/lapack/archive/v3.9.0.tar.gz && \
	tar -xzf v3.9.0.tar.gz && \
	cd lapack-3.9.0/ && \
	cp make.inc.example make.inc && \
	make || true && \
	cd .. && \
	rm v3.9.0.tar.gz && \
	rm -r lapack-3.9.0

RUN cd /home/libraries && \
	echo "installing ARMADILLO" && \
	wget --quiet http://sourceforge.net/projects/arma/files/armadillo-9.870.2.tar.xz && \
	tar -xJf armadillo-9.870.2.tar.xz && \
	cd armadillo-9.870.2/ && \
	cmake . && \
	make && \
	make install && \
	cd .. && \
	rm armadillo-9.870.2.tar.xz && \
	rm -r armadillo-9.870.2

COPY kernelo.cpython-37m-x86_64-linux-gnu.so /home/jovyan/kernelo.cpython-37m-x86_64-linux-gnu.so
	
EXPOSE 8888

USER 1000
	
	
