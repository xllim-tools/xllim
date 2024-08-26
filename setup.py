# from Cython.Distutils import build_ext
from setuptools.command.build_ext import build_ext
from numpy import get_include
import cyarma
from setuptools import find_packages, Extension, setup

setup(name='kernelo',
      version='0.1',
      # packages=['kernelo'],
      packages=find_packages(),
      # package_dir={'kernelo': 'kernelo'},
      description='Wrapper to Armadillo',
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("kernelo",
                               ["cython/kernelo.pyx"],
                               include_dirs = [get_include(), '/usr/include',
                                               '/usr/local/include',
                                               cyarma.include_dir],
                               library_dirs = ['/usr/lib', '/usr/local/lib'],
                               libraries=["armadillo", "lapack_atlas", "blas"],
                               language='c++',
                               compiler_directives={'embedsignature': True},
                              #  extra_compile_args=["-Ofast", "-DARMA_NO_DEBUG", "-std=c++11"], # optimized
                               extra_compile_args=["-O0", "-DARMA_NO_DEBUG", "-std=c++11", "-DPYTHON_LIBRARY_DIR='/home/luc/.local/lib/python3.10/site-packages'", "-DPYTHON_EXECUTABLE='/usr/bin/python3'"], # for debug
                               ),
                     ]
      )