from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from numpy import get_include
import cyarma

setup(name='kernel',
      version='0.1',
      packages=['kernel'],
      package_dir={'kernel': 'kernel'},
      description='Wrapper to Armadillo',
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("kernel",
                               ["cython/kernelo.pyx"],
                               include_dirs = [get_include(), '/usr/include',
                                               '/usr/local/include',
                                               cyarma.include_dir],
                               library_dirs = ['/usr/lib', '/usr/local/lib'],
                               libraries=["armadillo", "lapack_atlas", "blas"],
                               language='c++',
                               extra_compile_args=["-std=c++11"]
                               ),
                     ]
      )