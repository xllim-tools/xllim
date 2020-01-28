# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdio cimport printf

from functionalModelWrapper cimport FunctionalModel, HapkeAdapterConfig, HapkeModelConfig

cimport numpy as np
import numpy as np

# ---------------------------------- python classes definition ------------------------------------------- #

cdef class PyFunctionalModel:
    cdef shared_ptr[FunctionalModel] c_functional

    def get_D_dimension(self):
        return deref(self.c_functional).get_D_dimension()

    def get_L_dimension(self):
        return deref(self.c_functional).get_L_dimension()

    def from_physic(self,x):
        x_countiguous = np.ascontiguousarray(x)
        cdef double[::1] x_memview = x_countiguous
        deref(self.c_functional).from_physic(&x_memview[0], x_memview.shape[0])
        return x_countiguous

    def F(self,x):
        x_countiguous = np.ascontiguousarray(x)
        y_countiguous = np.ascontiguousarray(np.arange(self.get_D_dimension()),dtype=np.double)
        cdef double[::1] x_memview = x_countiguous
        cdef double[::1] y_memview = y_countiguous
        deref(self.c_functional).F(&x_memview[0],x_memview.shape[0],&y_memview[0],y_memview.shape[0])
        return y_countiguous

    cdef shared_ptr[FunctionalModel] getInstance(self):
            return self.c_functional

    @staticmethod
    cdef PyFunctionalModel create(shared_ptr[FunctionalModel] model):
        obj = <PyFunctionalModel>PyFunctionalModel.__new__(PyFunctionalModel)
        obj.c_functional = model
        return obj

cdef class PyHapkeAdapterConfig:
    cdef HapkeAdapterConfig config

    cdef HapkeAdapterConfig getInstance(self):
        return self.config

cdef class PyFourParamsHapkeAdapterConfig(PyHapkeAdapterConfig):
    def __cinit__(self, b0, h):
        version = "four"
        self.config.version = <string>version.encode('utf-8')
        self.config.b0 = b0
        self.config.h = h

cdef class PyThreeParamsHapkeAdapterConfig(PyHapkeAdapterConfig):
    def __cinit__(self, b0, h):
        version = "three"
        self.config.version = <string>version.encode('utf-8')
        self.config.b0 = b0
        self.config.h = h

cdef class PySixParamsHapkeAdapterConfig(PyHapkeAdapterConfig):
    def __cinit__(self):
        version = "six"
        self.config.version = <string>version.encode('utf-8')

cdef class PyHapkeModelConfig:
    cdef HapkeModelConfig config
    cdef double[:,::1] geometries_memview

    def __cinit__(self, version, adapter, geometries, theta_bar_scalling):
        self.geometries_memview = np.ascontiguousarray(geometries)
        cdef HapkeAdapterConfig hapkeAdapterConfig

        self.config.geometries = &self.geometries_memview[0,0]

        self.config.row_size = self.geometries_memview.shape[0]
        self.config.col_size = self.geometries_memview.shape[1]
        self.config.version = <string>version.encode('utf-8')
        self.config.theta_bar_scalling = theta_bar_scalling
        self.config.adapterConfig = (<PyHapkeAdapterConfig>adapter).getInstance()

    def create(self):
        return PyFunctionalModel.create(self.config.create())

