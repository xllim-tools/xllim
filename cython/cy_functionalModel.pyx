# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdio cimport printf

from functionalModelWrapper cimport FunctionalModel as CppFunctionalModel
from functionalModelWrapper cimport HapkeAdapterConfig as CppHapkeAdapterConfig
from functionalModelWrapper cimport HapkeModelConfig as CppHapkeModelConfig

cimport numpy as np
import numpy as np

# ---------------------------------- python classes definition ------------------------------------------- #

cdef class FunctionalModel:
    cdef shared_ptr[CppFunctionalModel] c_functional

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

    cdef shared_ptr[CppFunctionalModel] getInstance(self):
            return self.c_functional

    @staticmethod
    cdef FunctionalModel create(shared_ptr[CppFunctionalModel] model):
        obj = <FunctionalModel>FunctionalModel.__new__(FunctionalModel)
        obj.c_functional = model
        return obj

cdef class HapkeAdapterConfig:
    cdef CppHapkeAdapterConfig config

    cdef CppHapkeAdapterConfig getInstance(self):
        return self.config

cdef class FourParamsHapkeAdapterConfig(HapkeAdapterConfig):
    def __cinit__(self, b0, h):
        version = "four"
        self.config.version = <string>version.encode('utf-8')
        self.config.b0 = b0
        self.config.h = h

cdef class ThreeParamsHapkeAdapterConfig(HapkeAdapterConfig):
    def __cinit__(self, b0, h):
        version = "three"
        self.config.version = <string>version.encode('utf-8')
        self.config.b0 = b0
        self.config.h = h

cdef class SixParamsHapkeAdapterConfig(HapkeAdapterConfig):
    def __cinit__(self):
        version = "six"
        self.config.version = <string>version.encode('utf-8')

cdef class HapkeModelConfig:
    """
    This a Hapke model configuration wrapper for C++.
    """

    cdef CppHapkeModelConfig config
    cdef double[:,::1] geometries_memview

    def __cinit__(self, version, adapter, geometries, theta_bar_scalling):
        self.geometries_memview = np.ascontiguousarray(geometries)
        cdef CppHapkeAdapterConfig hapkeAdapterConfig

        self.config.geometries = &self.geometries_memview[0,0]

        self.config.row_size = self.geometries_memview.shape[0]
        self.config.col_size = self.geometries_memview.shape[1]
        self.config.version = <string>version.encode('utf-8')
        self.config.theta_bar_scalling = theta_bar_scalling
        self.config.adapterConfig = (<HapkeAdapterConfig>adapter).getInstance()

    def create(self):
        return FunctionalModel.create(self.config.create())

