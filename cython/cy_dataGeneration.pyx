# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from dataGenerationWrapper cimport StatModel, GaussianStatModelConfig , DependentGaussianStatModelConfig

cimport numpy as np
import numpy as np

# ---------------------------------- python classes definition ------------------------------------------- #

cdef class PyStatModel:
    cdef shared_ptr[StatModel] c_statModel

    @staticmethod
    cdef PyStatModel create(shared_ptr[StatModel] model):
        obj = <PyStatModel>PyStatModel.__new__(PyStatModel)
        obj.c_statModel = model
        return obj

    def gen_data(self, functionalModel, n):
        x_countiguous = np.ascontiguousarray(
            np.arange(functionalModel.get_L_dimension() * n).reshape(n,functionalModel.get_L_dimension()),
            dtype=np.double)

        y_countiguous = np.ascontiguousarray(
            np.arange(functionalModel.get_D_dimension() * n).reshape(n,functionalModel.get_D_dimension()),
            dtype=np.double)

        cdef double[:,::1] x_memview = x_countiguous
        cdef double[:,::1] y_memview = y_countiguous

        deref(self.c_statModel).gen_data((<PyFunctionalModel>functionalModel).getInstance(), n, &x_memview[0,0], &y_memview[0,0])

        return x_countiguous, y_countiguous

cdef class PyGaussianStatModelConfig:
    cdef GaussianStatModelConfig config

    def __cinit__(self, generatorType, covariance, seed):
        cdef double[::1] covariance_memview = np.ascontiguousarray(covariance)

        self.config.generatorType = <string>generatorType.encode('utf-8')
        self.config.covariance = &covariance_memview[0]
        self.config.cov_size = covariance_memview.shape[0]
        self.config.seed = seed

    def create(self):
        cdef shared_ptr[StatModel] model = self.config.create()
        return PyStatModel.create(model)


cdef class PyDependentGaussianStatModelConfig:
    cdef DependentGaussianStatModelConfig config

    def __cinit__(self, generatorType, r, seed):
        self.config.generatorType = <string>generatorType.encode('utf-8')
        self.config.r = r
        self.config.seed = seed

    def create(self):
        cdef shared_ptr[StatModel] model = self.config.create()
        return PyStatModel.create(model)

