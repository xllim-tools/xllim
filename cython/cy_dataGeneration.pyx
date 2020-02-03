# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from dataGenerationWrapper cimport StatModel as CppStatModel
from dataGenerationWrapper cimport GaussianStatModelConfig as CppGaussianStatModelConfig
from dataGenerationWrapper cimport DependentGaussianStatModelConfig as CppDependentGaussianStatModelConfig

cimport numpy as np
import numpy as np

# ---------------------------------- python classes definition ------------------------------------------- #

cdef class StatModel:
    cdef shared_ptr[CppStatModel] c_statModel
    cdef FunctionalModel functionalModel

    @staticmethod
    cdef StatModel create(shared_ptr[CppStatModel] model, functionalModel):
        obj = <StatModel>StatModel.__new__(StatModel, functionalModel)
        obj.c_statModel = model
        obj.functionalModel = functionalModel
        return obj

    def gen_data(self, n):
        cdef int dimension_D = (<FunctionalModel>self.functionalModel).get_D_dimension()
        cdef int dimension_L = (<FunctionalModel>self.functionalModel).get_L_dimension()

        x_countiguous = np.ascontiguousarray(
            np.arange(dimension_L * n).reshape(n, dimension_L), dtype=np.double)

        y_countiguous = np.ascontiguousarray(
            np.arange(dimension_D * n).reshape(n, dimension_D), dtype=np.double)

        cdef double[:,::1] x_memview = x_countiguous
        cdef double[:,::1] y_memview = y_countiguous

        deref(self.c_statModel).gen_data(n, &x_memview[0,0], dimension_L, &y_memview[0,0], dimension_D)

        return x_countiguous, y_countiguous



cdef class GaussianStatModelConfig:
    cdef CppGaussianStatModelConfig config
    cdef FunctionalModel functionalModel

    def __cinit__(self, generatorType, functionalModel , covariance, seed):
        cdef double[::1] covariance_memview = np.ascontiguousarray(covariance)

        self.config.generatorType = <string>generatorType.encode('utf-8')
        self.config.functionalModel = (<FunctionalModel>functionalModel).getInstance()
        self.config.covariance = &covariance_memview[0]
        self.config.cov_size = covariance_memview.shape[0]
        self.config.seed = seed
        self.functionalModel = functionalModel

    def create(self):
        cdef shared_ptr[CppStatModel] model = self.config.create()
        return StatModel.create(model, self.functionalModel)


cdef class DependentGaussianStatModelConfig:
    cdef CppDependentGaussianStatModelConfig config
    cdef FunctionalModel functionalModel

    def __cinit__(self, generatorType, functionalModel, r, seed):
        self.config.generatorType = <string>generatorType.encode('utf-8')
        self.config.functionalModel = (<FunctionalModel>functionalModel).getInstance()
        self.config.r = r
        self.config.seed = seed
        self.functionalModel = functionalModel

    def create(self):
        cdef shared_ptr[CppStatModel] model = self.config.create()
        return StatModel.create(model, self.functionalModel)

