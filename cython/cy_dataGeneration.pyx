# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from dataGenerationWrapper cimport StatModel, GaussianStatModel, DependentGaussianStatModel

cimport numpy as np
import numpy as np

# ---------------------------------- python classes definition ------------------------------------------- #

cdef class PyStatModel:
    cdef shared_ptr[StatModel] c_statModel

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



cdef class PyGaussianStatModel(PyStatModel):
    cdef shared_ptr[GaussianStatModel] c_gaussianModel

    def __cinit__(self, generatorType, covariance, cov_size, seed):
        covariance_countiguous = np.ascontiguousarray(covariance)
        cdef double[::1] covariance_memview = covariance_countiguous

        self.c_gaussianModel = shared_ptr[GaussianStatModel](new GaussianStatModel(
            <string>generatorType.encode('utf-8'),
            &covariance_memview[0],
            cov_size,
            seed)
        )
        self.c_statModel = <shared_ptr[StatModel]>(self.c_gaussianModel)

cdef class PyDependentGaussianStatModel(PyStatModel):
    cdef shared_ptr[DependentGaussianStatModel] c_dependentGaussianModel

    def __cinit__(self, generatorType, r, seed):
       self.c_dependentGaussianModel = shared_ptr[DependentGaussianStatModel](new DependentGaussianStatModel(
            <string>generatorType.encode('utf-8'),
            r,
            seed)
       )
       self.c_statModel = <shared_ptr[StatModel]>(self.c_dependentGaussianModel)