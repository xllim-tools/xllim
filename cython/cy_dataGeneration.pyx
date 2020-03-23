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
    """
    This interface defines the functions of a statistical model.

    Methods
    -------
    gen_data(self, n)
        Generates and returns a data set of low dimensional data X(n,L) and high dimensional data Y(n,D) using a functional model and a random data generator.

    """
    cdef shared_ptr[CppStatModel] c_statModel
    cdef FunctionalModel functionalModel

    @staticmethod
    cdef StatModel create(shared_ptr[CppStatModel] model, functionalModel):
        obj = <StatModel>StatModel.__new__(StatModel, functionalModel)
        obj.c_statModel = model
        obj.functionalModel = functionalModel
        return obj

    def gen_data(self, n):
        """
        gen_data(self, n)

        Generates and returns a data set of low dimensional data X(n,L) and high dimensional data Y(n,D) using a functional model and a random data generator. Y = F(X) + NOISE

        Parameters
        ----------
        int n
            Number of tuples to generate.

        """
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
    """
    This class wraps the parameters that configure a statistical model based on a normal distribution

    Constructor
    -----------
    GaussianStatModelConfig(generatorType, functionalModel , covariance, seed)

    string generatorType
        The type of the generator used by the model to generate data must be one of the following keywords :{"sobol","latin_cube","random"}
    FunctionalModel functionalModel
        The functional model that computes Y = F(X) where Y(N,D) and X(N,L)
    ndarray covariance
        1D array containing the D covariances used to add noise to F(X)
    int seed
        The seed used to initialize the random generator.

    """
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
    """
    This class wraps the parameters that configure a statistical model that is dependent on Y.

    Constructor
    -----------
    DependentGaussianStatModelConfig(generatorType, functionalModel , r, seed)

    string generatorType
        The type of the generator used by the model to generate data must be one of the following keywords :{"sobol","latin_cube","random"}
    FunctionalModel functionalModel
        The functional model that computes Y = F(X) where Y(N,D) and X(N,L)
    double r
        This percentage value is used to control the effect of the noise on the computed F(X)
    int seed
        The seed used to initialize the random generator.

    """

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

