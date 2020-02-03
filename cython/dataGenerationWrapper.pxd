from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from functionalModelWrapper cimport FunctionalModel

# ---------------------------------- header files declaration -------------------------------------------- #

cdef extern from "../src/dataGeneration/StatModel.h" namespace "DataGeneration":
    cdef cppclass StatModel:
        void gen_data(int n, double *x, int x_dimension, double *y, int y_dimension)

cdef extern from "../src/dataGeneration/creators.h" namespace "DataGeneration":
    cdef struct GaussianStatModelConfig:
        string generatorType
        shared_ptr[FunctionalModel] functionalModel
        double *covariance
        int cov_size
        unsigned seed

        shared_ptr[StatModel] create()

    cdef struct DependentGaussianStatModelConfig:
        string generatorType
        shared_ptr[FunctionalModel] functionalModel
        int r
        unsigned seed

        shared_ptr[StatModel] create()


# ---------------------------------- cpp files declaration ----------------------------------------------- #

cdef extern from "../src/dataGeneration/GaussianStatModel.cpp":
    pass

cdef extern from "../src/dataGeneration/DependentGaussianStatModel.cpp":
    pass

cdef extern from "../src/dataGeneration/GeneratorFactory.cpp":
    pass

cdef extern from "../src/dataGeneration/LatinCubeGenerator.cpp":
    pass

cdef extern from "../src/dataGeneration/RandomGenerator.cpp":
    pass

cdef extern from "../src/dataGeneration/SobolGenerator.cpp":
    pass