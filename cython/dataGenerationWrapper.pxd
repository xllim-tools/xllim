from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from functionalModelWrapper cimport FunctionalModel

# ---------------------------------- header files declaration -------------------------------------------- #

cdef extern from "../src/dataGeneration/StatModel.h" namespace "DataGeneration":
    cdef cppclass StatModel:
        void gen_data(shared_ptr[FunctionalModel] functionalModel, int n, double *x, double *y)

cdef extern from "../src/dataGeneration/GaussianStatModel.h" namespace "DataGeneration":
    cdef cppclass GaussianStatModel(StatModel):
        GaussianStatModel(string generatorType, double *covariance, int cov_size, unsigned seed) except +

cdef extern from "../src/dataGeneration/DependentGaussianStatModel.h" namespace "DataGeneration":
    cdef cppclass DependentGaussianStatModel(StatModel):
        DependentGaussianStatModel(string generatorType, int r, unsigned seed) except +


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