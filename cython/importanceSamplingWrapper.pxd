from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from dataGenerationWrapper cimport StatModel

# ---------------------------------- header files declaration -------------------------------------------- #

cdef extern from "../src/importanceSampling/ImportanceSamplingDiagnostic.h" namespace "importanceSampling":
    cdef struct ImportanceSamplingDiagnostic:
        unsigned nb_effective_sample
        double effective_sample_size
        double qn

cdef extern from "../src/importanceSampling/ImportanceSamplingResult.h" namespace "importanceSampling":
    cdef struct ImportanceSamplingResult:
        ImportanceSamplingDiagnostic diagnostic
        double *covariance
        double *mean

cdef extern from "../src/importanceSampling/proposition/ISProposition.h" namespace "importanceSampling":
    cdef cppclass ISProposition:
        unsigned getDimension()

cdef extern from "../src/importanceSampling/creators.h" namespace "importanceSampling":
    cdef struct GaussianMixturePropositionConfig:
        double *weights
        double *means
        double *covariances
        unsigned K
        unsigned L

        shared_ptr[ISProposition] create()

    cdef struct GaussianRegularizedPropositionConfig:
        double *means
        double *covariances
        unsigned L

        shared_ptr[ISProposition] create()

    cdef struct ImportanceSamplingConfig:
        unsigned N_Samples
        shared_ptr[StatModel] statModel

        shared_ptr[ImportanceSampler] create()

cdef extern from "../src/importanceSampling/ImportanceSampler.h" namespace "importanceSampling":
    cdef cppclass ImportanceSampler:
        void execute(shared_ptr[ISProposition] isProposition, double *y_obs, double *y_cov, unsigned size, shared_ptr[ImportanceSamplingResult] resultExport)


# ---------------------------------- cpp files declaration ----------------------------------------------- #
cdef extern from "../src/importanceSampling/ImportanceSampler.cpp":
    pass

cdef extern from "../src/importanceSampling/target/ISTargetDependent.cpp":
    pass

cdef extern from "../src/importanceSampling/proposition/GaussianMixtureProposition.cpp":
    pass

cdef extern from "../src/importanceSampling/proposition/GaussianRegularizedProposition.cpp":
    pass
