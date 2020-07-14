from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.memory cimport make_shared
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdio cimport printf

from importanceSamplingWrapper cimport ImportanceSamplingDiagnostic as CppImportanceSamplingDiagnostic
from importanceSamplingWrapper cimport ImportanceSamplingResult as CppImportanceSamplingResult
from importanceSamplingWrapper cimport GaussianMixturePropositionConfig as CppGaussianMixturePropositionConfig
from importanceSamplingWrapper cimport GaussianRegularizedPropositionConfig as CppGaussianRegularizedPropositionConfig
from importanceSamplingWrapper cimport ImportanceSamplingConfig as CppImportanceSamplingConfig
from importanceSamplingWrapper cimport ImportanceSampler as CppImportanceSampler
from importanceSamplingWrapper cimport ISProposition as CppISProposition

cimport numpy as np
import numpy as np

# ---------------------------------- python classes definition ------------------------------------------- #

class ImportanceSamplingDiagnostic:
    """
    This class wraps the result of the importance sampling diagnostic.

    Attributes
    ----------
    nb_effective_sample : int

    effective_sample_size : double

    qn : double

    """
    def __init__(self):
        self.nb_effective_sample = 0
        self.effective_sample_size = 0
        self.qn = 0

class ImportanceSamplingResult:
    """
    This class wraps the result of the importance sampling over a given prediction

    Attributes
    ----------
    diagnostic : ImportanceSamplingDiagnostic
        See the documentation of the class ImportanceSamplingDiagnostic

    mean : ndarray
        1D array (L) containing the enhanced prediction

    covariance : ndarray
        1D array (L) containing the error of the enhanced prediction

    """

    def __init__(self):
        self.diagnostic = ImportanceSamplingDiagnostic()
        self.covariance = 0
        self.mean = 0

cdef class GaussianMixturePropositionConfig:
    """
    This class wraps the parameters that configures a proposition law for the importance sampling algorithm that is based on the GMM used to compute the prediction by the mean

    Constructor
    -----------
    weights : ndarray
        1D array(K)

    means : ndarray
        2D array(L, K)

    covariances : ndarray
        3D array(L, L, K)
    """
    cdef shared_ptr[CppGaussianMixturePropositionConfig] config

    def __cinit__(self, weights, means, covariances):
        self.config = shared_ptr[CppGaussianMixturePropositionConfig](new CppGaussianMixturePropositionConfig())
        deref(self.config).K = weights.shape[0]
        deref(self.config).L = means.shape[0]

        cdef double[::1] weights_memview = np.ascontiguousarray(weights, dtype=np.double)
        cdef double[:,::1] means_memview = np.ascontiguousarray(means.reshape(deref(self.config).L,deref(self.config).K),dtype=np.double)
        cdef double[:,:,:] covariances_memview = covariances.reshape(deref(self.config).K, deref(self.config).L, deref(self.config).L)

        deref(self.config).weights = &weights_memview[0]
        deref(self.config).means = &means_memview[0,0]
        deref(self.config).covariances = &covariances_memview[0,0,0]

    def create(self):
        cdef shared_ptr[CppISProposition] proposition = deref(self.config).create()
        return ISProposition.create(proposition)


cdef class GaussianRegularizedPropositionConfig:
    """
    This class wraps the parameters used to configure a proposition law for importance sampling algorithm that enhance the prediction by the centers

    Constructor
    -----------
    means : ndarray
        1D array(L)

    covariance : ndarray
        2D array(L,L)

    """
    cdef shared_ptr[CppGaussianRegularizedPropositionConfig] config

    def __cinit__(self, means, covariances):
        self.config = shared_ptr[CppGaussianRegularizedPropositionConfig](new CppGaussianRegularizedPropositionConfig())
        deref(self.config).L = means.shape[0]

        cdef double[::1] means_memview = np.ascontiguousarray(means, dtype=np.double)
        cdef double[:,::1] covariances_memview = np.ascontiguousarray(covariances.reshape(deref(self.config).L,deref(self.config).L),dtype=np.double)

        deref(self.config).means = &means_memview[0]
        deref(self.config).covariances = &covariances_memview[0,0]

    def create(self):
        cdef shared_ptr[CppISProposition] proposition = deref(self.config).create()
        return ISProposition.create(proposition)


cdef class ISProposition:
    cdef shared_ptr[CppISProposition] proposition

    @staticmethod
    cdef ISProposition create(shared_ptr[CppISProposition] proposition):
        obj = <ISProposition>ISProposition.__new__(ISProposition)
        obj.proposition = proposition
        return obj

    def getDimension(self):
        return deref(self.proposition).getDimension()

    cdef shared_ptr[CppISProposition] getInstance(self):
        return self.proposition


cdef class ImportanceSamplingConfig:
    """
    This class wraps the parameters used to configure the importance sampler.

    Constructor
    -----------
    N_samples : int
        The number of samples to generate using the proposition law , that are used during the importance sampling algorithm

    statModel : StatModel
        The stat model object is used to construct the target law for the importance sampling algorithm

    """
    cdef shared_ptr[CppImportanceSamplingConfig] config

    def __cinit__(self, N_Samples, statModel):
        deref(self.config).N_Samples = N_Samples
        deref(self.config).statModel = (<StatModel>statModel).getInstance()

    def create(self):
        cdef shared_ptr[CppImportanceSampler] sampler = deref(self.config).create()
        return ImportanceSampler.create(sampler)

cdef class ImportanceSampler:
    """
    The importance sample enhances the prediction corresponding of a given observation and its error.

    Methods
    -------
    execute(proposition, y_obs, y_var)
        Executes the importance sampling algorithm given an observation, its error and a proposition law.

    """
    cdef shared_ptr[CppImportanceSampler] __c_sampler

    @staticmethod
    cdef ImportanceSampler create(shared_ptr[CppImportanceSampler] sampler):
        obj = <ImportanceSampler>ImportanceSampler.__new__(ImportanceSampler)
        obj.__c_sampler = sampler
        return obj

    def execute(self, proposition, y_obs, y_var):
        """
        execute(proposition, y_obs, y_var)

        Executes the importance sampling algorithm given an observation, its error and a proposition law.

        Parameters
        ----------
        proposition : ISProposition

        y_obs : ndarray
            1D array(D) is the observation that the sampler will enhance its prediction

        y_var : ndarray
            1D array(D) is the error over the observation

        Returns
        -------
        ImportanceSamplingResult
            See the documentation of the class ImportanceSamplingResult

        """

        cdef double[::1] y_obs_memview = np.ascontiguousarray(y_obs)
        cdef double[::1] var_obs_memview = np.ascontiguousarray(y_var)
        cdef shared_ptr[CppImportanceSamplingResult] cpp_result
        py_result = ImportanceSamplingResult()

        L = proposition.getDimension()

        py_result.mean = np.ascontiguousarray(np.arange(L), dtype=np.double)
        cdef double[::1] mean_memview = py_result.mean
        deref(cpp_result).mean = &mean_memview[0]

        py_result.covariance = np.ascontiguousarray(np.arange(L), dtype=np.double)
        cdef double[::1] covariance_memview = py_result.covariance
        deref(cpp_result).covariance = &covariance_memview[0]

        deref(self.__c_sampler).execute((<ISProposition>proposition).getInstance(), &y_obs_memview[0], &var_obs_memview[0], y_obs_memview.shape[0], cpp_result)

        return py_result