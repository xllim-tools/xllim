from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython cimport view

from learningModelWrapper cimport GLLiM as CppGLLiM
from learningModelWrapper cimport IGLLiMLearning as CppIGLLiMLearning
from learningModelWrapper cimport LearningConfig as CppLearningConfig
from learningModelWrapper cimport EMLearningConfig as CppEMLearningConfig
from learningModelWrapper cimport GMMLearningConfig as CppGMMLearningConfig
from learningModelWrapper cimport InitConfig as CppInitConfig
from learningModelWrapper cimport FixedInitConfig as CppFixedInitConfig
from learningModelWrapper cimport MultInitConfig as CppMultInitConfig
from learningModelWrapper cimport LearningModelFactory as CppLearningModelFactory

cimport numpy as np
import numpy as np

# ---------------------------------- python classes definition ------------------------------------------- #

class GLLiMParameters:
    """
    This class wraps the parameters of the GLLiM model

    Attributes
    ----------
    Pi : ndarray
        1D array (K)

    C : ndarray
        2D array (L * K)

    Gamma : ndarray
        3D array (L * L * K)

    B : ndarray
        2D array (D * K)

    A : ndarray
        3D array (D * L * K)

    Sigma : ndarray
        3D array (D * D * K)

    """
    def __init__(self):
        self.Pi = 0
        self.C = 0
        self.B = 0
        self.A = 0
        self.Gamma = 0
        self.Sigma = 0

cdef class LearningConfig:
    """
    This is an abstract class of GLLiM training configuration
    """
    cdef shared_ptr[CppLearningConfig] config

    cdef shared_ptr[CppLearningConfig] getInstance(self):
        return self.config

cdef class EMLearningConfig(LearningConfig):
    """
    This class wraps the parameters that configure the training step of the GLLiM model. It uses the GLLiM-EM algorithm that iterates over max_iteration times or stops if it reaches the ratio of likelihood set in the constructor of the class.

    Constructor
    -----------
    intmax_iteration : int
        Maximum number of iterations the algorithm may execute.

    ratio_ll : double
        If the increase of the algorithm's likelihood goes under this ratio it will immediately stop. (example 5%)

    floor : double
        The minimum values tolerated in the covariances of the GLLiM model. If smaller values are computed, it will be replaced by floor.

    """
    def __cinit__(self, max_iteration , ratio_ll, floor):
        self.config = shared_ptr[CppLearningConfig](
            new CppEMLearningConfig(max_iteration , ratio_ll, floor)
        )

cdef class GMMLearningConfig(LearningConfig):
    """
    This class wraps the parameters that configure the training step of the GLLiM model. The estimator computes the equivalent GMM of the GLLiM model, trains the GMM and computes the GLLiM Model parameters again from the trained GMM.

    Constructor
    -----------
    kmeans_iteration : int
        The number of kmeans algorithm iteration that the GMM will use to initialize the clusters of the model.

    em_iteration : int
        The number of iterations of the GMM-EM training algorithm.

    floor : double
        The minimum values tolerated in the covariances of the GLLiM model. If smaller values are computed, it will be replaced by floor.

    """
    def __cinit__(self, kmeans_iteration, em_iteration, floor):
        self.config = shared_ptr[CppLearningConfig](
            new CppGMMLearningConfig(kmeans_iteration, em_iteration, floor)
        )

cdef class InitConfig:
    """
    This is an abstract class of GLLiM initialization configuration
    """
    cdef shared_ptr[CppInitConfig] config

    cdef shared_ptr[CppInitConfig] getInstance(self):
        return self.config

cdef class FixedInitConfig(InitConfig):
    """
        This class wraps the parameters that configure the initialization step of the GLLiM model. The fixed initialization uses a GMM which is initialized with random and fixed values to compute the initial theta of the GLLiM model.

        Constructor
        -----------
        FixedInitConfig(seed, gmmLearningConfig)

        seed : int
            Used by a random generator to generate the means of the GMM.

        gmmLearningConfig : GMMLearningConfig
            Used to configure the estimator that trains the GMM.

    """
    def __cinit__(self, seed, gmmLearningConfig):
        self.config = shared_ptr[CppInitConfig](
            new CppFixedInitConfig(seed, (<GMMLearningConfig>gmmLearningConfig).getInstance(), EMLearningConfig(0,0,0).getInstance())
        )

cdef class MultInitConfig(InitConfig):
    """
        This class wraps the parameters used to configure the initialization step of the GLLiM model. The multi experiences initialization uses a GMM then the EM algorithm to initialize theta of the GLLiM model. It repeats the process nb_experiences times and in each experiences it runs the GLLiM-EM algorithm nb_iter_EM times. Only the best initialization is saved based on the maximum of likelihood obtained through the experiences.

        Constructor
        -----------
        seed : int
            Used by a random generator to generate the means of the GMM.

        nb_iter_EM : int
            The number of iteration the GLLiM-EM algorithm will execute.

        nb_experiences : int
            The number of time the initialization algorithm will be repeated.

        gmmLearningConfig : GMMLearningConfig
            Used to configure the estimator that trains the GMM.

    """
    def __cinit__(self, seed, nb_iter_EM, nb_experiences, gmmLearningConfig):
        self.config = shared_ptr[CppInitConfig](
            new CppMultInitConfig(seed, nb_iter_EM, nb_experiences, (<GMMLearningConfig>gmmLearningConfig).getInstance(), EMLearningConfig(0,0,0).getInstance())
        )


cdef class GLLiM:
    """
    The learning model class based on the Gaussian Locally Linear Mapping

    Constructor
    -----------
    GLLiM(D, L, K, GammaType, SigmaType, initConfig, learningConfig)

    D : int
        The dimension of a tuple of low dimensionality data

    L : int
        The dimension of a tuple of high dimensionality data

    K : int
        Number of Gaussians

    GammaType : string
        Type of gamma covariances, it must be one of the following keywords : {Full, Diag, Iso}

    SigmaType : string
        Type of sigma covariances, it must be one of the following keywords : {Full, Diag, Iso}

    initConfig : InitConfig
        This object wraps the parameters that configure the initialization step of the model. It should be either FixedInitConfig or MultInitConfig object.

    learningConfig : LearningConfig
        This object wraps the parameters that configure the training step of the model. It should be either GMMLearningConfig or EMLearningConfig object.


    Methods
    -------
    initialize(self, x, y)
        Initializes theta parameters. Must be called before starting the learning step of GLLiM

    train(self, x, y)
        Trains GLLiM model by updating its theta's parameters

    exportModel(self)
        Returns a GLLiMParameters object containing the parameters of theta in ndarray format

    importModel(self, gllimParameters)
        Sets the values of theta parameters in the C++ internal structure

    """

    cdef shared_ptr[CppIGLLiMLearning] __c_gllimLearning
    cdef CppGLLiM __c_gllimParameters
    cdef public object __py_GLLiMParameters


    def __cinit__(self, D, L, K, GammaType, SigmaType, initConfig, learningConfig):
        self.__py_GLLiMParameters = GLLiMParameters()
        self.__c_gllimParameters.K = K
        self.__c_gllimParameters.L = L
        self.__c_gllimParameters.D = D

        self.__py_GLLiMParameters.Pi = np.ascontiguousarray(np.arange(K),dtype=np.double)
        cdef double[::1] Pi_memview = self.__py_GLLiMParameters.Pi
        self.__c_gllimParameters.Pi = &Pi_memview[0]

        self.__py_GLLiMParameters.C = np.ascontiguousarray(np.arange(L * K).reshape(L, K), dtype=np.double)
        cdef double[:,::1] C_memview =  self.__py_GLLiMParameters.C
        self.__c_gllimParameters.C = &C_memview[0,0]

        self.__py_GLLiMParameters.B = np.ascontiguousarray(np.arange(D * K).reshape(D, K), dtype=np.double)
        cdef double[:,::1] B_memview = self.__py_GLLiMParameters.B
        self.__c_gllimParameters.B = &B_memview[0,0]

        self.__py_GLLiMParameters.A = np.arange(D * L * K, dtype=np.double).reshape(K, D, L)
        cdef double[:,:,:] A_memview = self.__py_GLLiMParameters.A
        self.__c_gllimParameters.A = &A_memview[0,0,0]

        self.__py_GLLiMParameters.Gamma = np.arange(L * L * K, dtype=np.double).reshape(K, L, L)
        cdef double[:,:,:] Gamma_memview = self.__py_GLLiMParameters.Gamma
        self.__c_gllimParameters.Gamma = &Gamma_memview[0,0,0]


        self.__py_GLLiMParameters.Sigma = np.arange(D * D * K, dtype=np.double).reshape(K, D, D)
        cdef double[:,:,:] Sigma_memview = self.__py_GLLiMParameters.Sigma
        self.__c_gllimParameters.Sigma = &Sigma_memview[0,0,0]

        self.__c_gllimLearning = CppLearningModelFactory.create(
            K,
            <string>GammaType.encode('utf-8'),
            <string>SigmaType.encode('utf-8'),
            (<InitConfig>initConfig).getInstance(),
            (<LearningConfig>learningConfig).getInstance()
        )

    def initialize(self, x, y):
        """
        initialize(self, x, y)

        Initializes GLLiM's theta parameters

        Parameters
        ----------
        x : ndarray
            2D array (N x L) containing the low dimensional data

        y : ndarray
            2D array (N x D) containing the high dimensional data

        """

        x_countiguous = np.ascontiguousarray(x)
        y_countiguous = np.ascontiguousarray(y)
        cdef double[:,::1] x_memview = x_countiguous
        cdef double[:,::1] y_memview = y_countiguous

        deref(self.__c_gllimLearning).initialize(
            &x_memview[0,0], x.shape[0], self.__c_gllimParameters.L,
            &y_memview[0,0], y.shape[0], self.__c_gllimParameters.D
        )

    def train(self, x, y):
        """
        train(self, x, y)

        Trains GLLiM's theta parameters

        Parameters
        ----------
        x : ndarray
            2D array (N x L) containing the low dimensional data

        y : ndarray
            2D array (N x D) containing the high dimensional data

        """

        x_countiguous = np.ascontiguousarray(x)
        y_countiguous = np.ascontiguousarray(y)
        cdef double[:,::1] x_memview = x_countiguous
        cdef double[:,::1] y_memview = y_countiguous

        deref(self.__c_gllimLearning).train(
             &x_memview[0,0], x_memview.shape[0], self.__c_gllimParameters.L,
             &y_memview[0,0], y_memview.shape[0], self.__c_gllimParameters.D
        )


    def exportModel(self):
        """
        exportModel(self)

        Exports the parameters of theta from the C++ structure as ndarrays encapsulated in an GLLiMParameters object.

        Returns
        -------
        GLLiMParameters : GLLiMParameters
            An object containing the parameters of theta in ndarray format

        """

        deref(self.__c_gllimLearning).getModel(self.__c_gllimParameters)
        return self.__py_GLLiMParameters


    def importModel(self, gllimParameters):
        """
        importModel(self, gllimParameters)

        Copies the parameters from the GLLiMParameters objects to the internal C++ structure theta.

        Parameters
        ----------
        gllimParameters : GLLiMParameters
            A GLLiMParameters object containing theta parameters

        """

        K = gllimParameters.Pi.shape[0]
        D = gllimParameters.Sigma.shape[1]
        L = gllimParameters.Gamma.shape[1]

        self.__c_gllimParameters.K = K
        self.__c_gllimParameters.L = L
        self.__c_gllimParameters.D = D

        self.__py_GLLiMParameters.Pi = np.ascontiguousarray(gllimParameters.Pi,dtype=np.double)
        cdef double[::1] Pi_memview = self.__py_GLLiMParameters.Pi
        self.__c_gllimParameters.Pi = &Pi_memview[0]

        self.__py_GLLiMParameters.C = np.ascontiguousarray(gllimParameters.C.reshape(L,K), dtype=np.double)
        cdef double[:,::1] C_memview =  self.__py_GLLiMParameters.C
        self.__c_gllimParameters.C = &C_memview[0,0]

        self.__py_GLLiMParameters.B = np.ascontiguousarray(gllimParameters.B.reshape(D,K), dtype=np.double)
        cdef double[:,::1] B_memview = self.__py_GLLiMParameters.B
        self.__c_gllimParameters.B = &B_memview[0,0]

        self.__py_GLLiMParameters.A = gllimParameters.A.reshape(K, D, L)
        cdef double[:,:,:] A_memview = self.__py_GLLiMParameters.A
        self.__c_gllimParameters.A = &A_memview[0,0,0]

        self.__py_GLLiMParameters.Gamma = gllimParameters.Gamma.reshape(K, L, L)
        cdef double[:,:,:] Gamma_memview = self.__py_GLLiMParameters.Gamma
        self.__c_gllimParameters.Gamma = &Gamma_memview[0,0,0]

        self.__py_GLLiMParameters.Sigma = gllimParameters.Sigma.reshape(K, D, D)
        cdef double[:,:,:] Sigma_memview = self.__py_GLLiMParameters.Sigma
        self.__c_gllimParameters.Sigma = &Sigma_memview[0,0,0]

        deref(self.__c_gllimLearning).setModel(self.__c_gllimParameters)

    def getInverse(self):
        """
        getInverse(self)

        Exports the parameters of theta* (inverse) from the C++ structure as ndarrays encapsulated in an GLLiMParameters object.

        Returns
        -------
        GLLiMParameters : GLLiMParameters
            An object containing the parameters of theta* (inverse) in ndarray format

        """

        cdef CppGLLiM c_inverseGllimParameters
        inverseGllimParameters = GLLiMParameters()

        K = self.__c_gllimParameters.K
        L = self.__c_gllimParameters.D
        D = self.__c_gllimParameters.L

        inverseGllimParameters.K = K
        inverseGllimParameters.D = D
        inverseGllimParameters.L = L

        c_inverseGllimParameters.K = K
        c_inverseGllimParameters.L = L
        c_inverseGllimParameters.D = D

        inverseGllimParameters.Pi = np.ascontiguousarray(np.arange(K),dtype=np.double)
        cdef double[::1] Pi_memview = inverseGllimParameters.Pi
        c_inverseGllimParameters.Pi = &Pi_memview[0]

        inverseGllimParameters.C = np.ascontiguousarray(np.arange(L * K).reshape(L, K), dtype=np.double)
        cdef double[:,::1] C_memview =  inverseGllimParameters.C
        c_inverseGllimParameters.C = &C_memview[0,0]

        inverseGllimParameters.B = np.ascontiguousarray(np.arange(D * K).reshape(D, K), dtype=np.double)
        cdef double[:,::1] B_memview = inverseGllimParameters.B
        c_inverseGllimParameters.B = &B_memview[0,0]

        inverseGllimParameters.A = np.arange(D * L * K, dtype=np.double).reshape(K, D, L)
        cdef double[:,:,:] A_memview = inverseGllimParameters.A
        c_inverseGllimParameters.A = &A_memview[0,0,0]

        inverseGllimParameters.Gamma = np.arange(L * L * K, dtype=np.double).reshape(K, L, L)
        cdef double[:,:,:] Gamma_memview = inverseGllimParameters.Gamma
        c_inverseGllimParameters.Gamma = &Gamma_memview[0,0,0]

        inverseGllimParameters.Sigma = np.arange(D * D * K, dtype=np.double).reshape(K, D, D)
        cdef double[:,:,:] Sigma_memview = inverseGllimParameters.Sigma
        c_inverseGllimParameters.Sigma = &Sigma_memview[0,0,0]

        deref(self.__c_gllimLearning).getInverse(c_inverseGllimParameters)

        return inverseGllimParameters

    def directLogDensity(self, x):
        """
        directLogDensity(self, x)

        Computes log density of P(Y|X=x) as a GMM with weights , means and covariances deduced from the GLLiM's parameters in theta.

        Parameters
        ----------
        x : ndarray
            1D array (L)

        Returns
        -------
        weights : ndarray
            1D array (K) containing the weights of the components of the GMM.

        means : ndarray
            2D array (D,K) containing the means of the components of the GMM.

        covariances : ndarray
            3D array (K,D,D) containing the covariances of the components of the GMM.

        """
        K = self.__c_gllimParameters.K
        L = self.__c_gllimParameters.L
        D = self.__c_gllimParameters.D

        x_countiguous = np.ascontiguousarray(x)
        cdef double[::1] x_memview = x_countiguous

        weights = np.ascontiguousarray(np.arange(K),dtype=np.double)
        cdef double[::1] weights_memview = weights

        means = np.ascontiguousarray(np.arange(D * K).reshape(D, K), dtype=np.double)
        cdef double[:,::1] means_memview = means

        covs = np.arange(D * D * K, dtype=np.double).reshape(K, D, D)
        cdef double[:,:,:] covs_memview = covs

        deref(self.__c_gllimLearning).directLogDensity(&x_memview[0], &weights_memview[0], &means_memview[0,0], &covs_memview[0,0,0])

        return weights, means, covs

    def inverseLogDensity(self, y):

        """
        inverseLogDensity(self, x)

        Computes log density of P(X|Y=y) as a GMM with weights , means and covariances deduced from the GLLiM's parameters in theta.

        Parameters
        ----------
        y : ndarray
            1D array (D)

        Returns
        -------
        weights : ndarray
            1D array (K) containing the weights of the components of the GMM.

        means : ndarray
            2D array (L,K) containing the means of the components of the GMM.

        covariances : ndarray
            3D array (K,L,L) containing the covariances of the components of the GMM.

        """
        K = self.__c_gllimParameters.K
        L = self.__c_gllimParameters.D
        D = self.__c_gllimParameters.L

        y_countiguous = np.ascontiguousarray(y)
        cdef double[::1] y_memview = y_countiguous

        weights = np.ascontiguousarray(np.arange(K),dtype=np.double)
        cdef double[::1] weights_memview = weights

        means = np.ascontiguousarray(np.arange(D * K).reshape(D, K), dtype=np.double)
        cdef double[:,::1] means_memview = means

        covs = np.arange(D * D * K, dtype=np.double).reshape(K, D, D)
        cdef double[:,:,:] covs_memview = covs

        deref(self.__c_gllimLearning).inverseLogDensity(&y_memview[0], &weights_memview[0], &means_memview[0,0], &covs_memview[0,0,0])

        return weights, means, covs

    def get_L_dimension(self):
        return self.__c_gllimParameters.L

    cdef shared_ptr[CppIGLLiMLearning] getInstance(self):
        return self.__c_gllimLearning