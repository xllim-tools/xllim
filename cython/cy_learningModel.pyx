from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

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

import my_python_objects as Obj

# ---------------------------------- python classes definition ------------------------------------------- #

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
    int max_iteration
        Maximum number of iterations the algorithm may execute.
    double ratio_ll
        If the algorithm likelihood increase goes under this ratio it will immediately stop.
    double floor
        The minimum values tolerated in the covariances of the GLLiM model. If smaller values are computed, it will be replaced by floor.

    """
    def __cinit__(self, max_iteration , ratio_ll, floor):
        self.config = shared_ptr[CppLearningConfig](
            new CppEMLearningConfig(max_iteration , ratio_ll, floor)
        )

cdef class GMMLearningConfig(LearningConfig):
    """
    This class wraps the pramaters that configure the training step of the GLLiM model. The estimator computes the equivalent GMM of the GLLiM model, trains the GMM and computes the GLLiM Model parameters again from the trained GMM.

    Constructor
    -----------
    int kmeans_iteration
        The number if kmeans algorithm iteration that the GMM will use to initialize the clusters of the model.
    int em_iteration
        The number of iterations of the GMM-EM training algorithm.
    double floor
        The minimum values tolerated in the covariances of the GLLiM model. If smaller values are computed, it will be replaced by floor.

    """
    def __cinit__(self, kmeans_iteration, em_iteration, floor):
        self.config = shared_ptr[CppLearningConfig](
            new CppGMMLearningConfig(kmeans_iteration, em_iteration, floor)
        )

cdef class InitConfig:
    """
    This is an abstract class if GLLiM initialization configuration
    """
    cdef shared_ptr[CppInitConfig] config

    cdef shared_ptr[CppInitConfig] getInstance(self):
        return self.config

cdef class FixedInitConfig(InitConfig):
    """
        This class wraps the parameters that configure configure the initialization step of the GLLiM model. The fixed initialization uses a GMM which is initialized with random and fixed values to compute the initial theta of the GLLiM model.

        Constructor
        -----------
        FixedInitConfig(seed, gmmLearningConfig)

        int seed
            Used by a random generator to generate the means of the GMM.
        GMMLearningConfig gmmLearningConfig
            Used to configure the estimator that trains the GMM.

    """
    def __cinit__(self, seed, gmmLearningConfig):
        self.config = shared_ptr[CppInitConfig](
            new CppFixedInitConfig(seed, (<GMMLearningConfig>gmmLearningConfig).getInstance(), EMLearningConfig(0,0,0).getInstance())
        )

cdef class MultInitConfig(InitConfig):
    """
        This class wraps the parameters used to configure the initialization step of the GLLiM model. The multi experiences initialization uses a GMM then the EM algorithm to initialize theta of the GLLiM model. It repeats the process nb_experiences times and in each experiences it run the GLLiM-EM algorithm nb_iter_EM  times. Only the best initialization is saved based on the maximum of likelihood obtained through the experiences.

        Constructor
        -----------
        int seed
            Used by a random generator to generate the means of the GMM.
        int nb_iter_EM
            The number of iteration the GLLiM-EM algorithm will execute.
        int nb_experiences
            The number of time the initialization algorithm will be repeated.
        GMMLearningConfig gmmLearningConfig
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

    int D
        The dimension of a tuple of low dimensionality data
    int L
        The dimension of a tuple of high dimensionality data
    int K
        Number of Gaussians
    string GammaType
        Type of gamma covariances, it must be one of the following keywords : {Full, Diag, Iso}
    string SigmaType
        Type of sigma covariances, it must be one of the following keywords : {Full, Diag, Iso}
    Initconfig initconfig
        This object wraps the parameters that configure the initialization step of the model. It should be either FixedInitConfig or MultInitConfig object.
    LearningConfig learningConfig
        This object wraps the parameters that configure the training step of the model. It should be either GMMLearningConfig or EMLearningConfig object.

        
    Methods
    -------
    initialize(self, x, y)
        Initializes theta parameters to before starting the learning step of GLLiM
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
        self.__py_GLLiMParameters = Obj.GLLiMParameters()
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

        self.__py_GLLiMParameters.A = np.arange(D * L * K, dtype=np.double).reshape(D, L, K)
        cdef double[:,:,:] A_memview = self.__py_GLLiMParameters.A
        self.__c_gllimParameters.A = &A_memview[0,0,0]

        self.__py_GLLiMParameters.Gamma = np.arange(L * L * K, dtype=np.double).reshape(L, L, K)
        cdef double[:,:,:] Gamma_memview = self.__py_GLLiMParameters.Gamma
        self.__c_gllimParameters.Gamma = &Gamma_memview[0,0,0]


        self.__py_GLLiMParameters.Sigma = np.arange(D * D * K, dtype=np.double).reshape(D, D, K)
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
            &x_memview[0,0], x_memview.shape[0], self.__c_gllimParameters.L,
            &y_memview[0,0], y_memview.shape[0], self.__c_gllimParameters.D
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
        GLLiMParameters
            An object containing the parameters of theta in ndarray format
        
        """
        
        deref(self.__c_gllimLearning).exportModel(self.__c_gllimParameters)
        return self.__py_GLLiMParameters
        

    def importModel(self, gllimParameters):
        """
        importModel(self, gllimParameters)
        
        Copies the parameters from the GLLiMParameters objects to the internal C++ structure theta.
        
        Parameters
        ----------
        gllimParameters :
            A GLLiMParameters object containing theta parameters
        
        """
        
        K = gllimParameters.Pi.shape[0]
        D = gllimParameters.Sigma.shape[0]
        L = gllimParameters.Gamma.shape[0]

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

        self.__py_GLLiMParameters.A = gllimParameters.A.reshape(D, L, K)
        cdef double[:,:,:] A_memview = self.__py_GLLiMParameters.A
        self.__c_gllimParameters.A = &A_memview[0,0,0]

        self.__py_GLLiMParameters.Gamma = gllimParameters.Gamma.reshape(L, L, K)
        cdef double[:,:,:] Gamma_memview = self.__py_GLLiMParameters.Gamma
        self.__c_gllimParameters.Gamma = &Gamma_memview[0,0,0]

        self.__py_GLLiMParameters.Sigma = gllimParameters.Sigma.reshape(D, D, K)
        cdef double[:,:,:] Sigma_memview = self.__py_GLLiMParameters.Sigma
        self.__c_gllimParameters.Sigma = &Sigma_memview[0,0,0]

        deref(self.__c_gllimLearning).importModel(self.__c_gllimParameters)


