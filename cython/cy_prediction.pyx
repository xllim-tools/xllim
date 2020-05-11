from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.memory cimport make_shared
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdio cimport printf

from predictionWrapper cimport PredictionResultExport as CppPredictionResultExport
from predictionWrapper cimport IPredictor as CppIPredictor
from predictionWrapper cimport PredictionConfig as CppPredictionConfig

cimport numpy as np
import numpy as np

# ---------------------------------- python classes definition ------------------------------------------- #
class MeanPredictionResultExport:
    """
    This class wraps the result of the prediction by the mean. It contains the prediction of the variables and their corresponding variances. It also includes the GMM used to compute the prediction which composed of K components.

    Attributes
    ----------

    mean : ndarray
        1D array (L) is the mean of the GMM which stands for the prediction

    variance : ndarray
        1D array (L) is the variance of the prediction

    gmm_weights : ndarray
        1D array (K) is the weights of the components in the GMM

    gmm_means : ndarray
        2D array (L * K) is the means of each component in the GMM

    gmm_covs : ndarray
        3D array (L * L * K) is the covariance matrices of each component in the GMM

    """
    def __init__(self):
        self.mean = 0
        self.variance = 0
        self.gmm_weights = 0
        self.gmm_means = 0
        self.gmm_covs = 0


class CenterPredictionResultExport:
    """
    This class wraps the result of the prediction by the centers. It contains K gaussian models represented by their weight , mean and matrix of covariances.

    Attributes
    ----------

    weights : ndarray
        1D array(K) is the weights of the centers

    means : ndarray
        2D array(L, K) is the centers that stands for the predictions

    covs : ndarray
        3D array(L, L, K) contains the covariances matrices of the centers
    """
    def __init__(self):
        self.weights = 0
        self.means = 0
        self.covs = 0

class PredictionResultExport:
    """
    This class aggregates the objects containing the result of the prediction by the mean and the prediction by the centers

    Attributes
    ----------

    meansPred : MeanPredictionResultExport
        See the documentation of the class "MeanPredictionResultExport"

    centersPred : CenterPredictionResultExport
        See the documentation of the calss "CenterPredictionResultExport"

    """
    def __init__(self):
        self.meansPred = MeanPredictionResultExport()
        self.centersPred = CenterPredictionResultExport()

cdef class PredictionConfig:
    """
    This class wraps the parameters used to configure the prediction module that offers two type pf predictions. One using the mean of the GMM computed from the GLLiM model and using the observation and it variance, while the second giving the pertinent centers of the GMM.

    Constructor
    -----------
    PredictionConfig(k_merged, k_pred_mean, threshold, gllim)

    k_merged : int
        The number of centers to obtain while using the prediction by the centers

    k_pred_mean : int
        The number of components that the GMM must be reduced to before returning the prediction by the mean

    threshold : double
        While reducing the size of the GMM during the prediction by the centers, only the components with a weight superior or equal to the threshold are kept.

    gllim : GLLiM
        The trained GLLiM model
    """

    cdef CppPredictionConfig config
    cdef GLLiM gllim

    def __cinit__(self, k_merged, k_pred_mean, threshold, gllim):
        self.config.k_merged = k_merged
        self.config.k_pred_mean = k_pred_mean
        self.config.threshold = threshold
        self.config.learningModel = (<GLLiM>gllim).getInstance()
        self.gllim = gllim

    def create(self):
        cdef shared_ptr[CppIPredictor] predictor = self.config.create()
        return Predictor.create(predictor, self.gllim, self.config.k_merged, self.config.k_pred_mean)

cdef class Predictor:
    """
    The prediction class provides two types of predictions, by the mean and by the centers. It can also regularize the centers if the context requires a regularity.

    Methods
    -------
    predict(y_obs, var_obs)
        Returns the prediction corresponding to an observation y_obs with an uncertainty var_obs.

    regularize(series):
        If the context requires a regularity in the predictions and given series of predictions , the method returns a permutation of it.

    """
    cdef shared_ptr[CppIPredictor] __c_predictor
    cdef GLLiM gllim
    cdef unsigned k_merged
    cdef unsigned k_pred_mean

    @staticmethod
    cdef Predictor create(shared_ptr[CppIPredictor] predictor, gllim, k_merged, k_pred_mean):
        obj = <Predictor>Predictor.__new__(Predictor, gllim)
        obj.__c_predictor = predictor
        obj.gllim = gllim
        obj.k_merged = k_merged
        obj.k_pred_mean = k_pred_mean
        return obj

    def predict(self, y_obs, var_obs):
        """
        predict(y_obs, var_obs)

        This method performs the prediction of the lower dimension variable given an observation and its error. The predictions includes two types, by the mean and by the centers

        Parameters
        ----------
        y_obs : ndarray
            1D array containing the observation of dimension D

        var_obs : ndarray
            1D array of dimension D containing the error of the observation

        Returns
        -------
        PredictionResultExport
            See the documentation fo the class PredictionResultExport

        """
        cdef double[::1] y_obs_memview = np.ascontiguousarray(y_obs)
        cdef double[::1] var_obs_memview = np.ascontiguousarray(var_obs)
        cdef CppPredictionResultExport cpp_result
        py_result = PredictionResultExport()
        L = self.gllim.get_L_dimension()

        # Prediciton by means result

        py_result.meansPred.mean = np.ascontiguousarray(np.arange(L), dtype=np.double)
        cdef double[::1] meansPred_mean_memview = py_result.meansPred.mean
        cpp_result.meanPred.mean = &meansPred_mean_memview[0]

        py_result.meansPred.variance = np.ascontiguousarray(np.arange(L), dtype=np.double)
        cdef double[::1] meansPred_variance_memview = py_result.meansPred.variance
        cpp_result.meanPred.variance = &meansPred_variance_memview[0]

        py_result.meansPred.gmm_weights = np.ascontiguousarray(np.arange(self.k_pred_mean), dtype=np.double)
        cdef double[::1] meansPred_gmm_weights_memview = py_result.meansPred.gmm_weights
        cpp_result.meanPred.gmm_weights = &meansPred_gmm_weights_memview[0]

        py_result.meansPred.gmm_means = np.ascontiguousarray(np.arange(L * self.k_pred_mean).reshape(L, self.k_pred_mean), dtype=np.double)
        cdef double[:,::1] meansPred_gmm_means_memview = py_result.meansPred.gmm_means
        cpp_result.meanPred.gmm_means = &meansPred_gmm_means_memview[0,0]

        py_result.meansPred.gmm_covs = np.arange(L * L * self.k_pred_mean, dtype=np.double).reshape(self.k_pred_mean, L, L)
        cdef double[:,:,:] meansPred_gmm_covs_memview = py_result.meansPred.gmm_covs
        cpp_result.meanPred.gmm_covs = &meansPred_gmm_covs_memview[0,0,0]


        # Prediction by centers result

        py_result.centersPred.weights = np.ascontiguousarray(np.arange(self.k_merged), dtype=np.double)
        cdef double[::1] centersPred_weights_memview = py_result.centersPred.weights
        cpp_result.centerPred.weights = &centersPred_weights_memview[0]

        py_result.centersPred.means = np.ascontiguousarray(np.arange(L * self.k_merged).reshape(L, self.k_merged), dtype=np.double)
        cdef double[:,::1] centersPred_means_memview = py_result.centersPred.means
        cpp_result.centerPred.means = &centersPred_means_memview[0,0]

        py_result.centersPred.covs = np.arange(L * L * self.k_merged, dtype=np.double).reshape(self.k_merged, L, L)
        cdef double[:,:,:] centersPred_covs_memview = py_result.centersPred.covs
        cpp_result.centerPred.covs = &centersPred_covs_memview[0,0,0]

        deref(self.__c_predictor).predict(&y_obs_memview[0], &var_obs_memview[0], y_obs_memview.shape[0], make_shared[CppPredictionResultExport](cpp_result))

        return py_result

    def regularize(self, series):
        """
        regularize(series)

        If the context requires a regularity in the predictions and given series of predictions , the method returns a permutation of it.

        Parameters
        ----------
        series : ndarray
            3D array (L, K, N) containing series of predictions per observation

        Returns
        -------
        permutations : ndarray
            2D array (N , K) containing a permutation of the indices of the centers per observation.

        """
        cdef unsigned rows = series.shape[1]
        cdef unsigned cols = series.shape[2]
        cdef unsigned slices = series.shape[0]

        cdef double[:,:,:] series_memview = series

        permutations = np.ascontiguousarray(np.arange(cols * slices).reshape(cols, slices), dtype=np.double)
        cdef double[:,::1] permutations_memview = permutations

        deref(self.__c_predictor).regularize(&series_memview[0,0,0], rows, cols, slices, &permutations_memview[0,0])

        return permutations





