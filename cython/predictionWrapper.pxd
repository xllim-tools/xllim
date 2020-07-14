from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from learningModelWrapper cimport IGLLiMLearning

# ---------------------------------- header files declaration -------------------------------------------- #

cdef extern from "../src/prediction/PredictionResultExport.h" namespace "prediction":
    cdef cppclass MeanPredictionResultExport:
        double *mean
        double *variance
        double *gmm_weights
        double *gmm_means
        double *gmm_covs
        MeanPredictionResultExport() except +

    cdef cppclass CenterPredictionResultExport:
        double *weights
        double *means
        double *covs
        CenterPredictionResultExport() except +

    cdef cppclass PredictionResultExport:
        shared_ptr[MeanPredictionResultExport] meanPred
        shared[CenterPredictionResultExport] centerPred
        PredictionResultExport() except +

cdef extern from "../src/prediction/IPredictor.h" namespace "prediction":
    cdef cppclass IPredictor:
        void predict(double *y_obs, double *var_obs, unsigned size, shared_ptr[PredictionResultExport] resultExport)
        void regularize(const double *series , unsigned rows, unsigned cols, unsigned slices, double *permutations)

cdef extern from "../src/prediction/creators.h" namespace "prediction":
    cdef cppclass PredictionConfig:
        unsigned k_merged
        unsigned k_pred_mean
        double threshold
        shared_ptr[IGLLiMLearning] learningModel

        PredictionConfig() except +
        shared_ptr[IPredictor] create()


# ---------------------------------- cpp files declaration ----------------------------------------------- #
cdef extern from "../src/prediction/Predictor.cpp":
    pass