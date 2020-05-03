from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from learningModelWrapper cimport IGLLiMLearning

# ---------------------------------- header files declaration -------------------------------------------- #

cdef extern from "../src/prediction/PredictionResultExport.h" namespace "prediction":
    cdef struct MeanPredictionResultExport:
        double *mean
        double *variance
        double *gmm_weights
        double *gmm_means
        double *gmm_covs

    cdef struct CenterPredictionResultExport:
        double *weights;
        double *means;
        double *covs;

    cdef struct PredictionResultExport:
        MeanPredictionResultExport meanPred
        CenterPredictionResultExport centerPred

cdef extern from "../src/prediction/IPredictor.h" namespace "prediction":
    cdef cppclass IPredictor:
        void predict(double *y_obs, double *var_obs, unsigned size, shared_ptr[PredictionResultExport] resultExport)
        void regularize(const double *series , unsigned rows, unsigned cols, unsigned slices, double *permutations)

cdef extern from "../src/prediction/creators.h" namespace "prediction":
    cdef struct PredictionConfig:
        unsigned k_merged
        unsigned k_pred_mean
        double threshold
        shared_ptr[IGLLiMLearning] learningModel

        shared_ptr[IPredictor] create()


# ---------------------------------- cpp files declaration ----------------------------------------------- #
cdef extern from "../src/prediction/Predictor.cpp":
    pass