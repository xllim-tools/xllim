from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

# ---------------------------------- header files declaration -------------------------------------------- #
cdef extern from "../src/learningModel/gllim/GLLiM.h" namespace "learningModel":
    cdef cppclass GLLiM:
        GLLiM() except +
        unsigned K
        unsigned L
        unsigned D
        double *Pi
        double *C
        double *B
        double *Gamma
        double *Sigma
        double *A

cdef extern from "../src/learningModel/gllim/IGLLiMLearning.h" namespace "learningModel":
    cdef cppclass IGLLiMLearning:
        void train(double *x, int x_rows, int x_cols, double *y, int y_rows, int y_cols)
        void initialize(double *x, int x_rows, int x_cols, double *y, int y_rows, int y_cols)
        void exportModel(GLLiM &gllim);
        void importModel(GLLiM &gllim);

cdef extern from "../src/learningModel/configs/LearningConfig.h" namespace "learningModel":
    cdef cppclass LearningConfig:
        pass

    cdef cppclass EMLearningConfig(LearningConfig):
        int max_iteration
        double ratio_ll
        double floor
        EMLearningConfig(int , double , double) except +

    cdef cppclass GMMLearningConfig(LearningConfig):
        int kmeans_iteration
        int em_iteration
        double floor
        GMMLearningConfig(int , int , double ) except +

cdef extern from "../src/learningModel/configs/InitConfig.h" namespace "learningModel":
    cdef cppclass InitConfig:
        pass

    cdef cppclass FixedInitConfig(InitConfig):
        unsigned seed
        shared_ptr[LearningConfig] gmmLearningConfig
        shared_ptr[LearningConfig] emLearningConfig
        FixedInitConfig(unsigned, shared_ptr[LearningConfig], shared_ptr[LearningConfig])

    cdef cppclass MultInitConfig(InitConfig):
        unsigned seed
        unsigned nb_iter_EM
        unsigned nb_experiences
        shared_ptr[LearningConfig] gmmLearningConfig
        shared_ptr[LearningConfig] emLearningConfig
        MultInitConfig(unsigned ,unsigned , unsigned , shared_ptr[LearningConfig] ,shared_ptr[LearningConfig] )


cdef extern from "../src/learningModel/LearningModelFactory.h" namespace "learningModel":
    cdef cppclass LearningModelFactory:
        @staticmethod
        shared_ptr[IGLLiMLearning] create(unsigned k, string GammaType, string SigmaType, shared_ptr[InitConfig] initConfig, shared_ptr[LearningConfig] learningConfig)


# ---------------------------------- cpp files declaration ----------------------------------------------- #

cdef extern from "../src/learningModel/covariances/Fullcovariance.cpp":
    pass

cdef extern from "../src/learningModel/covariances/DiagCovariance.cpp":
    pass

cdef extern from "../src/learningModel/covariances/IsoCovariance.cpp":
    pass

cdef extern from "../src/helpersFunctions/Helpers.cpp":
    pass

