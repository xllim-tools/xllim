from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

# ---------------------------------- header files declaration -------------------------------------------- #

cdef extern from "../src/functionalModel/FunctionalModel.h" namespace "Functional":
    cdef cppclass FunctionalModel:
        void F(double *x, int size_x, double *y, int size_y)
        int get_D_dimension()
        int get_L_dimension()
        void from_physic(double *x, int size)
        void to_physic(double *x, int size)

cdef extern from "../src/functionalModel/HapkeModel/HapkeAdapter.h" namespace "Functional":
    cdef cppclass HapkeAdapter:
        pass

cdef extern from "../src/functionalModel/creators.h" namespace "Functional":
    cdef cppclass HapkeAdapterConfig:
        string version
        double b0
        double h
        HapkeAdapterConfig() except +

    cdef cppclass HapkeModelConfig:
        string version
        shared_ptr[HapkeAdapterConfig] adapterConfig
        double *geometries
        int row_size
        int col_size
        double theta_bar_scalling

        HapkeModelConfig() except +
        shared_ptr[FunctionalModel] create()

    cdef cppclass ShkuratovModelConfig:
        double *geometries
        int row_size
        int col_size
        double *scalingCoeffs
        double *offset

        ShkuratovModelConfig() except +
        shared_ptr[FunctionalModel] create()

    cdef cppclass TestModelConfig:
        TestModelConfig() except +
        shared_ptr[FunctionalModel] create()

    cdef cppclass ExternalModelConfig:
        string className
        string fileName
        string filePath

        ExternalModelConfig() except +
        shared_ptr[FunctionalModel] create()





# ---------------------------------- cpp files declaration -------------------------------------------- #

cdef extern from "../src/functionalModel/HapkeModel/HapkeVersions/Hapke02Model.cpp":
    pass

cdef extern from "../src/functionalModel/HapkeModel/HapkeVersions/Hapke93Model.cpp":
    pass

cdef extern from "../src/functionalModel/HapkeModel/HapkeVersions/HapkeModel.cpp":
    pass

cdef extern from "../src/functionalModel/HapkeModel/HapkeAdapters/FourParamsModel.cpp":
    pass

cdef extern from "../src/functionalModel/HapkeModel/HapkeAdapters/SixParamsModel.cpp":
    pass

cdef extern from "../src/functionalModel/HapkeModel/HapkeAdapters/ThreeParamsModel.cpp":
    pass

cdef extern from "../src/functionalModel/ShkuratovModel/ShkuratovModel.cpp":
    pass

cdef extern from "../src/functionalModel/ExternalModel/ExternalFunctionalModel.cpp":
    pass

cdef extern from "../src/functionalModel/TestModel/TestModel.cpp":
    pass