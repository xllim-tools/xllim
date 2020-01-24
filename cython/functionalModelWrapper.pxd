from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

# ---------------------------------- header files declaration -------------------------------------------- #

cdef extern from "../src/physicalModel/FunctionalModel.h" namespace "Functional":
    cdef cppclass FunctionalModel:
        void F(double *x, int size_x, double *y, int size_y)
        int get_D_dimension()
        int get_L_dimension()
        void from_physic(double *x, int size)

cdef extern from "../src/physicalModel/HapkeModel.h" namespace "Functional":
    cdef cppclass HapkeModel(FunctionalModel):
        HapkeModel(double *geometries,
                   int row_size,
                   int col_size,
                   shared_ptr[HapkeAdapter] adapter,
                   double theta_bar_scaling) except +

cdef extern from "../src/physicalModel/Hapke02Model.h" namespace "Functional":
    cdef cppclass Hapke02Model(HapkeModel):
        Hapke02Model(double *geometries,
                     int row_size,
                     int col_size,
                     shared_ptr[HapkeAdapter] adapter,
                     double theta_bar_scaling) except +

cdef extern from "../src/physicalModel/Hapke93Model.h" namespace "Functional":
    cdef cppclass Hapke93Model(HapkeModel):
        Hapke93Model(double *geometries,
                     int row_size,
                     int col_size,
                     shared_ptr[HapkeAdapter] adapter,
                     double theta_bar_scaling) except +

cdef extern from "../src/physicalModel/HapkeAdapter.h" namespace "Functional":
    cdef cppclass HapkeAdapter:
        pass

cdef extern from "../src/physicalModel/SixParamsModel.h" namespace "Functional":
    cdef cppclass SixParamsModel(HapkeAdapter):
        SixParamsModel() except +

cdef extern from "../src/physicalModel/FourParamsModel.h" namespace "Functional":
    cdef cppclass FourParamsModel(HapkeAdapter):
        FourParamsModel(double b0, double h) except +

cdef extern from "../src/physicalModel/ThreeParamsModel.h" namespace "Functional":
    cdef cppclass ThreeParamsModel(HapkeAdapter):
        ThreeParamsModel(double b0, double h) except +


# ---------------------------------- cpp files declaration -------------------------------------------- #

cdef extern from "../src/physicalModel/Hapke02Model.cpp":
    pass

cdef extern from "../src/physicalModel/Hapke93Model.cpp":
    pass

cdef extern from "../src/physicalModel/HapkeModel.cpp":
    pass

cdef extern from "../src/physicalModel/FourParamsModel.cpp":
    pass

cdef extern from "../src/physicalModel/SixParamsModel.cpp":
    pass

cdef extern from "../src/physicalModel/ThreeParamsModel.cpp":
    pass
