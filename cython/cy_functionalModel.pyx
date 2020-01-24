# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from functionalModelWrapper cimport FunctionalModel, HapkeModel, Hapke02Model, Hapke93Model, HapkeAdapter, SixParamsModel, FourParamsModel, ThreeParamsModel

cimport numpy as np
import numpy as np

# ---------------------------------- python classes definition ------------------------------------------- #

cdef class PyFunctionalModel:
    cdef shared_ptr[FunctionalModel] c_functional

    def get_D_dimension(self):
        return deref(self.c_functional).get_D_dimension()

    def get_L_dimension(self):
        return deref(self.c_functional).get_L_dimension()

    def from_physic(self,x):
        x_countiguous = np.ascontiguousarray(x)
        cdef double[::1] x_memview = x_countiguous
        deref(self.c_functional).from_physic(&x_memview[0], x_memview.shape[0])
        return x_countiguous

    def F(self,x):
        x_countiguous = np.ascontiguousarray(x)
        y_countiguous = np.ascontiguousarray(np.arange(self.get_D_dimension()),dtype=np.double)
        cdef double[::1] x_memview = x_countiguous
        cdef double[::1] y_memview = y_countiguous
        deref(self.c_functional).F(&x_memview[0],x_memview.shape[0],&y_memview[0],y_memview.shape[0])
        return y_countiguous

    cdef shared_ptr[FunctionalModel] getInstance(self):
            return self.c_functional

cdef class PyHapkeAdapter:
    cdef shared_ptr[HapkeAdapter] c_adapter

    cdef shared_ptr[HapkeAdapter] getInstance(self):
        return self.c_adapter

cdef class PySixParamsModel(PyHapkeAdapter):
    cdef shared_ptr[SixParamsModel] c_six_adapter

    def __cinit__(self):
        self.c_six_adapter = shared_ptr[SixParamsModel](new SixParamsModel())
        self.c_adapter = <shared_ptr[HapkeAdapter]>self.c_six_adapter

cdef class PyFourParamsModel(PyHapkeAdapter):
    cdef shared_ptr[FourParamsModel] c_four_adapter

    def __cinit__(self, double b0, double h):
        self.c_four_adapter = shared_ptr[FourParamsModel](new FourParamsModel(b0, h))
        self.c_adapter = <shared_ptr[HapkeAdapter]>self.c_four_adapter

cdef class PyThreeParamsModel(PyHapkeAdapter):
    cdef shared_ptr[ThreeParamsModel] c_three_adapter

    def __cinit__(self, double b0, double h):
        self.c_three_adapter = shared_ptr[ThreeParamsModel](new ThreeParamsModel(b0, h))
        self.c_adapter = <shared_ptr[HapkeAdapter]>self.c_three_adapter

cdef class PyHapke02Model(PyFunctionalModel):
    cdef shared_ptr[Hapke02Model] c_model

    def __cinit__(self, geometries,  row_size,  col_size, adapter, theta_bar_scaling):
        cdef double[:,::1] geometries_array = np.ascontiguousarray(geometries)
        cdef shared_ptr[HapkeAdapter] c_adapter = (<PyHapkeAdapter>adapter).getInstance()
        self.c_model = shared_ptr[Hapke02Model](
            new Hapke02Model(&geometries_array[0,0], row_size, col_size, c_adapter, theta_bar_scaling)
        )
        self.c_functional = <shared_ptr[FunctionalModel]>(self.c_model)

cdef class PyHapke93Model(PyFunctionalModel):
    cdef shared_ptr[Hapke93Model] c_model

    def __cinit__(self, geometries,  row_size,  col_size, adapter, theta_bar_scaling):
        cdef double[:,::1] geometries_array = np.ascontiguousarray(geometries)
        cdef shared_ptr[HapkeAdapter] c_adapter = (<PyHapkeAdapter>adapter).getInstance()
        self.c_model = shared_ptr[Hapke93Model](
            new Hapke93Model(&geometries_array[0,0], row_size, col_size, c_adapter, theta_bar_scaling)
        )
        self.c_functional = <shared_ptr[FunctionalModel]>(self.c_model)