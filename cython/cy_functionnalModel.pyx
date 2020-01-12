# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

cimport numpy as np
import numpy as np

cdef extern from "../src/physicalModel/FunctionnalModelFactory.cpp":
    pass

cdef extern from "../src/physicalModel/Hapke02Model.cpp":
    pass

cdef extern from "../src/physicalModel/Hapke93Model.cpp":
    pass

cdef extern from "../src/physicalModel/HapkeModel.cpp":
    pass

cdef extern from "../src/physicalModel/FunctionnalModel.h":
    cdef cppclass FunctionnalModel:
        void F(double *x, int size_x, double *y, int size_y)
        int get_D_dimension()
        int get_L_dimension()
        void to_physic(double *x, int size)
        void from_physic(double *x, int size)

cdef extern from "../src/physicalModel/FunctionnalModelFactory.h":
    cdef cppclass FunctionnalModelFactory:
        @staticmethod
        shared_ptr[FunctionnalModel] getModel(string type, double *data, int row_size, int col_size)

cdef class PyFunctionnalModel:
    cdef shared_ptr[FunctionnalModel] c_model

    @staticmethod
    cdef PyFunctionnalModel create(shared_ptr[FunctionnalModel] model):
        obj = <PyFunctionnalModel>PyFunctionnalModel.__new__(PyFunctionnalModel)
        obj.c_model = model
        return obj

    def get_D_dimension(self):
        return deref(self.c_model).get_D_dimension()

    def get_L_dimension(self):
        return deref(self.c_model).get_L_dimension()

    def to_physic(self,x):
        x_countiguous = np.ascontiguousarray(x)
        cdef double[::1] x_memview = x_countiguous
        deref(self.c_model).to_physic(&x_memview[0], x_memview.shape[0])
        return x_countiguous

    def from_physic(self,x):
        x_countiguous = np.ascontiguousarray(x)
        cdef double[::1] x_memview = x_countiguous
        deref(self.c_model).from_physic(&x_memview[0], x_memview.shape[0])
        return x_countiguous

    def F(self,x):
        x_countiguous = np.ascontiguousarray(x)
        y_countiguous = np.ascontiguousarray(np.arange(self.get_D_dimension()),dtype=np.double)
        cdef double[::1] x_memview = x_countiguous
        cdef double[::1] y_memview = y_countiguous
        deref(self.c_model).F(&x_memview[0],x_memview.shape[0],&y_memview[0],y_memview.shape[0])
        return y_countiguous

cdef class PyFunctionnalModelFactory:
    cdef FunctionnalModelFactory c_factory

    def getModel(self,type,data):
        cdef double[:,::1] t2 = np.ascontiguousarray(data)
        cdef shared_ptr[FunctionnalModel] model = self.c_factory.getModel(<string>type.encode('utf-8'),&t2[0,0], data.shape[0],data.shape[1])
        return PyFunctionnalModel.create(model)

