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
        void F(vector[double] &x, vector[double] &y)
        vector[double] F(vector[double] &x)
        vector[vector[double]] F(vector[vector[double]] &x)
        int get_D_dimension()
        int get_L_dimension()
        vector[double] nomalize(vector[double] x)
        vector[double] invNormalize(vector[double] x)

cdef extern from "../src/physicalModel/FunctionnalModelFactory.h":
    cdef cppclass FunctionnalModelFactory:
        @staticmethod
        shared_ptr[FunctionnalModel] getModel(string type, vector[vector[double]] &data)



cdef vector[double] arrayToVector(np.ndarray[np.double_t,ndim=1] array):
    cdef long size = array.size
    cdef vector[double] vec
    cdef long i
    for i in range(size):
        vec.push_back(array[i])
    return vec

cdef np.ndarray[np.double_t,ndim=1] VectorToArray(vector[double] vec):
    cdef long size = vec.size()
    array = np.empty(size)
    array = np.empty(size)
    cdef long i
    for i in range(size):
        array[i] = vec[i]
    return array

cdef vector[vector[double]] ArrayToMat(np.ndarray[np.double_t,ndim=2] array):
    cdef vector[vector[double]] mat
    cdef vector[double] vec
    cdef long i
    cdef long j
    for i in range(array.shape[0]):
        vec.clear()
        for j in range(array.shape[1]):
            vec.push_back(array[i,j])
        mat.push_back(vec)
    return mat

cdef np.ndarray[np.double_t,ndim=2] MatToArray(vector[vector[double]] vec):
    array = np.empty([vec.size(),vec[0].size()])
    cdef long i,j
    for i in range(vec.size()):
        for j in range(vec[i].size()):
            array[i,j] = vec[i][j]
    return array



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

    def normalize(self,x):
        cdef x_vector = arrayToVector(x)
        cdef x_normlized = deref(self.c_model).nomalize(x_vector)
        return VectorToArray(x_normlized)

    def invNormalize(self,x):
        cdef x_vector = arrayToVector(x)
        cdef x_non_normlized = deref(self.c_model).invNormalize(x_vector)
        return VectorToArray(x_non_normlized)


    def F(self,x,y=None):
        cdef vector[vector[double]] x_mat
        cdef vector[vector[double]] y_mat
        cdef vector[double] x_vector
        cdef vector[double] y_vector

        if len(x.shape) == 2 :
            x_mat = ArrayToMat(x)
            y_mat = deref(self.c_model).F(x_mat)
            return MatToArray(y_mat)

        if len(x.shape) == 1 :
            x_vector = arrayToVector(x)
            if y is not None :
                y_vector = arrayToVector(y)
                deref(self.c_model).F(x_vector,y_vector)
                for i in range(y_vector.size()):
                    y[i] = y_vector[i]
            else :
                y = deref(self.c_model).F(x_vector)
                return VectorToArray(y)

cdef class PyFunctionnalModelFactory:
    cdef FunctionnalModelFactory c_factory

    def getModel(self,type,data):
        cdef vector[vector[double]] c_data
        cdef vector[double] vec
        cdef long i
        cdef long j
        for i in range(data.shape[0]):
            vec.clear()
            for j in range(data.shape[1]):
                vec.push_back(data[i,j])
            c_data.push_back(vec)
        cdef shared_ptr[FunctionnalModel] model = self.c_factory.getModel(<string>type.encode('utf-8'),c_data)
        return PyFunctionnalModel.create(model)

