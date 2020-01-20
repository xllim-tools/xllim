cdef extern from "../src/physicalModel/FunctionnalModelFactory.cpp":
    pass

cdef extern from "../src/physicalModel/Hapke02Model.cpp":
    pass

cdef extern from "../src/physicalModel/Hapke93Model.cpp":
    pass

cdef extern from "../src/physicalModel/HapkeModel.cpp":
    pass

cdef extern from "../src/physicalModel/FunctionalModel.h":
    cdef cppclass FunctionalModel:
        void F(vector[double] &x, vector[double] &y)
        vector[double] F(vector[double] &x)
        vector[vector[double]] F(vector[vector[double]] &x)
        int get_D_dimension()
        int get_L_dimension()
        vector[double] to_physic(vector[double] x)
        vector[double] invNormalize(vector[double] x)

cdef extern from "../src/physicalModel/FunctionnalModelFactory.h":
    cdef cppclass FunctionnalModelFactory:
        @staticmethod
        shared_ptr[FunctionalModel] getModel(string type, vector[vector[double]] &geometries)