from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from learningModelWrapper cimport GLLiM as CppGLLiM
from learningModelWrapper cimport IGLLiMLearning as CppIGLLiMLearning
from learningModelWrapper cimport LearningConfig as CppLearningConfig
from learningModelWrapper cimport EMLearningConfig as CppEMLearningConfig
from learningModelWrapper cimport GMMLearningConfig as CppGMMLearningConfig
from learningModelWrapper cimport InitConfig as CppInitConfig
from learningModelWrapper cimport FixedInitConfig as CppFixedInitConfig
from learningModelWrapper cimport MultInitConfig as CppMultInitConfig
from learningModelWrapper cimport LearningModelFactory as CppLearningModelFactory

cimport numpy as np
import numpy as np

import my_python_objects as Obj

# ---------------------------------- python classes definition ------------------------------------------- #



cdef class LearningConfig:
    cdef shared_ptr[CppLearningConfig] config

    cdef shared_ptr[CppLearningConfig] getInstance(self):
        return self.config

cdef class EMLearningConfig(LearningConfig):
    def __cinit__(self, max_iteration , ratio_ll, floor):
        self.config = shared_ptr[CppLearningConfig](
            new CppEMLearningConfig(max_iteration , ratio_ll, floor)
        )

cdef class GMMLearningConfig(LearningConfig):
    def __cinit__(self, kmeans_iteration, em_iteration, floor):
        self.config = shared_ptr[CppLearningConfig](
            new CppGMMLearningConfig(kmeans_iteration, em_iteration, floor)
        )

cdef class InitConfig:
    cdef shared_ptr[CppInitConfig] config

    cdef shared_ptr[CppInitConfig] getInstance(self):
        return self.config

cdef class FixedInitConfig(InitConfig):
    def __cinit__(self, seed, gmmLearningConfig):
        self.config = shared_ptr[CppInitConfig](
            new CppFixedInitConfig(seed, (<GMMLearningConfig>gmmLearningConfig).getInstance(), EMLearningConfig(0,0,0).getInstance())
        )

cdef class MultInitConfig(InitConfig):
    def __cinit__(self, seed, nb_iter_EM, nb_experiences, gmmLearningConfig):
        self.config = shared_ptr[CppInitConfig](
            new CppMultInitConfig(seed, nb_iter_EM, nb_experiences, (<GMMLearningConfig>gmmLearningConfig).getInstance(), EMLearningConfig(0,0,0).getInstance())
        )


cdef class GLLiM:
    cdef shared_ptr[CppIGLLiMLearning] c_gllimLearning
    cdef CppGLLiM c_gllimParameters
    cdef public object py_GLLiMParameters


    def __cinit__(self, D, L, K, GammaType, SigmaType, initConfig, learningConfig, py_GLLiMParameters_):
        assert isinstance(py_GLLiMParameters_, Obj.GLLiMParameters)
        self.py_GLLiMParameters = py_GLLiMParameters_
        self.c_gllimParameters.K = K
        self.c_gllimParameters.L = L
        self.c_gllimParameters.D = D

        print(self.c_gllimParameters.K)
        print(L)
        print(D)

        print("lol0")

        self.py_GLLiMParameters.Pi = np.ascontiguousarray(np.arange(K),dtype=np.double)
        print("lol01")
        cdef double[::1] Pi_memview = self.py_GLLiMParameters.Pi
        print("lol02")
        self.c_gllimParameters.Pi = &Pi_memview[0]

        print("lol1")

        self.py_GLLiMParameters.C = np.ascontiguousarray(np.arange(L * K).reshape(L, K), dtype=np.double)
        cdef double[:,::1] C_memview =  self.py_GLLiMParameters.C
        self.c_gllimParameters.C = &C_memview[0,0]

        print("lol2")

        self.py_GLLiMParameters.B = np.ascontiguousarray(np.arange(D * K).reshape(D, K), dtype=np.double)
        cdef double[:,::1] B_memview = self.py_GLLiMParameters.B
        self.c_gllimParameters.B = &B_memview[0,0]

        print("lol3")

        self.py_GLLiMParameters.A = np.arange(D * L * K, dtype=np.double).reshape(D, L, K)
        cdef double[:,:,:] A_memview = self.py_GLLiMParameters.A
        self.c_gllimParameters.A = &A_memview[0,0,0]

        print("lol4")

        self.py_GLLiMParameters.Gamma = np.arange(L * L * K, dtype=np.double).reshape(L, L, K)
        cdef double[:,:,:] Gamma_memview = self.py_GLLiMParameters.Gamma
        self.c_gllimParameters.Gamma = &Gamma_memview[0,0,0]

        print("lol5")

        self.py_GLLiMParameters.Sigma = np.arange(D * D * K, dtype=np.double).reshape(D, D, K)
        cdef double[:,:,:] Sigma_memview = self.py_GLLiMParameters.Sigma
        self.c_gllimParameters.Sigma = &Sigma_memview[0,0,0]

        print("lol6")

        self.c_gllimLearning = CppLearningModelFactory.create(
            K,
            <string>GammaType.encode('utf-8'),
            <string>SigmaType.encode('utf-8'),
            (<InitConfig>initConfig).getInstance(),
            (<LearningConfig>learningConfig).getInstance()
        )

        print("lol7")

    def initialize(self, x, y):
        x_countiguous = np.ascontiguousarray(x)
        y_countiguous = np.ascontiguousarray(y)
        cdef double[:,::1] x_memview = x_countiguous
        cdef double[:,::1] y_memview = y_countiguous

        deref(self.c_gllimLearning).initialize(
            &x_memview[0,0], x_memview.shape[0], self.c_gllimParameters.L,
            &y_memview[0,0], y_memview.shape[0], self.c_gllimParameters.D
        )

    def train(self, x, y):
        x_countiguous = np.ascontiguousarray(x)
        y_countiguous = np.ascontiguousarray(y)
        cdef double[:,::1] x_memview = x_countiguous
        cdef double[:,::1] y_memview = y_countiguous

        deref(self.c_gllimLearning).train(
             &x_memview[0,0], x_memview.shape[0], self.c_gllimParameters.L,
             &y_memview[0,0], y_memview.shape[0], self.c_gllimParameters.D
        )


    def exportModel(self):
        deref(self.c_gllimLearning).exportModel(self.c_gllimParameters)

    def importModel(self, Pi, C, Gamma, A, B, Sigma):
        K = Pi.shape[0]
        D = Sigma.shape[0]
        L = Gamma.shape[0]

        self.c_gllimParameters.K = K
        self.c_gllimParameters.L = L
        self.c_gllimParameters.D = D

        self._Pi = np.ascontiguousarray(Pi,dtype=np.double)
        cdef double[::1] Pi_memview = self._Pi
        self.c_gllimParameters.Pi = &Pi_memview[0]

        self._C = np.ascontiguousarray(C.reshape(L, K), dtype=np.double)
        cdef double[:,::1] C_memview =  self._C
        self.c_gllimParameters.C = &C_memview[0,0]

        self._B = np.ascontiguousarray(B.reshape(D, K), dtype=np.double)
        cdef double[:,::1] B_memview = self._B
        self.c_gllimParameters.B = &B_memview[0,0]

        self._A = A.reshape(D, L, K)
        cdef double[:,:,:] A_memview = self._A
        self.c_gllimParameters.A = &A_memview[0,0,0]

        self._Gamma = Gamma.reshape(L, L, K)
        cdef double[:,:,:] Gamma_memview = self._Gamma
        self.c_gllimParameters.Gamma = &Gamma_memview[0,0,0]

        self._Sigma = Sigma.reshape(D, D, K)
        cdef double[:,:,:] Sigma_memview = self._Sigma
        self.c_gllimParameters.Sigma = &Sigma_memview[0,0,0]

        deref(self.c_gllimLearning).importModel(self.c_gllimParameters)


