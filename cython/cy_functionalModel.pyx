# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdio cimport printf

from functionalModelWrapper cimport FunctionalModel as CppFunctionalModel
from functionalModelWrapper cimport HapkeAdapterConfig as CppHapkeAdapterConfig
from functionalModelWrapper cimport HapkeModelConfig as CppHapkeModelConfig
from functionalModelWrapper cimport ShkuratovModelConfig as CppShkuratovModelConfig

cimport numpy as np
import numpy as np

# ---------------------------------- python classes definition ------------------------------------------- #

cdef class FunctionalModel:
    """
    This class is an interface of the functional model.

    Methods
    -------
    get_D_dimension(self)
        returns the D dimension of the model.

    get_L_dimension(self)
        returns the L dimension of the model.

    from_physic(self,x)
        returns an 1D array where the values of x are normalized

    F(self,x)
        Returns an 1D array Y = F(X)

    """
    cdef shared_ptr[CppFunctionalModel] c_functional

    def get_D_dimension(self):
        """
        get_D_dimension()

        Returns the D dimension of the model.

        Returns
        -------
        int
            The D dimension of the model.
        """
        return deref(self.c_functional).get_D_dimension()

    def get_L_dimension(self):
        """
        get_L_dimension()

        Returns the L dimension of the model.

        Returns
        -------
        int
            The L dimension of the model.
        """
        return deref(self.c_functional).get_L_dimension()

    def from_physic(self,x):
        """
        from_physic(x)

        Returns
        -------
        ndarray
            1D array where the values of x are normalized

        """
        x_countiguous = np.ascontiguousarray(x)
        cdef double[::1] x_memview = x_countiguous
        deref(self.c_functional).from_physic(&x_memview[0], x_memview.shape[0])
        return x_countiguous

    def F(self,x):
        """
        F(x)

        Computes and returns Y = F(X)

        Returns
        -------
        ndarray
            1D array containing Y = F(X)
        """
        x_countiguous = np.ascontiguousarray(x)
        y_countiguous = np.ascontiguousarray(np.arange(self.get_D_dimension()),dtype=np.double)
        cdef double[::1] x_memview = x_countiguous
        cdef double[::1] y_memview = y_countiguous
        deref(self.c_functional).F(&x_memview[0],x_memview.shape[0],&y_memview[0],y_memview.shape[0])
        return y_countiguous

    cdef shared_ptr[CppFunctionalModel] getInstance(self):
            return self.c_functional

    @staticmethod
    cdef FunctionalModel create(shared_ptr[CppFunctionalModel] model):
        obj = <FunctionalModel>FunctionalModel.__new__(FunctionalModel)
        obj.c_functional = model
        return obj

cdef class HapkeAdapterConfig:
    cdef CppHapkeAdapterConfig config

    cdef CppHapkeAdapterConfig getInstance(self):
        return self.config

cdef class FourParamsHapkeAdapterConfig(HapkeAdapterConfig):
    """
    This class configures and creates an adapter of Hapke model using four photometric parameters

    Constructor
    -----------
    FourParamsHapkeAdapterConfig(b0, h)

    double b0
        Amplitude of the opposition effect
    double h
        Angular width of the opposition effect

    """
    def __cinit__(self, b0, h):
        version = "four"
        self.config.version = <string>version.encode('utf-8')
        self.config.b0 = b0
        self.config.h = h

cdef class ThreeParamsHapkeAdapterConfig(HapkeAdapterConfig):
    """
    This class configures and creates an adapter of Hapke model using three photometric parameters

    Constructor
    -----------
    ThreeParamsHapkeAdapterConfig(b0, h)

    double b0
        Amplitude of the opposition effect
    double h
        Angular width of the opposition effect

    """
    def __cinit__(self, b0, h):
        version = "three"
        self.config.version = <string>version.encode('utf-8')
        self.config.b0 = b0
        self.config.h = h

cdef class SixParamsHapkeAdapterConfig(HapkeAdapterConfig):
    """
    This class configures and creates an adapter of Hapke model using six photometric parameters

    Constructor
    -----------
    SixParamsHapkeAdapterConfig()

    """
    def __cinit__(self):
        version = "six"
        self.config.version = <string>version.encode('utf-8')

cdef class HapkeModelConfig:
    """
    This class wraps the parameters that configure the Hapke model

    Constructor
    -----------
    HapkeModelConfig(version, adapter, geometries, theta_bar_scalling)

    string version
        The version of the hapke model must be one of the following keywords : {"2002","1993"}.
    HapkeAdapterConfig adapter
        This object is used to create a Hapke model adapter.
    ndarray geometries
        2D array containing N geometries with D dimensions.
    double theta_bar_scalling
        Used to transform theta_bar between physical and mathematical spaces.

    """

    cdef CppHapkeModelConfig config
    cdef double[:,::1] geometries_memview

    def __cinit__(self, version, adapter, geometries, theta_bar_scalling):
        self.geometries_memview = np.ascontiguousarray(geometries)
        cdef CppHapkeAdapterConfig hapkeAdapterConfig

        self.config.geometries = &self.geometries_memview[0,0]

        self.config.row_size = self.geometries_memview.shape[0]
        self.config.col_size = self.geometries_memview.shape[1]
        self.config.version = <string>version.encode('utf-8')
        self.config.theta_bar_scalling = theta_bar_scalling
        self.config.adapterConfig = (<HapkeAdapterConfig>adapter).getInstance()

    def create(self):
        return FunctionalModel.create(self.config.create())

cdef class ShkuratovModelConfig:
    """
    This class wraps the parameters that configure the Shkuratov model

    Constructor
    -----------
    ShkuratovModel(geometries, scalingCoeffs, offset)

    ndarray geometries
        2D array containing N geometries with D dimensions.
    ndarray scalingCoeffs
        1D array containing 5 values used to normalize the photometric variables of the model, where normalized_x = (x - offset)/scalingCoeff
    ndarray offset
        1D array containing 5 values used to normalize the photometric variables of the model, where normalized_x = (x - offset)/scalingCoeff

    """
    cdef CppShkuratovModelConfig config
    cdef double[:,::1] geometries_memview
    cdef double[::1] scalingCoeffs_memview
    cdef double[::1] offset_memview

    def __cinit__(self, geometries, scalingCoeffs, offset):
        self.geometries_memview = np.ascontiguousarray(geometries)
        self.config.geometries = &self.geometries_memview[0,0]

        self.config.row_size = self.geometries_memview.shape[0]
        self.config.col_size = self.geometries_memview.shape[1]

        self.scalingCoeffs_memview = np.ascontiguousarray(scalingCoeffs)
        self.config.scalingCoeffs = &self.scalingCoeffs_memview[0]

        self.offset_memview = np.ascontiguousarray(offset)
        self.config.offset = &self.offset_memview[0]

    def create(self):
        return FunctionalModel.create(self.config.create())

