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
from functionalModelWrapper cimport ExternalModelConfig as CppExternalModelConfig

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

    to_physic(self,x)
            returns an 1D array where the values of x are in physical space

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

    def to_physic(self,x):
            """
            to_physic(x)

            Returns
            -------
            ndarray
                1D array where the values of x are in physical space

            """
            x_countiguous = np.ascontiguousarray(x)
            cdef double[::1] x_memview = x_countiguous
            deref(self.c_functional).to_physic(&x_memview[0], x_memview.shape[0])
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
    cdef shared_ptr[CppHapkeAdapterConfig] config

    cdef shared_ptr[CppHapkeAdapterConfig] getInstance(self):
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
        self.config = shared_ptr[CppHapkeAdapterConfig](new CppHapkeAdapterConfig())
        version = "four"
        deref(self.config).b0 = b0
        deref(self.config).h = h
        deref(self.config).version = <string>version.encode('utf-8')

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
        self.config = shared_ptr[CppHapkeAdapterConfig](new CppHapkeAdapterConfig())
        version = "three"
        deref(self.config).b0 = b0
        deref(self.config).h = h
        deref(self.config).version = <string>version.encode('utf-8')

cdef class SixParamsHapkeAdapterConfig(HapkeAdapterConfig):
    """
    This class configures and creates an adapter of Hapke model using six photometric parameters

    Constructor
    -----------
    SixParamsHapkeAdapterConfig()

    """
    def __cinit__(self):
        self.config = shared_ptr[CppHapkeAdapterConfig](new CppHapkeAdapterConfig())
        version = "six"
        deref(self.config).version = <string>version.encode('utf-8')

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

    cdef shared_ptr[CppHapkeModelConfig] config
    cdef double[:,::1] geometries_memview

    def __cinit__(self, version, adapter, geometries, theta_bar_scalling):
        self.geometries_memview = np.ascontiguousarray(geometries)
        self.config = shared_ptr[CppHapkeModelConfig](new CppHapkeModelConfig())

        deref(self.config).geometries = &self.geometries_memview[0,0]
        deref(self.config).row_size = self.geometries_memview.shape[0]
        deref(self.config).col_size = self.geometries_memview.shape[1]
        deref(self.config).version = <string>version.encode('utf-8')
        deref(self.config).theta_bar_scalling = theta_bar_scalling
        deref(self.config).adapterConfig = (<HapkeAdapterConfig>adapter).getInstance()

    def create(self):
        return FunctionalModel.create(deref(self.config).create())

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
    cdef shared_ptr[CppShkuratovModelConfig] config
    cdef double[:,::1] geometries_memview
    cdef double[::1] scalingCoeffs_memview
    cdef double[::1] offset_memview

    def __cinit__(self, geometries, scalingCoeffs, offset):
        self.config = shared_ptr[CppShkuratovModelConfig](new CppShkuratovModelConfig())

        self.geometries_memview = np.ascontiguousarray(geometries)
        deref(self.config).geometries = &self.geometries_memview[0,0]

        deref(self.config).row_size = self.geometries_memview.shape[0]
        deref(self.config).col_size = self.geometries_memview.shape[1]

        self.scalingCoeffs_memview = np.ascontiguousarray(scalingCoeffs)
        deref(self.config).scalingCoeffs = &self.scalingCoeffs_memview[0]

        self.offset_memview = np.ascontiguousarray(offset)
        deref(self.config).offset = &self.offset_memview[0]

    def create(self):
        return FunctionalModel.create(deref(self.config).create())

cdef class ExternalModelConfig:
    cdef shared_ptr[CppExternalModelConfig] config

    def __cinit__(self, className, fileName, filePath):
        self.config = shared_ptr[CppExternalModelConfig](new CppExternalModelConfig())
        deref(self.config).className = <string>className.encode('utf-8')
        deref(self.config).fileName = <string>fileName.encode('utf-8')
        deref(self.config).filePath = <string>filePath.encode('utf-8')

    def create(self):
        return FunctionalModel.create(deref(self.config).create())

