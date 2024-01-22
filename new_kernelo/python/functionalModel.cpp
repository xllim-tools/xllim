#include <pybind11/pybind11.h>
#include <carma>
#include <armadillo>

#include "../src/FunctionalModel.hpp"
#include "../src/TestModel.hpp"
#include "../src/ShkuratovModel.hpp"
#include "../src/HapkeModel.hpp"
#include "../src/ExternalPythonModel.hpp"

using namespace Functional;
namespace py = pybind11;


PYBIND11_MODULE(newkernelo, m)
{
    py::class_<FunctionalModel> (m, "FunctionalModel")
        .def("F", [](FunctionalModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);
                arma::vec y_arma;
                self.F(x_arma, y_arma);
                py::array_t<double> y_arr = carma::col_to_arr(y_arma).squeeze();
                return y_arr; },
            R"mydelimiter(
                This method calculates y = F(x).

                :param x: A one-dimensional Numpy array with shape (L,) corresponding to the functional model dimensions.
                :type x: ndarray of shape (L,)

                :return: The Numpy array with shape (D,) resulting from F(x).
                :rtype: ndarray of shape (D,)
            )mydelimiter") // kernelo.FunctionalModel.__doc__.
        .def("get_D_dimension", &FunctionalModel::get_D_dimension,
            R"mydelimiter(
                This method returns the D dimension of the problem.

                :return: The D dimension of the model.
                :rtype: int
            )mydelimiter")
        .def("get_L_dimension", &FunctionalModel::get_L_dimension,
            R"mydelimiter(
                This method returns the L dimension of the problem.

                :return: The L dimension of the model.
                :rtype: int
            )mydelimiter")
        .def("to_physic", [](FunctionalModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);                      // Convert the NumPy array to a Carma vector with copy=true because we want to argument to keep unmodified
                self.to_physic(x_arma);                                             // Call the C++ function
                py::array_t<double> y_arr = carma::col_to_arr(x_arma).squeeze();    // Convert the Carma vector back to a NumPy array + squeeze the array into shape (x,)
                return y_arr; },
            R"mydelimiter(
                This method transforms the values of x from the mathematical space to the physical space.

                :param x: A one-dimensional Numpy array to transform with shape (L,) corresponding to the functional model dimensions.
                :type x: ndarray of shape (L,)

                :return: The transformed Numpy array.
                :rtype: ndarray of shape (L,)
            )mydelimiter")
        .def("from_physic", [](FunctionalModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);
                self.from_physic(x_arma);
                py::array_t<double> y_arr = carma::col_to_arr(x_arma).squeeze();
                return y_arr;
            }, R"mydelimiter(
                This method transforms the values of x from the physical space to the mathematical space.

                :param x: A one-dimensional Numpy array to normalize with shape (L,) corresponding to the functional model dimensions.
                :type x: ndarray of shape (L,)

                :return: The transformed Numpy array.
                :rtype: ndarray of shape (L,)
            )mydelimiter")
        .doc() = R"mydelimiter(

            The base class :class:`FunctionalModel` is an abstract class representing the functional model.
            It offers the functional method "F" which requires that the parameters of X be
            in mathematical space. It contains normalization methods to transform
            X from and to physical space. It also allows to retrieve the dimensions
            of the problem.


            Derived classes
            ---------------
            +----------------------------+---------------------------------------------------------------------------------------------------------------+
            | :ref:`HapkeModel`          | The ``HapkeModel`` class describes the Hapke photometric model.                                               |
            +----------------------------+---------------------------------------------------------------------------------------------------------------+
            | :ref:`ShkuratovModel`      | The ``ShkuratovModel`` class describes the Shkuratov photometric model.                                       |
            +----------------------------+---------------------------------------------------------------------------------------------------------------+
            | :ref:`ExternalPythonModel` | The ``ExternalPythonModel`` class allows to import a python script in order to use your own functional model. |
            +----------------------------+---------------------------------------------------------------------------------------------------------------+
            | :ref:`TestModel`           | The ``TestModel`` class describes a simple non-linear model                                                   |
            +----------------------------+---------------------------------------------------------------------------------------------------------------+

            Methods
            -------
            +------------------------+------------------------------------------------------------------------------+
            | **F** (*X*)            | Apply the model function on vector *x*                                       |
            +------------------------+------------------------------------------------------------------------------+
            | **get_D_dimension** () | get the dimension **D** of the model - ie. dim(*Y*)                          |
            +------------------------+------------------------------------------------------------------------------+
            | **get_L_dimension** () | get the dimension **L** of the model - ie. dim(*X*)                          |
            +------------------------+------------------------------------------------------------------------------+
            | **to_physic** (*X*)    | Get a transformed vector of *X* from mathematical domain to physical domain. |
            +------------------------+------------------------------------------------------------------------------+
            | **from_physic** (*X*)  | Get a transformed vector of *X* from physical model to mathematical domain.  |
            +------------------------+------------------------------------------------------------------------------+
            )mydelimiter"; // kernelo.Testmodel.__doc__

    py::class_<TestModel, FunctionalModel>(m, "TestModel")
        .def(py::init<>())
        .doc() = R"mydelimiter(

            The class :class:`TestModel` is low-dimensional functional model implemented in order to test the GLLiM method with simple but not trivial example.
            The functional F is designed so as to exhibit 2 solutions with D=9 and L=4. :math:`F = A ◦ G ◦ H`, where
             - :math:`A` is a (DxL) injective matrix,
            
                .. math::
                    A = \frac{1}{2} \begin{pmatrix}
                        1 & 2 & 2 & 1 \\
                        0 & 0.5 & 0 & 0 \\
                        0 & 0 & 1 & 0 \\
                        0 & 0 & 0 & 3 \\
                        0.2 & 0 & 0 & 0 \\
                        0 & -0.5 & 0 & 0 \\
                        -0.2 & 0 & -1 & 0 \\
                        -1 & 0 & 2 & 0 \\
                        0 & 0 & 0 & -0.7
                    \end{pmatrix}

             - :math:`G(x)` = :math:`(exp(x_1), exp(x_2), exp(x_3), exp(x_4))` and 
             - :math:`H(x)` = :math:`(x_1, x_2, 4(x_3-0.5)^2, x_4)`.
            
            The resulting F is therefore non-linear and yields two solutions for each observation, denoted by xobs,1 and xobs,2 = 1 - xobs,1.

        )mydelimiter";

    py::class_<ShkuratovModel, FunctionalModel>(m, "ShkuratovModel")
        .def(py::init<mat, std::string, vec, vec>(), py::arg("geometries"), py::arg("variant"), py::arg("scaling_coeffs"), py::arg("offset"))
        .doc() = R"mydelimiter(
            The class :class:`ShkuratovModel` is a representation of the Shkuratov's photometric model.
            For more details check the scientific :ref:`documentation <DocShkuratovModel>`.

            There are two variants of the Shkuratov's formulation, the original model with 5 parameters and a reduced model with 3 parameters.
            The *scaling_coeffs* and *offset* arguments are  to perform affine transformation between the physical space and the mathematical space, such as, 
            :math:`(\text{to_physic}(x))_{1 \leq i \leq L} = (\text{scaling_coeffs}_i x_i + \text{offset}_i)_{1 \leq i \leq L}`.

            :param geometries: The matrix of geometries that will be used by the model. The shape of the Numpy array should be (n_geometries,3).
                The three geometric angles must be this particular order :
                 - At [:,0] the incidence angle (*inc*) also called the solar zenith angle (*sza*)
                 - At [:,1] the emergence angle (*eme*) also called the vertical zenith angle (*vza*)
                 - At [:,2] the phi angle (*phi*)
                Thus the geometries matrix should look like this

                >>> geometries
                array([ [inc_0, eme_0, phi_0],
                        [inc_1, eme_1, phi_1],
                        [inc_2, eme_2, phi_2],
                                ...          
                        [inc_n, eme_n, phi_n]])                

            :type geometries: ndarray of shape (n_geometries,3)
            :param variant: The variant a the Hapke's model among {'3p', '5p'}. '3p' refers to the 3-parameters variant- and '5p' refers to the original 5-parameters model.
            :type variant: string
            :param scaling_coeffs: A set of coefficients used in the transformation between physical and mathematical spaces.
            :type scaling_coeffs: ndarray of shape (L,)
            :param offset: A set of offsets used in the transformation between physical and mathematical spaces
            :type offset: ndarray of shape (L,)
        )mydelimiter";
    
    py::class_<HapkeModel, FunctionalModel>(m, "HapkeModel")
        .def(py::init<mat, std::string, std::string, double, double, double>(), py::arg("geometries"), py::arg("variant"), py::arg("adapter"), py::arg("theta_bar_scaling"), py::arg("b0"), py::arg("h"))
        .doc() = R"mydelimiter(
            The class :class:`HapkeModel` is a representation of the Hapke's photometric model.
            For more details check the scientific :ref:`documentation <DocHapkeModel>`.

            There are two variants of the Hapke's formulation, the initial 1993 version and the 2002 version.
            The calculation in the class are initially designed for a 4 parameters model. The class can
            use a adapter to adapt different variants of the model for example : the model with 3 or 6 parameters.

            See : Hapke B. 1993 Theory of Reflectance and Emittance Spectroscopy. Topics in Remote Sensing.
            Cambridge University Press, Cambridge, UK.

            See : Schmidt F. and Fernando J. 2015 Realistic uncertainties on Hapke model parameters from
            photometric measurement. Icarus, 260 :73 - 93, 2015.

            :param geometries: The matrix of geometries that will be used by the model. The shape of the Numpy array should be (n_geometries,3).
                The three geometric angles must be this particular order :
                 - At [:,0] the incidence angle (*inc*) also called the solar zenith angle (*sza*)
                 - At [:,1] the emergence angle (*eme*) also called the vertical zenith angle (*vza*)
                 - At [:,2] the phi angle (*phi*)
                Thus the geometries matrix should look like this

                >>> geometries
                array([ [inc_0, eme_0, phi_0],
                        [inc_1, eme_1, phi_1],
                        [inc_2, eme_2, phi_2],
                                ...          
                        [inc_n, eme_n, phi_n]])                

            :type geometries: ndarray of shape (n_geometries,3)
            :param variant: The variant a the Hapke's model among {'1993', '2002'}.
            :type variant: string
            :param adapter: This argument allows to adapt the Hapke's model parameter number among {'three', 'four', 'six'}.
            :type adapter: string
            :param theta_bar_scaling: Value used to transform theta_bar between physical and mathematical spaces.
            :type theta_bar_scaling: float
            :param b0: The amplitude of the opposition effect
            :type b0: float
            :param h: The angular width of the opposition effect
            :type h: float
        )mydelimiter";
        
    py::class_<ExternalPythonModel, FunctionalModel>(m, "ExternalPythonModel")
        .def(py::init<std::string, std::string, std::string>(), py::arg("className"), py::arg("fileName"), py::arg("filePath"))
        .doc() = R"mydelimiter(
            
            The PlanetGLLiM software allows you to edit your own physical model. To do so you must provide a Python file (.py) 
            describing your physical model as a Python class. The template of such python module is described below.

            >>> # IMPORTS
            >>> import numpy as np
            >>>
            >>> class PhysicalModel(object):
            >>>    """ This is a python class defining a functional model. 
            >>>    
            >>>    This class is composed of 5 mandatory functions:
            >>>        - F: the functional model F describing the physical model. F takes photometries as arguments and return reflectances
            >>>        - get_D_dimension: returns the dimension of Y (reflectances)
            >>>        - get_L_dimension: return de dimension of X (photometries)
            >>>        - to_physic: converts the X data from mathematical framework (0<X<1) to physical framework
            >>>        - from_physic: converts the X data from physical framework to mathematical framework (0<X<1)
            >>>
            >>>    Note that some class constants, other functions and class constructors can be declared.
            >>>
            >>>    Geometries : if your physical model requires geometries as with Shkuratov, the structure and the values of the geometries
            >>>        must be declared within this Python file.
            >>>
            >>>    See the Planet-Gllim documentation for more informations (https://gitlab.inria.fr/kernelo-mistis/planet-gllim-front-end/-/wikis/home)
            >>>    """
            >>>    
            >>>    #################################################################################################
            >>>    ##                          CLASS CONSTANTS (OPTIONAL)                                         ##
            >>>    #################################################################################################
            >>>
            >>>    # Geometries of the physical model MUST be declared here - directly in the python file. 
            >>>
            >>>
            >>>    #################################################################################################
            >>>    ##                          CORE FUNCTIONS (MANDATORY)                                         ##
            >>>    #################################################################################################
            >>>
            >>>    def F(self):
            >>>        pass
            >>>
            >>>    def get_D_dimension(self):
            >>>        pass
            >>>
            >>>    def get_L_dimension(self):
            >>>        pass
            >>>
            >>>    def to_physic(self):
            >>>        pass
            >>>
            >>>    def from_physic(self):
            >>>        pass
            >>>
            >>>
            >>>    #################################################################################################
            >>>    ##                          OTHER FUNCTIONS (OPTIONAL)                                         ##
            >>>    #################################################################################################
            >>>
            >>>    def __init__(self, *args, **kwargs):
            >>>        pass
            >>>
            >>>    def optional_but_useful_function():
            >>>        pass

            If the functional model shows interest it can be added in further development in Kernelo as a built-in class to 
            improve speed performance.

            :param className: The exact name of the python class.
            :type className: string
            :param fileName: The exact name of the python module without the .py extension.
            :type fileName: string
            :param filePath: The directory path of the python module.
            :type filePath: string
        )mydelimiter";

    // m.doc() = R"mydelimiter(
    //     Kernelo
    //     -----------------------
    //     Functional
    //     Learning
    //     DataGeneration
    //     ...
    // )mydelimiter"; // kernelo.__doc__
}
