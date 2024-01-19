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
                Computes and returns Y = F(X)

                Returns
                -------
                ndarray
                    1D array containing Y = F(X)
            )mydelimiter") // kernelo.FunctionalModel.__doc__.
        .def("get_D_dimension", &FunctionalModel::get_D_dimension,
            R"mydelimiter(
                Returns the D dimension of the model.

                Returns
                -------
                int
                    The D dimension of the model.
            )mydelimiter")
        .def("get_L_dimension", &FunctionalModel::get_L_dimension,
            R"mydelimiter(
                Returns the L dimension of the model.

                Returns
                -------
                int
                    The L dimension of the model.
            )mydelimiter")
        .def("to_physic", [](FunctionalModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);                      // Convert the NumPy array to a Carma vector with copy=true because we want to argument to keep unmodified
                self.to_physic(x_arma);                                             // Call the C++ function
                py::array_t<double> y_arr = carma::col_to_arr(x_arma).squeeze();    // Convert the Carma vector back to a NumPy array + squeeze the array into shape (x,)
                return y_arr; },
            R"mydelimiter(
                to_physic(x)

                Returns
                -------
                ndarray
                    1D array where the values of x are in physical space
            )mydelimiter")
        .def("from_physic", [](FunctionalModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);
                self.from_physic(x_arma);
                py::array_t<double> y_arr = carma::col_to_arr(x_arma).squeeze();
                return y_arr;
            }, R"mydelimiter(
                from_physic(x)

                Returns
                -------
                ndarray
                    1D array where the values of x are normalized
            )mydelimiter")
        .doc() = R"mydelimiter(

            Some general presentation.


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
            blabla Test Model ....
            
            Parameters
            -----------
            ShkuratovModel(geometries, variant, scalingCoeffs, offset)

            ndarray geometries
                2D array containing N geometries with D dimensions.
            string variant
                The variant of the model corresponding to the number of parameters. It must be one of the following keywords : {"5p","3p"}.
            ndarray scalingCoeffs
                1D array containing 5 values used to normalize the photometric variables of the model, where normalized_x = (x - offset)/scalingCoeff
            ndarray offset
                1D array containing 5 values used to normalize the photometric variables of the model, where normalized_x = (x - offset)/scalingCoeff
        
        )mydelimiter";

    py::class_<ShkuratovModel, FunctionalModel>(m, "ShkuratovModel")
        .def(py::init<mat, std::string, vec, vec>(), py::arg("geometries"), py::arg("variant"), py::arg("scaling_coeffs"), py::arg("offset"))
        .doc() = R"mydelimiter(
            blabla Shkuratov
            
            Parameters
            -----------
            ShkuratovModel(geometries, variant, scalingCoeffs, offset)

            ndarray geometries
                2D array containing N geometries with D dimensions.
            string variant
                The variant of the model corresponding to the number of parameters. It must be one of the following keywords : {"5p","3p"}.
            ndarray scalingCoeffs
                1D array containing 5 values used to normalize the photometric variables of the model, where normalized_x = (x - offset)/scalingCoeff
            ndarray offset
                1D array containing 5 values used to normalize the photometric variables of the model, where normalized_x = (x - offset)/scalingCoeff
        
        )mydelimiter";
    
    py::class_<HapkeModel, FunctionalModel>(m, "HapkeModel")
        .def(py::init<mat, std::string, std::string, double, double, double>(), py::arg("geometries"), py::arg("variant"), py::arg("adapter"), py::arg("theta_bar_scaling"), py::arg("b0"), py::arg("h"))
        .doc() = R"mydelimiter(
            Hapke blabla

            Parameters
            -----------
            string version
                The version of the hapke model must be one of the following keywords : {"2002","1993"}.
            HapkeAdapterConfig adapter
                This object is used to create a Hapke model adapter.
            ndarray geometries
                2D array containing N geometries with D dimensions.
            double theta_bar_scalling
                Used to transform theta_bar between physical and mathematical spaces.
        )mydelimiter";
        
    py::class_<ExternalPythonModel, FunctionalModel>(m, "ExternalPythonModel")
        .def(py::init<std::string, std::string, std::string>(), py::arg("className"), py::arg("fileName"), py::arg("filePath"))
        .doc() = R"mydelimiter(
            External Python module balbal

            Parameters
            -----------
            string className
                The version of the hapke model must be one of the following keywords : {"2002","1993"}.
            string fileName
                This object is used to create a Hapke model adapter.
            string filePath
                2D array containing N geometries with D dimensions.
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
