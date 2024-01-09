#include <pybind11/pybind11.h>
#include <carma> // implicit call of carma within pybind11: carma automatic conversion. see documentation https://carma.readthedocs.io/
#include <armadillo>

#include "../src/FunctionalModel.hpp"
#include "../src/TestModel.hpp"
#include "../src/ShkuratovModel.hpp"
#include "../src/HapkeModel.hpp"

using namespace Functional;

namespace py = pybind11;

// Du coup on écrit le max en C++ avec les fonctions plus adaptée (signatures différentes)
// Ici on expose seulement les fonctions utiles (eg y=F(x)). Le code pybind11 reste épuré et on s'enbête pas avec des fonctions de conversion..
// A la limite on pourrait faire des fonctions bind_TestModel(..){py::class_<> ...} et tout mettre dans un PYBUND11_MODULE() en bas de page
// Influence de carma ?? OUI => automatic conversion

// 2 methodes pour inclure CARMA:
//      soit C/C dans le code en dur carma/include et faire #include <carma.carma.h> dans le C++ pybind11
//      soit comme dans la docu officiel avec cmake (instalation dans /usr/local/include/) ... mouais
//  -> c'est plus clair si le carma est directement dans le code, discretmement dans python/carma, à côté des binding files
// Même "problème" avec pybind11 (submodule or pip/cmake installation)

// examples below from https://carma.readthedocs.io/en/latest/examples.html

// py::array_t<double> F(py::array_t<double> photometry) {

//     // convert to armadillo matrix without copying.
//     // Note the size of the matrix cannot be changed when borrowing
//     arma::vec photometry_arma = carma::arr_to_mat<double>(photometry);

//     // useful code
//     arma::vec reflectances_arma;
//     TestModel::F(photometry_arma, reflectances_arma);

//     // convert to Numpy array and copy out
//     return carma::mat_to_arr(reflectances_arma, true);
// }

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
                Functional Model
                -----------------------
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
            )mydelimiter"; // kernelo.Testmodel.__doc__

    py::class_<TestModel, FunctionalModel>(m, "TestModel")
        .def(py::init<>())
        .doc() = R"mydelimiter(
                TestModel
                -----------------------
                derived from Functional
                F(x) = 1/2A*exp(HX) ...
            )mydelimiter";

    py::class_<ShkuratovModel, FunctionalModel>(m, "ShkuratovModel")
        .def(py::init<mat, std::string, vec, vec>(), py::arg("geometries"), py::arg("variant"), py::arg("scaling_coeffs"), py::arg("offset"),
            R"mydelimiter(
                Constructor
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

            )mydelimiter")
        .doc() = R"mydelimiter(
                ShkuratovModel
                -----------------------
                derived from Functional.
                Some more details.
                F(x) = alpha*cos(i) ...
            )mydelimiter";
    
    py::class_<HapkeModel, FunctionalModel>(m, "HapkeModel")
        .def(py::init<mat, std::string, std::string, double, double, double>(), py::arg("geometries"), py::arg("variant"), py::arg("adapter"), py::arg("theta_bar_scaling"), py::arg("b0"), py::arg("h"),
            R"mydelimiter(
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
            )mydelimiter")
        .doc() = R"mydelimiter(
                HapkeModel
                -----------------------
                derived from Functional.
                Some more details.
                F(x) = alpha*cos(i) ...
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
