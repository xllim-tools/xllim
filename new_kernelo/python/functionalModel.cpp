#include <pybind11/pybind11.h>
#include <carma> // implicit call of carma within pybind11: carma automatic conversion. see documentation https://carma.readthedocs.io/
#include <armadillo>

#include "../src/FunctionalModel.hpp"
#include "../src/TestModel.hpp"
#include "../src/ShkuratovModel.hpp"

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
    py::class_<TestModel>(m, "TestModel")
        .doc() = R"pbdoc(
            TestModel
            -----------------------
            derived from Functional
            F(x) = 1/2A*exp(HX) ...
        )pbdoc"; // kernelo.Testmodel.__doc__
        .def(py::init<>())
        .def("F", [](TestModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);
                arma::vec y_arma;
                self.F(x_arma, y_arma);
                py::array_t<double> y_arr = carma::col_to_arr(y_arma).squeeze();
                return y_arr;
            }, 
            R"pbdoc(
                Add two numbers
                Some other explanation about the add function.
            )pbdoc") // kernelo.Testmodel.__doc__. Note that the identation in R"pbdoc() is kept
        .def("F", static_cast<arma::vec (FunctionalModel::*)(arma::vec)>(&FunctionalModel::F), R"pbdoc(
            Add two numbers
            Some other explanation about the add function.
            )pbdoc") // kernelo.Testmodel.__doc__. Note that the identation in R"pbdoc() is kept
        .def("get_D_dimension", &TestModel::get_D_dimension)
        .def("get_L_dimension", &TestModel::get_L_dimension)
        .def("to_physic", [](TestModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);                      // Convert the NumPy array to a Carma vector with copy=true because we want to argument to keep unmodified
                self.to_physic(x_arma);                                             // Call the C++ function
                py::array_t<double> y_arr = carma::col_to_arr(x_arma).squeeze();    // Convert the Carma vector back to a NumPy array + squeeze the array into shape (x,)
                return y_arr;
            })
        .def("from_physic", [](TestModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);
                self.from_physic(x_arma);
                py::array_t<double> y_arr = carma::col_to_arr(x_arma).squeeze();
                return y_arr;
            })


    py::class_<ShkuratovModel>(m, "ShkuratovModel")
        .def(py::init<mat, std::string, vec, vec>(),
             // .def(py::init<py::array_t<double>, std::string, py::array_t<double>, py::array_t<double>>(),//py::array_t<double>mat, std::string, vec, vec>(),
             py::arg("geometries"), py::arg("variant"), py::arg("scaling_coeffs"), py::arg("offset"))
        .def("F", [](ShkuratovModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);
                arma::vec y_arma;
                self.F(x_arma, y_arma);
                py::array_t<double> y_arr = carma::col_to_arr(y_arma).squeeze();
                return y_arr;
            }, R"pbdoc(
                Add two numbers
                Some other explanation about the add function.
            )pbdoc")
        .def("get_D_dimension", &ShkuratovModel::get_D_dimension)
        .def("get_L_dimension", &ShkuratovModel::get_L_dimension)
        .def("to_physic", [](ShkuratovModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);
                self.to_physic(x_arma);
                py::array_t<double> y_arr = carma::col_to_arr(x_arma).squeeze();
                return y_arr;
            })
        .def("from_physic", [](ShkuratovModel &self, py::array_t<double> x)
             {
                arma::vec x_arma = carma::arr_to_col(x, true);
                self.from_physic(x_arma);
                py::array_t<double> y_arr = carma::col_to_arr(x_arma).squeeze();
                return y_arr;
             })
        .doc() = R"pbdoc(
            ShkuratovModel
            -----------------------
            derived from Functional
            F(x) = 1/2A*exp(HX) ...
        )pbdoc";

    m.doc() = R"pbdoc(
        Kernelo
        -----------------------
        Functional
        Learning
        DataGeneration
        ...
    )pbdoc"; // kernelo.__doc__
}
