#include <pybind11/pybind11.h>
#include <carma> // implicit call of carma within pybind11: carma automatic conversion. see documentation https://carma.readthedocs.io/
#include <armadillo>

#include "../src/TestModel.hpp"
#include "../src/FunctionalModel.hpp"

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
//     arma::rowvec photometry_arma = carma::arr_to_mat<double>(photometry);

//     // useful code
//     arma::rowvec reflectances_arma;
//     TestModel::F(photometry_arma, reflectances_arma);

//     // convert to Numpy array and copy out
//     return carma::mat_to_arr(reflectances_arma, true);
// }



PYBIND11_MODULE(cmake_example, m) {

    py::class_<TestModel>(m, "TestModel")
        .def(py::init<>())
        .def("F", static_cast<void (TestModel::*)(arma::rowvec, arma::rowvec &)>(&TestModel::F), "F return reference")
        .def("F", static_cast<arma::rowvec (FunctionalModel::*)(arma::rowvec)>(&FunctionalModel::F), "F return value")
        .def("get_D_dimension", &TestModel::get_D_dimension)
        .def("get_L_dimension", &TestModel::get_L_dimension);
        
    
    m.doc() = R"pbdoc(
        Pybind11 example
        -----------------------
        F
        get_D_dimension
        get_L_dimension
    )pbdoc";

    // m.def("F", &TestModel::F, R"pbdoc(
    //     Add two numbers

    //     Some other explanation about the add function.
    // )pbdoc");

    // m.def("get_D_dimension", &TestModel::get_D_dimension, R"pbdoc(
    //     Add two numbers

    //     Some other explanation about the add function.
    // )pbdoc");

    // m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
    //     Subtract two numbers

    //     Some other explanation about the subtract function.
    // )pbdoc");

        // m.attr("__version__") = "0.0.1";
}
