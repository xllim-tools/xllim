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



PYBIND11_MODULE(newkernelo, m) {

    py::class_<TestModel>(m, "TestModel")
        .def(py::init<>())
        .def("F", static_cast<arma::rowvec (FunctionalModel::*)(arma::rowvec)>(&FunctionalModel::F), R"pbdoc(
            Add two numbers
            Some other explanation about the add function.
            )pbdoc") // kernelo.Testmodel.__doc__. Note that the identation in R"pbdoc() is kept
        .def("get_D_dimension", &TestModel::get_D_dimension)
        .def("get_L_dimension", &TestModel::get_L_dimension)
        // .def("to_physic", &TestModel::to_physic, pybind11::return_value_policy::reference)
        .def("to_physic", [](TestModel &self, py::array_t<double> x){
            auto carmaVec = carma::arr_to_row(x,true); // Convert the NumPy array to a Carma vector with copy=true because we want to argument to keep unmodified
            self.to_physic(carmaVec); // Call the C++ function
            return carma::row_to_arr(carmaVec); // Convert the Carma vector back to a NumPy array
        })
        // .def("to_physics", &FunctionalModel::to_physics)
        // .def("from_physic", &TestModel::from_physic)//, pybind11::return_value_policy::reference)
        .def("from_physic", [](TestModel &self, py::array_t<double> x){
            auto carmaVec = carma::arr_to_row(x,true); // Convert the NumPy array to a Carma vector with copy=true because we want to argument to keep unmodified
            self.from_physic(carmaVec); // Call the C++ function
            return carma::row_to_arr(carmaVec); // Convert the Carma vector back to a NumPy array
        })
        .doc() = R"pbdoc(
            TestModel
            -----------------------
            derived from Functional
            F(x) = 1/2A*exp(HX) ...
        )pbdoc"; // kernelo.Testmodel.__doc__
        
    
    m.doc() = R"pbdoc(
        Kernelo
        -----------------------
        Functional
        Learning
        DataGeneration
        ...
    )pbdoc"; // kernelo.__doc__

}
