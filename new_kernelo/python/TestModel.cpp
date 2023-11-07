#include <pybind11/pybind11.h>
#include <armadillo>
#include <carma>

#include "../src/TestModel.hpp"

// #define STRINGIFY(x) #x
// #define MACRO_STRINGIFY(x) STRINGIFY(x)
using namespace Functional;

namespace py = pybind11;

py::array_t<double> F(py::array_t<double> photometry,  &reflectances) {
    TestModel::F(photometry, reflectances);
}

PYBIND11_MODULE(cmake_example, m) {

    py::class_<TestModel>(m, "TestModel")
        .def(py::init<>())
        m.def("F", &F,
            R"pbdoc(
                Example function for automatic conversion.

                Parameters
                ----------
                mat : np.array
                    input array

                Returns
                -------
                result : np.array
                    output array
            )pbdoc",
            py::arg("x"),
            py::arg("y")
            )
        .def("get_D_dimension", &TestModel::get_D_dimension);
    
    // m.doc() = R"pbdoc(
    //     Pybind11 example plugin
    //     -----------------------

    //     .. currentmodule:: cmake_example

    //     .. autosummary::
    //        :toctree: _generate

    //        F
    //        get_D_dimension
    //        get_L_dimension
    // )pbdoc";

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

    m.attr("__version__") = "0.0.1";
}
