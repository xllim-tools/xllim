#include "ExternalPythonModel.hpp"
#include <pybind11/embed.h>
#include <carma>

namespace py = pybind11;
using namespace py::literals;
using namespace Functional;


ExternalPythonModel::ExternalPythonModel(const std::string &className, const std::string &fileName, const std::string &filePath) {

    if (!Py_IsInitialized()) { // Check if the Python interpreter is already running
        py::scoped_interpreter guard{}; // If not, initialize it
    }

    py::module_ sys = py::module_::import("sys");
    py::list path = sys.attr("path");
    path.attr("append")(filePath.c_str());

    this->pModule = py::module_::import(fileName.c_str());
    this->pObj = pModule.attr(className.c_str())();
}

void ExternalPythonModel::F(vec x, vec &y) {
    py::array_t<double> x_arr = carma::col_to_arr(x).squeeze();     // Convert the Carma vector to a NumPy array + squeeze the array into shape (x,)
    auto result = pObj.attr("F")(x_arr).cast<py::array_t<double>>();
    y = carma::arr_to_col(result, true);                            // Convert the NumPy array to a Carma vector with copy because result object is modify
}

int ExternalPythonModel::get_D_dimension() {
    return pObj.attr("get_D_dimension")().cast<int>();
}

int ExternalPythonModel::get_L_dimension() {
    return pObj.attr("get_L_dimension")().cast<int>();
}

void ExternalPythonModel::to_physic(vec &x) {
    py::array_t<double> x_arr = carma::col_to_arr(x).squeeze();     // Convert the Carma vector to a NumPy array + squeeze the array into shape (x,)
    py::object result = pObj.attr("to_physic")(x_arr);
    x = carma::arr_to_col(x_arr);                                   // Convert the NumPy array to a Carma vector
}

void ExternalPythonModel::from_physic(vec &x) {
    py::array_t<double> x_arr = carma::col_to_arr(x).squeeze();     // Convert the Carma vector to a NumPy array + squeeze the array into shape (x,)
    py::object result = pObj.attr("from_physic")(x_arr);
    x = carma::arr_to_col(x_arr);                                   // Convert the NumPy array to a Carma vector
}
