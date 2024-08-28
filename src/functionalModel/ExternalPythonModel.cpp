#include "ExternalPythonModel.hpp"
#include <pybind11/embed.h>
#include <carma>

namespace py = pybind11;
using namespace py::literals;

ExternalPythonModel::ExternalPythonModel(const std::string &className, const std::string &fileName, const std::string &filePath)
{

    if (!Py_IsInitialized())
    {                                   // Check if the Python interpreter is already running
        py::scoped_interpreter guard{}; // If not, initialize it
    }

    py::module_ sys = py::module_::import("sys");
    py::list path = sys.attr("path");
    path.attr("append")(filePath.c_str());

    pModule_ = py::module_::import(fileName.c_str());
    pObj_ = pModule_.attr(className.c_str())();
}

void ExternalPythonModel::F(vec x, vec &y)
{
    py::array_t<double> x_arr = carma::col_to_arr(x).squeeze(); // Convert the Carma vector to a NumPy array + squeeze the array into shape (x,)
    auto result = pObj_.attr("F")(x_arr).cast<py::array_t<double>>();
    y = carma::arr_to_col(result, true); // Convert the NumPy array to a Carma vector with copy because result object is modify
}

unsigned ExternalPythonModel::getDimensionY()
{
    return pObj_.attr("getDimensionY")().cast<unsigned>();
}

unsigned ExternalPythonModel::getDimensionX()
{
    return pObj_.attr("getDimensionX")().cast<unsigned>();
}

void ExternalPythonModel::toPhysic(vec &x)
{
    py::array_t<double> x_arr = carma::col_to_arr(x).squeeze(); // Convert the Carma vector to a NumPy array + squeeze the array into shape (x,)
    py::object result = pObj_.attr("toPhysic")(x_arr);
    x = carma::arr_to_col(x_arr); // Convert the NumPy array to a Carma vector
}

void ExternalPythonModel::fromPhysic(vec &x)
{
    py::array_t<double> x_arr = carma::col_to_arr(x).squeeze(); // Convert the Carma vector to a NumPy array + squeeze the array into shape (x,)
    py::object result = pObj_.attr("fromPhysic")(x_arr);
    x = carma::arr_to_col(x_arr); // Convert the NumPy array to a Carma vector
}
