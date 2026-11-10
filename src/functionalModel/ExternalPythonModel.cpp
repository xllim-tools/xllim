#include "ExternalPythonModel.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
// #include <pybind11/embed.h>

namespace py = pybind11;
using namespace py::literals;

namespace
{
// Conversions between arma::vec and NumPy arrays always COPY the data: no memory is ever shared between Armadillo
// and NumPy, so the destruction of one object can never leave the other pointing to freed memory (see issue #67).
// The vectors are small (dimension L or D) and each call already costs a Python call, so the copies are negligible.

// arma::vec -> new 1D NumPy array
py::array_t<double> toNumpy(const vec &v)
{
    return py::array_t<double>(static_cast<py::ssize_t>(v.n_elem), v.memptr());
}

// Array-like returned by the Python model -> new arma::vec.
// If expectedSize != 0, the number of values is checked.
vec toArma(py::handle obj, const char *method, arma::uword expectedSize = 0)
{
    const std::string prefix = std::string("ExternalPythonModel: ") + method + "()";
    if (obj.is_none())
        throw std::runtime_error(prefix + " returned None, an array of floats is expected");

    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr)
        throw std::runtime_error(prefix + " must return an array of floats");

    const auto size = static_cast<arma::uword>(arr.size());
    if (expectedSize != 0 && size != expectedSize)
        throw std::runtime_error(prefix + " returned " + std::to_string(size) + " values, " +
                                 std::to_string(expectedSize) + " expected");

    return vec(arr.data(), size); // This Armadillo constructor copies the data
}

// Calls pyModel.<method>(x) for toPhysic / fromPhysic. The Python method may either modify its argument in place
// or return the converted array: the returned value is used when it is not None.
void applyConversion(const py::object &pyModel, const char *method, vec &x)
{
    py::array_t<double> x_arr = toNumpy(x);
    py::object result = pyModel.attr(method)(x_arr);
    x = toArma(result.is_none() ? py::handle(x_arr) : py::handle(result), method, x.n_elem);
}
} // namespace

ExternalPythonModel::ExternalPythonModel(const std::string &className, const std::string &fileName, const std::string &filePath)
{
    py::gil_scoped_acquire acquire; // Ensure GIL is held

    py::module_ sys = py::module_::import("sys");
    py::list path = sys.attr("path");
    path.attr("append")(filePath.c_str());

    pModule_ = py::module_::import(fileName.c_str());
    pObj_ = pModule_.attr(className.c_str())();
}

void ExternalPythonModel::F(vec x, vec &y)
{
    py::gil_scoped_acquire acquire; // Ensure GIL is held
    py::object result = pObj_.attr("F")(toNumpy(x));
    y = toArma(result, "F");
}

unsigned ExternalPythonModel::getDimensionY()
{
    py::gil_scoped_acquire acquire; // Ensure GIL is held
    return pObj_.attr("getDimensionY")().cast<unsigned>();
}

unsigned ExternalPythonModel::getDimensionX()
{
    py::gil_scoped_acquire acquire; // Ensure GIL is held
    return pObj_.attr("getDimensionX")().cast<unsigned>();
}

void ExternalPythonModel::toPhysic(vec &x)
{
    py::gil_scoped_acquire acquire; // Ensure GIL is held
    applyConversion(pObj_, "toPhysic", x);
}

void ExternalPythonModel::fromPhysic(vec &x)
{
    py::gil_scoped_acquire acquire; // Ensure GIL is held
    applyConversion(pObj_, "fromPhysic", x);
}
