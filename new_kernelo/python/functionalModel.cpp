#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <carma>
#include <armadillo>

// #include "../src/functionalModel/FunctionalModel.hpp"
#include "../src/functionalModel/TestModel.hpp"
#include "../src/functionalModel/ShkuratovModel.hpp"
#include "../src/functionalModel/HapkeModel.hpp"
#include "../src/functionalModel/ExternalPythonModel.hpp"

namespace py = pybind11;

void bind_functional_model(pybind11::module& m)
{
    // PYBIND11_NUMPY_DTYPE(Dummy, predictions, predictions_variance);
    py::class_<ImportanceSamplingResult>(m, "ImportanceSamplingResult")
        .def(py::init<unsigned, unsigned>())
        .def_readwrite("predictions", &ImportanceSamplingResult::predictions)
        .def_readwrite("predictions_variance", &ImportanceSamplingResult::predictions_variance)
        .def_readwrite("nb_effective_sample", &ImportanceSamplingResult::nb_effective_sample)
        .def_readwrite("effective_sample_size", &ImportanceSamplingResult::effective_sample_size)
        .def_readwrite("qn", &ImportanceSamplingResult::qn);
    
    py::class_<FunctionalModel, std::shared_ptr<FunctionalModel> > (m, "FunctionalModel")
        .def("F", [](FunctionalModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);
                arma::vec y_arma;
                self.F(x_arma, y_arma);
                py::array_t<double> y_arr = carma::col_to_arr(y_arma).squeeze();
                return y_arr; })
        .def("getDimensionY", &FunctionalModel::getDimensionY)
        .def("getDimensionX", &FunctionalModel::getDimensionX)
        .def("toPhysic", [](FunctionalModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);                      // Convert the NumPy array to a Carma vector with copy=true because we want to argument to keep unmodified
                self.toPhysic(x_arma);                                             // Call the C++ function
                py::array_t<double> y_arr = carma::col_to_arr(x_arma).squeeze();    // Convert the Carma vector back to a NumPy array + squeeze the array into shape (x,)
                return y_arr; })
        .def("fromPhysic", [](FunctionalModel &self, py::array_t<double> x)
            {
                arma::vec x_arma = carma::arr_to_col(x, true);
                self.fromPhysic(x_arma);
                py::array_t<double> y_arr = carma::col_to_arr(x_arma).squeeze();
                return y_arr;
            })
        .def("genData", py::overload_cast<unsigned, const std::string &, double, unsigned>(&FunctionalModel::genData))
        .def("genData", py::overload_cast<unsigned, const std::string &, vec &, unsigned>(&FunctionalModel::genData))
        .def("importanceSampling", &FunctionalModel::importanceSampling);
        

    py::class_<TestModel, std::shared_ptr<TestModel>, FunctionalModel>(m, "TestModel")
        .def(py::init<>());

    py::class_<ShkuratovModel, std::shared_ptr<ShkuratovModel>, FunctionalModel>(m, "ShkuratovModel")
        .def(py::init<mat, std::string, vec, vec>(), py::arg("geometries"), py::arg("variant"), py::arg("scaling_coeffs"), py::arg("offset"));
    
    py::class_<HapkeModel, std::shared_ptr<HapkeModel>, FunctionalModel>(m, "HapkeModel")
        .def(py::init<mat, std::string, std::string, double, double, double>(), py::arg("geometries"), py::arg("variant"), py::arg("adapter"), py::arg("theta_bar_scaling"), py::arg("b0"), py::arg("h"));
        
    py::class_<ExternalPythonModel, std::shared_ptr<ExternalPythonModel>, FunctionalModel>(m, "ExternalPythonModel")
        .def(py::init<std::string, std::string, std::string>(), py::arg("className"), py::arg("fileName"), py::arg("filePath"));
}
