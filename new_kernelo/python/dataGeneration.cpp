#include <pybind11/pybind11.h>
#include <carma>
#include <armadillo>

// #include "../src/dataGeneration/generator/Generator.hpp"
// #include "../src/dataGeneration/generator/RandomGenerator.hpp"
#include "../src/dataGeneration/statModel/StatModel.hpp"
#include "../src/dataGeneration/statModel/GaussianStatModel.hpp"
#include "../src/dataGeneration/statModel/DependentGaussianStatModel.hpp"
#include "../src/functionalModel/FunctionalModel.hpp"

using namespace DataGeneration;
namespace py = pybind11;


void bind_data_generation(pybind11::module &m)
{
    py::class_<StatModel>(m, "StatModel")
        .def("gen_data", &StatModel::gen_data,
             R"mydelimiter(
                StatModel
            )mydelimiter");
    py::class_<GaussianStatModel, StatModel>(m, "GaussianStatModel")
        .def(py::init<std::string &, std::shared_ptr<FunctionalModel>, vec, unsigned int>(), py::arg("generator_type"), py::arg("functional_model"), py::arg("covariance"), py::arg("seed"))
        .doc() = R"mydelimiter(
                GaussianStatModel
            )mydelimiter";
    py::class_<DependentGaussianStatModel, StatModel>(m, "DependentGaussianStatModel")
        .def(py::init<std::string &, std::shared_ptr<FunctionalModel>, double, unsigned int>(), py::arg("generator_type"), py::arg("functional_model"), py::arg("r"), py::arg("seed"))
        .doc() = R"mydelimiter(
                DependentGaussianStatModel
            )mydelimiter";
}
