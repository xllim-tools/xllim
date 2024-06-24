#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <carma>
#include <armadillo>

#include "../src/xllimSolver/gllim.hpp"
#include "../src/xllimSolver/jgmm.hpp"

namespace py = pybind11;

void bind_gllim(pybind11::module &m)
{
    py::class_<GLLiMParameters>(m, "GLLiMParameters")
        .def(py::init<unsigned, unsigned, unsigned>())
        .def_readwrite("L", &GLLiMParameters::L)
        .def_readwrite("D", &GLLiMParameters::D)
        .def_readwrite("K", &GLLiMParameters::K)
        .def_readwrite("Pi", &GLLiMParameters::Pi)
        .def_readwrite("C", &GLLiMParameters::C)
        .def_readwrite("Gamma", &GLLiMParameters::Gamma)
        .def_readwrite("A", &GLLiMParameters::A)
        .def_readwrite("B", &GLLiMParameters::B)
        .def_readwrite("Sigma", &GLLiMParameters::Sigma);

    py::class_<MeanPredictionResult>(m, "MeanPredictionResult")
        .def_readwrite("mean", &MeanPredictionResult::mean)
        .def_readwrite("variance", &MeanPredictionResult::variance)
        .def_readwrite("gmm_weights", &MeanPredictionResult::gmm_weights)
        .def_readwrite("gmm_means", &MeanPredictionResult::gmm_means)
        .def_readwrite("gmm_covs", &MeanPredictionResult::gmm_covs);

    py::class_<CenterPredictionResult>(m, "CenterPredictionResult")
        .def_readwrite("weights", &CenterPredictionResult::weights)
        .def_readwrite("means", &CenterPredictionResult::means)
        .def_readwrite("covs", &CenterPredictionResult::covs);

    py::class_<PredictionResult>(m, "PredictionResult")
        .def_readwrite("meanPredResult", &PredictionResult::meanPredResult)
        .def_readwrite("centerPredResult", &PredictionResult::centerPredResult);

    py::class_<GLLiM, std::shared_ptr<GLLiM>>(m, "GLLiM")
        .def(py::init<unsigned, unsigned, unsigned, std::string, std::string>(), py::arg("L"), py::arg("D"), py::arg("K"), py::arg("gamma_type"), py::arg("sigma_type"))
        .def("getParams", &GLLiM::getParams)
        .def("getDimensions", &GLLiM::getDimensions)
        .def("getParamPi", &GLLiM::getParamPi)
        .def("getParamA", &GLLiM::getParamA)
        .def("getParamC", &GLLiM::getParamC)
        .def("getParamGamma", &GLLiM::getParamGamma)
        .def("getParamB", &GLLiM::getParamB)
        .def("getParamSigma", &GLLiM::getParamSigma)

        .def("setParams", &GLLiM::setParams)
        .def("setParamPi", &GLLiM::setParamPi)
        .def("setParamA", &GLLiM::setParamA)
        .def("setParamC", &GLLiM::setParamC)
        .def("setParamGamma", &GLLiM::setParamGamma)
        .def("setParamB", &GLLiM::setParamB)
        .def("setParamSigma", &GLLiM::setParamSigma)

        .def("getInverse", &GLLiM::getInverse)

        .def("directDensities", py::overload_cast<const mat &, const vec &>(&GLLiM::directDensities))
        .def("directDensities", py::overload_cast<const mat &>(&GLLiM::directDensities))
        .def("inverseDensities", py::overload_cast<const mat &, const mat &>(&GLLiM::inverseDensities))
        .def("inverseDensities", py::overload_cast<const mat &>(&GLLiM::inverseDensities))

        .def("train", &GLLiM::train);

    // Do we need to expose this class ?
    py::class_<JGMM, std::shared_ptr<JGMM>>(m, "JGMM")
        .def(py::init<>())
        .def("train", &JGMM::train)
        .def("getPosterior", &JGMM::getPosterior)

        .doc() = R"mydelimiter(
            GLLiM class
        )mydelimiter";
}
