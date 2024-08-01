#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <carma>
#include <armadillo>

#include "../src/xllimSolver/gllim.hpp"
#include "../src/xllimSolver/jgmm.hpp"
#include "../src/xllimSolver/factory.hpp"
#include "../src/xllimSolver/gllimStructures/gllimParametersArma.hpp"

namespace py = pybind11;
void bind_gllim(pybind11::module &m)
{
    py::class_<GLLiMBase, std::shared_ptr<GLLiMBase>>(m, "GLLiMBase");

    py::class_<GLLiMParametersBase, std::shared_ptr<GLLiMParametersBase>>(m, "GLLiMParametersBase");

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

    m.def("GLLiM", &create_gllim, "A function to create GLLiM instances",
          py::arg("L"), py::arg("D"), py::arg("K"), py::arg("gamma_type"), py::arg("sigma_type"));
    m.def("GLLiMParameters", &create_gllim_parameters, "A function to create a GLLiM parameters structure",
          py::arg("L"), py::arg("D"), py::arg("K"), py::arg("gamma_type"), py::arg("sigma_type"));
}

template <typename TGamma, typename TSigma>
void bind_gllim_templates(pybind11::module &m, const std::string &str)
{
    std::string GLLiMParameters_pyname = std::string("_GLLiMParameters") + str;
    py::class_<GLLiMParametersArma<TGamma, TSigma>, std::shared_ptr<GLLiMParametersArma<TGamma, TSigma>>>(m, GLLiMParameters_pyname.c_str())
        .def(py::init<unsigned, unsigned, unsigned>())
        .def_readwrite("Pi", &GLLiMParametersArma<TGamma, TSigma>::Pi)
        .def_readwrite("A", &GLLiMParametersArma<TGamma, TSigma>::A)
        .def_readwrite("B", &GLLiMParametersArma<TGamma, TSigma>::B)
        .def_readwrite("C", &GLLiMParametersArma<TGamma, TSigma>::C)
        .def_readwrite("Gamma", &GLLiMParametersArma<TGamma, TSigma>::Gamma)
        .def_readwrite("Sigma", &GLLiMParametersArma<TGamma, TSigma>::Sigma);

    std::string GLLiM_pyname = std::string("_GLLiM") + str;
    py::class_<GLLiM<TGamma, TSigma>, std::shared_ptr<GLLiM<TGamma, TSigma>>>(m, GLLiM_pyname.c_str()) // exposed to Pybind11 and hide in Python with the underscore
        .def(py::init<unsigned, unsigned, unsigned, std::string, std::string>(), py::arg("L"), py::arg("D"), py::arg("K"), py::arg("gamma_type"), py::arg("sigma_type"))

        .def("getDimensions", &GLLiM<TGamma, TSigma>::getDimensions, "some info :)")
        .def("getConstraints", &GLLiM<TGamma, TSigma>::getConstraints)
        .def("getParams", &GLLiM<TGamma, TSigma>::getParamsArma)
        .def("getParamPi", &GLLiM<TGamma, TSigma>::getParamPi)
        .def("getParamA", &GLLiM<TGamma, TSigma>::getParamA)
        .def("getParamB", &GLLiM<TGamma, TSigma>::getParamB)
        .def("getParamC", &GLLiM<TGamma, TSigma>::getParamC)
        .def("getParamGamma", &GLLiM<TGamma, TSigma>::getParamGammaArma)
        .def("getParamSigma", &GLLiM<TGamma, TSigma>::getParamSigmaArma)

        .def("setParams", &GLLiM<TGamma, TSigma>::setParamsArma)
        .def("setParamPi", &GLLiM<TGamma, TSigma>::setParamPi)
        .def("setParamA", &GLLiM<TGamma, TSigma>::setParamA)
        .def("setParamB", &GLLiM<TGamma, TSigma>::setParamB)
        .def("setParamC", &GLLiM<TGamma, TSigma>::setParamC)
        .def("setParamGamma", &GLLiM<TGamma, TSigma>::setParamGammaArma)
        .def("setParamSigma", &GLLiM<TGamma, TSigma>::setParamSigmaArma)

        .def("getInverse", &GLLiM<TGamma, TSigma>::getInverseArma)

        .def("directDensities", py::overload_cast<const mat &, const vec &>(&GLLiM<TGamma, TSigma>::directDensities))
        .def("directDensities", py::overload_cast<const mat &>(&GLLiM<TGamma, TSigma>::directDensities))
        .def("inverseDensities", py::overload_cast<const mat &, const mat &>(&GLLiM<TGamma, TSigma>::inverseDensities))
        .def("inverseDensities", py::overload_cast<const mat &>(&GLLiM<TGamma, TSigma>::inverseDensities))

        .def("initialize", &GLLiM<TGamma, TSigma>::initialize)
        .def("train", &GLLiM<TGamma, TSigma>::train)

        //    // Do we need to expose this class ?
        // py::class_<JGMM, std::shared_ptr<JGMM>>(m, "JGMM")
        //     .def(py::init<>())
        //     .def("train", &JGMM::train)
        //     .def("getPosterior", &JGMM::getPosterior)

        .doc() = R"mydelimiter(
            GLLiM class
        )mydelimiter";
}
