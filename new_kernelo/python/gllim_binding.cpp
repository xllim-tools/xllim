#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
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

    py::class_<InitialisationInsights>(m, "InitialisationInsights")
        .def_readwrite("time", &InitialisationInsights::time)
        .def_readwrite("start_time", &InitialisationInsights::start_time)
        .def_readwrite("end_time", &InitialisationInsights::end_time)
        .def_readwrite("N_obs", &InitialisationInsights::N_obs)
        .def_readwrite("gllim_em_iteration", &InitialisationInsights::gllim_em_iteration)
        .def_readwrite("gllim_em_floor", &InitialisationInsights::gllim_em_floor)
        .def_readwrite("gmm_kmeans_iteration", &InitialisationInsights::gmm_kmeans_iteration)
        .def_readwrite("gmm_em_iteration", &InitialisationInsights::gmm_em_iteration)
        .def_readwrite("gmm_floor", &InitialisationInsights::gmm_floor)
        .def_readwrite("nb_experiences", &InitialisationInsights::nb_experiences);

    py::class_<TrainingInsights>(m, "TrainingInsights")
        .def_readwrite("time", &TrainingInsights::time)
        .def_readwrite("start_time", &TrainingInsights::start_time)
        .def_readwrite("end_time", &TrainingInsights::end_time)
        .def_readwrite("N_obs", &TrainingInsights::N_obs)
        .def_readwrite("max_iteration", &TrainingInsights::max_iteration)
        .def_readwrite("ratio_ll", &TrainingInsights::ratio_ll)
        .def_readwrite("floor", &TrainingInsights::floor);

    py::class_<Insights>(m, "Insights")
        .def_readwrite("time", &Insights::time)
        .def_readwrite("log_likelihood", &Insights::log_likelihood)
        .def_readwrite("initialisation", &Insights::initialisation)
        .def_readwrite("training", &Insights::training);

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

        .def("getInsights", &GLLiM<TGamma, TSigma>::getInsights)

        .def("directDensities", py::overload_cast<const mat &, const vec &, int>(&GLLiM<TGamma, TSigma>::directDensities), py::arg("x"), py::arg("x_incertitude"), py::arg("verbose") = 1)
        .def("directDensities", py::overload_cast<const mat &, int>(&GLLiM<TGamma, TSigma>::directDensities), py::arg("x"), py::arg("verbose") = 1)
        .def("inverseDensities", py::overload_cast<const mat &, const mat &, int>(&GLLiM<TGamma, TSigma>::inverseDensities), py::arg("y"), py::arg("y_incertitude"), py::arg("verbose") = 1)
        .def("inverseDensities", py::overload_cast<const mat &, int>(&GLLiM<TGamma, TSigma>::inverseDensities), py::arg("y"), py::arg("verbose") = 1)

        .def("initialize", &GLLiM<TGamma, TSigma>::initialize)
        .def("train", &GLLiM<TGamma, TSigma>::train)

        .doc() = R"mydelimiter(
            GLLiM class
        )mydelimiter";
}
