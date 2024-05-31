#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <carma>
#include <armadillo>

#include "../src/xllimSolver/gllim.hpp"

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

    py::class_<GLLiM, std::shared_ptr<GLLiM>>(m, "GLLiM")
        .def(py::init<unsigned, unsigned, unsigned>(), py::arg("L"), py::arg("D"), py::arg("K"))
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

        .def("directDensities", &GLLiM::directDensities)

        .doc() = R"mydelimiter(
            GLLiM class
        )mydelimiter";
}
