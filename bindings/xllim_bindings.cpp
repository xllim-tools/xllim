// #include <pybind11/pybind11.h>
#include "functionalModel_bindings.cpp"
#include "gllim_bindings.cpp"
#include "../src/utils/utils.hpp"

PYBIND11_MODULE(_core, m)
{
    // Define the Python module and initialize function
    m.doc() = "Your module documentation";
    bind_functional_model(m);
    bind_gllim_templates<FullCovariance, FullCovariance>(m, "_full_full");
    bind_gllim_templates<FullCovariance, DiagCovariance>(m, "_full_diag");
    bind_gllim_templates<FullCovariance, IsoCovariance>(m, "_full_iso");
    bind_gllim_templates<DiagCovariance, FullCovariance>(m, "_diag_full");
    bind_gllim_templates<DiagCovariance, DiagCovariance>(m, "_diag_siag");
    bind_gllim_templates<DiagCovariance, IsoCovariance>(m, "_diag_iso");
    bind_gllim_templates<IsoCovariance, FullCovariance>(m, "_iso_full");
    bind_gllim_templates<IsoCovariance, DiagCovariance>(m, "_iso_diag");
    bind_gllim_templates<IsoCovariance, IsoCovariance>(m, "_iso_iso");
    bind_gllim(m);

    // TODO add a miscellaneous file/namespace (different from utils) and a miscellaneous_bindings.cpp
    m.def("regularize", &utils::regularize,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
}
