// #include <pybind11/pybind11.h>
#include "functionalModel.cpp"
#include "gllim_binding.cpp"

PYBIND11_MODULE(newkernelo, m)
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
}
