#include <pybind11/pybind11.h>
#include "functionalModel.cpp"
#include "dataGeneration.cpp"
#include "gllim_binding.cpp"

PYBIND11_MODULE(newkernelo, m) {
    // Define the Python module and initialize function
    m.doc() = "Your module documentation";
    bind_functional_model(m);
    bind_data_generation(m);
    bind_gllim(m);
}