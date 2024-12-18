#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <carma>
#include <armadillo>

// #include "../src/functionalModel/FunctionalModel.hpp"
#include "../src/functionalModel/TestModel.hpp"
#include "../src/functionalModel/ShkuratovModel.hpp"
#include "../src/functionalModel/HapkeModel.hpp"
#include "../src/functionalModel/ExternalPythonModel.hpp"

namespace py = pybind11;

// For some reason the automatic conversion between list<tuple(ndarray, ndarray, ndarray) and std::vector<std::tuple<vec, mat, cube>>
// fails for K <= L. That is whay this explicit conversion function is needed at the moment (pybind11 2.11.1 carma 0.6.7)
std::vector<std::tuple<vec, mat, cube>> parse_proposition_gmms(const py::list &proposition_gmms_py)
{
    std::vector<std::tuple<vec, mat, cube>> proposition_gmms;
    for (size_t i = 0; i < proposition_gmms_py.size(); ++i)
    {
        auto item = proposition_gmms_py[i].cast<py::tuple>();
        // Cast each component of the tuple to the correct Armadillo type
        vec weights = carma::arr_to_col(item[0].cast<py::array_t<double>>());
        mat means = carma::arr_to_mat(item[1].cast<py::array_t<double>>());
        cube covs = carma::arr_to_cube(item[2].cast<py::array_t<double>>());

        proposition_gmms.emplace_back(weights, means, covs);
    }
    return proposition_gmms;
}

void bind_functional_model(pybind11::module &m)
{
    // PYBIND11_NUMPY_DTYPE(Dummy, predictions, predictions_variance);
    py::class_<ImportanceSamplingResult>(m, "ImportanceSamplingResult")
        .def(py::init<unsigned, unsigned>())
        .def_readwrite("predictions", &ImportanceSamplingResult::predictions)
        .def_readwrite("predictions_variance", &ImportanceSamplingResult::predictions_variance)
        .def_readwrite("nb_effective_sample", &ImportanceSamplingResult::nb_effective_sample)
        .def_readwrite("effective_sample_size", &ImportanceSamplingResult::effective_sample_size)
        .def_readwrite("qn", &ImportanceSamplingResult::qn);

    py::class_<FunctionalModel, std::shared_ptr<FunctionalModel>>(m, "FunctionalModel")
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
                return y_arr; })
        .def("genData", py::overload_cast<unsigned, const std::string &, double, unsigned>(&FunctionalModel::genData),
             py::arg("N"), py::arg("generator_type"), py::arg("noise_ratio"), py::arg("seed") = 0,
             py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("genData", py::overload_cast<unsigned, const std::string &, vec &, unsigned>(&FunctionalModel::genData),
             py::arg("N"), py::arg("generator_type"), py::arg("covariance"), py::arg("seed") = 0,
             py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

        // ! The implementation in pybind11/iostream.h is NOT thread safe. Multiple threads writing to a redirected ostream concurrently cause data races and potentially buffer overflows.
        .def("importanceSampling", [](FunctionalModel &self, py::list proposition_gmms_py, const mat &y, const mat &y_err, const vec &covariance, unsigned N_0, unsigned B, unsigned J, int verbose, unsigned seed)
            {
                auto proposition_gmms = parse_proposition_gmms(proposition_gmms_py);
                return self.importanceSampling(proposition_gmms, y, y_err, covariance, N_0, B, J, verbose, seed); }, py::arg("proposition_gmms"), py::arg("y"), py::arg("y_err"), py::arg("covariance"), py::arg("N_0"), py::arg("B") = 0, py::arg("J") = 0, py::arg("verbose") = 1, py::arg("seed") = 0, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        
        .def("importanceSampling", py::overload_cast<FullGMMResult, const mat, const mat, const vec, const unsigned, const unsigned, const unsigned, int, unsigned>(&FunctionalModel::importanceSampling), py::arg("predictions"), py::arg("y"), py::arg("y_err"), py::arg("covariance"), py::arg("N_0"), py::arg("B") = 0, py::arg("J") = 0, py::arg("verbose") = 1, py::arg("seed") = 0, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        
        .def("importanceSampling", py::overload_cast<MergedGMMResult, const mat, const mat, const vec, const unsigned, const unsigned, const unsigned, int, unsigned>(&FunctionalModel::importanceSampling), py::arg("predictions"), py::arg("y"), py::arg("y_err"), py::arg("covariance"), py::arg("N_0"), py::arg("B") = 0, py::arg("J") = 0, py::arg("verbose") = 1, py::arg("seed") = 0, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        
        .def("importanceSampling", py::overload_cast<MergedGMMResult, unsigned, const mat, const mat, const vec, const unsigned, const unsigned, const unsigned, int, unsigned>(&FunctionalModel::importanceSampling), py::arg("predictions"), py::arg("idx_gaussian"), py::arg("y"), py::arg("y_err"), py::arg("covariance"), py::arg("N_0"), py::arg("B") = 0, py::arg("J") = 0, py::arg("verbose") = 1, py::arg("seed") = 0, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    py::class_<TestModel, std::shared_ptr<TestModel>, FunctionalModel>(m, "TestModel")
        .def(py::init<>());

    py::class_<ShkuratovModel, std::shared_ptr<ShkuratovModel>, FunctionalModel>(m, "ShkuratovModel")
        .def(py::init<mat, std::string, vec, vec>(), py::arg("geometries"), py::arg("variant"), py::arg("scaling_coeffs"), py::arg("offset"));

    py::class_<HapkeModel, std::shared_ptr<HapkeModel>, FunctionalModel>(m, "HapkeModel")
        .def(py::init<mat, std::string, std::string, double, double, double>(), py::arg("geometries"), py::arg("variant"), py::arg("adapter"), py::arg("theta_bar_scaling"), py::arg("b0"), py::arg("h"));

    py::class_<ExternalPythonModel, std::shared_ptr<ExternalPythonModel>, FunctionalModel>(m, "ExternalPythonModel")
        .def(py::init<std::string, std::string, std::string>(), py::arg("className"), py::arg("fileName"), py::arg("filePath"));
}
