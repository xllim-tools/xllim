#ifndef XLLIM_EXTERNALPYTHONMODEL_H
#define XLLIM_EXTERNALPYTHONMODEL_H
#pragma GCC diagnostic ignored "-Wattributes" // Do not display (harmless) warning on this file. py::module_ and py::object
// Warning: ‘Functional::ExternalPythonModel’ declared with greater visibility than the type of its field ‘Functional::ExternalPythonModel::pModule’ [-Wattributes]

#include "FunctionalModel.hpp"
#include <pybind11/pybind11.h>
// #include <pybind11/embed.h>

namespace py = pybind11;
using namespace py::literals;

/**
 * @class ExternalPythonModel
 * @details This class allows to extend the library dynamically with functional models written with Python. This avoid
 * to rebuild the library for new functional models. This use is meant to make protoypes of new functional models and
 * test them before coding them with C++.
 */
class ExternalPythonModel : public FunctionalModel
{
public:
    /**
     * @brief Constructor
     * @param className : The name of the concrete class that defines the required external model.
     * @param fileName : The name of the file where the source code of the external model is written.
     * @param filePath : The path to the file where the source code of the external model is written.
     */
    ExternalPythonModel(const std::string &className, const std::string &fileName, const std::string &filePath);
    void F(vec photometry, vec &reflectances) final;
    unsigned getDimensionY() final;
    unsigned getDimensionX() final;
    void toPhysic(vec &x) final;
    void fromPhysic(vec &x) final;

private:
    py::module_ pModule_;
    py::object pObj_;
};

#endif // XLLIM_EXTERNALPYTHONMODEL_H
