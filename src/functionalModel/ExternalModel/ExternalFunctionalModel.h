/**
 * @file ExternalFunctionalModel.h
 * @author Sami DJOUADI
 * @version 1.2
 * @date 22/04/2020
 */

#ifndef KERNELO_EXTERNALFUNCTIONALMODEL_H
#define KERNELO_EXTERNALFUNCTIONALMODEL_H

#include "../FunctionalModel.h"
#include <Python.h>
#include "pyhelper.hpp"

namespace Functional {
    /**
     * @class ExternalFunctionalModel
     * @details This class allows to extend the library dynamically with functional models written with Python. This avoid
     * to rebuild the library for new functional models. This use is meant to make protoypes of new functional models and
     * test them before coding them with C++.
     *
     */
    class ExternalFunctionalModel: public FunctionalModel{
    public:

        /**
         * @brief Constructor
         * @param className : The name of the concrete class that defines the required external model.
         * @param fileName : The name of the file where the source code of the external model is written.
         * @param filePath : The path to the file where the source code of the external model is written.
         */
        ExternalFunctionalModel(const std::string &className, const std::string &fileName, const std::string &filePath);

        void F(rowvec x, rowvec &y) override;
        int get_D_dimension() override;
        int get_L_dimension() override;
        void to_physic(rowvec &x) override;
        void to_physic(double *x, unsigned int size) override;
        void from_physic(double *x, unsigned int size) override;

    private:
        CPyObject pModule; /**< Is a reference to the module containing the class describing the functional model in Python code. */
        CPyObject py_obj; /**< Is a reference to an instantiation of the class describing the functional model in Python code. */
    };
}

#endif //KERNELO_EXTERNALFUNCTIONALMODEL_H
