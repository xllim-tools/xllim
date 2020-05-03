//
// Created by reverse-proxy on 22‏/4‏/2020.
//

#ifndef KERNELO_EXTERNALFUNCTIONALMODEL_H
#define KERNELO_EXTERNALFUNCTIONALMODEL_H

#include "../FunctionalModel.h"
#include <Python.h>
#include "pyhelper.hpp"

namespace Functional {
    class ExternalFunctionalModel: public FunctionalModel{
    public:

        ExternalFunctionalModel(const std::string &className, const std::string &fileName, const std::string &filePath);
        void F(rowvec x, rowvec &y) override;
        int get_D_dimension() override;
        int get_L_dimension() override;
        void to_physic(rowvec &x) override;
        void from_physic(double *x, int size) override;

    private:
        //CPyInstance pyInstance;
        CPyObject pModule;
        CPyObject py_obj;
    };
}

#endif //KERNELO_EXTERNALFUNCTIONALMODEL_H
