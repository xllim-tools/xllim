//
// Created by reverse-proxy on 6‏/1‏/2020.
//

#ifndef KERNELO_DATAGENERATOR_H
#define KERNELO_DATAGENERATOR_H

#include <armadillo>
#include "../physicalModel/FunctionnalModel.h"

using namespace arma;

namespace DataGeneration{
    class StatModel{
    public:
        virtual void gen_data(std::shared_ptr<FunctionnalModel> functionnalModel, int n, double *x, double *y) = 0;
        virtual double density_X_Y(mat x, mat y) = 0;
    };
}

#endif //KERNELO_DATAGENERATOR_H
