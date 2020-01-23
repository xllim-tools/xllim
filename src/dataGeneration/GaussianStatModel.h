//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#ifndef KERNELO_GAUSSSTATMODEL_H
#define KERNELO_GAUSSSTATMODEL_H

#include "StatModel.h"
#include "GeneratorStrategy.h"
#include "../physicalModel/FunctionalModel.h"
#include <memory>

using namespace Functional;

namespace DataGeneration{
    class GaussianStatModel : public StatModel{
    public :
        GaussianStatModel(std::string generatorType, const double *covariance, int cov_size, unsigned seed);
        std::tuple<mat, mat> gen_data(std::shared_ptr<FunctionalModel> functionalModel, int n) final;
        double density_X_Y(mat x, mat y) final;

    private:
        std::shared_ptr<GeneratorStrategy> generator;
        rowvec covariance;
        unsigned seed;
    };

}



#endif //KERNELO_GAUSSSTATMODEL_H
