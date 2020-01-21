//
// Created by reverse-proxy on 17‏/1‏/2020.
//

#ifndef KERNELO_DEPENDENTGAUSSIANSTATMODEL_H
#define KERNELO_DEPENDENTGAUSSIANSTATMODEL_H

#include "StatModel.h"
#include "GeneratorStrategy.h"
#include "../physicalModel/FunctionalModel.h"
#include <memory>

using namespace Functional;

namespace DataGeneration{
    class DependentGaussianStatModel : public StatModel{
    public:
        DependentGaussianStatModel(std::string generatorType, int r);
        std::tuple<mat, mat> gen_data(std::shared_ptr<FunctionalModel> functionnalModel, int n, unsigned seed) final;
        double density_X_Y(mat x, mat y) final;

    private:
        std::shared_ptr<GeneratorStrategy> generator;
        double r;
    };
}



#endif //KERNELO_DEPENDENTGAUSSIANSTATMODEL_H
