//
// Created by reverse-proxy on 17‏/1‏/2020.
//

#ifndef KERNELO_DEPENDENTGAUSSIANSTATMODEL_H
#define KERNELO_DEPENDENTGAUSSIANSTATMODEL_H

#include "StatModel.h"
#include "GeneratorStrategy.h"
#include "../physicalModel/FunctionnalModel.h"
#include <memory>

namespace DataGeneration{
    class DependentGaussianStatModel : public StatModel{
    public:
        DependentGaussianStatModel(std::string generatorType, int r);
        void gen_data(std::shared_ptr<FunctionnalModel> functionnalModel, int n, double *x, double *y) final;
        double density_X_Y(mat x, mat y) final;

    private:
        std::shared_ptr<GeneratorStrategy> generator;
        double r;
    };
}



#endif //KERNELO_DEPENDENTGAUSSIANSTATMODEL_H
