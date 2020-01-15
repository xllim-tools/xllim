//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#ifndef KERNELO_GAUSSSTATMODEL_H
#define KERNELO_GAUSSSTATMODEL_H

#include "StatModel.h"
#include "GeneratorStrategy.h"
#include "../physicalModel/FunctionnalModel.h"
#include <memory>

namespace DataGeneration{
    class GaussianStatModel : StatModel{
    public :
        GaussianStatModel(std::string generatorType, const double *covariance, int cov_size);
        void gen_data(FunctionnalModel &functionnalModel, int n, double *x, double *y) override;
        double density_X_Y(mat x, mat y) override;

    private:
        std::shared_ptr<GeneratorStrategy> generator;
        rowvec covariance;
    };

}



#endif //KERNELO_GAUSSSTATMODEL_H
