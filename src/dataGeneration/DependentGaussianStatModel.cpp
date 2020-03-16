/**
 * @file DependentGaussianStatModel.h
 * @brief Class implementation of a gaussian statistical model where noise depends on generated X.
 * @author Sami DJOUADI
 * @version 1.0
 * @date 17/01/2020
 */

#include "DependentGaussianStatModel.h"
#include "GeneratorFactory.h"
#include <omp.h>

#include <utility>


using namespace DataGeneration;

DependentGaussianStatModel::DependentGaussianStatModel(
        const std::string& generatorType,
        std::shared_ptr<FunctionalModel> functionalModel,
        int r,
        unsigned seed) {
    this->generator = GeneratorFactory::create(generatorType);
    this->functionalModel = std::move(functionalModel);
    this->r = r;
    this->seed = seed;
}

std::tuple<mat, mat> DependentGaussianStatModel::gen_data(int n) {
    int dimension_D = functionalModel->get_D_dimension();
    int dimension_L = functionalModel->get_L_dimension();

    mat x_arma = mat(n,dimension_L);
    mat y_arma = mat(n,dimension_D);


    // generate X
    generator->execute(x_arma);

    // create a vector of random values under a normal distribution with 0 mean and 1 variance
    std::normal_distribution<double> normalDistribution(0, 1);
    std::mt19937_64 engine;
    engine.seed(seed);

    rowvec noise(dimension_D);

    // generate Y
    rowvec y_temp(dimension_D);

    #pragma omp parallel for
    for(unsigned i=0; i<n; i++){
        // calculate F(X)
        functionalModel->F(x_arma.row(i),y_temp);

        // add noise
        for(unsigned j=0; j<dimension_D;j++){
            noise(j) = normalDistribution(engine);
            y_arma(i,j) = y_temp(j) + noise(j) * sqrt(y_temp(j)/r);
        }
    }

    return std::tuple<mat, mat>(x_arma,y_arma);
}

double DependentGaussianStatModel::density_X_Y(mat x, mat y) {
    return 0;
}
