/**
 * @file GaussianStatModel.cpp
 * @brief Gaussian statistical model class definition
 * @author Sami DJOUADI
 * @version 1.0
 * @date 13/01/2020
 */
#include "GaussianStatModel.h"
#include "GeneratorFactory.h"
#include <omp.h>

using namespace std;
using namespace DataGeneration;


GaussianStatModel::GaussianStatModel(
        const std::string& generatorType,
        std::shared_ptr<FunctionalModel> functionalModel,
        const double *covariance,
        int cov_size,
        unsigned seed) {
    this->generator = GeneratorFactory::create(generatorType);
    this->functionalModel = std::move(functionalModel);

    //Transform cov from double* to arma::rowvec
    this->covariance = rowvec(covariance, cov_size);
    this->seed = seed;
}


double GaussianStatModel::density_X_Y(mat x, mat y) {
    //try something
    return 0;
}

std::tuple<mat, mat> GaussianStatModel::gen_data(int n) {
    int dimension_D = functionalModel->get_D_dimension();
    int dimension_L = functionalModel->get_L_dimension();

    mat x_arma = mat(n,dimension_L);
    mat y_arma = mat(n,dimension_D);

    // generate X
    generator->execute(x_arma, seed);

    // create a vector of random values under a normal distribution with 0 mean and 1 variance
    std::normal_distribution<double> normalDistribution(0, 1);
    std::mt19937_64 engine;
    engine.seed(seed);

    // generate Y
    rowvec noise(dimension_D);
    rowvec y_temp(dimension_D);

    #pragma omp parallel for
    for(unsigned i=0; i<n; i++){
        // calculate F(X)
        functionalModel->F(x_arma.row(i),y_temp);

        // add noise
        for(unsigned j=0; j<dimension_D;j++){
            noise(j) = normalDistribution(engine);
            y_arma(i,j) += noise(j) * sqrt(covariance(j));
        }
    }

    return std::tuple<mat, mat>(x_arma,y_arma);
}




