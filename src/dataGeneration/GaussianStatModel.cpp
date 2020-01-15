//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#include "GaussianStatModel.h"

#include <utility>
#include "GeneratorFactory.h"

using namespace DataGeneration;


GaussianStatModel::GaussianStatModel(std::string generatorType, const double *covariance, int cov_size) {
    generator = GeneratorFactory::create(std::move(generatorType));

    //Transform cov from double* to arma::rowvec
    this->covariance = rowvec(cov_size);
    for(unsigned j=0; j<cov_size; j++){
        this->covariance(j) = covariance[j];
    }
}

void GaussianStatModel::gen_data(FunctionnalModel &functionnalModel, int n, double *x, double *y) {
    int dimension_D = functionnalModel.get_D_dimension();
    int dimension_L = functionnalModel.get_L_dimension();
    rowvec y_temp = rowvec(dimension_D);

    //Genrate x
    generator->execute(n, dimension_L, x);

    // create a vector of random values under a normal distribution with 0 mean and 1 variance
    rowvec noise = randn<rowvec>(dimension_D);
    noise  = noise % sqrt(covariance);

    //Generate Y
    for(unsigned i=0; i<n; i++){
        //functionnalModel.F(x,y_temp);
        for(unsigned j=0; j<dimension_D;j++){
            y[i*dimension_D+j] = y_temp(j) + noise(j);
        }
    }
}

double GaussianStatModel::density_X_Y(mat x, mat y) {
    return 0;
}




