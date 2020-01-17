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
    this->covariance = rowvec(covariance, cov_size);
}

void GaussianStatModel::gen_data(std::shared_ptr<FunctionnalModel> functionnalModel, int n, double *x, double *y) {
    int dimension_D = functionnalModel->get_D_dimension();
    int dimension_L = functionnalModel->get_L_dimension();
    auto *y_temp = new double[dimension_D];

    // generate X
    generator->execute(n, dimension_L, x);

    // create a vector of random values under a normal distribution with 0 mean and 1 variance
    std::normal_distribution<double> normalDistribution(0, 1);
    std::mt19937_64 engine;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(seed & 0xffffffff), uint32_t(seed>>32)};
    engine.seed(ss);

    rowvec noise(dimension_D);

    // generate Y
    for(unsigned i=0; i<n; i++){
        // normalize vector to physical intervals
        //functionnalModel->to_physic(&x[i*dimension_L],dimension_L);

        // calucalte F(X)
        functionnalModel->F(&x[i*dimension_L],dimension_L,y_temp,dimension_D);

        // add noise
        for(unsigned j=0; j<dimension_D;j++){
            noise(j) = normalDistribution(engine);
            y[i*dimension_D+j] = y_temp[j] + noise(j) * sqrt(covariance(j));
        }
    }

    delete[] y_temp;
}

double GaussianStatModel::density_X_Y(mat x, mat y) {
    //try something
    return 0;
}




