//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#include "GaussianStatModel.h"

#include <utility>
#include <chrono>
#include "GeneratorFactory.h"

using namespace std;
using namespace DataGeneration;


GaussianStatModel::GaussianStatModel(std::string generatorType, const double *covariance, int cov_size) {
    generator = GeneratorFactory::create(std::move(generatorType));

    //Transform cov from double* to arma::rowvec
    this->covariance = rowvec(covariance, cov_size);
}


double GaussianStatModel::density_X_Y(mat x, mat y) {
    //try something
    return 0;
}

std::tuple<mat, mat> GaussianStatModel::gen_data(std::shared_ptr<FunctionalModel> functionalModel, int n, unsigned seed) {
    mat x_arma = mat(n,functionalModel->get_L_dimension());
    mat y_arma = mat(n,functionalModel->get_D_dimension());
    int dimension_D = functionalModel->get_D_dimension();
    int dimension_L = functionalModel->get_L_dimension();

    // generate X
    //auto start1 = chrono::high_resolution_clock::now();

    generator->execute(x_arma, seed);
    //auto end1 = chrono::high_resolution_clock::now();
    // create a vector of random values under a normal distribution with 0 mean and 1 variance
    std::normal_distribution<double> normalDistribution(0, 1);
    std::mt19937_64 engine;

    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //std::seed_seq ss{uint32_t(seed & 0xffffffff), uint32_t(seed>>32)};
    engine.seed(seed);

    // generate Y
    rowvec noise(dimension_D);
    rowvec y_temp(dimension_D);

    //auto start2 = chrono::high_resolution_clock::now();

    for(unsigned i=0; i<n; i++){
        // calculate F(X)
        functionalModel->F(x_arma.row(i),y_temp);

        // add noise
        for(unsigned j=0; j<dimension_D;j++){
            noise(j) = normalDistribution(engine);
            y_arma(i,j) += noise(j) * sqrt(covariance(j));
        }
    }

    //auto end2 = chrono::high_resolution_clock::now();

    //cout << chrono::duration_cast<chrono::microseconds>(end2-start1).count() << endl;
    //cout << chrono::duration_cast<chrono::microseconds>(end2-start2).count() << endl;
    //cout << chrono::duration_cast<chrono::microseconds>(end1-start1).count() << endl;

    return std::tuple<mat, mat>(x_arma,y_arma);
}




