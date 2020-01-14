//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#include "GaussianStatModel.h"

#include <utility>
#include <random>
#include "GeneratorFactory.h"

using namespace DataGeneration;

GaussianStatModel::GaussianStatModel(std::string generatorType, double variance) {
    generator = GeneratorFactory::create(std::move(generatorType));
    this->variance = variance;
}

void GaussianStatModel::gen_data(int n, FunctionnalModel &functionnalModel, mat x, mat y) {

    x = mat(n,functionnalModel.get_L_dimension());
    rowvec y_temp = rowvec(functionnalModel.get_D_dimension());

    generator->execute(n, 0, x);

    for(unsigned i=0; i<n; i++){
        //functionalModel.F(x.row(i),y_temp)
        y.row(i) = y_temp;
    }

    // Add noise to Y

    //default random engine
    std::default_random_engine generator;

    // create normal distribution with 0 mean and "variance" variance
    std::normal_distribution<double> normal_dist(0.0,variance);

    // generate epsilon

}

double GaussianStatModel::density_X_Y(mat x, mat y) {
    return 0;
}




