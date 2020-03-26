//
// Created by reverse-proxy on 26‏/3‏/2020.
//

#ifndef KERNELO_MULTIVARIATEGAUSSIAN_H
#define KERNELO_MULTIVARIATEGAUSSIAN_H

#include "armadillo"

using namespace arma;

namespace prediction{
    struct MultivariateGaussian{
        double weight;
        vec mean;
        mat covariance;

        /*bool operator<(const MultivariateGaussian &g) const{
            return weight < g.weight;
        }*/
    };

    bool compareByWeight(const MultivariateGaussian &g1, const MultivariateGaussian &g2){
        return g1.weight > g2.weight;
    }
}

#endif //KERNELO_MULTIVARIATEGAUSSIAN_H
