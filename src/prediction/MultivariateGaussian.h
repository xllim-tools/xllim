/**
 * @file MultivariateGaussian.h
 * @author Sami DJOUADI
 * @version 1.2
 * @date 26/03/2020
 */

#ifndef KERNELO_MULTIVARIATEGAUSSIAN_H
#define KERNELO_MULTIVARIATEGAUSSIAN_H

#include "armadillo"

using namespace arma;

namespace prediction{
    /**
     * @struct MultivariateGaussian
     */
    struct MultivariateGaussian{
        double weight; /**< weight of the gaussian distribution in the GMM where it belongs*/
        vec mean; /**< the mean of the gaussian distribution */
        mat covariance; /** the covariance matrix of gaussian distribution */

        /*bool operator<(const MultivariateGaussian &g) const{
            return weight < g.weight;
        }*/
    };
}

#endif //KERNELO_MULTIVARIATEGAUSSIAN_H
