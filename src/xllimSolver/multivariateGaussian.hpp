#ifndef MULTIVARIATEGAUSSIAN_HPP
#define MULTIVARIATEGAUSSIAN_HPP

#include "armadillo"

using namespace arma;

/**
 * @struct MultivariateGaussian
 */
class MultivariateGaussian
{
public:
    double weight;  // weight of the gaussian distribution in the GMM where it belongs
    vec mean;       // the mean of the gaussian distribution
    mat covariance; // the covariance matrix of gaussian distribution
};

#endif // MULTIVARIATEGAUSSIAN_HPP
