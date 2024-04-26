#ifndef UTILS_HPP
#define UTILS_HPP

#include <armadillo>
using namespace arma;

namespace utils
{
    double logSumExp(const vec &elements);
    vec logSumExp(const mat &x, const int axis);
    double weightedLogSumExp(const double &log_p1, const double &log_p2, const unsigned &c1, const unsigned &c2);
    double logDensity(const vec &x, const vec &mean, const mat &covariance);
    double logDensity(const vec &x, const vec &weight, const mat &mean, const cube &covariance);
    mat logDensity(const mat &x, const vec &weight, const mat &mean, const cube &covariance);
    // arma::vec Mahalanobis(arma::mat const &x, arma::vec const &center, arma::mat const &cov);
}

#endif // UTILS_HPP
