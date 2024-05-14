#ifndef UTILS_HPP
#define UTILS_HPP

#include <armadillo>
#include <omp.h>
using namespace arma;

namespace utils
{
    double logSumExp(const vec &x);
    vec logSumExp(const mat &x, const int axis);
    double weightedLogSumExp(const double &log_p1, const double &log_p2, const unsigned &c1, const unsigned &c2);
    double logDensity(const vec &x, const vec &mean, const mat &covariance);
    double logDensity(const vec &x, const vec &weight, const mat &mean, const cube &covariance);
    mat logDensity(const mat &x, const rowvec &weight, const mat &mean, const cube &covariance);
    void inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat);
    double mvnrm_arma_fast_chol(arma::rowvec const &x,arma::rowvec const &mean, arma::mat &chol, bool const logd = true);
    vec dmvnrm_arma_fast_chol(arma::mat const &x, arma::rowvec const &mean, arma::mat &chol, bool const logd = true);
    mat safe_cholesky(mat &Sigma);
    // arma::vec Mahalanobis(arma::mat const &x, arma::vec const &center, arma::mat const &cov);
}

#endif // UTILS_HPP
